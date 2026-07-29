"""Leakage-safe demographic environments for journal evaluation protocols."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def assert_disjoint_subjects(
    train_subject_ids: Sequence[object],
    *other_split_subject_ids: Sequence[object],
) -> None:
    """Raise when any subject occurs in more than one supplied split."""
    splits = (train_subject_ids,) + other_split_subject_ids
    subject_sets = [
        set(np.asarray(subject_ids, dtype=object).reshape(-1).tolist())
        for subject_ids in splits
    ]
    for first_index in range(len(subject_sets)):
        for second_index in range(first_index + 1, len(subject_sets)):
            overlap = subject_sets[first_index].intersection(
                subject_sets[second_index]
            )
            if overlap:
                preview = sorted(map(str, overlap))[:5]
                raise ValueError(
                    "Subject leakage between splits {} and {}: {}".format(
                        first_index, second_index, preview
                    )
                )


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("Number of bins must be at least one")
    if not np.all(np.isfinite(values)):
        raise ValueError("Demographic values must be finite")
    if n_bins == 1:
        return np.empty(0, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, quantiles)).astype(float)


def _normalise_sex(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "unknown"
        text = str(int(value)) if float(value).is_integer() else str(value)
    else:
        text = str(value).strip().lower()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
    text = text.strip().lower()
    aliases = {
        "0": "female",
        "f": "female",
        "female": "female",
        "woman": "female",
        "1": "male",
        "m": "male",
        "male": "male",
        "man": "male",
        "2": "female",
    }
    return aliases.get(text, text or "unknown")


@dataclass
class DemographicEnvironmentBuilder:
    """Fit demographic bin boundaries on training data and reuse them safely.

    Default ``mode='age_sex'`` keeps groups denser for GroupDRO.  When
    ``mode='age_sex_education'``, a training group smaller than
    ``min_group_size`` (or failing ``min_per_class``) falls back to
    ``age x sex``, then to ``age``.
    """

    n_age_bins: int = 3
    n_education_bins: int = 3
    min_group_size: int = 5
    min_per_class: int = 1
    mode: str = "age_sex"
    labels: Optional[Sequence[int]] = None

    def fit(
        self,
        age: Sequence[float],
        sex: Sequence[object],
        education: Sequence[float],
        subject_ids: Optional[Sequence[object]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> "DemographicEnvironmentBuilder":
        ages, sexes, educations = self._validate_inputs(
            age, sex, education, subject_ids
        )
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be at least one")
        if self.mode not in {"age_sex", "age_sex_education"}:
            raise ValueError("mode must be 'age_sex' or 'age_sex_education'")
        label_array = (
            None
            if labels is None
            else np.asarray(labels, dtype=int).reshape(-1)
        )
        if label_array is not None and len(label_array) != len(ages):
            raise ValueError("labels must match demographic input length")

        self.age_edges_ = _quantile_edges(ages, self.n_age_bins)
        self.education_edges_ = _quantile_edges(
            educations, self.n_education_bins
        )
        age_bins = np.digitize(ages, self.age_edges_, right=False)
        education_bins = np.digitize(
            educations, self.education_edges_, right=False
        )

        detailed = [
            ("age_sex_education", int(a), s, int(e))
            for a, s, e in zip(age_bins, sexes, education_bins)
        ]
        age_sex = [("age_sex", int(a), s) for a, s in zip(age_bins, sexes)]
        self._detailed_groups_ = (
            self._frequent_keys(detailed, label_array)
            if self.mode == "age_sex_education"
            else set()
        )
        self._age_sex_groups_ = self._frequent_keys(age_sex, label_array)

        selected = [
            self._select_key(detailed_key, age_sex_key)
            for detailed_key, age_sex_key in zip(detailed, age_sex)
        ]
        age_fallbacks = {
            ("age", age_bin) for age_bin in range(len(self.age_edges_) + 1)
        }
        unique_keys = sorted(set(selected).union(age_fallbacks), key=repr)
        self.environment_to_id_ = {
            key: index for index, key in enumerate(unique_keys)
        }
        self.environment_names_ = {
            index: "|".join(map(str, key))
            for key, index in self.environment_to_id_.items()
        }
        self.train_subject_ids_ = (
            None
            if subject_ids is None
            else frozenset(np.asarray(subject_ids, dtype=object).tolist())
        )
        self.is_fitted_ = True
        return self

    def transform(
        self,
        age: Sequence[float],
        sex: Sequence[object],
        education: Sequence[float],
        subject_ids: Optional[Sequence[object]] = None,
        validate_no_train_overlap: bool = False,
    ) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("DemographicEnvironmentBuilder must be fitted")
        ages, sexes, educations = self._validate_inputs(
            age, sex, education, subject_ids
        )
        if validate_no_train_overlap:
            if subject_ids is None or self.train_subject_ids_ is None:
                raise ValueError(
                    "subject_ids are required in fit and transform for leakage validation"
                )
            overlap = self.train_subject_ids_.intersection(
                np.asarray(subject_ids, dtype=object).tolist()
            )
            if overlap:
                raise ValueError(
                    "Subject leakage from training split: {}".format(
                        sorted(map(str, overlap))[:5]
                    )
                )

        age_bins = np.digitize(ages, self.age_edges_, right=False)
        education_bins = np.digitize(
            educations, self.education_edges_, right=False
        )
        result = []
        for age_bin, sex_value, education_bin in zip(
            age_bins, sexes, education_bins
        ):
            detailed = (
                "age_sex_education",
                int(age_bin),
                sex_value,
                int(education_bin),
            )
            age_sex = ("age_sex", int(age_bin), sex_value)
            key = self._select_key(detailed, age_sex)
            # Every age bin present after digitize has a training age fallback.
            result.append(self.environment_to_id_[key])
        return np.asarray(result, dtype=np.int64)

    def fit_transform(
        self,
        age: Sequence[float],
        sex: Sequence[object],
        education: Sequence[float],
        subject_ids: Optional[Sequence[object]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        return self.fit(
            age, sex, education, subject_ids=subject_ids, labels=labels
        ).transform(age, sex, education)

    def to_dict(self) -> dict:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("DemographicEnvironmentBuilder must be fitted")

        def encode_key(key: Tuple) -> list:
            return [key[0], *[item if not isinstance(item, (np.integer,)) else int(item) for item in key[1:]]]

        return {
            "mode": self.mode,
            "n_age_bins": int(self.n_age_bins),
            "n_education_bins": int(self.n_education_bins),
            "min_group_size": int(self.min_group_size),
            "min_per_class": int(self.min_per_class),
            "age_edges": np.asarray(self.age_edges_, dtype=float).tolist(),
            "education_edges": np.asarray(self.education_edges_, dtype=float).tolist(),
            "detailed_groups": [encode_key(key) for key in sorted(self._detailed_groups_, key=repr)],
            "age_sex_groups": [encode_key(key) for key in sorted(self._age_sex_groups_, key=repr)],
            "environment_to_id": [
                [encode_key(key), int(index)]
                for key, index in sorted(
                    self.environment_to_id_.items(), key=lambda item: item[1]
                )
            ],
            "names": {
                str(index): name for index, name in self.environment_names_.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DemographicEnvironmentBuilder":
        instance = cls(
            n_age_bins=int(payload.get("n_age_bins", 3)),
            n_education_bins=int(payload.get("n_education_bins", 3)),
            min_group_size=int(payload.get("min_group_size", 5)),
            min_per_class=int(payload.get("min_per_class", 1)),
            mode=str(payload.get("mode", "age_sex")),
        )

        def decode_key(parts) -> Tuple:
            values = list(parts)
            head = values[0]
            rest = []
            for item in values[1:]:
                if isinstance(item, str) and item not in {"female", "male", "unknown"}:
                    try:
                        rest.append(int(item))
                        continue
                    except ValueError:
                        pass
                rest.append(item if not isinstance(item, (np.integer,)) else int(item))
            # age / age_sex bins are ints; sex strings stay strings
            if head in {"age", "age_sex", "age_sex_education"} and rest:
                rest[0] = int(rest[0])
            if head == "age_sex_education" and len(rest) >= 3:
                rest[-1] = int(rest[-1])
            return (head, *rest)

        instance.age_edges_ = np.asarray(payload["age_edges"], dtype=float)
        instance.education_edges_ = np.asarray(payload["education_edges"], dtype=float)
        instance._detailed_groups_ = {
            decode_key(key) for key in payload.get("detailed_groups", [])
        }
        instance._age_sex_groups_ = {
            decode_key(key) for key in payload.get("age_sex_groups", [])
        }
        mapping = payload.get("environment_to_id")
        if mapping:
            instance.environment_to_id_ = {
                decode_key(key): int(index) for key, index in mapping
            }
        else:
            # Legacy manifests only stored string names.
            names = payload.get("names", {})
            instance.environment_to_id_ = {}
            for index_text, name in names.items():
                parts = str(name).split("|")
                instance.environment_to_id_[decode_key(parts)] = int(index_text)
        instance.environment_names_ = {
            index: "|".join(map(str, key))
            for key, index in instance.environment_to_id_.items()
        }
        instance.is_fitted_ = True
        return instance

    def _select_key(self, detailed: Tuple, age_sex: Tuple) -> Tuple:
        if self.mode == "age_sex_education" and detailed in self._detailed_groups_:
            return detailed
        if age_sex in self._age_sex_groups_:
            return age_sex
        return ("age", age_sex[1])

    def _frequent_keys(
        self,
        keys: Iterable[Tuple],
        labels: Optional[np.ndarray] = None,
    ) -> set:
        counts = {}
        class_counts = {}
        for index, key in enumerate(keys):
            counts[key] = counts.get(key, 0) + 1
            if labels is not None:
                class_counts.setdefault(key, {})
                label = int(labels[index])
                class_counts[key][label] = class_counts[key].get(label, 0) + 1
        kept = set()
        for key, count in counts.items():
            if count < self.min_group_size:
                continue
            if labels is not None:
                per_class = class_counts.get(key, {})
                if len(per_class) < 2:
                    continue
                if min(per_class.values()) < self.min_per_class:
                    continue
            kept.add(key)
        return kept

    @staticmethod
    def _validate_inputs(age, sex, education, subject_ids):
        ages = np.asarray(age, dtype=float).reshape(-1)
        raw_sexes = np.asarray(sex, dtype=object).reshape(-1)
        educations = np.asarray(education, dtype=float).reshape(-1)
        if not (len(ages) == len(raw_sexes) == len(educations)):
            raise ValueError("age, sex, and education must have equal length")
        if len(ages) == 0:
            raise ValueError("At least one observation is required")
        if subject_ids is not None and len(subject_ids) != len(ages):
            raise ValueError("subject_ids must match demographic input length")
        if not np.all(np.isfinite(ages)) or not np.all(np.isfinite(educations)):
            raise ValueError("age and education must contain finite values")
        sexes = np.asarray([_normalise_sex(value) for value in raw_sexes])
        return ages, sexes, educations
