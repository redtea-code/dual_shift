"""Leakage-safe NIfTI dataset utilities for the journal experiments."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_FEMALE = {"0", "2", "f", "female", "woman"}
_MALE = {"1", "m", "male", "man"}


def _first_column(columns: Sequence[str], aliases: Sequence[str], required=True):
    by_lower = {str(column).lower(): column for column in columns}
    for alias in aliases:
        if str(alias).lower() in by_lower:
            return by_lower[str(alias).lower()]
    if required:
        raise ValueError(f"None of columns {list(aliases)} found in {list(columns)}")
    return None


def _date(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    return pd.to_datetime(str(value).replace("_", "-"), errors="coerce")


def parse_journal_folder(folder: str) -> dict:
    """Parse ``subject-date-label`` from the right, preserving IDs with dashes."""
    match = re.match(
        r"^(?P<subject>.+)-(?P<date>\d{4}[_-]\d{1,2}[_-]\d{1,2})-(?P<label>[^-]+)$",
        folder,
    )
    if not match:
        raise ValueError(
            f"Invalid sample folder {folder!r}; expected subject-YYYY_MM_DD-label"
        )
    return {
        "subject_id": match.group("subject"),
        "date": match.group("date"),
        "raw_label": match.group("label"),
    }


def _normalise_label_mapping(mapping: Mapping | None) -> dict[str, int]:
    if mapping is None:
        return {"0": 0, "1": 1}
    result = {str(key): int(value) for key, value in mapping.items()}
    if not result:
        raise ValueError("label_mapping cannot be empty")
    values = sorted(set(result.values()))
    if values != list(range(len(values))):
        raise ValueError("label_mapping values must be contiguous and zero-based")
    return result


def _normalise_diagnosis_value(value, label_mapping: Mapping[str, int]) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
    return label_mapping.get(text)


class JournalNiftiDataset(Dataset):
    """Index NIfTI paths eagerly, but load and normalize image data on demand."""

    def __init__(
        self,
        root: str,
        metadata_csv: str,
        image_filename: str,
        columns: Mapping[str, Sequence[str]],
        *,
        label_mapping: Mapping | None = None,
        image_shape: Sequence[int] | None = None,
        sex_mapping: Mapping | None = None,
        intensity_normalization: str = "none",
        max_match_days: Optional[int] = 90,
        require_diagnosis_match: bool = True,
        on_diagnosis_mismatch: str = "exclude",
    ):
        self.root = os.path.abspath(root)
        self.metadata_csv = os.path.abspath(metadata_csv)
        self.image_filename = image_filename
        self.label_mapping = _normalise_label_mapping(label_mapping)
        self.image_shape = tuple(image_shape) if image_shape else None
        self.sex_mapping = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in (sex_mapping or {}).items()
        }
        self.intensity_normalization = str(intensity_normalization).lower()
        self.max_match_days = (
            None if max_match_days is None else int(max_match_days)
        )
        self.require_diagnosis_match = bool(require_diagnosis_match)
        if on_diagnosis_mismatch not in {"exclude", "keep"}:
            raise ValueError("on_diagnosis_mismatch must be 'exclude' or 'keep'")
        self.on_diagnosis_mismatch = on_diagnosis_mismatch
        if self.intensity_normalization not in {"none", "nonzero_zscore"}:
            raise ValueError(
                "intensity_normalization must be 'none' or 'nonzero_zscore'"
            )
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not os.path.isfile(self.metadata_csv):
            raise FileNotFoundError(f"Metadata CSV does not exist: {self.metadata_csv}")

        table = pd.read_csv(self.metadata_csv)
        id_col = _first_column(table.columns, columns["id"])
        date_col = _first_column(table.columns, columns["date"])
        age_col = _first_column(table.columns, columns["age"])
        sex_col = _first_column(table.columns, columns["sex"])
        education_col = _first_column(
            table.columns, columns.get("education", ()), required=False
        )
        diagnosis_col = _first_column(
            table.columns,
            columns.get("diagnosis", ("DIAGNOSIS", "DX", "label", "LABEL")),
            required=False,
        )
        table = table.copy()
        table["_journal_id"] = table[id_col].astype(str).str.strip()
        table["_journal_date"] = table[date_col].map(_date)

        audit = {
            "exact_match": 0,
            "days_1_30": 0,
            "days_31_60": 0,
            "days_61_90": 0,
            "excluded_over_max_days": 0,
            "excluded_no_metadata": 0,
            "excluded_invalid_scan_date": 0,
            "excluded_no_valid_metadata_date": 0,
            "excluded_label_mismatch": 0,
            "excluded_unmapped_folder_label": 0,
            "match_days": [],
            "excluded": [],
            "subject_label_conflicts": [],
        }

        records = []
        for path in sorted(Path(self.root).glob(f"*/{self.image_filename}")):
            parsed = parse_journal_folder(path.parent.name)
            raw_label = parsed["raw_label"]
            if raw_label not in self.label_mapping:
                audit["excluded_unmapped_folder_label"] += 1
                audit["excluded"].append(
                    {
                        "folder": path.parent.name,
                        "reason": "unmapped_folder_label",
                        "raw_label": raw_label,
                    }
                )
                continue
            rows = table[table["_journal_id"] == parsed["subject_id"]]
            if rows.empty:
                audit["excluded_no_metadata"] += 1
                audit["excluded"].append(
                    {
                        "folder": path.parent.name,
                        "reason": "no_metadata",
                    }
                )
                continue
            scan_date = _date(parsed["date"])
            if scan_date is pd.NaT or pd.isna(scan_date):
                audit["excluded_invalid_scan_date"] += 1
                audit["excluded"].append(
                    {
                        "folder": path.parent.name,
                        "reason": "invalid_scan_date",
                        "scan_date": parsed["date"],
                    }
                )
                continue
            valid_meta = rows[rows["_journal_date"].notna()]
            if valid_meta.empty:
                audit["excluded_no_valid_metadata_date"] += 1
                audit["excluded"].append(
                    {
                        "folder": path.parent.name,
                        "reason": "no_valid_metadata_date",
                    }
                )
                continue
            distances = (valid_meta["_journal_date"] - scan_date).abs()
            best_index = distances.idxmin()
            row = valid_meta.loc[best_index]
            day_delta = distances.loc[best_index]
            days = float(day_delta / np.timedelta64(1, "D"))

            audit["match_days"].append(days)
            if days == 0:
                audit["exact_match"] += 1
            elif days <= 30:
                audit["days_1_30"] += 1
            elif days <= 60:
                audit["days_31_60"] += 1
            elif days <= 90:
                audit["days_61_90"] += 1
            if self.max_match_days is not None and days > self.max_match_days:
                audit["excluded_over_max_days"] += 1
                audit["excluded"].append(
                    {
                        "folder": path.parent.name,
                        "reason": "over_max_days",
                        "days": days,
                    }
                )
                continue

            folder_label = self.label_mapping[raw_label]
            if diagnosis_col is not None and self.require_diagnosis_match:
                diagnosis_label = _normalise_diagnosis_value(
                    row[diagnosis_col], self.label_mapping
                )
                if diagnosis_label is not None and diagnosis_label != folder_label:
                    audit["excluded_label_mismatch"] += 1
                    audit["excluded"].append(
                        {
                            "folder": path.parent.name,
                            "reason": "diagnosis_mismatch",
                            "folder_label": folder_label,
                            "diagnosis_label": diagnosis_label,
                        }
                    )
                    if self.on_diagnosis_mismatch == "exclude":
                        continue

            records.append(
                {
                    "path": str(path),
                    "folder": path.parent.name,
                    "subject_id": parsed["subject_id"],
                    "scan_date": parsed["date"],
                    "label": folder_label,
                    "age": pd.to_numeric(row[age_col], errors="coerce"),
                    "sex": self._canonical_sex(row[sex_col]),
                    "education": (
                        np.nan
                        if education_col is None
                        else pd.to_numeric(row[education_col], errors="coerce")
                    ),
                    "metadata_date": str(row[date_col]),
                    "match_days": None if not np.isfinite(days) else float(days),
                }
            )
        if not records:
            raise ValueError(
                f"No usable {image_filename} samples with matching metadata under {root}"
            )
        # Audit subjects whose scans disagree on CN/AD labels (longitudinal or error).
        by_subject: dict = {}
        for record in records:
            by_subject.setdefault(record["subject_id"], set()).add(int(record["label"]))
        for subject_id, label_set in sorted(by_subject.items(), key=lambda item: str(item[0])):
            if len(label_set) > 1:
                audit["subject_label_conflicts"].append(
                    {
                        "subject_id": subject_id,
                        "labels": sorted(label_set),
                        "n_scans": sum(
                            1 for record in records if record["subject_id"] == subject_id
                        ),
                        "policy": "subject_mean_uses_majority_label",
                    }
                )
        match_days = np.asarray(audit["match_days"], dtype=float)
        finite_days = match_days[np.isfinite(match_days)]
        self.match_audit = {
            **{
                key: value
                for key, value in audit.items()
                if key not in {"match_days", "subject_label_conflicts"}
            },
            "n_kept": len(records),
            "max_days_limit": self.max_match_days,
            "median_match_days": (
                float(np.median(finite_days)) if len(finite_days) else None
            ),
            "max_match_days_observed": (
                float(np.max(finite_days)) if len(finite_days) else None
            ),
            "n_subject_label_conflicts": len(audit["subject_label_conflicts"]),
            "subject_label_conflicts": audit["subject_label_conflicts"][:200],
            "excluded": audit["excluded"][:200],
            "n_excluded_logged": len(audit["excluded"]),
            "study_unit": "scan_rows_with_subject_mean_aggregation",
        }
        self.records = records
        self.subject_ids = np.asarray([r["subject_id"] for r in records], dtype=object)
        self.labels = np.asarray([r["label"] for r in records], dtype=np.int64)
        self.raw_age = np.asarray([r["age"] for r in records], dtype=float)
        self.raw_sex = np.asarray([r["sex"] for r in records], dtype=object)
        self.raw_education = np.asarray(
            [r["education"] for r in records], dtype=float
        )

    def _canonical_sex(self, value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "unknown"
        if isinstance(value, (int, np.integer)):
            text = str(int(value))
        elif isinstance(value, (float, np.floating)):
            text = str(int(value)) if float(value).is_integer() else str(value)
        else:
            text = str(value).strip().lower()
            if text.endswith(".0") and text.replace(".", "", 1).isdigit():
                text = text[:-2]
        text = text.strip().lower()
        mapped = self.sex_mapping.get(text, text)
        if mapped not in {"male", "female", "unknown"}:
            return "unknown"
        return mapped

    def __len__(self):
        return len(self.records)

    def _load_image(self, path: str) -> torch.Tensor:
        array = np.asarray(nib.load(path).dataobj, dtype=np.float32)
        array = np.nan_to_num(array, copy=False)
        if self.intensity_normalization == "nonzero_zscore":
            mask = array != 0
            values = array[mask] if mask.any() else array.reshape(-1)
            std = float(values.std())
            array = (array - float(values.mean())) / max(std, 1e-6)
        image = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        if self.image_shape and tuple(image.shape[1:]) != self.image_shape:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return image

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "image": self._load_image(record["path"]),
            "label": int(record["label"]),
            "age": float(record["age"]),
            "sex": str(record["sex"]),
            "education": float(record["education"]),
            "subject_id": record["subject_id"],
            "folder": record["folder"],
        }


@dataclass
class CovariatePreprocessor:
    """Train-only median imputation and stable age/education scaling."""

    scale_continuous: bool = True
    # unknown sex is imputed to the train-mode sex code for BOTH the model input
    # and environment construction (same semantic).
    impute_unknown_sex: bool = True

    def fit(self, age, sex, education, subject_ids=None):
        age = np.asarray(age, dtype=float)
        education = np.asarray(education, dtype=float)
        if not len(age):
            raise ValueError("Cannot fit covariates on an empty training split")
        self.age_median_ = self._median(age, fallback=0.0)
        self.education_median_ = self._median(education, fallback=0.0)
        age_filled = np.where(np.isfinite(age), age, self.age_median_)
        education_filled = np.where(
            np.isfinite(education), education, self.education_median_
        )
        self.age_mean_ = float(age_filled.mean())
        self.age_scale_ = max(float(age_filled.std()), 1e-6)
        self.education_mean_ = float(education_filled.mean())
        self.education_scale_ = max(float(education_filled.std()), 1e-6)
        normalized_sex = [self._sex(value) for value in sex]
        known = [value for value in normalized_sex if value is not None]
        self.sex_impute_ = (
            max((0, 1), key=lambda value: (known.count(value), -value)) if known else 0
        )
        self.sex_impute_name_ = "female" if self.sex_impute_ == 0 else "male"
        self.train_subject_ids_ = (
            None if subject_ids is None else frozenset(map(str, subject_ids))
        )
        self.is_fitted_ = True
        return self

    @staticmethod
    def _median(values, fallback):
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if len(finite) else float(fallback)

    @staticmethod
    def _sex(value):
        text = str(value).strip().lower()
        if text in _FEMALE:
            return 0
        if text in _MALE:
            return 1
        return None

    def transform_sex_names(self, sex) -> np.ndarray:
        """Return imputed sex names aligned with model covariates."""
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("CovariatePreprocessor must be fitted first")
        names = []
        for value in sex:
            code = self._sex(value)
            if code is None:
                code = self.sex_impute_
            names.append("female" if code == 0 else "male")
        return np.asarray(names, dtype=object)

    def transform(self, age, sex, education):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("CovariatePreprocessor must be fitted first")
        age = np.asarray(age, dtype=float)
        education = np.asarray(education, dtype=float)
        age_missing = (~np.isfinite(age)).astype(np.float32)
        education_missing = (~np.isfinite(education)).astype(np.float32)
        sex_codes = [self._sex(value) for value in sex]
        sex_missing = np.asarray(
            [1.0 if code is None else 0.0 for code in sex_codes], dtype=np.float32
        )
        age = np.where(np.isfinite(age), age, self.age_median_)
        education = np.where(
            np.isfinite(education), education, self.education_median_
        )
        sex_values = np.asarray(
            [
                self.sex_impute_ if code is None else code
                for code in sex_codes
            ],
            dtype=float,
        )
        if self.scale_continuous:
            age = (age - self.age_mean_) / self.age_scale_
            education = (
                education - self.education_mean_
            ) / self.education_scale_
        covariates = np.column_stack((age, sex_values, education)).astype(np.float32)
        masks = {
            "age_missing": age_missing,
            "sex_missing": sex_missing,
            "education_missing": education_missing,
        }
        return covariates, masks

    def transform_covariates_only(self, age, sex, education):
        """Back-compat: return only the imputed covariate matrix."""
        covariates, _masks = self.transform(age, sex, education)
        return covariates

    def to_dict(self):
        keys = (
            "age_median_",
            "education_median_",
            "age_mean_",
            "age_scale_",
            "education_mean_",
            "education_scale_",
            "sex_impute_",
            "sex_impute_name_",
        )
        payload = {key.rstrip("_"): getattr(self, key) for key in keys}
        payload["scale_continuous"] = bool(self.scale_continuous)
        payload["impute_unknown_sex"] = bool(self.impute_unknown_sex)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "CovariatePreprocessor":
        instance = cls(
            scale_continuous=bool(payload.get("scale_continuous", True)),
            impute_unknown_sex=bool(payload.get("impute_unknown_sex", True)),
        )
        instance.age_median_ = float(payload["age_median"])
        instance.education_median_ = float(payload["education_median"])
        instance.age_mean_ = float(payload["age_mean"])
        instance.age_scale_ = float(payload["age_scale"])
        instance.education_mean_ = float(payload["education_mean"])
        instance.education_scale_ = float(payload["education_scale"])
        instance.sex_impute_ = int(payload["sex_impute"])
        instance.sex_impute_name_ = str(
            payload.get(
                "sex_impute_name",
                "female" if instance.sex_impute_ == 0 else "male",
            )
        )
        instance.is_fitted_ = True
        return instance


class JournalManifestDataset(Dataset):
    """Index NIfTI volumes from a frozen scan manifest (Manifest-first path)."""

    def __init__(
        self,
        root: str,
        manifest_csv: str,
        image_filename: str,
        *,
        label_mapping: Mapping | None = None,
        image_shape: Sequence[int] | None = None,
        intensity_normalization: str = "none",
        require_diagnosis_match: bool = True,
        on_diagnosis_mismatch: str = "exclude",
        max_match_days: Optional[int] = 90,
    ):
        from data.scan_manifest import load_scan_manifest, row_acquisition_dict

        self.root = os.path.abspath(root)
        self.manifest_csv = os.path.abspath(manifest_csv)
        self.image_filename = image_filename
        self.label_mapping = _normalise_label_mapping(label_mapping)
        self.image_shape = tuple(image_shape) if image_shape else None
        self.intensity_normalization = str(intensity_normalization).lower()
        self.require_diagnosis_match = bool(require_diagnosis_match)
        self.max_match_days = None if max_match_days is None else int(max_match_days)
        if on_diagnosis_mismatch not in {"exclude", "keep"}:
            raise ValueError("on_diagnosis_mismatch must be 'exclude' or 'keep'")
        self.on_diagnosis_mismatch = on_diagnosis_mismatch
        if self.intensity_normalization not in {"none", "nonzero_zscore"}:
            raise ValueError(
                "intensity_normalization must be 'none' or 'nonzero_zscore'"
            )
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not os.path.isfile(self.manifest_csv):
            raise FileNotFoundError(
                f"Scan manifest does not exist: {self.manifest_csv}"
            )

        manifest = load_scan_manifest(self.manifest_csv)
        by_folder = {
            str(row["pre_folder"]).strip(): row
            for _, row in manifest.iterrows()
        }
        audit = {
            "exact_match": 0,
            "days_1_30": 0,
            "days_31_60": 0,
            "days_61_90": 0,
            "excluded_over_max_days": 0,
            "excluded_no_metadata": 0,
            "excluded_invalid_scan_date": 0,
            "excluded_label_mismatch": 0,
            "excluded_unmapped_folder_label": 0,
            "excluded_manifest_missing_image": 0,
            "match_days": [],
            "excluded": [],
            "subject_label_conflicts": [],
            "source": "scan_manifest",
            "manifest_csv": self.manifest_csv,
        }
        records = []
        for path in sorted(Path(self.root).glob(f"*/{self.image_filename}")):
            folder = path.parent.name
            parsed = parse_journal_folder(folder)
            raw_label = parsed["raw_label"]
            if raw_label not in self.label_mapping:
                audit["excluded_unmapped_folder_label"] += 1
                audit["excluded"].append(
                    {
                        "folder": folder,
                        "reason": "unmapped_folder_label",
                        "raw_label": raw_label,
                    }
                )
                continue
            row = by_folder.get(folder)
            if row is None:
                audit["excluded_no_metadata"] += 1
                audit["excluded"].append(
                    {"folder": folder, "reason": "no_manifest_row"}
                )
                continue
            days = pd.to_numeric(row.get("csv_match_days"), errors="coerce")
            if np.isfinite(days):
                days = float(days)
                audit["match_days"].append(days)
                if days == 0:
                    audit["exact_match"] += 1
                elif days <= 30:
                    audit["days_1_30"] += 1
                elif days <= 60:
                    audit["days_31_60"] += 1
                elif days <= 90:
                    audit["days_61_90"] += 1
                if self.max_match_days is not None and days > self.max_match_days:
                    audit["excluded_over_max_days"] += 1
                    audit["excluded"].append(
                        {
                            "folder": folder,
                            "reason": "over_max_days",
                            "days": days,
                        }
                    )
                    continue
            folder_label = self.label_mapping[raw_label]
            if self.require_diagnosis_match and "csv_label" in row.index:
                diagnosis_label = _normalise_diagnosis_value(
                    row.get("csv_label"), self.label_mapping
                )
                if diagnosis_label is not None and diagnosis_label != folder_label:
                    audit["excluded_label_mismatch"] += 1
                    audit["excluded"].append(
                        {
                            "folder": folder,
                            "reason": "diagnosis_mismatch",
                            "folder_label": folder_label,
                            "diagnosis_label": diagnosis_label,
                        }
                    )
                    if self.on_diagnosis_mismatch == "exclude":
                        continue
            image_path = str(path)
            declared = row.get("image_path")
            if (
                isinstance(declared, str)
                and declared.strip()
                and os.path.abspath(declared) != os.path.abspath(image_path)
                and not os.path.isfile(declared)
            ):
                audit["excluded_manifest_missing_image"] += 1
                audit["excluded"].append(
                    {
                        "folder": folder,
                        "reason": "manifest_image_path_missing",
                        "image_path": declared,
                    }
                )
                continue
            sex_value = row.get("sex", "missing")
            if str(sex_value).strip().lower() in {"missing", "unknown", "nan", ""}:
                sex_value = "unknown"
            records.append(
                {
                    "path": image_path,
                    "folder": folder,
                    "subject_id": parsed["subject_id"],
                    "scan_date": parsed["date"],
                    "label": folder_label,
                    "age": float(pd.to_numeric(row.get("age"), errors="coerce")),
                    "sex": str(sex_value),
                    "education": float(
                        pd.to_numeric(row.get("education"), errors="coerce")
                    ),
                    "metadata_date": str(row.get("csv_visdate", "")),
                    "match_days": None if not np.isfinite(days) else float(days),
                    "acquisition": row_acquisition_dict(row),
                }
            )
        if not records:
            raise ValueError(
                f"No usable {image_filename} samples with matching manifest under {root}"
            )
        by_subject: dict = {}
        for record in records:
            by_subject.setdefault(record["subject_id"], set()).add(int(record["label"]))
        for subject_id, label_set in sorted(
            by_subject.items(), key=lambda item: str(item[0])
        ):
            if len(label_set) > 1:
                audit["subject_label_conflicts"].append(
                    {
                        "subject_id": subject_id,
                        "labels": sorted(label_set),
                        "n_scans": sum(
                            1
                            for record in records
                            if record["subject_id"] == subject_id
                        ),
                        "policy": "subject_mean_uses_majority_label",
                    }
                )
        match_days = np.asarray(audit["match_days"], dtype=float)
        finite_days = match_days[np.isfinite(match_days)]
        self.match_audit = {
            **{
                key: value
                for key, value in audit.items()
                if key not in {"match_days", "subject_label_conflicts"}
            },
            "n_kept": len(records),
            "max_days_limit": self.max_match_days,
            "median_match_days": (
                float(np.median(finite_days)) if len(finite_days) else None
            ),
            "max_match_days_observed": (
                float(np.max(finite_days)) if len(finite_days) else None
            ),
            "n_subject_label_conflicts": len(audit["subject_label_conflicts"]),
            "subject_label_conflicts": audit["subject_label_conflicts"][:200],
            "excluded": audit["excluded"][:200],
            "n_excluded_logged": len(audit["excluded"]),
            "study_unit": "scan_rows_with_subject_mean_aggregation",
        }
        self.records = records
        self.subject_ids = np.asarray(
            [record["subject_id"] for record in records], dtype=object
        )
        self.labels = np.asarray([record["label"] for record in records], dtype=np.int64)
        self.raw_age = np.asarray([record["age"] for record in records], dtype=float)
        self.raw_sex = np.asarray([record["sex"] for record in records], dtype=object)
        self.raw_education = np.asarray(
            [record["education"] for record in records], dtype=float
        )

    def __len__(self):
        return len(self.records)

    def _load_image(self, path: str) -> torch.Tensor:
        array = np.asarray(nib.load(path).dataobj, dtype=np.float32)
        array = np.nan_to_num(array, copy=False)
        if self.intensity_normalization == "nonzero_zscore":
            mask = array != 0
            values = array[mask] if mask.any() else array.reshape(-1)
            std = float(values.std())
            array = (array - float(values.mean())) / max(std, 1e-6)
        image = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        if self.image_shape and tuple(image.shape[1:]) != self.image_shape:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return image

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "image": self._load_image(record["path"]),
            "label": int(record["label"]),
            "age": float(record["age"]),
            "sex": str(record["sex"]),
            "education": float(record["education"]),
            "subject_id": record["subject_id"],
            "folder": record["folder"],
            "acquisition": record["acquisition"],
        }


def build_journal_dataset(config: Mapping, cohort: str):
    """Factory: Manifest-first when ``scan_manifest.enabled``, else legacy CSV join."""
    spec = config["cohorts"][cohort]
    match_cfg = config.get("metadata_match", {})
    scan_cfg = config.get("scan_manifest", {}) or {}
    common = dict(
        label_mapping=config["task"].get("label_mapping"),
        image_shape=(config.get("training") or {}).get("image_shape"),
        intensity_normalization=spec.get("intensity_normalization", "none"),
        max_match_days=match_cfg.get("max_days", 90),
        require_diagnosis_match=bool(match_cfg.get("require_diagnosis_match", True)),
        on_diagnosis_mismatch=str(match_cfg.get("on_diagnosis_mismatch", "exclude")),
    )
    if bool(scan_cfg.get("enabled", False)):
        root = scan_cfg.get("root", "F:/ADNI/scan_manifests")
        files = scan_cfg.get("files") or {}
        filename = files.get(cohort) or f"{cohort}_scan_manifest.csv"
        manifest_csv = os.path.join(str(root), str(filename))
        return JournalManifestDataset(
            root=spec["image_root"],
            manifest_csv=manifest_csv,
            image_filename=spec["image_filename"],
            **common,
        )
    return JournalNiftiDataset(
        root=spec["image_root"],
        metadata_csv=spec["metadata_csv"],
        image_filename=spec["image_filename"],
        columns=spec["columns"],
        sex_mapping=spec.get("sex_mapping"),
        **common,
    )


class JournalSubset(Dataset):
    """Attach train-fitted covariates and environment IDs to a dataset split."""

    def __init__(
        self,
        dataset: JournalNiftiDataset | JournalManifestDataset,
        indices: Sequence[int],
        preprocessor: CovariatePreprocessor,
        environment_ids: Sequence[int],
    ):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.environment_ids = np.asarray(environment_ids, dtype=np.int64)
        if len(self.indices) != len(self.environment_ids):
            raise ValueError("indices and environment_ids must have equal length")
        self.covariates, masks = preprocessor.transform(
            dataset.raw_age[self.indices],
            dataset.raw_sex[self.indices],
            dataset.raw_education[self.indices],
        )
        self.age_missing = np.asarray(masks["age_missing"], dtype=np.float32)
        self.sex_missing = np.asarray(masks["sex_missing"], dtype=np.float32)
        self.education_missing = np.asarray(
            masks["education_missing"], dtype=np.float32
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        source = self.dataset[source_index]
        item = {
            "image": source["image"],
            "label": source["label"],
            "covariates": torch.from_numpy(self.covariates[index]),
            "age_missing": torch.tensor(self.age_missing[index], dtype=torch.float32),
            "sex_missing": torch.tensor(self.sex_missing[index], dtype=torch.float32),
            "education_missing": torch.tensor(
                self.education_missing[index], dtype=torch.float32
            ),
            "environment_id": int(self.environment_ids[index]),
            "subject_id": source["subject_id"],
            "folder": source["folder"],
        }
        if "acquisition" in source:
            item["acquisition"] = source["acquisition"]
        return item


__all__ = [
    "CovariatePreprocessor",
    "JournalManifestDataset",
    "JournalNiftiDataset",
    "JournalSubset",
    "build_journal_dataset",
    "parse_journal_folder",
]
