"""Source-train-only acquisition / protocol descriptor encoder."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


MANUFACTURER_CANON = {
    "siemens": "Siemens",
    "ge": "GE",
    "ge medical systems": "GE",
    "philips": "Philips",
    "philips medical systems": "Philips",
}

SEQUENCE_CANON = {
    "mprage": "MPRAGE",
    "mp-rage": "MPRAGE",
    "ir-fspgr": "IR-FSPGR",
    "ir_fspgr": "IR-FSPGR",
    "fspgr": "IR-FSPGR",
    "spgr": "SPGR",
}

TR_MODE_VOCAB = ["short_cycle", "inversion_cycle", "missing"]
CONTINUOUS_FIELDS = (
    "tr_raw",
    "te_ms",
    "ti_ms",
    "flip_angle",
    "slice_thickness",
    "pixel_spacing_x",
    "pixel_spacing_y",
    "acceleration",
)


def canonicalize_manufacturer(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Missing"
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return "Missing"
    return MANUFACTURER_CANON.get(text.lower(), "Other")


def canonicalize_field_strength(value: Any) -> str:
    number = np.nan if value is None else float(pd_to_float(value))
    if not np.isfinite(number):
        return "Missing"
    if abs(number - 1.5) < 0.2:
        return "1.5T"
    if abs(number - 3.0) < 0.2:
        return "3T"
    return "Other"


def canonicalize_sequence_family(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Other"
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return "Other"
    return SEQUENCE_CANON.get(text, SEQUENCE_CANON.get(text.replace(" ", ""), "Other"))


def canonicalize_model_family(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Missing"
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return "Missing"
    token = text.replace("-", " ").replace("_", " ").split()[0]
    return token[:32] if token else "Missing"


def parse_acceleration(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return float("nan")
    if "unacceler" in text:
        return 0.0
    if "acceler" in text:
        return 1.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def pd_to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class AcquisitionDescriptorEncoder(nn.Module):
    """Fit categorical vocabularies on source-train only, then embed + MLP."""

    def __init__(
        self,
        embedding_dim: int = 8,
        hidden_dim: int = 64,
        out_dim: int = 32,
        min_model_count: int = 8,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.min_model_count = int(min_model_count)
        self.is_fitted_ = False
        self.manufacturer_to_id: Dict[str, int] = {}
        self.field_to_id: Dict[str, int] = {}
        self.model_to_id: Dict[str, int] = {}
        self.sequence_to_id: Dict[str, int] = {}
        self.tr_mode_to_id = {name: index for index, name in enumerate(TR_MODE_VOCAB)}
        self.cont_mean_: Dict[str, float] = {}
        self.cont_std_: Dict[str, float] = {}
        # Placeholders replaced in fit()
        self.manufacturer_emb = nn.Embedding(1, embedding_dim)
        self.field_emb = nn.Embedding(1, embedding_dim)
        self.model_emb = nn.Embedding(1, embedding_dim)
        self.sequence_emb = nn.Embedding(1, embedding_dim)
        self.tr_mode_emb = nn.Embedding(len(TR_MODE_VOCAB), embedding_dim)
        self.mlp = nn.Identity()

    def fit(self, acquisitions: Sequence[Mapping[str, Any]]) -> "AcquisitionDescriptorEncoder":
        manufacturers = [canonicalize_manufacturer(row.get("manufacturer")) for row in acquisitions]
        fields = [canonicalize_field_strength(row.get("field_strength")) for row in acquisitions]
        models = [canonicalize_model_family(row.get("scanner_model")) for row in acquisitions]
        sequences = [
            canonicalize_sequence_family(row.get("sequence_family")) for row in acquisitions
        ]

        def _vocab(values: Sequence[str], *, always: Sequence[str], rare_to: str) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            kept = [name for name in always if name in counts or name in {"Missing", "Other"}]
            for name, count in sorted(counts.items()):
                if name in kept:
                    continue
                if count >= self.min_model_count or rare_to is None:
                    kept.append(name)
            if rare_to not in kept:
                kept.append(rare_to)
            if "Missing" not in kept:
                kept.append("Missing")
            return {name: index for index, name in enumerate(kept)}

        self.manufacturer_to_id = _vocab(
            manufacturers, always=["Siemens", "GE", "Philips", "Other", "Missing"], rare_to="Other"
        )
        self.field_to_id = _vocab(
            fields, always=["1.5T", "3T", "Other", "Missing"], rare_to="Other"
        )
        # Collapse rare scanner models.
        model_counts: Dict[str, int] = {}
        for value in models:
            model_counts[value] = model_counts.get(value, 0) + 1
        model_names = ["Missing", "Other"]
        for name, count in sorted(model_counts.items()):
            if name in {"Missing", "Other"}:
                continue
            if count >= self.min_model_count:
                model_names.append(name)
        self.model_to_id = {name: index for index, name in enumerate(model_names)}
        self.sequence_to_id = _vocab(
            sequences,
            always=["MPRAGE", "IR-FSPGR", "SPGR", "Other", "Missing"],
            rare_to="Other",
        )

        for field in CONTINUOUS_FIELDS:
            values = []
            for row in acquisitions:
                if field == "acceleration":
                    number = parse_acceleration(row.get("acceleration"))
                else:
                    number = pd_to_float(row.get(field, row.get("tr_raw")))
                    if field == "tr_raw":
                        number = pd_to_float(row.get("tr_raw", row.get("tr_ms")))
                if np.isfinite(number):
                    values.append(float(number))
            if values:
                arr = np.asarray(values, dtype=float)
                self.cont_mean_[field] = float(arr.mean())
                self.cont_std_[field] = float(max(arr.std(), 1e-6))
            else:
                self.cont_mean_[field] = 0.0
                self.cont_std_[field] = 1.0

        self.manufacturer_emb = nn.Embedding(len(self.manufacturer_to_id), self.embedding_dim)
        self.field_emb = nn.Embedding(len(self.field_to_id), self.embedding_dim)
        self.model_emb = nn.Embedding(len(self.model_to_id), self.embedding_dim)
        self.sequence_emb = nn.Embedding(len(self.sequence_to_id), self.embedding_dim)
        cat_dim = self.embedding_dim * 5
        cont_dim = len(CONTINUOUS_FIELDS) * 2  # value + missing
        in_dim = cat_dim + cont_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.is_fitted_ = True
        return self

    def _lookup(self, mapping: Mapping[str, int], key: str, other_key: str = "Other") -> int:
        if key in mapping:
            return mapping[key]
        if "Missing" in mapping and key == "Missing":
            return mapping["Missing"]
        return mapping.get(other_key, mapping.get("Missing", 0))

    def encode_rows(self, acquisitions: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not self.is_fitted_:
            raise RuntimeError("AcquisitionDescriptorEncoder must be fitted on source train")
        manufacturers = []
        fields = []
        models = []
        sequences = []
        tr_modes = []
        continuous = []
        missing = []
        for row in acquisitions:
            manufacturers.append(
                self._lookup(self.manufacturer_to_id, canonicalize_manufacturer(row.get("manufacturer")))
            )
            fields.append(
                self._lookup(self.field_to_id, canonicalize_field_strength(row.get("field_strength")))
            )
            model_name = canonicalize_model_family(row.get("scanner_model"))
            if model_name not in self.model_to_id:
                model_name = "Other" if model_name != "Missing" else "Missing"
            models.append(self._lookup(self.model_to_id, model_name))
            sequences.append(
                self._lookup(
                    self.sequence_to_id,
                    canonicalize_sequence_family(row.get("sequence_family")),
                )
            )
            tr_mode = str(row.get("tr_mode") or "missing")
            if tr_mode not in self.tr_mode_to_id:
                tr_mode = "missing"
            tr_modes.append(self.tr_mode_to_id[tr_mode])
            cont_vals = []
            miss_vals = []
            for field in CONTINUOUS_FIELDS:
                if field == "acceleration":
                    number = parse_acceleration(row.get("acceleration"))
                elif field == "tr_raw":
                    number = pd_to_float(row.get("tr_raw", row.get("tr_ms")))
                else:
                    number = pd_to_float(row.get(field))
                if np.isfinite(number):
                    scaled = (float(number) - self.cont_mean_[field]) / self.cont_std_[field]
                    cont_vals.append(scaled)
                    miss_vals.append(0.0)
                else:
                    cont_vals.append(0.0)
                    miss_vals.append(1.0)
            continuous.append(cont_vals)
            missing.append(miss_vals)
        return {
            "manufacturer_id": torch.tensor(manufacturers, dtype=torch.long),
            "field_strength_id": torch.tensor(fields, dtype=torch.long),
            "scanner_model_id": torch.tensor(models, dtype=torch.long),
            "sequence_family_id": torch.tensor(sequences, dtype=torch.long),
            "tr_mode_id": torch.tensor(tr_modes, dtype=torch.long),
            "continuous": torch.tensor(continuous, dtype=torch.float32),
            "continuous_missing": torch.tensor(missing, dtype=torch.float32),
        }

    def forward_from_encoded(self, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.manufacturer_emb(encoded["manufacturer_id"]),
            self.field_emb(encoded["field_strength_id"]),
            self.model_emb(encoded["scanner_model_id"]),
            self.sequence_emb(encoded["sequence_family_id"]),
            self.tr_mode_emb(encoded["tr_mode_id"]),
            encoded["continuous"],
            encoded["continuous_missing"],
        ]
        return self.mlp(torch.cat(parts, dim=1))

    def forward(self, acquisitions: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        encoded = self.encode_rows(acquisitions)
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        return self.forward_from_encoded(encoded)

    def domain_keys(self, acquisitions: Sequence[Mapping[str, Any]]) -> list[str]:
        keys = []
        for row in acquisitions:
            keys.append(
                "|".join(
                    [
                        canonicalize_manufacturer(row.get("manufacturer")),
                        canonicalize_field_strength(row.get("field_strength")),
                        canonicalize_model_family(row.get("scanner_model")),
                        canonicalize_sequence_family(row.get("sequence_family")),
                    ]
                )
            )
        return keys

    def to_state_dict_extra(self) -> Dict[str, Any]:
        return {
            "manufacturer_to_id": self.manufacturer_to_id,
            "field_to_id": self.field_to_id,
            "model_to_id": self.model_to_id,
            "sequence_to_id": self.sequence_to_id,
            "cont_mean_": self.cont_mean_,
            "cont_std_": self.cont_std_,
            "is_fitted_": self.is_fitted_,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "min_model_count": self.min_model_count,
        }

    def load_state_dict_extra(self, payload: Mapping[str, Any]) -> None:
        self.manufacturer_to_id = dict(payload["manufacturer_to_id"])
        self.field_to_id = dict(payload["field_to_id"])
        self.model_to_id = dict(payload["model_to_id"])
        self.sequence_to_id = dict(payload["sequence_to_id"])
        self.cont_mean_ = dict(payload["cont_mean_"])
        self.cont_std_ = dict(payload["cont_std_"])
        self.embedding_dim = int(payload.get("embedding_dim", self.embedding_dim))
        self.hidden_dim = int(payload.get("hidden_dim", self.hidden_dim))
        self.out_dim = int(payload.get("out_dim", self.out_dim))
        self.min_model_count = int(payload.get("min_model_count", self.min_model_count))
        self.manufacturer_emb = nn.Embedding(len(self.manufacturer_to_id), self.embedding_dim)
        self.field_emb = nn.Embedding(len(self.field_to_id), self.embedding_dim)
        self.model_emb = nn.Embedding(len(self.model_to_id), self.embedding_dim)
        self.sequence_emb = nn.Embedding(len(self.sequence_to_id), self.embedding_dim)
        self.tr_mode_emb = nn.Embedding(len(TR_MODE_VOCAB), self.embedding_dim)
        cat_dim = self.embedding_dim * 5
        cont_dim = len(CONTINUOUS_FIELDS) * 2
        in_dim = cat_dim + cont_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.is_fitted_ = bool(payload.get("is_fitted_", True))
