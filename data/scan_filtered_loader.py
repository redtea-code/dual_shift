"""Data loading primitives for the ADNI-1.5T scan-filtered protocol.

This module is deliberately separate from the legacy paired-subject holdout
path.  It filters scan rows before splitting subjects and refuses to load a
manifest whose field-strength contract is violated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data.journal_dataset import JournalManifestDataset

PROTOCOL_NAME = "ADNI_1.5T_scan_filtered"
PROTOCOL_VERSION = "scan_filtered_v1_2026-08-08"


def normalize_field_strength(value: Any) -> str:
    """Map common numeric/string encodings to ``1.5T``, ``3T`` or ``unknown``."""
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "unknown"
    text = str(value).strip().lower().replace("tesla", "t")
    if text in {"1.5", "1.5t", "1,5", "1,5t"}:
        return "1.5T"
    if text in {"3", "3.0", "3t", "3.0t"}:
        return "3T"
    number = pd.to_numeric(value, errors="coerce")
    if np.isfinite(number):
        if abs(float(number) - 1.5) < 1e-6:
            return "1.5T"
        if abs(float(number) - 3.0) < 1e-6:
            return "3T"
    return "unknown"


def filter_scan_manifest(
    table: pd.DataFrame,
    *,
    cohort: str,
    require_image_aligned: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the protocol's scan-level filter and return rows plus audit data.

    ADNI keeps only 1.5T rows; NACC keeps only 3T rows.  Subject IDs are never
    removed merely because another scan from the same subject had a different
    field strength.
    """
    if "field_strength" not in table.columns:
        raise ValueError("scan manifest requires a field_strength column")
    if "subject_id" not in table.columns:
        raise ValueError("scan manifest requires a subject_id column")
    cohort_upper = str(cohort).upper()
    expected = "1.5T" if cohort_upper == "ADNI" else "3T" if cohort_upper == "NACC" else None
    if expected is None:
        raise ValueError(f"unsupported cohort {cohort!r}; expected ADNI or NACC")

    work = table.copy()
    work["_protocol_field_strength"] = work["field_strength"].map(normalize_field_strength)
    if require_image_aligned and "image_aligned" not in work.columns:
        raise ValueError("scan manifest requires image_aligned for strict filtering")
    if require_image_aligned and "image_aligned" in work.columns:
        aligned = work["image_aligned"].map(
            lambda value: value is True
            or (isinstance(value, (int, np.integer, float, np.floating)) and bool(value) and np.isfinite(value))
            or str(value).strip().lower() in {"true", "1", "yes"}
        )
    else:
        aligned = pd.Series(True, index=work.index)
    keep = (work["_protocol_field_strength"] == expected) & aligned

    subject_strengths = (
        work.assign(_subject=work["subject_id"].astype(str))
        .groupby("_subject")["_protocol_field_strength"]
        .agg(lambda values: sorted(set(values)))
    )
    kept = work.loc[keep].copy()
    kept["protocol_name"] = PROTOCOL_NAME
    kept["protocol_version"] = PROTOCOL_VERSION
    kept["paired_origin"] = kept["subject_id"].astype(str).map(
        lambda subject: "had_3T" if "3T" in subject_strengths.get(str(subject), []) else "never_3T"
    )
    kept = kept.drop(columns=["_protocol_field_strength"], errors="ignore").reset_index(drop=True)
    strength_counts = work["_protocol_field_strength"].value_counts(dropna=False).to_dict()
    audit = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "cohort": cohort_upper,
        "expected_field_strength": expected,
        "n_input_rows": int(len(work)),
        "n_kept_rows": int(len(kept)),
        "n_dropped_rows": int(len(work) - len(kept)),
        "input_field_strength_counts": {str(k): int(v) for k, v in strength_counts.items()},
        "n_kept_subjects": int(kept["subject_id"].astype(str).nunique()),
        "n_subjects_with_retained_1p5T_and_dropped_3T": int(
            (kept.loc[kept["paired_origin"] == "had_3T", "subject_id"].astype(str).nunique())
            if cohort_upper == "ADNI" else 0
        ),
        "n_subjects_never_observed_3T": int(
            (kept.loc[kept["paired_origin"] == "never_3T", "subject_id"].astype(str).nunique())
            if cohort_upper == "ADNI" else 0
        ),
        "require_image_aligned": bool(require_image_aligned),
    }
    if kept.empty:
        raise ValueError(f"protocol filter retained no rows for {cohort_upper}")
    return kept, audit


def write_filtered_manifest(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    cohort: str,
    audit_json: str | Path | None = None,
) -> dict[str, Any]:
    """Filter a CSV and write a frozen protocol manifest plus optional audit."""
    table = pd.read_csv(input_csv)
    filtered, audit = filter_scan_manifest(table, cohort=cohort)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output, index=False)
    audit["input_csv"] = str(Path(input_csv).resolve())
    audit["output_csv"] = str(output.resolve())
    if audit_json is not None:
        Path(audit_json).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_json).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def subject_level_split(
    table: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
) -> dict[str, list[int] | list[str]]:
    """Split filtered rows by subject, returning disjoint row indices.

    Subjects are shuffled within their majority-label stratum, then allocated
    with largest-remainder counts.  This preserves subject-level isolation and
    gives reproducible approximate 6/2/2 proportions without scan-level split.
    """
    if "subject_id" not in table.columns or "label" not in table.columns:
        raise ValueError("subject_level_split requires subject_id and label columns")
    ratios = tuple(float(x) for x in ratios)
    if len(ratios) != 3 or any(x < 0 for x in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three non-negative values summing to 1")
    work = table.reset_index(drop=True)
    grouped = work.groupby(work["subject_id"].astype(str), sort=True)
    subjects = []
    for subject, rows in grouped:
        labels = rows["label"].dropna().astype(int).value_counts()
        subjects.append((str(subject), int(labels.index[0]) if len(labels) else -1))
    rng = np.random.default_rng(int(seed))
    strata: dict[int, list[str]] = {}
    for subject, label in subjects:
        strata.setdefault(label, []).append(subject)
    assignments = {"train": [], "val": [], "test": []}
    names = tuple(assignments)
    for label in sorted(strata):
        pool = np.asarray(strata[label], dtype=object)
        rng.shuffle(pool)
        exact = np.asarray(ratios) * len(pool)
        counts = np.floor(exact).astype(int)
        for index in np.argsort(-(exact - counts))[: int(len(pool) - counts.sum())]:
            counts[index] += 1
        start = 0
        for name, count in zip(names, counts):
            assignments[name].extend(pool[start : start + int(count)].tolist())
            start += int(count)
    row_indices = {
        name: np.flatnonzero(work["subject_id"].astype(str).isin(subjects_for)).astype(int).tolist()
        for name, subjects_for in assignments.items()
    }
    if set(assignments["train"]) & set(assignments["val"]) or set(assignments["train"]) & set(assignments["test"]) or set(assignments["val"]) & set(assignments["test"]):
        raise AssertionError("subject split overlap")
    return {"seed": int(seed), "ratios": list(ratios), "subjects": assignments, "indices": row_indices}


class ScanFilteredManifestDataset(JournalManifestDataset):
    """Strict loader for a previously frozen scan-filtered manifest."""

    def __init__(self, *args, cohort: str, expose_acquisition: bool = False, **kwargs):
        manifest_csv = kwargs.get("manifest_csv") or (args[1] if len(args) > 1 else None)
        if manifest_csv is None:
            raise TypeError("manifest_csv is required")
        table = pd.read_csv(manifest_csv)
        filtered, audit = filter_scan_manifest(table, cohort=cohort)
        if len(filtered) != len(table):
            raise ValueError(
                f"manifest {manifest_csv} is not frozen for {cohort}: "
                f"contains {len(table) - len(filtered)} rows outside the protocol"
            )
        if "protocol_version" not in table.columns or not table["protocol_version"].eq(PROTOCOL_VERSION).all():
            raise ValueError("manifest is missing the new protocol_version marker")
        super().__init__(*args, **kwargs)
        self.scan_filtered_audit = audit
        self.expose_acquisition = bool(expose_acquisition)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if not self.expose_acquisition:
            item.pop("acquisition", None)
        return item


__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "normalize_field_strength",
    "filter_scan_manifest",
    "write_filtered_manifest",
    "subject_level_split",
    "ScanFilteredManifestDataset",
]
