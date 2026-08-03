"""Audit frozen scan-manifest acquisition fields before scan-aware training.

The audit is intentionally label-free: it reports data availability and cohort
shift candidates without using diagnostic labels or target-domain tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scan_manifest import ACQUISITION_FIELDS, load_scan_manifest


NUMERIC_FIELDS = frozenset(
    {
        "field_strength",
        "tr_raw",
        "te_ms",
        "ti_ms",
        "flip_angle",
        "slice_thickness",
        "pixel_spacing_x",
        "pixel_spacing_y",
        "acceleration",
    }
)


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "nan", "None", "<NA>"}


def field_strength_bin(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(number):
        return "unknown"
    if abs(float(number) - 1.5) < 0.2:
        return "1.5T"
    if abs(float(number) - 3.0) < 0.2:
        return "3T"
    return "other"


def _numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"n_observed": 0, "min": None, "p01": None, "median": None, "p99": None, "max": None}
    return {
        "n_observed": int(len(numeric)),
        "min": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "median": float(numeric.median()),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def audit_manifest(table: pd.DataFrame, cohort: str) -> dict[str, list[dict[str, Any]]]:
    """Return normalized audit tables for one frozen scan manifest."""
    missing = set(ACQUISITION_FIELDS) - set(table.columns)
    work = table.copy()
    for field in missing:
        work[field] = np.nan

    field_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for field in ACQUISITION_FIELDS:
        values = work[field]
        missing_count = int(values.map(_missing).sum())
        row: dict[str, Any] = {
            "cohort": cohort,
            "field": field,
            "field_type": "numeric" if field in NUMERIC_FIELDS else "categorical",
            "n_rows": int(len(work)),
            "n_missing": missing_count,
            "missing_rate": float(missing_count / len(work)) if len(work) else 0.0,
            "column_present": field not in missing,
        }
        if field in NUMERIC_FIELDS:
            row.update(_numeric_summary(values))
        field_rows.append(row)

        if field not in NUMERIC_FIELDS:
            normalized = values.map(lambda value: "Missing" if _missing(value) else str(value).strip())
            for value, count in normalized.value_counts(dropna=False).items():
                category_rows.append(
                    {"cohort": cohort, "field": field, "value": str(value), "n_rows": int(count)}
                )

    field_bins = work["field_strength"].map(field_strength_bin)
    strength_rows = [
        {"cohort": cohort, "field_strength": str(name), "n_rows": int(count)}
        for name, count in field_bins.value_counts().items()
    ]

    conflict_rows: list[dict[str, Any]] = []
    if "subject_id" in work.columns:
        for field in ACQUISITION_FIELDS:
            normalized = work[field].map(lambda value: "Missing" if _missing(value) else str(value).strip())
            cardinality = normalized.groupby(work["subject_id"].astype(str)).nunique(dropna=False)
            conflict_rows.append(
                {
                    "cohort": cohort,
                    "field": field,
                    "n_subjects": int(cardinality.size),
                    "n_subjects_multiple_values": int((cardinality > 1).sum()),
                }
            )

    return {
        "field_summary": field_rows,
        "category_distribution": category_rows,
        "field_strength_summary": strength_rows,
        "subject_parameter_conflicts": conflict_rows,
    }


def _flatten(items: Iterable[Mapping[str, list[dict[str, Any]]]], key: str) -> list[dict[str, Any]]:
    return [row for item in items for row in item[key]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("journal_dual_shift_apis_v2_claim.yaml"))
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scan_cfg = config.get("scan_manifest") or {}
    root = args.manifest_root or Path(scan_cfg["root"])
    files = scan_cfg.get("files") or {}
    if not files:
        raise SystemExit("scan_manifest.files is required")
    output = args.output_dir or Path("outputs") / "scan_parameter_audit" / date.today().isoformat()
    output.mkdir(parents=True, exist_ok=True)

    audits = []
    provenance = []
    for cohort, filename in files.items():
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"Missing frozen scan manifest: {path}")
        table = load_scan_manifest(path)
        audits.append(audit_manifest(table, str(cohort)))
        provenance.append({"cohort": str(cohort), "manifest": str(path), "n_rows": int(len(table))})

    for key in (
        "field_summary",
        "category_distribution",
        "field_strength_summary",
        "subject_parameter_conflicts",
    ):
        pd.DataFrame(_flatten(audits, key)).to_csv(output / f"{key}.csv", index=False)
    (output / "scan_parameter_audit.json").write_text(
        json.dumps({"config": str(args.config), "manifests": provenance}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote scan-parameter audit to {output}")


if __name__ == "__main__":
    main()
