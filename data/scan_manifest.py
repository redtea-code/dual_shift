"""Build and load frozen scan-level manifests for dual-shift / journal training.

Manifests live under ``F:/ADNI/scan_manifests`` by default. Clinical covariates
and acquisition fields are frozen at build time; train-time z-score / vocab
fitting still happens only on the source-train split.

This module intentionally avoids torch/nibabel so the build CLI can run in a
lightweight pandas-only environment.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

RULE_VERSION = "scan_manifest_v1_2026-07-22"
TR_SHORT_CYCLE_MAX_MS = 100.0
EDUCATION_MISSING_CODES = {99, 999, -1, -9}


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


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sex(value, sex_mapping: Mapping[str, str] | None = None) -> str:
    mapping = {
        str(key).strip().lower(): str(val).strip().lower()
        for key, val in (sex_mapping or {}).items()
    }
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = str(int(value)) if float(value).is_integer() else str(value)
    else:
        text = str(value).strip().lower()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
    text = text.strip().lower()
    mapped = mapping.get(text, text)
    if mapped in {"male", "m"}:
        return "male"
    if mapped in {"female", "f"}:
        return "female"
    # Without an explicit map, keep common raw codes: 1=male, 2=female.
    if text in {"male", "m", "1"}:
        return "male"
    if text in {"female", "f", "2"}:
        return "female"
    return "missing"


def _clean_education(value) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(number):
        return float("nan")
    if float(number) in EDUCATION_MISSING_CODES or int(number) in EDUCATION_MISSING_CODES:
        return float("nan")
    return float(number)


def classify_tr_mode(tr_ms) -> str:
    value = pd.to_numeric(tr_ms, errors="coerce")
    if not np.isfinite(value):
        return "missing"
    if float(value) < TR_SHORT_CYCLE_MAX_MS:
        return "short_cycle"
    return "inversion_cycle"


def infer_metadata_match_quality(row: pd.Series) -> str:
    image_id = row.get("source_image_id")
    has_image_id = (
        image_id is not None
        and not (isinstance(image_id, float) and np.isnan(image_id))
        and str(image_id).strip() not in {"", "nan", "None"}
    )
    scanner_fields = [
        row.get("manufacturer"),
        row.get("field_strength"),
        row.get("scanner_model"),
        row.get("tr_ms"),
    ]
    missing_scanner = sum(
        1
        for value in scanner_fields
        if value is None
        or (isinstance(value, float) and np.isnan(value))
        or str(value).strip() in {"", "nan", "None"}
    )
    if missing_scanner >= 3:
        return "site_fallback"
    if has_image_id:
        return "exact_image"
    return "same_day_inferred"


def _protocol_missing_count(row: pd.Series) -> int:
    fields = [
        "manufacturer",
        "field_strength",
        "scanner_model",
        "sequence_family",
        "tr_ms",
        "te_ms",
        "ti_ms",
        "flip_angle",
        "slice_thickness",
        "pixel_spacing_x",
        "pixel_spacing_y",
        "acceleration",
        "source_image_id",
    ]
    count = 0
    for field in fields:
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            count += 1
        elif str(value).strip() in {"", "nan", "None"}:
            count += 1
    return count


def _has_value(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip() not in {"", "nan", "None"}


def dedupe_pre_folder(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one row per non-null ``pre_folder``; export dropped/orphan candidates."""
    work = table.copy()
    work["_row_id"] = np.arange(len(work))
    work["pre_folder"] = work["pre_folder"].astype(object)
    null_mask = work["pre_folder"].isna() | (
        work["pre_folder"].astype(str).str.strip().isin({"", "nan", "None"})
    )
    orphans = work.loc[null_mask].copy()
    orphans["dedupe_action"] = "drop_null_pre_folder"
    orphans["dedupe_reason"] = "pre_folder_missing"

    valid = work.loc[~null_mask].copy()
    valid["pre_folder"] = valid["pre_folder"].astype(str).str.strip()
    valid["_has_image_id"] = valid["source_image_id"].map(_has_value).astype(int)
    valid["_protocol_missing"] = valid.apply(_protocol_missing_count, axis=1)

    valid = valid.sort_values(
        by=["pre_folder", "_has_image_id", "_protocol_missing", "_row_id"],
        ascending=[True, False, True, True],
    )
    kept = valid.drop_duplicates("pre_folder", keep="first").copy()
    dropped = valid.loc[~valid["_row_id"].isin(kept["_row_id"])].copy()
    if not dropped.empty:
        dropped["dedupe_action"] = "drop_duplicate_pre_folder"
        dropped["dedupe_reason"] = "lower_priority_than_kept_row"
    kept_ids = set(kept["_row_id"].tolist())
    candidate_parts = []
    if not orphans.empty:
        candidate_parts.append(orphans)
    if not dropped.empty:
        candidate_parts.append(dropped)
    # Also log kept representatives for duplicate groups (for audit).
    dup_folders = valid["pre_folder"].value_counts()
    dup_folders = dup_folders[dup_folders > 1].index
    if len(dup_folders):
        kept_reps = kept[kept["pre_folder"].isin(dup_folders)].copy()
        kept_reps["dedupe_action"] = "keep"
        kept_reps["dedupe_reason"] = "preferred_row_for_duplicate_pre_folder"
        candidate_parts.append(kept_reps)

    candidates = (
        pd.concat(candidate_parts, ignore_index=True)
        if candidate_parts
        else pd.DataFrame()
    )
    kept = kept.drop(
        columns=["_row_id", "_has_image_id", "_protocol_missing"], errors="ignore"
    )
    if not candidates.empty:
        candidates = candidates.drop(
            columns=["_has_image_id", "_protocol_missing"], errors="ignore"
        )
    return kept.reset_index(drop=True), candidates.reset_index(drop=True)


def _load_clinical_table(
    metadata_csv: str | Path,
    columns: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    table = pd.read_csv(metadata_csv)
    id_col = _first_column(table.columns, columns["id"])
    date_col = _first_column(table.columns, columns["date"])
    age_col = _first_column(table.columns, columns["age"])
    sex_col = _first_column(table.columns, columns["sex"])
    education_col = _first_column(
        table.columns, columns.get("education", ()), required=False
    )
    out = pd.DataFrame(
        {
            "_journal_id": table[id_col].astype(str).str.strip(),
            "_journal_date": table[date_col].map(_date),
            "csv_age": pd.to_numeric(table[age_col], errors="coerce"),
            "csv_sex": table[sex_col],
            "csv_education": (
                np.nan
                if education_col is None
                else pd.to_numeric(table[education_col], errors="coerce")
            ),
        }
    )
    return out


def attach_clinical_covariates(
    scan_table: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sex_mapping: Mapping | None = None,
) -> pd.DataFrame:
    """Join age/sex/education onto scan rows using subject_id + csv_visdate."""
    out = scan_table.copy()
    out["_subject"] = out["subject_id"].astype(str).str.strip()
    out["_visit"] = out["csv_visdate"].map(_date)

    clinical = clinical.copy()
    clinical["_subject"] = clinical["_journal_id"].astype(str).str.strip()
    clinical["_visit"] = clinical["_journal_date"]

    merged = out.merge(
        clinical[
            ["_subject", "_visit", "csv_age", "csv_sex", "csv_education"]
        ].drop_duplicates(["_subject", "_visit"], keep="first"),
        how="left",
        on=["_subject", "_visit"],
    )

    # Fallback: if csv_visdate missing/unmatched, nearest clinical date within row.
    unmatched = merged["csv_age"].isna() & merged["csv_sex"].isna()
    if unmatched.any():
        fallback_rows = []
        clin_by_subject = {
            key: group for key, group in clinical.groupby("_subject", sort=False)
        }
        for idx in merged.index[unmatched]:
            subject = merged.at[idx, "_subject"]
            scan_date = _date(merged.at[idx, "scan_date"])
            group = clin_by_subject.get(subject)
            if group is None or group.empty or scan_date is pd.NaT:
                continue
            valid = group[group["_visit"].notna()]
            if valid.empty:
                continue
            distances = (valid["_visit"] - scan_date).abs()
            best = distances.idxmin()
            fallback_rows.append(
                (
                    idx,
                    valid.at[best, "csv_age"],
                    valid.at[best, "csv_sex"],
                    valid.at[best, "csv_education"],
                )
            )
        for idx, age, sex, edu in fallback_rows:
            merged.at[idx, "csv_age"] = age
            merged.at[idx, "csv_sex"] = sex
            merged.at[idx, "csv_education"] = edu

    merged["age"] = pd.to_numeric(merged["csv_age"], errors="coerce")
    merged["sex"] = merged["csv_sex"].map(
        lambda value: _canonical_sex(value, sex_mapping)
    )
    merged["education"] = merged["csv_education"].map(_clean_education)
    merged["age_missing"] = (~np.isfinite(merged["age"].to_numpy(dtype=float))).astype(
        int
    )
    merged["sex_missing"] = (merged["sex"] == "missing").astype(int)
    merged["education_missing"] = (
        ~np.isfinite(merged["education"].to_numpy(dtype=float))
    ).astype(int)
    merged["tr_mode"] = merged["tr_ms"].map(classify_tr_mode)
    merged["tr_raw"] = pd.to_numeric(merged["tr_ms"], errors="coerce")
    merged["metadata_match_quality"] = merged.apply(infer_metadata_match_quality, axis=1)
    merged["image_aligned"] = False
    return merged.drop(columns=["_subject", "_visit"], errors="ignore")


def mark_image_alignment(
    manifest: pd.DataFrame,
    image_root: str | Path,
    image_filename: str,
) -> pd.DataFrame:
    root = Path(image_root)
    folders = {
        path.parent.name
        for path in root.glob(f"*/{image_filename}")
        if path.is_file()
    }
    out = manifest.copy()
    out["image_aligned"] = out["pre_folder"].astype(str).isin(folders)
    out["image_path"] = out["pre_folder"].map(
        lambda folder: str(root / str(folder) / image_filename)
        if str(folder) in folders
        else ""
    )
    return out


@dataclass
class CohortManifestSpec:
    cohort: str
    scan_site_csv: str | Path
    clinical_csv: str | Path
    image_root: str | Path
    image_filename: str
    columns: Mapping[str, Sequence[str]]
    sex_mapping: Mapping | None = None


def build_cohort_manifest(spec: CohortManifestSpec) -> dict[str, Any]:
    raw = pd.read_csv(spec.scan_site_csv)
    kept, candidates = dedupe_pre_folder(raw)
    clinical = _load_clinical_table(spec.clinical_csv, spec.columns)
    enriched = attach_clinical_covariates(
        kept, clinical, sex_mapping=spec.sex_mapping
    )
    enriched = mark_image_alignment(
        enriched, spec.image_root, spec.image_filename
    )
    aligned = enriched[enriched["image_aligned"]].copy()
    unaligned = enriched[~enriched["image_aligned"]].copy()
    if not unaligned.empty:
        unaligned = unaligned.copy()
        unaligned["dedupe_action"] = unaligned.get(
            "dedupe_action", "drop_unaligned_pre_folder"
        )
        if "dedupe_reason" not in unaligned.columns:
            unaligned["dedupe_reason"] = "pre_folder_not_in_image_root"
        else:
            unaligned["dedupe_reason"] = unaligned["dedupe_reason"].fillna(
                "pre_folder_not_in_image_root"
            )
        candidates = pd.concat([candidates, unaligned], ignore_index=True)

    audit = {
        "cohort": spec.cohort,
        "rule_version": RULE_VERSION,
        "n_raw_rows": int(len(raw)),
        "n_manifest_rows": int(len(aligned)),
        "n_candidate_audit_rows": int(len(candidates)),
        "n_image_folders": int(
            len(list(Path(spec.image_root).glob(f"*/{spec.image_filename}")))
        ),
        "n_pre_folder_null": int(
            raw["pre_folder"].isna().sum()
            if "pre_folder" in raw.columns
            else 0
        ),
        "field_strength_counts": (
            aligned["field_strength"].astype(str).value_counts(dropna=False).to_dict()
            if "field_strength" in aligned.columns
            else {}
        ),
        "manufacturer_counts": (
            aligned["manufacturer"].astype(str).value_counts(dropna=False).to_dict()
            if "manufacturer" in aligned.columns
            else {}
        ),
        "tr_mode_counts": aligned["tr_mode"].value_counts(dropna=False).to_dict(),
        "metadata_match_quality_counts": aligned["metadata_match_quality"]
        .value_counts(dropna=False)
        .to_dict(),
        "n_age_missing": int(aligned["age_missing"].sum()) if len(aligned) else 0,
        "n_sex_missing": int(aligned["sex_missing"].sum()) if len(aligned) else 0,
        "n_education_missing": int(aligned["education_missing"].sum())
        if len(aligned)
        else 0,
        "inputs": {
            "scan_site_csv": str(spec.scan_site_csv),
            "clinical_csv": str(spec.clinical_csv),
            "image_root": str(spec.image_root),
            "image_filename": spec.image_filename,
        },
    }
    return {
        "manifest": aligned.reset_index(drop=True),
        "candidates": candidates.reset_index(drop=True),
        "audit": audit,
    }


def default_output_columns(manifest: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset",
        "subject_id",
        "scan_date",
        "pre_folder",
        "pre_label",
        "csv_visdate",
        "csv_label",
        "csv_match_days",
        "csv_matched",
        "csv_age",
        "csv_sex",
        "csv_education",
        "age",
        "sex",
        "education",
        "age_missing",
        "sex_missing",
        "education_missing",
        "site_id",
        "site_id_alt",
        "institution_name",
        "manufacturer",
        "scanner_model",
        "field_strength",
        "software_version",
        "receive_coil",
        "series_description",
        "series_type",
        "sequence_family",
        "sequence_name",
        "scanning_sequence",
        "acquisition_type",
        "acquisition_plane",
        "acceleration",
        "tr_ms",
        "tr_raw",
        "tr_mode",
        "te_ms",
        "ti_ms",
        "flip_angle",
        "slice_thickness",
        "pixel_spacing_x",
        "pixel_spacing_y",
        "matrix_rows",
        "matrix_cols",
        "n_slices",
        "acquisition_matrix",
        "source_path",
        "source_image_id",
        "metadata_match_quality",
        "image_aligned",
        "image_path",
    ]
    return [column for column in preferred if column in manifest.columns]


def write_manifest_bundle(
    *,
    output_dir: str | Path,
    cohort_results: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = {}
    combined_audit = {
        "rule_version": RULE_VERSION,
        "output_dir": str(output),
        "cohorts": {},
    }
    for cohort, payload in cohort_results.items():
        manifest_path = output / f"{cohort}_scan_manifest.csv"
        columns = default_output_columns(payload["manifest"])
        payload["manifest"][columns].to_csv(manifest_path, index=False)
        written[f"{cohort}_manifest"] = str(manifest_path)
        candidates = payload["candidates"]
        if cohort.upper() == "NACC" or not candidates.empty:
            cand_path = output / f"{cohort.lower()}_pre_folder_dedupe_candidates.csv"
            candidates.to_csv(cand_path, index=False)
            written[f"{cohort}_candidates"] = str(cand_path)
        combined_audit["cohorts"][cohort] = payload["audit"]
        combined_audit["cohorts"][cohort]["manifest_sha256"] = _sha256_file(
            manifest_path
        )
        combined_audit["cohorts"][cohort]["n_written"] = int(len(payload["manifest"]))

    audit_path = output / "scanner_protocol_audit.json"
    meta_path = output / "build_meta.json"
    audit_path.write_text(
        json.dumps(combined_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "rule_version": RULE_VERSION,
        "output_dir": str(output),
        "files": written,
        "cohort_row_counts": {
            cohort: int(len(payload["manifest"]))
            for cohort, payload in cohort_results.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    written["scanner_protocol_audit"] = str(audit_path)
    written["build_meta"] = str(meta_path)
    return {"files": written, "audit": combined_audit, "meta": meta}


def load_scan_manifest(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"pre_folder", "subject_id", "age", "sex", "education"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    return table


ACQUISITION_FIELDS = (
    "manufacturer",
    "field_strength",
    "scanner_model",
    "sequence_family",
    "tr_raw",
    "tr_mode",
    "te_ms",
    "ti_ms",
    "flip_angle",
    "slice_thickness",
    "pixel_spacing_x",
    "pixel_spacing_y",
    "acceleration",
    "site_id",
    "metadata_match_quality",
    "source_image_id",
)


def row_acquisition_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {}
    for field in ACQUISITION_FIELDS:
        value = row.get(field, None)
        if field in {
            "tr_raw",
            "te_ms",
            "ti_ms",
            "flip_angle",
            "slice_thickness",
            "pixel_spacing_x",
            "pixel_spacing_y",
            "field_strength",
        }:
            number = pd.to_numeric(value, errors="coerce")
            if not np.isfinite(number):
                payload[field] = float("nan")
                payload[f"{field}_missing"] = 1
            else:
                payload[field] = float(number)
                payload[f"{field}_missing"] = 0
        else:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                payload[field] = ""
                payload[f"{field}_missing"] = 1
            else:
                text = str(value).strip()
                if text in {"", "nan", "None"}:
                    payload[field] = ""
                    payload[f"{field}_missing"] = 1
                else:
                    payload[field] = text
                    payload[f"{field}_missing"] = 0
    return payload
