"""Aggregate APIS v2 claim E1 metrics (primary: balanced_accuracy).

Formal gate requires every preregistered seed and direction to be present.
Paired bootstrap requires identical subject sets and labels (no silent
intersection). Smoke paths are refused.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = ("ce_only", "mixstyle", "metadata", "metadata_xda", "apis_v2")
DIRECTIONS = ("adni_to_nacc", "nacc_to_adni")
PRIMARY = "balanced_accuracy"
PRIMARY_BASELINES = ("mixstyle", "metadata_xda")
REQUIRED_PROTOCOL_REVISION = 2
KEYS = (
    "balanced_accuracy",
    "auc",
    "macro_f1",
    "sensitivity",
    "specificity",
    "accuracy",
    "brier",
    "ece",
)


def _metric(block: Dict[str, Any], key: str):
    metrics = (block.get("target") or {}).get("metrics") or {}
    return metrics.get(key)


def _field_metric(block: Dict[str, Any], stratum: str, key: str):
    strata = (block.get("target") or {}).get("metrics_by_field_strength") or {}
    metrics = strata.get(stratum) or {}
    return metrics.get(key)


def _load_variant(run_dir: Path, variant: str) -> Optional[Dict[str, Any]]:
    path = run_dir / variant / "journal_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": mean, "std": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": len(vals)}


def _load_pred_table(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    subjects, folders, labels, probs = [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        prob_cols = sorted(
            [c for c in fields if c.startswith("probability_")],
            key=lambda name: int(name.split("_")[1]),
        )
        if not prob_cols:
            return None
        for row in reader:
            subjects.append(str(row["subject_id"]))
            folders.append(str(row.get("folder", "")))
            labels.append(int(row["label"]))
            probs.append([float(row[col]) for col in prob_cols])
    return (
        np.asarray(subjects, dtype=object),
        np.asarray(folders, dtype=object),
        np.asarray(labels, dtype=int),
        np.asarray(probs, dtype=float),
    )


def _subject_mean_table(subjects, folders, labels, probs):
    from training.journal_metrics import aggregate_subject_predictions

    agg_probs, agg_labels, _, agg_subjects = aggregate_subject_predictions(
        probs,
        labels,
        subjects,
        folders=folders,
        label_conflict="earliest_visit",
    )
    # Deterministic folder token per kept subject (earliest visit already applied).
    agg_folders = []
    for subject in agg_subjects.tolist():
        mask = subjects == subject
        folder_vals = folders[mask]
        dated = sorted((_parse_folder_date(f), str(f)) for f in folder_vals.tolist())
        agg_folders.append(dated[0][1] if dated else "")
    return (
        np.asarray(agg_subjects, dtype=object),
        np.asarray(agg_folders, dtype=object),
        np.asarray(agg_labels, dtype=int),
        np.asarray(agg_probs, dtype=float),
    )


def _parse_folder_date(folder: object):
    from training.journal_metrics import _parse_folder_date as parse

    return parse(folder)


def _balanced_accuracy(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(labels, probs.argmax(axis=1)))


def _stratified_subject_indices(labels: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    pieces = []
    for label in np.unique(labels):
        pool = np.flatnonzero(labels == label)
        if len(pool) == 0:
            continue
        pieces.append(rng.choice(pool, size=len(pool), replace=True))
    if not pieces:
        return rng.randint(0, len(labels), size=len(labels))
    indices = np.concatenate(pieces)
    rng.shuffle(indices)
    return indices


def _paired_delta_bootstrap(
    path_a: Path,
    path_b: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    loaded_a = _load_pred_table(path_a)
    loaded_b = _load_pred_table(path_b)
    if loaded_a is None or loaded_b is None:
        raise ValueError(f"missing prediction CSV for paired bootstrap: {path_a} / {path_b}")
    sub_a, fold_a, lab_a, prob_a = _subject_mean_table(*loaded_a)
    sub_b, fold_b, lab_b, prob_b = _subject_mean_table(*loaded_b)
    if sub_a.tolist() != sub_b.tolist():
        raise ValueError(
            "paired bootstrap requires identical subject order/set; "
            f"got {len(sub_a)} vs {len(sub_b)} subjects"
        )
    if not np.array_equal(lab_a, lab_b):
        raise ValueError("paired bootstrap requires identical subject labels")
    if fold_a.tolist() != fold_b.tolist():
        raise ValueError("paired bootstrap requires identical subject folder tokens")
    point = _balanced_accuracy(lab_a, prob_a) - _balanced_accuracy(lab_a, prob_b)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_bootstrap):
        draw = _stratified_subject_indices(lab_a, rng)
        boots.append(
            _balanced_accuracy(lab_a[draw], prob_a[draw])
            - _balanced_accuracy(lab_a[draw], prob_b[draw])
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta_balanced_accuracy": float(point),
        "ci95": [float(lo), float(hi)],
        "n_subjects": int(len(sub_a)),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "positive_for_a": bool(point > 0 and lo > 0),
        "bootstrap": "subject_stratified_by_label",
        "aggregation": "earliest_visit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/journal/dual_shift_apis_v2/claim/e1",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/journal/dual_shift_apis_v2/claim/e1",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Dev-only: skip formal completeness gate (never for claim freeze).",
    )
    args = parser.parse_args()
    if "smoke" in str(args.seed_root).replace("\\", "/").lower():
        raise SystemExit("refuse to aggregate smoke/ as claim E1 evidence")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    paired = {d: {} for d in DIRECTIONS}
    missing = []
    for seed in seeds:
        for direction in DIRECTIONS:
            run_dir = args.seed_root / f"seed{seed}" / direction
            loaded = {v: _load_variant(run_dir, v) for v in VARIANTS}
            for variant in VARIANTS:
                metrics = loaded[variant]
                row = {
                    "seed": seed,
                    "direction": direction,
                    "variant": variant,
                    "present": metrics is not None,
                }
                if metrics is None:
                    missing.append(f"seed{seed}/{direction}/{variant}")
                else:
                    if metrics.get("claim_protocol_revision") != REQUIRED_PROTOCOL_REVISION:
                        missing.append(
                            f"seed{seed}/{direction}/{variant}:stale_revision"
                        )
                        row["present"] = False
                    for key in KEYS:
                        row[key] = _metric(metrics, key)
                    for stratum in ("1.5T", "3T"):
                        row[f"{stratum}_{PRIMARY}"] = _field_metric(
                            metrics, stratum, PRIMARY
                        )
                rows.append(row)
            if all(loaded.get(v) is not None for v in ("apis_v2", "mixstyle")):
                paired[direction][f"seed{seed}_vs_mixstyle"] = _paired_delta_bootstrap(
                    run_dir / "apis_v2" / "target_predictions.csv",
                    run_dir / "mixstyle" / "target_predictions.csv",
                    n_bootstrap=args.bootstrap,
                    seed=seed,
                )
            if all(loaded.get(v) is not None for v in ("apis_v2", "metadata_xda")):
                paired[direction][f"seed{seed}_vs_metadata_xda"] = _paired_delta_bootstrap(
                    run_dir / "apis_v2" / "target_predictions.csv",
                    run_dir / "metadata_xda" / "target_predictions.csv",
                    n_bootstrap=args.bootstrap,
                    seed=seed,
                )
            if all(loaded.get(v) is not None for v in ("apis_v2", "metadata")):
                paired[direction][f"seed{seed}_vs_metadata"] = _paired_delta_bootstrap(
                    run_dir / "apis_v2" / "target_predictions.csv",
                    run_dir / "metadata" / "target_predictions.csv",
                    n_bootstrap=args.bootstrap,
                    seed=seed,
                )

    complete = len(missing) == 0
    if not complete and not args.allow_incomplete:
        raise SystemExit(
            "claim E1 incomplete; refusing formal gate. missing="
            + ",".join(missing[:20])
            + ("..." if len(missing) > 20 else "")
        )

    summary: Dict[str, Any] = {
        "seeds": seeds,
        "primary_metric": PRIMARY,
        "complete": complete,
        "missing": missing,
        "rows": rows,
        "paired_bootstrap": paired,
        "aggregates": {},
        "claim_gate": {},
    }
    for direction in DIRECTIONS:
        summary["aggregates"][direction] = {}
        for variant in VARIANTS:
            summary["aggregates"][direction][variant] = {}
            for key in KEYS:
                vals = [
                    r[key]
                    for r in rows
                    if r["direction"] == direction
                    and r["variant"] == variant
                    and r.get("present")
                    and key in r
                ]
                summary["aggregates"][direction][variant][key] = _mean_std(vals)
        for baseline in PRIMARY_BASELINES:
            deltas = []
            for seed in seeds:
                a = next(
                    (
                        r[PRIMARY]
                        for r in rows
                        if r["seed"] == seed
                        and r["direction"] == direction
                        and r["variant"] == "apis_v2"
                        and r.get("present")
                    ),
                    None,
                )
                b = next(
                    (
                        r[PRIMARY]
                        for r in rows
                        if r["seed"] == seed
                        and r["direction"] == direction
                        and r["variant"] == baseline
                        and r.get("present")
                    ),
                    None,
                )
                if a is not None and b is not None:
                    deltas.append(float(a) - float(b))
            summary["aggregates"][direction][f"delta_apis_v2_minus_{baseline}"] = (
                _mean_std(deltas)
            )
            seed_tests = [
                paired[direction].get(f"seed{seed}_vs_{baseline}")
                for seed in seeds
            ]
            all_seeds_present = all(t is not None for t in seed_tests)
            summary["claim_gate"][f"{direction}_vs_{baseline}"] = {
                "seed_mean_delta": _mean_std(deltas),
                "n_preregistered_seeds": len(seeds),
                "n_seeds_with_pred_ci": int(sum(1 for t in seed_tests if t)),
                "n_seeds_ci_positive": int(
                    sum(1 for t in seed_tests if t and t.get("positive_for_a"))
                ),
                "all_preregistered_seeds_present": bool(all_seeds_present and complete),
                "all_preregistered_ci_positive": bool(all_seeds_present)
                and bool(complete)
                and all(bool(t.get("positive_for_a")) for t in seed_tests),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "gate_report_claim.json"
    out_csv = args.output_dir / "metrics_table_claim.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = [
        "seed",
        "direction",
        "variant",
        "present",
        *KEYS,
        f"1.5T_{PRIMARY}",
        f"3T_{PRIMARY}",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "wrote": str(out_json),
                "complete": complete,
                "claim_gate": summary["claim_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
