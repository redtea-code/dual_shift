"""Aggregate APIS v2 claim E1 metrics (primary: balanced_accuracy).

Smoke outputs must not be passed as --seed-root. Compares apis_v2 against
mixstyle and metadata with seed-level mean 螖 and optional subject-level
paired bootstrap when prediction CSVs are present.
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

VARIANTS = ("ce_only", "mixstyle", "metadata", "apis_v2")
DIRECTIONS = ("adni_to_nacc", "nacc_to_adni")
PRIMARY = "balanced_accuracy"
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
        # Historical alias directory name from smoke runs.
        if variant == "apis_v2":
            path = run_dir / "apis_only" / "journal_metrics.json"
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


def _load_pred_probs(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    subjects, labels, probs = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        prob_cols = sorted(
            [c for c in reader.fieldnames or [] if c.startswith("probability_")],
            key=lambda name: int(name.split("_")[1]),
        )
        if not prob_cols:
            return None
        for row in reader:
            subjects.append(str(row["subject_id"]))
            labels.append(int(row["label"]))
            probs.append([float(row[col]) for col in prob_cols])
    return np.asarray(subjects), np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def _subject_mean_table(subjects, labels, probs):
    unique = []
    seen = set()
    for subject in subjects.tolist():
        if subject not in seen:
            unique.append(subject)
            seen.add(subject)
    agg_probs, agg_labels, agg_subjects = [], [], []
    for subject in unique:
        mask = subjects == subject
        agg_probs.append(probs[mask].mean(axis=0))
        counts = np.bincount(labels[mask], minlength=probs.shape[1])
        agg_labels.append(int(np.argmax(counts)))
        agg_subjects.append(subject)
    return (
        np.asarray(agg_subjects, dtype=object),
        np.asarray(agg_labels, dtype=int),
        np.asarray(agg_probs, dtype=float),
    )


def _balanced_accuracy(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(labels, probs.argmax(axis=1)))


def _paired_delta_bootstrap(
    path_a: Path,
    path_b: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    loaded_a = _load_pred_probs(path_a)
    loaded_b = _load_pred_probs(path_b)
    if loaded_a is None or loaded_b is None:
        return None
    sub_a, lab_a, prob_a = _subject_mean_table(*loaded_a)
    sub_b, lab_b, prob_b = _subject_mean_table(*loaded_b)
    common = sorted(set(sub_a.tolist()).intersection(sub_b.tolist()))
    if len(common) < 5:
        return None
    index_a = {s: i for i, s in enumerate(sub_a.tolist())}
    index_b = {s: i for i, s in enumerate(sub_b.tolist())}
    labs = np.asarray([lab_a[index_a[s]] for s in common], dtype=int)
    pa = np.asarray([prob_a[index_a[s]] for s in common], dtype=float)
    pb = np.asarray([prob_b[index_b[s]] for s in common], dtype=float)
    point = _balanced_accuracy(labs, pa) - _balanced_accuracy(labs, pb)
    rng = np.random.RandomState(seed)
    boots = []
    n = len(common)
    for _ in range(n_bootstrap):
        draw = rng.randint(0, n, size=n)
        boots.append(
            _balanced_accuracy(labs[draw], pa[draw])
            - _balanced_accuracy(labs[draw], pb[draw])
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta_balanced_accuracy": float(point),
        "ci95": [float(lo), float(hi)],
        "n_subjects": int(n),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "positive_for_a": bool(point > 0 and lo > 0),
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
    args = parser.parse_args()
    if "smoke" in str(args.seed_root).replace("\\", "/").lower():
        raise SystemExit("refuse to aggregate smoke/ as claim E1 evidence")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    paired = {d: {} for d in DIRECTIONS}
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
                if metrics is not None:
                    for key in KEYS:
                        row[key] = _metric(metrics, key)
                    for stratum in ("1.5T", "3T"):
                        row[f"{stratum}_{PRIMARY}"] = _field_metric(
                            metrics, stratum, PRIMARY
                        )
                rows.append(row)
            if loaded.get("apis_v2") and loaded.get("mixstyle"):
                paired[direction][f"seed{seed}_vs_mixstyle"] = _paired_delta_bootstrap(
                    run_dir / "apis_v2" / "target_predictions.csv",
                    run_dir / "mixstyle" / "target_predictions.csv",
                    n_bootstrap=args.bootstrap,
                    seed=seed,
                )
            if loaded.get("apis_v2") and loaded.get("metadata"):
                paired[direction][f"seed{seed}_vs_metadata"] = _paired_delta_bootstrap(
                    run_dir / "apis_v2" / "target_predictions.csv",
                    run_dir / "metadata" / "target_predictions.csv",
                    n_bootstrap=args.bootstrap,
                    seed=seed,
                )

    summary: Dict[str, Any] = {
        "seeds": seeds,
        "primary_metric": PRIMARY,
        "rows": rows,
        "paired_bootstrap": paired,
        "aggregates": {},
        "claim_preview": {},
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
        for baseline in ("mixstyle", "metadata"):
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
            present_tests = [t for t in seed_tests if t]
            summary["claim_preview"][f"{direction}_vs_{baseline}"] = {
                "seed_mean_delta": _mean_std(deltas),
                "n_seeds_with_pred_ci": len(present_tests),
                "n_seeds_ci_positive": int(
                    sum(1 for t in present_tests if t.get("positive_for_a"))
                ),
                "all_present_ci_positive": bool(present_tests)
                and all(t.get("positive_for_a") for t in present_tests),
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
                "claim_preview": summary["claim_preview"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
