"""Summarize APIC v3 training histories from completed screening artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HISTORY_FIELDS = (
    "train_loss",
    "val_loss",
    "val_auc",
    "val_balanced_accuracy",
    "selection_eligible",
    "selection_reasons",
    "clean_ce",
    "shift_ce",
    "js",
    "feature_consistency",
    "intervention_penalty",
    "apis_coefficient_l2",
    "valid_intervention_frac",
    "apis_feature_strength",
    "style_confidence",
    "style_entropy",
    "style_delta",
    "condition_gate",
    "style_memory_valid_slots",
    "style_memory_total_assignments",
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: Iterable) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else None


def _window(history: list[dict], first: int, last: int, key: str) -> float | None:
    return _mean(
        row.get(key)
        for row in history
        if first <= int(row.get("epoch", -1)) <= last
    )


def summarize_history(payload: dict) -> dict:
    history = list(payload.get("history") or [])
    if not history:
        raise ValueError("journal_metrics.json has no epoch history")
    scored = [row for row in history if _finite(row.get("val_auc"))]
    best = max(scored, key=lambda row: float(row["val_auc"])) if scored else {}
    last_epoch = max(int(row.get("epoch", 0)) for row in history)
    early_l2 = _window(history, 6, 10, "apis_coefficient_l2")
    late_l2 = _window(history, max(1, last_epoch - 9), last_epoch, "apis_coefficient_l2")
    ratio = None
    if early_l2 is not None and early_l2 > 0 and late_l2 is not None:
        ratio = late_l2 / early_l2
    target = ((payload.get("target") or {}).get("metrics") or {})
    audit = payload.get("apic_v3_audit") or {}
    return {
        "variant": payload.get("variant"),
        "epochs": len(history),
        "max_logged_composite_epoch": best.get("epoch"),
        "max_logged_validation_composite": best.get("val_auc"),
        "max_composite_epoch_val_loss": best.get("val_loss"),
        "best_checkpoint_epoch": payload.get("best_checkpoint_epoch"),
        "best_checkpoint_selection": payload.get("best_checkpoint_selection"),
        "final_train_loss": history[-1].get("train_loss"),
        "final_val_loss": history[-1].get("val_loss"),
        "early_l2_epoch6_10": early_l2,
        "late_l2_last10": late_l2,
        "late_to_early_l2_ratio": ratio,
        "early_feature_strength_epoch6_10": _window(
            history, 6, 10, "apis_feature_strength"
        ),
        "late_feature_strength_last10": _window(
            history, max(1, last_epoch - 9), last_epoch, "apis_feature_strength"
        ),
        "target_balanced_accuracy": target.get("balanced_accuracy"),
        "target_auc": target.get("auc"),
        "target_sensitivity": target.get("sensitivity"),
        "target_specificity": target.get("specificity"),
        "memory_valid_slots": audit.get("valid_slots"),
        "memory_total_assignments": audit.get("total_assignments"),
    }


def _identity(metrics_path: Path, root: Path) -> dict:
    relative = metrics_path.relative_to(root)
    parts = relative.parts
    seed = next((item for item in parts if item.startswith("seed")), "")
    direction = next(
        (item for item in parts if item in {"adni_to_nacc", "nacc_to_adni"}), ""
    )
    return {
        "artifact_root": str(root),
        "seed": seed.removeprefix("seed"),
        "direction": direction,
        "metrics_path": str(metrics_path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more screening s1 roots containing seed*/direction/variant outputs.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary_rows: list[dict] = []
    history_rows: list[dict] = []
    for raw_root in args.roots:
        root = Path(raw_root).resolve()
        paths = sorted(root.glob("seed*/**/journal_metrics.json"))
        if not paths:
            raise FileNotFoundError(f"No journal_metrics.json files under {root}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            identity = _identity(path, root)
            summary_rows.append({**identity, **summarize_history(payload)})
            for epoch in payload.get("history") or []:
                history_rows.append(
                    {
                        **identity,
                        "variant": payload.get("variant"),
                        "epoch": epoch.get("epoch"),
                        "phase": epoch.get("phase"),
                        **{key: epoch.get(key) for key in HISTORY_FIELDS},
                    }
                )
    output = Path(args.output_dir).resolve()
    _write_csv(output / "apic_v3_history_summary.csv", summary_rows)
    _write_csv(output / "apic_v3_epoch_history.csv", history_rows)
    (output / "apic_v3_history_summary.json").write_text(
        json.dumps(_json_safe(summary_rows), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summary_rows)} run summaries and {len(history_rows)} epochs to {output}")


if __name__ == "__main__":
    main()
