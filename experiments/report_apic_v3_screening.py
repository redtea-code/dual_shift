"""Build the pre-registered two-seed Gate S1 report for APIC v3."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apic_v3_protocol import APIC_V3_PRIMARY_VARIANTS, config_fingerprint
from experiments.run_apic_v3_screening import DEFAULT_CONFIGS, DIRECTIONS


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _task_name(config: dict) -> str:
    mapping = {int(key): int(value) for key, value in config["task"]["label_mapping"].items()}
    if mapping == {1: 0, 3: 1}:
        return "CN_vs_AD"
    if mapping == {2: 0, 3: 1}:
        return "MCI_vs_AD"
    raise ValueError(f"Unsupported APIC v3 task mapping: {mapping}")


def _load_rows(config_paths: list[Path]) -> list[dict]:
    rows = []
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        task = _task_name(config)
        root = PROJECT_ROOT / str(config["output_root"]) / "s1"
        for seed in (42, 43):
            seeded = copy.deepcopy(config)
            seeded["seed"] = seed
            expected_hash = config_fingerprint(seeded)
            for direction, slug in DIRECTIONS:
                for variant in APIC_V3_PRIMARY_VARIANTS:
                    path = root / f"seed{seed}" / slug / variant / "journal_metrics.json"
                    if not path.exists():
                        raise FileNotFoundError(f"Missing required screening result: {path}")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("config_hash") != expected_hash:
                        raise ValueError(f"Config hash mismatch: {path}")
                    if payload.get("input_modalities") != "X":
                        raise ValueError(f"Primary result is not image-only: {path}")
                    if bool(payload.get("uses_acquisition_metadata", True)):
                        raise ValueError(f"Primary result consumed acquisition metadata: {path}")
                    metrics = payload["target"]["metrics"]
                    audit = payload.get("apic_v3_audit") or {}
                    rows.append(
                        {
                            "task": task,
                            "direction": direction,
                            "seed": seed,
                            "variant": variant,
                            "balanced_accuracy": metrics.get("balanced_accuracy"),
                            "auc": metrics.get("auc"),
                            "macro_f1": metrics.get("macro_f1"),
                            "sensitivity": metrics.get("sensitivity"),
                            "specificity": metrics.get("specificity"),
                            "valid_memory_slots": audit.get("valid_slots"),
                        }
                    )
    return rows


def _config_hashes_seed42(config_paths: list[Path]) -> dict[str, str]:
    hashes = {}
    for path in config_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        hashes[path.name] = config_fingerprint(config)
    return hashes


def evaluate_gate(rows: list[dict], *, config_hashes_seed42: dict[str, str] | None = None) -> dict:
    keyed = {
        (row["task"], row["direction"], int(row["seed"]), row["variant"]): row
        for row in rows
    }
    cells = []
    wins_both = 0
    collapse_ok = True
    memory_ok = True
    for task in ("CN_vs_AD", "MCI_vs_AD"):
        for direction, _ in DIRECTIONS:
            deltas_ce = []
            deltas_mix = []
            for seed in (42, 43):
                apic = keyed[(task, direction, seed, "apic_v3_x")]
                ce = keyed[(task, direction, seed, "ce_x")]
                mix = keyed[(task, direction, seed, "mixstyle_x")]
                values = (
                    apic["balanced_accuracy"],
                    ce["balanced_accuracy"],
                    mix["balanced_accuracy"],
                )
                if not all(_finite(value) for value in values):
                    raise ValueError(f"Non-finite primary BA for {task} {direction} seed{seed}")
                delta_ce = float(values[0]) - float(values[1])
                delta_mix = float(values[0]) - float(values[2])
                deltas_ce.append(delta_ce)
                deltas_mix.append(delta_mix)
                wins_both += int(delta_ce > 0 and delta_mix > 0)
                for name in ("sensitivity", "specificity"):
                    collapse_ok = collapse_ok and _finite(apic.get(name)) and float(
                        apic[name]
                    ) >= 0.15
                memory_ok = memory_ok and int(apic.get("valid_memory_slots") or 0) >= 2
            mean_ce = sum(deltas_ce) / 2.0
            mean_mix = sum(deltas_mix) / 2.0
            cells.append(
                {
                    "task": task,
                    "direction": direction,
                    "mean_delta_ba_vs_ce": mean_ce,
                    "mean_delta_ba_vs_mixstyle": mean_mix,
                    "positive_vs_both": mean_ce > 0 and mean_mix > 0,
                }
            )
    positive_cells = sum(int(cell["positive_vs_both"]) for cell in cells)
    worst_delta = min(
        min(cell["mean_delta_ba_vs_ce"], cell["mean_delta_ba_vs_mixstyle"])
        for cell in cells
    )
    checks = {
        "positive_cells_ge_3_of_4": positive_cells >= 3,
        "seed_cells_winning_both_ge_5_of_8": wins_both >= 5,
        "worst_mean_delta_not_below_minus_0_02": worst_delta >= -0.02,
        "no_class_collapse": bool(collapse_ok),
        "style_memory_established": bool(memory_ok),
    }
    return {
        "gate": "APIC-V3-S1",
        "pass": all(checks.values()),
        "checks": checks,
        "positive_task_direction_cells": positive_cells,
        "seed_cells_winning_both": wins_both,
        "worst_mean_delta_ba": worst_delta,
        "config_hashes_seed42": config_hashes_seed42 or {},
        "cells": cells,
        "interpretation": (
            "Go to Phase P; two-seed results are not confirmatory."
            if all(checks.values())
            else "No-Go; do not start X+D or seeds 44--46."
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--output",
        default="outputs/journal/apic_v3_screening_gate_s1.json",
    )
    args = parser.parse_args(argv)
    paths = [Path(item) if Path(item).is_absolute() else PROJECT_ROOT / item for item in args.configs]
    rows = _load_rows(paths)
    report = evaluate_gate(rows, config_hashes_seed42=_config_hashes_seed42(paths))
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(output.with_suffix(".csv"), rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
