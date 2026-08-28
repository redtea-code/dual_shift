"""Evaluate a frozen original-CAPM checkpoint with the residual pilot.

The report contains four pre-registered cells: source control, source with the
target-stat correction, target control, and target with the correction.  Only
the last target cell is the proposed deployment path; source-adapted output is
an explicit preservation diagnostic rather than a tuning signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.journal_dataset import JournalSubset
from experiments.train_journal import (
    _dataset,
    _fit_protocol,
    _load_frozen_split,
    journal_collate,
    seed_everything,
)
from Model.ablation.residual_adaptation import (
    CAPMResidualStats,
    TARGET_SPLIT_SCHEMA,
    build_capm_residual_model,
)
from training.journal_metrics import compute_journal_metrics


def _load_model_state(model: torch.nn.Module, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a model_state/state_dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = [name for name in missing if name.startswith("residual_adapter.")]
    disallowed_missing = [name for name in missing if name not in allowed_missing]
    if disallowed_missing or unexpected:
        raise ValueError(
            "checkpoint does not match original_capm backbone: "
            f"missing={disallowed_missing[:5]}, unexpected={unexpected[:5]}"
        )


def _target_indices(target: Any, subjects: list[str]) -> np.ndarray:
    wanted = set(map(str, subjects))
    indices = np.asarray(
        [index for index, subject in enumerate(target.subject_ids.tolist()) if str(subject) in wanted],
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError("target test subjects do not match the target dataset")
    return indices


def _make_model(config: dict[str, Any], stats: CAPMResidualStats):
    ablation_cfg = config.get("scale_table_ablation") or {}
    input_shape = tuple(
        int(value)
        for value in ablation_cfg.get(
            "input_shape", config.get("training", {}).get("image_shape", (160, 196, 160))
        )
    )
    layers = tuple(int(value) for value in ablation_cfg.get("layers", (2, 2, 2, 2)))
    return build_capm_residual_model(
        stats,
        preset=str(ablation_cfg.get("preset", "layer4_pixel")),
        num_classes=2,
        layers=layers,
        spatial_shape=tuple(int(value) for value in ablation_cfg.get("spatial_shape", (4, 4, 4))),
        transformer_dim=int(ablation_cfg.get("transformer_dim", 128)),
        num_heads=int(ablation_cfg.get("num_heads", 4)),
        transformer_dropout=float(ablation_cfg.get("transformer_dropout", 0.1)),
        classifier_dropout=float(ablation_cfg.get("classifier_dropout", 0.3)),
        gate_init=float(ablation_cfg.get("gate_init", 0.95)),
        input_shape=input_shape,
        max_strength=float((config.get("capm_residual") or {}).get("max_strength", 0.25)),
    )


@torch.no_grad()
def _collect(model, loader, *, device: torch.device, apply_adaptation: bool) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[object] = []
    folders: list[object] = []
    environments: list[np.ndarray] = []
    audit_values: dict[str, list[float]] = {}
    try:
        for batch in loader:
            image = batch["image"].to(device)
            table = batch["covariates"].to(device)
            output, audit = model(
                image,
                table,
                apply_adaptation=apply_adaptation,
                return_audit=True,
            )
            logits.append(output.detach().cpu().numpy())
            labels.append(batch["label"].detach().cpu().numpy())
            environments.append(batch["environment_id"].detach().cpu().numpy())
            subjects.extend(batch["subject_id"])
            folders.extend(batch["folder"])
            for key, value in audit.items():
                if value.numel() == 1:
                    audit_values.setdefault(key, []).append(float(value.detach().cpu().reshape(-1)[0]))
    finally:
        model.train(was_training)
    prediction = np.concatenate(logits, axis=0)
    target = np.concatenate(labels, axis=0)
    metrics = compute_journal_metrics(
        prediction,
        target,
        np.concatenate(environments, axis=0),
        subject_ids=subjects,
        folders=folders,
        aggregate=str("subject_mean"),
    )
    metrics["audit_mean"] = {
        key: float(np.mean(values)) for key, values in audit_values.items() if values
    }
    metrics["n_scans"] = int(len(target))
    return metrics


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed_everything(args.seed)
    with Path(args.target_split).open(encoding="utf-8") as handle:
        target_split = json.load(handle)
    if target_split.get("schema") != TARGET_SPLIT_SCHEMA:
        raise ValueError("target split schema does not match the CAPM residual pilot")
    if target_split.get("target_labels_read") is not False or target_split.get("target_metrics_read") is not False:
        raise ValueError("target adaptation split must certify labels/metrics were not read")

    source_name, target_name = args.direction.split("_to_", maxsplit=1)
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    train_indices, val_indices, source_test_indices, source_manifest = _load_frozen_split(
        source, args.source_split
    )
    target_test_indices = _target_indices(target, target_split["target_test_subjects"])
    environment_config = config.get("environments") or {
        "mode": "age_sex",
        "n_age_bins": 3,
        "n_education_bins": 3,
        "min_group_size": 8,
        "min_per_class": 2,
        "scale_continuous": True,
    }
    preprocessor, builder, train_env, val_env, source_env, target_env = _fit_protocol(
        source,
        train_indices,
        val_indices,
        source_test_indices,
        target,
        environment_config,
        frozen_manifest=source_manifest,
        target_indices=target_test_indices,
    )
    source_set = JournalSubset(source, source_test_indices, preprocessor, source_env)
    target_set = JournalSubset(target, target_test_indices, preprocessor, target_env)
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": 0, "collate_fn": journal_collate}
    source_loader = DataLoader(source_set, **loader_kwargs)
    target_loader = DataLoader(target_set, **loader_kwargs)

    stats = CAPMResidualStats.load(args.stats)
    model = _make_model(config, stats).to(args.device)
    _load_model_state(model, args.checkpoint)
    device = torch.device(args.device)
    results = {
        "source_control": _collect(model, source_loader, device=device, apply_adaptation=False),
        "source_adapted_diagnostic": _collect(model, source_loader, device=device, apply_adaptation=True),
        "target_control": _collect(model, target_loader, device=device, apply_adaptation=False),
        "target_adapted": _collect(model, target_loader, device=device, apply_adaptation=True),
    }
    report = {
        "schema": "dualshift_capm_residual_pilot_report_v1",
        "direction": args.direction,
        "config": str(args.config),
        "source_split": str(args.source_split),
        "target_split": str(args.target_split),
        "stats": str(args.stats),
        "checkpoint": str(args.checkpoint),
        "target_labels_read_for_adaptation": False,
        "target_metrics_read_for_final_report": True,
        "selection_source_validation_only": True,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC", "NACC_to_ADNI"))
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_pilot(args), indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
