"""Evaluate a frozen DS-041 residual-distribution transport model.

The evaluator reports matched control/adapted source and target cells. Target
labels are read only in the final exploratory report; the statistics artifact
and transport path are checked for the label-blind source-free contract.
"""

from __future__ import annotations

import argparse
import hashlib
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
from Model.ablation.capm_residual_distribution_alignment import (
    RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA,
    CAPMResidualDistributionAlignment3D,
    ResidualDistributionStats,
    apply_bounded_intensity_perturbation,
    apply_fourier_amplitude_perturbation,
    audit_synthetic_perturbation,
    load_task_support_projector_artifact,
)
from Model.ablation.capm_frequency_grl import TaskSupportProjector
from training.journal_metrics import compute_journal_metrics


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_state(model: torch.nn.Module, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a model_state/state_dict mapping")
    # The source checkpoint is authoritative only for the frozen backbone and
    # original CAPM. Never let a stale DS-041 transport/projector buffer in a
    # reused checkpoint override the separately verified artifacts.
    backbone_state = {
        name: value
        for name, value in state.items()
        if not name.startswith(("transport.", "projector."))
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    allowed_prefixes = ("transport.", "projector.")
    disallowed_missing = [name for name in missing if not name.startswith(allowed_prefixes)]
    if disallowed_missing or unexpected:
        raise ValueError(
            "checkpoint does not match original_capm backbone: "
            f"missing={disallowed_missing[:5]}, unexpected={unexpected[:5]}"
        )


def _target_indices(target: Any, subjects: list[str]) -> np.ndarray:
    wanted = set(map(str, subjects))
    indices = np.asarray(
        [i for i, subject in enumerate(target.subject_ids) if str(subject) in wanted], dtype=np.int64
    )
    if not len(indices):
        raise ValueError("target split subjects do not match target dataset")
    return indices


def _subject_digest(subject_ids: Any) -> str:
    payload = json.dumps(
        sorted({str(value) for value in subject_ids}), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_model(
    config: dict[str, Any],
    stats: ResidualDistributionStats,
    projector: TaskSupportProjector | None,
    *,
    scope: str,
) -> CAPMResidualDistributionAlignment3D:
    ablation_cfg = config.get("scale_table_ablation") or {}
    input_shape = tuple(
        int(value)
        for value in ablation_cfg.get(
            "input_shape", config.get("training", {}).get("image_shape", (160, 196, 160))
        )
    )
    layers = tuple(int(value) for value in ablation_cfg.get("layers", (1, 1, 1, 1)))
    return CAPMResidualDistributionAlignment3D(
        stats=stats,
        projector=projector,
        transport_scope=scope,
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
        max_strength=float((config.get("capm_residual_distribution") or {}).get("max_strength", 0.25)),
    )


@torch.no_grad()
def _collect(
    model: CAPMResidualDistributionAlignment3D,
    loader: DataLoader,
    *,
    device: torch.device,
    apply_transport: bool,
) -> dict[str, Any]:
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
            output, audit = model(
                batch["image"].to(device),
                batch["covariates"].to(device),
                apply_transport=apply_transport,
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
        aggregate="subject_mean",
    )
    metrics["audit_mean"] = {
        key: float(np.mean(values)) for key, values in audit_values.items() if values
    }
    metrics["n_scans"] = int(len(target))
    # Keep row-level predictions alongside aggregate metrics so the audit
    # artifact remains independently inspectable. These are read only after
    # the label-blind adaptation/statistics phase.
    probabilities = torch.softmax(torch.from_numpy(prediction), dim=1).numpy()
    metrics["predictions"] = prediction.argmax(axis=1).astype(int).tolist()
    metrics["probabilities"] = probabilities.astype(float).tolist()
    metrics["labels"] = target.astype(int).tolist()
    metrics["subject_ids"] = [str(value) for value in subjects]
    metrics["folders"] = [str(value) for value in folders]
    metrics["environment_ids"] = np.concatenate(environments, axis=0).astype(int).tolist()
    return metrics


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(args.target_split).open(encoding="utf-8") as handle:
        target_split = json.load(handle)
    runtime_cfg = config.get("capm_residual_distribution") or {}
    if args.seed is None:
        args.seed = int(target_split.get("seed", runtime_cfg.get("split_seed", config.get("seed", 43))))
    seed_everything(args.seed)
    if target_split.get("schema") != RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA:
        raise ValueError("target split schema does not match DS-041")
    if str(target_split.get("direction")) != str(args.direction):
        raise ValueError("target split direction does not match requested direction")
    if target_split.get("target_labels_read") is not False or target_split.get("target_metrics_read") is not False:
        raise ValueError("target split must certify labels/metrics were not read")
    stats = ResidualDistributionStats.load(args.stats)
    if stats.metadata.get("target_labels_read") is not False:
        raise ValueError("stats artifact does not certify target_labels_read=false")
    if stats.metadata.get("target_metrics_read") is not False:
        raise ValueError("stats artifact does not certify target_metrics_read=false")
    stats_direction = stats.metadata.get("direction")
    if stats_direction is not None and str(stats_direction) != str(args.direction):
        raise ValueError("statistics direction does not match requested direction")
    recorded_scope = stats.metadata.get("transport_scope")
    if recorded_scope is not None and str(recorded_scope) != args.scope:
        raise ValueError(
            f"stats transport scope {recorded_scope!r} does not match requested {args.scope!r}"
        )
    configured_strength = float(runtime_cfg.get("max_strength", 0.25))
    recorded_strength = stats.metadata.get("max_strength")
    if recorded_strength is not None and not np.isclose(float(recorded_strength), configured_strength):
        raise ValueError("statistics max_strength does not match the resolved configuration")
    projector = None
    if args.scope == "residual":
        if not args.projector:
            raise ValueError("--projector is required for residual scope")
        projector = load_task_support_projector_artifact(args.projector)

    source_name, target_name = args.direction.split("_to_", maxsplit=1)
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    train_indices, val_indices, source_test_indices, source_manifest = _load_frozen_split(
        source, args.source_split
    )
    target_adapt_subjects = set(map(str, target_split.get("target_adapt_subjects", [])))
    target_test_subjects = set(map(str, target_split["target_test_subjects"]))
    if not target_adapt_subjects or target_adapt_subjects.intersection(target_test_subjects):
        raise ValueError("target split must contain non-empty, disjoint adaptation/test subjects")
    if target_split.get("target_adapt_subject_digest") != _subject_digest(target_adapt_subjects):
        raise ValueError("target adaptation subject digest does not match target split")
    if target_split.get("target_test_subject_digest") != _subject_digest(target_test_subjects):
        raise ValueError("target test subject digest does not match target split")
    available_target_subjects = set(map(str, target.subject_ids))
    if not target_adapt_subjects.issubset(available_target_subjects):
        raise ValueError("target adaptation split contains unknown subjects")
    if not target_test_subjects.issubset(available_target_subjects):
        raise ValueError("target test split contains unknown subjects")
    target_test_indices = _target_indices(target, target_split["target_test_subjects"])
    environment_config = config.get("environments") or {
        "mode": "age_sex",
        "n_age_bins": 3,
        "n_education_bins": 3,
        "min_group_size": 8,
        "min_per_class": 2,
        "scale_continuous": True,
    }
    preprocessor, _builder, train_env, val_env, source_env, target_env = _fit_protocol(
        source,
        train_indices,
        val_indices,
        source_test_indices,
        target,
        environment_config,
        frozen_manifest=source_manifest,
        target_indices=target_test_indices,
    )
    source_val_set = JournalSubset(source, val_indices, preprocessor, val_env)
    source_set = JournalSubset(source, source_test_indices, preprocessor, source_env)
    target_set = JournalSubset(target, target_test_indices, preprocessor, target_env)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": journal_collate,
    }
    source_loader = DataLoader(source_set, **loader_kwargs)
    source_val_loader = DataLoader(source_val_set, **loader_kwargs)
    target_loader = DataLoader(target_set, **loader_kwargs)
    model = _make_model(config, stats, projector, scope=args.scope).to(args.device)
    _load_model_state(model, args.checkpoint)
    device = torch.device(args.device)
    results = {
        "source_validation": _collect(model, source_val_loader, device=device, apply_transport=False),
        "source_control": _collect(model, source_loader, device=device, apply_transport=False),
        "source_adapted_diagnostic": _collect(model, source_loader, device=device, apply_transport=True),
        "target_control": _collect(model, target_loader, device=device, apply_transport=False),
        "target_adapted": _collect(model, target_loader, device=device, apply_transport=True),
    }
    report = {
        "schema": "dualshift_ds041_residual_distribution_pilot_v1",
        "status": "exploratory_pilot",
        "direction": args.direction,
        "seed": int(args.seed),
        "config": str(args.config),
        "source_split": str(args.source_split),
        "target_split": str(args.target_split),
        "target_split_sha256": _file_sha256(args.target_split),
        "stats": str(args.stats),
        "stats_sha256": _file_sha256(args.stats),
        "config_sha256": _file_sha256(args.config),
        "pilot_code_sha256": _file_sha256(__file__),
        "alignment_module_sha256": _file_sha256(
            PROJECT_ROOT / "Model" / "ablation" / "capm_residual_distribution_alignment.py"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "projector_artifact_sha256": (
            None if args.projector is None else _file_sha256(args.projector)
        ),
        "projector": None if projector is None else projector.to_dict(),
        "transport_scope": args.scope,
        "adaptation_capm_table": "source_training_covariate_mean",
        "target_test_capm_table": "ordinary_target_covariates_final_report_only",
        "results": results,
        "target_labels_read_for_adaptation": False,
        "target_metrics_read_for_final_report": True,
        "selection_source_validation_only": True,
        "source_test_subject_digest": _subject_digest(source.subject_ids[source_test_indices]),
        "target_test_subject_digest": _subject_digest(target.subject_ids[target_test_indices]),
        "model_signature": model.experiment_signature(),
    }
    if args.synthetic_audit:
        clean_batches: list[torch.Tensor] = []
        table_batches: list[torch.Tensor] = []
        for batch in source_loader:
            clean_batches.append(batch["image"])
            table_batches.append(batch["covariates"])
        clean_images = torch.cat(clean_batches, dim=0).to(device)
        synthetic_images = apply_bounded_intensity_perturbation(
            clean_images, scale=args.synthetic_scale, bias=args.synthetic_bias
        )
        synthetic_images = apply_fourier_amplitude_perturbation(
            synthetic_images, amplitude_scale=args.synthetic_fourier_scale
        )
        report["synthetic_perturbation"] = audit_synthetic_perturbation(
            model,
            clean_images,
            synthetic_images,
            torch.cat(table_batches, dim=0).to(device),
            device=device,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    output.with_name("audit.json").write_text(
        json.dumps(
            {
                "schema": "dualshift_ds041_residual_distribution_audit_v1",
                "direction": args.direction,
                "seed": int(args.seed),
                "transport_scope": args.scope,
                "adaptation_capm_table": "source_training_covariate_mean",
                "target_test_capm_table": "ordinary_target_covariates_final_report_only",
                "components": stats.components,
                "target_labels_read_for_adaptation": False,
                "target_metrics_read_for_final_report": True,
                "selection_source_validation_only": True,
                "stats_sha256": report["stats_sha256"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "target_split_sha256": report["target_split_sha256"],
                "config_sha256": report["config_sha256"],
                "pilot_code_sha256": report["pilot_code_sha256"],
                "alignment_module_sha256": report["alignment_module_sha256"],
                "projector_artifact_sha256": report["projector_artifact_sha256"],
                "target_test_subject_digest": report["target_test_subject_digest"],
                "model_signature": report["model_signature"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output.with_name("predictions.json").write_text(
        json.dumps(
            {
                "source_control": results["source_control"].get("predictions", []),
                "source_validation": results["source_validation"].get("predictions", []),
                "source_adapted_diagnostic": results["source_adapted_diagnostic"].get("predictions", []),
                "target_control": results["target_control"].get("predictions", []),
                "target_adapted": results["target_adapted"].get("predictions", []),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    torch.save({"model_state": model.state_dict(), "report_schema": report["schema"]}, output.with_name("best.pt"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC", "NACC_to_ADNI"))
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--projector")
    parser.add_argument("--scope", choices=("full", "residual"), default="residual")
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-audit", action="store_true")
    parser.add_argument("--synthetic-scale", type=float, default=1.05)
    parser.add_argument("--synthetic-bias", type=float, default=0.02)
    parser.add_argument("--synthetic-fourier-scale", type=float, default=1.01)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run_pilot(args)


if __name__ == "__main__":
    main()
