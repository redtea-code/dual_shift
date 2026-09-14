"""Build a label-free feature-frequency prior for one UDA direction and seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.frequency_uda import (
    ImageSubjectSubset,
    build_frequency_prior_from_loaders,
    earliest_subject_indices,
    save_target_adaptation_split,
    split_target_adaptation_indices,
)
from experiments.train_journal import _dataset, _load_frozen_split
from Model.ablation.frequency_uda import FrequencyGuidedScaleTable3D, FrequencyPrior


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _placeholder_prior() -> FrequencyPrior:
    summary = {"count": 1, "mean": [1 / 3, 1 / 3, 1 / 3], "std": [1.0, 1.0, 1.0]}
    return FrequencyPrior.from_summaries(summary, summary, metadata={"placeholder": True})


def _model_from_config(config: dict, checkpoint: str | Path) -> FrequencyGuidedScaleTable3D:
    ablation = config.get("scale_table_ablation") or {}
    model = FrequencyGuidedScaleTable3D(
        prior=_placeholder_prior(),
        preset="layer5_pixel",
        num_classes=len(set(config["task"]["label_mapping"].values())),
        layers=tuple(int(value) for value in ablation.get("layers", (2, 2, 2, 2))),
        spatial_shape=tuple(int(value) for value in ablation.get("spatial_shape", (4, 4, 4))),
        transformer_dim=int(ablation.get("transformer_dim", 128)),
        num_heads=int(ablation.get("num_heads", 4)),
        transformer_dropout=float(ablation.get("transformer_dropout", 0.1)),
        classifier_dropout=float(ablation.get("classifier_dropout", 0.3)),
        gate_init=float(ablation.get("gate_init", 0.95)),
        input_shape=tuple(int(value) for value in ablation.get("input_shape", (160, 196, 160))),
    )
    model.load_source_baseline(checkpoint)
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC", "NACC_to_ADNI"))
    parser.add_argument("--source-split-manifest", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--prior-output", required=True, type=Path)
    parser.add_argument("--target-split-output", required=True, type=Path)
    parser.add_argument("--adaptation-fraction", required=True, type=float)
    parser.add_argument("--adaptation-seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source_name, target_name = args.direction.split("_to_")
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    source_train_indices, _, _, _ = _load_frozen_split(
        source, str(args.source_split_manifest)
    )
    adaptation_indices, test_indices = split_target_adaptation_indices(
        target.subject_ids,
        adaptation_fraction=args.adaptation_fraction,
        seed=args.adaptation_seed,
    )
    source_prior_indices = earliest_subject_indices(
        source.subject_ids, source_train_indices, source.records
    )
    target_prior_indices = earliest_subject_indices(
        target.subject_ids, adaptation_indices, target.records
    )
    source_set = ImageSubjectSubset(source, source_prior_indices)
    target_adapt_set = ImageSubjectSubset(target, target_prior_indices)
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False}
    source_loader = DataLoader(source_set, **loader_kwargs)
    target_adapt_loader = DataLoader(target_adapt_set, **loader_kwargs)
    resolved_device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = _model_from_config(config, args.source_checkpoint).to(resolved_device)

    target_split = save_target_adaptation_split(
        args.target_split_output,
        target.subject_ids,
        adaptation_indices,
        test_indices,
        direction=args.direction,
        adaptation_fraction=args.adaptation_fraction,
        seed=args.adaptation_seed,
        metadata={
            "source_split_manifest_sha256": _sha256(args.source_split_manifest),
            "source_checkpoint_sha256": _sha256(args.source_checkpoint),
            "config_sha256": _sha256(args.config),
        },
    )
    prior = build_frequency_prior_from_loaders(
        model,
        source_loader,
        target_adapt_loader,
        device=resolved_device,
        output_path=args.prior_output,
        metadata={
            "direction": args.direction,
            "source_split_manifest_sha256": _sha256(args.source_split_manifest),
            "source_checkpoint_sha256": _sha256(args.source_checkpoint),
            "target_split_sha256": _sha256(args.target_split_output),
            "target_test_subject_digest": target_split["target_test_subject_digest"],
            "source_prior_selection": "earliest_visit_then_folder",
            "target_adapt_prior_selection": "earliest_visit_then_folder",
        },
    )
    print(
        json.dumps(
            {
                "prior_output": str(args.prior_output),
                "target_split_output": str(args.target_split_output),
                "source_count": prior.source_count,
                "target_adapt_count": prior.target_count,
                "discrepancy": prior.discrepancy,
                "target_labels_read": False,
                "target_metrics_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
