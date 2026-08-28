"""Build the label-free CAPM residual statistics artifact.

The command uses a frozen source ``original_capm`` checkpoint, a frozen source
split, and an explicit subject-disjoint target ``T_adapt`` split.  The two
image-only loaders expose only ``image`` and ``subject_id`` to the statistics
builder; target labels are never read in this preparation stage.
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
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.journal_dataset import CovariatePreprocessor
from experiments.train_journal import _dataset, _load_frozen_split, _make_model, seed_everything
from Model.ablation.residual_adaptation import (
    build_capm_residual_stats_from_loaders,
    save_capm_residual_target_split,
)


class ImageSubjectSubset(Dataset):
    """Expose only image and subject ID for label-blind statistics building."""

    def __init__(self, dataset: Any, indices: list[int] | np.ndarray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset.records[int(self.indices[index])]
        return {
            "image": self.dataset._load_image(record["path"]),
            "subject_id": record["subject_id"],
        }


def split_target_adaptation_indices(
    subject_ids: Any,
    *,
    adaptation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split target rows by subject without inspecting labels or predictions."""
    if not (0.0 < adaptation_fraction < 1.0):
        raise ValueError("adaptation_fraction must be strictly between zero and one")
    subject_array = np.asarray(subject_ids, dtype=object)
    unique_subjects = np.asarray(
        sorted({str(value) for value in subject_array.tolist()}), dtype=object
    )
    if len(unique_subjects) < 2:
        raise ValueError("At least two target subjects are required for adapt/test separation")
    generator = np.random.default_rng(int(seed))
    shuffled = unique_subjects[generator.permutation(len(unique_subjects))]
    n_adapt = min(
        max(1, int(round(len(unique_subjects) * adaptation_fraction))),
        len(unique_subjects) - 1,
    )
    adaptation_subjects = set(shuffled[:n_adapt].tolist())
    adaptation_indices = np.asarray(
        [index for index, subject in enumerate(subject_array.tolist()) if str(subject) in adaptation_subjects],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, subject in enumerate(subject_array.tolist()) if str(subject) not in adaptation_subjects],
        dtype=np.int64,
    )
    adapt_ids = set(map(str, subject_array[adaptation_indices].tolist()))
    test_ids = set(map(str, subject_array[test_indices].tolist()))
    if not len(adaptation_indices) or not len(test_indices) or adapt_ids.intersection(test_ids):
        raise RuntimeError("Target adaptation/test split is not subject-disjoint")
    return adaptation_indices, test_indices


def earliest_subject_indices(
    subject_ids: Any,
    indices: Any,
    records: list[dict[str, Any]],
) -> np.ndarray:
    """Choose one reproducible earliest scan per subject."""
    subject_array = np.asarray(subject_ids, dtype=object)
    selected: dict[str, tuple[tuple[str, str], int]] = {}
    for raw_index in indices:
        index = int(raw_index)
        record = records[index]
        key = (str(record.get("scan_date") or ""), str(record.get("folder") or ""))
        subject = str(subject_array[index])
        previous = selected.get(subject)
        if previous is None or key < previous[0]:
            selected[subject] = (key, index)
    return np.asarray(sorted(item[1] for item in selected.values()), dtype=np.int64)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_state(model: torch.nn.Module, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a model_state/state_dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "source checkpoint is not an exact original_capm model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def build_stats(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed_everything(args.seed)
    source_name, target_name = args.direction.split("_to_", maxsplit=1)
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    source_train, _, _, source_manifest = _load_frozen_split(source, args.source_split)
    adaptation_indices, test_indices = split_target_adaptation_indices(
        target.subject_ids,
        adaptation_fraction=args.adaptation_fraction,
        seed=args.seed,
    )
    split_payload = save_capm_residual_target_split(
        args.target_split,
        target.subject_ids,
        adaptation_indices,
        test_indices,
        direction=args.direction,
        adaptation_fraction=args.adaptation_fraction,
        seed=args.seed,
        metadata={
            "method": "capm_conditioned_residual_adaptation",
            "source_split": str(args.source_split),
            "source_split_sha256": _file_sha256(args.source_split),
            "target_labels_read": False,
            "target_metrics_read": False,
        },
    )

    ablation_cfg = config.get("scale_table_ablation") or {}
    preset = str(ablation_cfg.get("preset", "layer4_pixel"))
    if preset != "layer4_pixel":
        raise ValueError("CAPM residual pilot requires preset=layer4_pixel")
    model = _make_model(config, num_classes=2, variant="original_capm")
    _load_model_state(model, args.checkpoint)
    model.to(args.device)

    # CAPM statistics use a fixed source-only reference table.  This keeps the
    # T_adapt loader image-only while matching the frozen source preprocessing.
    preprocessing_cfg = config.get("environments") or {}
    preprocessor = CovariatePreprocessor(
        scale_continuous=bool(preprocessing_cfg.get("scale_continuous", True))
    ).fit(
        source.raw_age[source_train],
        source.raw_sex[source_train],
        source.raw_education[source_train],
        source.subject_ids[source_train],
    )
    if source_manifest.get("covariate_preprocessor"):
        preprocessor = CovariatePreprocessor.from_dict(source_manifest["covariate_preprocessor"])
    reference_covariates, _ = preprocessor.transform(
        source.raw_age[source_train],
        source.raw_sex[source_train],
        source.raw_education[source_train],
    )
    reference_table = torch.from_numpy(reference_covariates.mean(axis=0)).to(args.device)

    source_indices = earliest_subject_indices(source.subject_ids, source_train, source.records)
    target_indices = earliest_subject_indices(target.subject_ids, adaptation_indices, target.records)
    source_loader = DataLoader(
        ImageSubjectSubset(source, source_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    target_loader = DataLoader(
        ImageSubjectSubset(target, target_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    metadata = {
        "direction": args.direction,
        "preset": preset,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": _file_sha256(args.checkpoint),
        "source_split": str(args.source_split),
        "source_split_sha256": _file_sha256(args.source_split),
        "target_split": str(args.target_split),
        "target_split_schema": split_payload["schema"],
        "source_subject_count": len(set(map(str, source.subject_ids[source_indices].tolist()))),
        "target_adapt_subject_count": len(
            set(map(str, target.subject_ids[target_indices].tolist()))
        ),
        "target_labels_read": False,
        "target_metrics_read": False,
        "capm_reference_table": [float(value) for value in reference_table.cpu().tolist()],
    }
    stats = build_capm_residual_stats_from_loaders(
        model,
        source_loader,
        target_loader,
        device=args.device,
        output_path=args.output,
        reference_table=reference_table,
        metadata=metadata,
    )
    return {
        "stats": stats.to_dict(),
        "target_split": split_payload,
        "source_manifest_schema": source_manifest.get("schema"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC", "NACC_to_ADNI"))
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adaptation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = build_stats(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
