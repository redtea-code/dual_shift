"""Build DS-041 source-free residual distribution statistics.

The source checkpoint and optional task projector are loaded locally, then the
target adaptation view exposes only ``image`` and ``subject_id``. Target
labels/covariates are not exposed to the adaptation loader or read by the
statistics path.
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
from Model.ablation.capm_residual_distribution_alignment import (
    RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA,
    build_residual_distribution_stats_from_loaders,
    load_task_support_projector_artifact,
    save_residual_distribution_target_split,
)


class ImageSubjectSubset(Dataset):
    """Expose only image and subject ID to enforce the label-blind contract."""

    def __init__(self, dataset: Any, indices: np.ndarray) -> None:
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
    """Split target rows by subject without inspecting labels or metadata."""
    if not (0.0 < adaptation_fraction < 1.0):
        raise ValueError("adaptation_fraction must be strictly between zero and one")
    subject_array = np.asarray(subject_ids, dtype=object)
    unique_subjects = np.asarray(sorted({str(value) for value in subject_array}), dtype=object)
    if len(unique_subjects) < 2:
        raise ValueError("at least two target subjects are required")
    generator = np.random.default_rng(int(seed))
    shuffled = unique_subjects[generator.permutation(len(unique_subjects))]
    n_adapt = min(max(1, int(round(len(unique_subjects) * adaptation_fraction))), len(unique_subjects) - 1)
    adaptation_subjects = set(shuffled[:n_adapt].tolist())
    adaptation_indices = np.asarray(
        [i for i, subject in enumerate(subject_array) if str(subject) in adaptation_subjects], dtype=np.int64
    )
    test_indices = np.asarray(
        [i for i, subject in enumerate(subject_array) if str(subject) not in adaptation_subjects], dtype=np.int64
    )
    if not len(adaptation_indices) or not len(test_indices):
        raise RuntimeError("target adaptation/test split is empty")
    if set(map(str, subject_array[adaptation_indices])).intersection(
        set(map(str, subject_array[test_indices]))
    ):
        raise RuntimeError("target adaptation/test split is not subject-disjoint")
    return adaptation_indices, test_indices


def earliest_subject_indices(
    subject_ids: Any,
    indices: Any,
    records: list[dict[str, Any]],
) -> np.ndarray:
    """Select one deterministic earliest scan per subject."""
    subject_array = np.asarray(subject_ids, dtype=object)
    selected: dict[str, tuple[tuple[str, str], int]] = {}
    for raw_index in indices:
        index = int(raw_index)
        record = records[index]
        key = (str(record.get("scan_date") or ""), str(record.get("folder") or ""))
        subject = str(subject_array[index])
        if subject not in selected or key < selected[subject][0]:
            selected[subject] = (key, index)
    return np.asarray(sorted(item[1] for item in selected.values()), dtype=np.int64)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_digest(subject_ids: Any) -> str:
    payload = json.dumps(
        sorted({str(value) for value in subject_ids}), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_model_state(model: torch.nn.Module, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a model_state/state_dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The source checkpoint predates DS-041 and therefore has no transport or
    # projector buffers. Those buffers are initialized from the separately
    # built statistics/projector artifacts below; every backbone/CAPM key must
    # still match exactly.
    allowed_missing = ("transport.", "projector.")
    disallowed_missing = [
        name for name in missing if not name.startswith(allowed_missing)
    ]
    if disallowed_missing or unexpected:
        raise ValueError(
            "source checkpoint does not match original_capm: "
            f"missing={disallowed_missing[:5]}, unexpected={unexpected[:5]}"
        )


def build_stats(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime_cfg = config.get("capm_residual_distribution") or {}
    if args.seed is None:
        args.seed = int(runtime_cfg.get("split_seed", config.get("seed", 43)))
    if args.adaptation_fraction is None:
        args.adaptation_fraction = float(runtime_cfg.get("adaptation_fraction", 0.5))
    seed_everything(args.seed)
    source_name, target_name = args.direction.split("_to_", maxsplit=1)
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    source_train, _, _, source_manifest = _load_frozen_split(source, args.source_split)
    adapt_indices, test_indices = split_target_adaptation_indices(
        target.subject_ids, adaptation_fraction=args.adaptation_fraction, seed=args.seed
    )
    split_payload = save_residual_distribution_target_split(
        args.target_split,
        target.subject_ids,
        adapt_indices,
        test_indices,
        direction=args.direction,
        adaptation_fraction=args.adaptation_fraction,
        seed=args.seed,
        metadata={
            "method": "capm_conditioned_source_free_residual_distribution_alignment",
            "source_split": str(args.source_split),
            "source_split_sha256": _file_sha256(args.source_split),
            "target_labels_read": False,
            "target_metrics_read": False,
        },
    )
    projector = None
    if args.scope == "residual":
        if not args.projector:
            raise ValueError("--projector is required for residual scope")
        projector = load_task_support_projector_artifact(args.projector)

    ablation_cfg = config.get("scale_table_ablation") or {}
    if str(ablation_cfg.get("preset", "layer4_pixel")) != "layer4_pixel":
        raise ValueError("DS-041 requires preset=layer4_pixel")
    model = _make_model(config, num_classes=2, variant="original_capm")
    _load_model_state(model, args.checkpoint)
    model.to(args.device)

    environment_cfg = config.get("environments") or {}
    preprocessor = CovariatePreprocessor(
        scale_continuous=bool(environment_cfg.get("scale_continuous", True))
    ).fit(
        source.raw_age[source_train], source.raw_sex[source_train],
        source.raw_education[source_train], source.subject_ids[source_train],
    )
    if source_manifest.get("covariate_preprocessor"):
        preprocessor = CovariatePreprocessor.from_dict(source_manifest["covariate_preprocessor"])
    reference_covariates, _ = preprocessor.transform(
        source.raw_age[source_train], source.raw_sex[source_train], source.raw_education[source_train]
    )
    reference_table = torch.from_numpy(reference_covariates.mean(axis=0)).to(args.device)
    source_indices = earliest_subject_indices(source.subject_ids, source_train, source.records)
    target_indices = earliest_subject_indices(target.subject_ids, adapt_indices, target.records)
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": False, "num_workers": 0}
    source_loader = DataLoader(ImageSubjectSubset(source, source_indices), **loader_kwargs)
    target_loader = DataLoader(ImageSubjectSubset(target, target_indices), **loader_kwargs)
    metadata = {
        "direction": args.direction,
        "transport_scope": args.scope,
        "components": int(args.components),
        "gmm_iterations": int(args.gmm_iterations),
        "max_strength": float(runtime_cfg.get("max_strength", 0.25)),
        "config_sha256": _file_sha256(args.config),
        "builder_code_sha256": _file_sha256(__file__),
        "alignment_module_sha256": _file_sha256(
            PROJECT_ROOT / "Model" / "ablation" / "capm_residual_distribution_alignment.py"
        ),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": _file_sha256(args.checkpoint),
        "projector_artifact_sha256": (
            None if args.projector is None else _file_sha256(args.projector)
        ),
        "source_split": str(args.source_split),
        "source_split_sha256": _file_sha256(args.source_split),
        "target_split": str(args.target_split),
        "target_split_sha256": _file_sha256(args.target_split),
        "target_split_schema": RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA,
        "source_subject_count": len(set(map(str, source.subject_ids[source_indices]))),
        "target_adapt_subject_count": len(set(map(str, target.subject_ids[target_indices]))),
        "source_subject_digest": _subject_digest(source.subject_ids[source_indices]),
        "target_adapt_subject_digest": _subject_digest(target.subject_ids[target_indices]),
        "capm_reference_table": [float(value) for value in reference_table.cpu().tolist()],
        "target_labels_read": False,
        "target_metrics_read": False,
    }
    stats = build_residual_distribution_stats_from_loaders(
        model, source_loader, target_loader, device=args.device,
        output_path=args.output, reference_table=reference_table,
        projector=projector, components=args.components,
        gmm_iterations=args.gmm_iterations, metadata=metadata,
    )
    return {
        "stats": stats.to_dict(),
        "target_split": split_payload,
        "projector": None if projector is None else projector.to_dict(),
        "source_manifest_schema": source_manifest.get("schema"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC", "NACC_to_ADNI"))
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--projector")
    parser.add_argument("--scope", choices=("full", "residual"), default="residual")
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, choices=(1, 2), default=2)
    parser.add_argument("--gmm-iterations", type=int, default=30)
    parser.add_argument("--adaptation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.gmm_iterations < 1:
        raise ValueError("batch-size and gmm-iterations must be positive")
    result = build_stats(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
