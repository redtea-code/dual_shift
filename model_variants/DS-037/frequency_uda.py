"""Leakage-safe utilities for the frequency-guided UDA preparation stage.

This module intentionally has no target-label argument.  It partitions target
subjects by identifier, extracts only ``layer4`` image features, and writes the
label-free source/target-adaptation frequency prior consumed by the new model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from Model.ablation.frequency_uda import (
    FeatureSpectrumAccumulator,
    FrequencyPrior,
)


TARGET_SPLIT_SCHEMA = "dualshift_frequency_uda_target_split_v1"


class ImageSubjectSubset(Dataset):
    """Emit only image tensors and subject IDs for frequency-prior extraction."""

    def __init__(self, dataset: Any, indices: Sequence[int]):
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


def _subject_digest(subject_ids: Sequence[object]) -> str:
    payload = "\n".join(sorted(map(str, subject_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_target_adaptation_indices(
    subject_ids: Sequence[object],
    *,
    adaptation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split target data by subject without inspecting labels or predictions."""
    if not (0.0 < adaptation_fraction < 1.0):
        raise ValueError("adaptation_fraction must be strictly between zero and one")
    subject_array = np.asarray(subject_ids, dtype=object)
    unique_subjects = np.asarray(sorted({str(value) for value in subject_array.tolist()}), dtype=object)
    if len(unique_subjects) < 2:
        raise ValueError("At least two target subjects are required for adapt/test separation")
    generator = np.random.default_rng(int(seed))
    shuffled = unique_subjects[generator.permutation(len(unique_subjects))]
    n_adapt = min(max(1, int(round(len(unique_subjects) * adaptation_fraction))), len(unique_subjects) - 1)
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
    subject_ids: Sequence[object],
    indices: Sequence[int],
    records: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Choose one reproducible earliest scan per subject for prior estimation."""
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


def save_target_adaptation_split(
    path: str | Path,
    subject_ids: Sequence[object],
    adaptation_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    direction: str,
    adaptation_fraction: float,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the target partition without labels, scores, or predictions."""
    subject_array = np.asarray(subject_ids, dtype=object)
    adapt_ids = sorted({str(subject_array[int(index)]) for index in adaptation_indices})
    test_ids = sorted({str(subject_array[int(index)]) for index in test_indices})
    if not adapt_ids or not test_ids or set(adapt_ids).intersection(test_ids):
        raise ValueError("target adaptation/test subjects must be non-empty and disjoint")
    payload = {
        "schema": TARGET_SPLIT_SCHEMA,
        "direction": str(direction),
        "adaptation_fraction": float(adaptation_fraction),
        "seed": int(seed),
        "target_labels_read": False,
        "target_metrics_read": False,
        "target_adapt_subjects": adapt_ids,
        "target_test_subjects": test_ids,
        "target_adapt_subject_digest": _subject_digest(adapt_ids),
        "target_test_subject_digest": _subject_digest(test_ids),
        "metadata": dict(metadata or {}),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


@torch.no_grad()
def build_frequency_prior_from_loaders(
    model: Any,
    source_loader: Iterable[dict[str, Any]],
    target_adapt_loader: Iterable[dict[str, Any]],
    *,
    device: torch.device | str,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> FrequencyPrior:
    """Create a prior from source and target-adaptation images only.

    The loader contract is deliberately limited to the ``image`` key.  Labels,
    target logits, metrics, and checkpoint selection are not read here.
    """
    if not hasattr(model, "extract_layer4"):
        raise TypeError("model must expose extract_layer4(image) for frequency prior building")
    previous_mode = bool(model.training)
    model.eval()
    source_accumulator = FeatureSpectrumAccumulator()
    target_accumulator = FeatureSpectrumAccumulator()
    source_subjects: list[object] = []
    target_subjects: list[object] = []

    def accumulate(loader: Iterable[dict[str, Any]], accumulator: FeatureSpectrumAccumulator, subjects: list[object]) -> None:
        for batch in loader:
            if "image" not in batch:
                raise KeyError("frequency-prior loader batches must contain image")
            images = batch["image"].to(device)
            features: Tensor = model.extract_layer4(images)
            accumulator.update(features)
            subjects.extend(batch.get("subject_id", []))

    try:
        accumulate(source_loader, source_accumulator, source_subjects)
        accumulate(target_adapt_loader, target_accumulator, target_subjects)
    finally:
        model.train(previous_mode)

    supplied = dict(metadata or {})
    supplied.update(
        {
            "method": "label_free_layer4_feature_frequency_prior",
            "target_labels_read": False,
            "target_metrics_read": False,
            "source_subject_digest": _subject_digest(source_subjects),
            "target_adapt_subject_digest": _subject_digest(target_subjects),
            "source_subject_count": len(set(map(str, source_subjects))),
            "target_adapt_subject_count": len(set(map(str, target_subjects))),
        }
    )
    prior = FrequencyPrior.from_summaries(
        source_accumulator.summary(), target_accumulator.summary(), metadata=supplied
    )
    prior.save(output_path)
    return prior


__all__ = [
    "TARGET_SPLIT_SCHEMA",
    "ImageSubjectSubset",
    "build_frequency_prior_from_loaders",
    "earliest_subject_indices",
    "save_target_adaptation_split",
    "split_target_adaptation_indices",
]
