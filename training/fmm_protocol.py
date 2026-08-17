"""Dataset views and subject-level split helpers for the FMM experiment."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FMMDatasetView(Dataset):
    """Index a journal dataset without exposing fields unused by FMM.

    The target adaptation view deliberately omits ``label``.  This is stronger
    than merely ignoring the value in the loss: the training loader cannot
    return a target diagnosis label at all.
    """

    def __init__(self, dataset, indices: Sequence[int], *, include_label: bool):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.include_label = bool(include_label)
        self.subject_ids = np.asarray(dataset.subject_ids)[self.indices]
        self.labels = (
            np.asarray(dataset.labels)[self.indices] if self.include_label else None
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.dataset[int(self.indices[index])]
        result = {
            "image": item["image"],
            "subject_id": item["subject_id"],
            "folder": item.get("folder", str(item["subject_id"])),
        }
        if self.include_label:
            result["label"] = int(item["label"])
        return result


class SyntheticFMMDataset(Dataset):
    """Small deterministic dataset used by the runner smoke test."""

    def __init__(self, images: torch.Tensor, labels: Sequence[int] | None, cohort: str):
        if images.ndim != 5 or images.shape[1] != 1:
            raise ValueError("synthetic images must have shape [N,1,D,H,W]")
        self.images = images.float().contiguous()
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.labels) != len(images):
            raise ValueError("labels and images must have equal length")
        self.subject_ids = np.asarray([f"{cohort}{index:04d}" for index in range(len(images))], dtype=object)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        result = {"image": self.images[index], "subject_id": str(self.subject_ids[index])}
        if self.labels is not None:
            result["label"] = int(self.labels[index])
        return result


def _subject_labels(dataset, subjects: Sequence[str]) -> list[int]:
    labels = np.asarray(dataset.labels)
    subject_ids = np.asarray(dataset.subject_ids)
    return [int(np.bincount(labels[subject_ids == subject]).argmax()) for subject in subjects]


def _can_stratify(labels: Sequence[int], test_size: float) -> bool:
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    present = counts[counts > 0]
    holdout = max(1, round(len(labels) * float(test_size)))
    return bool(len(present) > 1 and present.min() >= 2 and holdout >= len(present))


def split_source_indices(dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("source split ratios must be positive and sum to one")
    subjects = np.unique(np.asarray(dataset.subject_ids, dtype=object))
    if len(subjects) < 3:
        raise ValueError("source split needs at least three subjects")
    labels = _subject_labels(dataset, subjects)
    holdout_ratio = float(val_ratio + test_ratio)
    train_subjects, holdout_subjects = train_test_split(
        subjects,
        test_size=holdout_ratio,
        random_state=int(seed),
        stratify=labels if _can_stratify(labels, holdout_ratio) else None,
    )
    holdout_labels = _subject_labels(dataset, holdout_subjects)
    relative_test = float(test_ratio / holdout_ratio)
    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=relative_test,
        random_state=int(seed),
        stratify=holdout_labels if _can_stratify(holdout_labels, relative_test) else None,
    )
    indices = tuple(
        np.flatnonzero(np.isin(np.asarray(dataset.subject_ids), subjects_part)).astype(np.int64)
        for subjects_part in (train_subjects, val_subjects, test_subjects)
    )
    assert_disjoint_subjects(*(np.asarray(dataset.subject_ids)[part] for part in indices))
    return indices


def split_target_indices(dataset, adapt_ratio: float, seed: int):
    """Split target subjects without consulting ``dataset.labels``."""
    if not 0.0 < float(adapt_ratio) < 1.0:
        raise ValueError("target adapt_ratio must be strictly between zero and one")
    subjects = np.unique(np.asarray(dataset.subject_ids, dtype=object))
    if len(subjects) < 2:
        raise ValueError("target split needs at least two subjects")
    rng = np.random.default_rng(int(seed))
    subjects = subjects[rng.permutation(len(subjects))]
    cut = min(max(1, int(round(len(subjects) * float(adapt_ratio)))), len(subjects) - 1)
    adapt_subjects, test_subjects = subjects[:cut], subjects[cut:]
    subject_ids = np.asarray(dataset.subject_ids)
    adapt = np.flatnonzero(np.isin(subject_ids, adapt_subjects)).astype(np.int64)
    test = np.flatnonzero(np.isin(subject_ids, test_subjects)).astype(np.int64)
    assert_disjoint_subjects(subject_ids[adapt], subject_ids[test])
    return adapt, test


def assert_disjoint_subjects(*groups: Sequence[str]) -> None:
    sets = [set(map(str, np.asarray(group).tolist())) for group in groups]
    for left_index, left in enumerate(sets):
        for right in sets[left_index + 1 :]:
            overlap = left & right
            if overlap:
                raise AssertionError(f"subject overlap detected: {sorted(overlap)[:5]}")


def subject_digest(subject_ids: Sequence[str]) -> str:
    payload = json.dumps(sorted(set(map(str, np.asarray(subject_ids).tolist()))), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "FMMDatasetView",
    "SyntheticFMMDataset",
    "assert_disjoint_subjects",
    "split_source_indices",
    "split_target_indices",
    "subject_digest",
]
