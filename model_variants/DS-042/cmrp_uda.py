"""Target split helpers for the DS-042 label-blind adaptation path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TARGET_SPLIT_SCHEMA = "ds042_cmrp_target_split_v1"


def _digest(subjects: Sequence[str]) -> str:
    payload = "\n".join(sorted(map(str, subjects))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_target_adaptation_indices(
    subject_ids: Sequence[Any],
    candidate_indices: Sequence[int],
    *,
    adaptation_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split target rows by subject without reading labels or covariates."""
    if not (0.0 < float(adaptation_fraction) < 1.0):
        raise ValueError("adaptation_fraction must be strictly between zero and one")
    subject_array = np.asarray(subject_ids, dtype=object)
    candidate = np.asarray(candidate_indices, dtype=np.int64)
    unique = np.asarray(sorted({str(subject_array[int(index)]) for index in candidate}), dtype=object)
    if len(unique) < 2:
        raise ValueError("at least two target subjects are required for adaptation/test split")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_adapt = min(max(1, int(round(len(shuffled) * float(adaptation_fraction)))), len(shuffled) - 1)
    adapt_subjects = sorted(map(str, shuffled[:n_adapt].tolist()))
    test_subjects = sorted(map(str, shuffled[n_adapt:].tolist()))
    adapt_set = set(adapt_subjects)
    test_set = set(test_subjects)
    adapt_indices = np.asarray(
        [int(index) for index in candidate.tolist() if str(subject_array[int(index)]) in adapt_set],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [int(index) for index in candidate.tolist() if str(subject_array[int(index)]) in test_set],
        dtype=np.int64,
    )
    if not len(adapt_indices) or not len(test_indices) or adapt_set.intersection(test_set):
        raise RuntimeError("target adaptation/test split is not non-empty and subject-disjoint")
    payload = {
        "schema": TARGET_SPLIT_SCHEMA,
        "seed": int(seed),
        "adaptation_fraction": float(adaptation_fraction),
        "target_adapt_subjects": adapt_subjects,
        "target_test_subjects": test_subjects,
        "target_adapt_subject_digest": _digest(adapt_subjects),
        "target_test_subject_digest": _digest(test_subjects),
        "target_labels_read": False,
        "target_metrics_read": False,
    }
    return adapt_indices, test_indices, payload


def save_target_split(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["TARGET_SPLIT_SCHEMA", "save_target_split", "split_target_adaptation_indices"]
