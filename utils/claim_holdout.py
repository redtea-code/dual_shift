"""APIS v2 claim hold-out helpers (paired field-strength subjects)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence, Set


def load_holdout_subjects(
    path: str | Path,
    *,
    key: str = "subjects_le_30d",
) -> Set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if key not in payload:
        raise KeyError(f"hold-out JSON missing {key!r}; keys={sorted(payload)}")
    return {str(item) for item in payload[key]}


def holdout_file_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def filter_indices_excluding_subjects(
    subject_ids: Sequence[object],
    indices: Sequence[int],
    excluded: Iterable[str],
) -> list[int]:
    blocked = {str(item) for item in excluded}
    kept = []
    for index in indices:
        if str(subject_ids[int(index)]) not in blocked:
            kept.append(int(index))
    return kept


def filter_indices_including_subjects(
    subject_ids: Sequence[object],
    indices: Sequence[int],
    included: Iterable[str],
) -> list[int]:
    allowed = {str(item) for item in included}
    kept = []
    for index in indices:
        if str(subject_ids[int(index)]) in allowed:
            kept.append(int(index))
    return kept


def assert_no_holdout_leak(
    subject_ids: Sequence[object],
    indices: Sequence[int],
    excluded: Iterable[str],
    *,
    split_name: str,
) -> None:
    blocked = {str(item) for item in excluded}
    leaked = sorted(
        {
            str(subject_ids[int(index)])
            for index in indices
            if str(subject_ids[int(index)]) in blocked
        }
    )
    if leaked:
        raise AssertionError(
            f"hold-out leak in {split_name}: {leaked[:10]}"
            + (" ..." if len(leaked) > 10 else "")
        )


def assert_index_sets_disjoint(
    subject_ids: Sequence[object],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    *,
    left_name: str,
    right_name: str,
) -> None:
    left = {str(subject_ids[int(i)]) for i in left_indices}
    right = {str(subject_ids[int(i)]) for i in right_indices}
    overlap = sorted(left & right)
    if overlap:
        raise AssertionError(
            f"{left_name} and {right_name} share subjects: {overlap[:10]}"
            + (" ..." if len(overlap) > 10 else "")
        )


def assert_no_subject_id_overlap(
    left_subjects: Iterable[object],
    right_subjects: Iterable[object],
    *,
    left_name: str,
    right_name: str,
) -> None:
    left = {str(item) for item in left_subjects}
    right = {str(item) for item in right_subjects}
    overlap = sorted(left & right)
    if overlap:
        raise AssertionError(
            f"{left_name} and {right_name} subject-id overlap: {overlap[:10]}"
            + (" ..." if len(overlap) > 10 else "")
        )


def eligible_subjects(
    subject_ids: Sequence[object],
    excluded: Iterable[str],
) -> Set[str]:
    blocked = {str(item) for item in excluded}
    return {str(item) for item in subject_ids if str(item) not in blocked}
