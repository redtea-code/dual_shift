"""APIS v2 claim hold-out helpers (paired field-strength subjects)."""
from __future__ import annotations

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
