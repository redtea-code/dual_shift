import numpy as np
import pandas as pd
import pytest

from data.scan_filtered_loader import (
    PROTOCOL_VERSION,
    filter_scan_manifest,
    normalize_field_strength,
    subject_level_split,
)


def _table():
    return pd.DataFrame(
        [
            {"subject_id": "A", "field_strength": 1.5, "label": 0, "image_aligned": True},
            {"subject_id": "A", "field_strength": "3T", "label": 0, "image_aligned": True},
            {"subject_id": "B", "field_strength": "1.5T", "label": 1, "image_aligned": True},
            {"subject_id": "C", "field_strength": "unknown", "label": 1, "image_aligned": True},
        ]
    )


def test_field_strength_normalization():
    assert normalize_field_strength(1.5) == "1.5T"
    assert normalize_field_strength("3.0T") == "3T"
    assert normalize_field_strength(np.nan) == "unknown"


def test_adni_keeps_paired_subject_15t_and_marks_origin():
    filtered, audit = filter_scan_manifest(_table(), cohort="ADNI")
    assert filtered["subject_id"].tolist() == ["A", "B"]
    assert filtered["protocol_version"].eq(PROTOCOL_VERSION).all()
    assert filtered.loc[filtered["subject_id"] == "A", "paired_origin"].iloc[0] == "had_3T"
    assert audit["n_subjects_with_retained_1p5T_and_dropped_3T"] == 1


def test_nacc_requires_3t_and_unknown_is_dropped():
    filtered, _ = filter_scan_manifest(_table(), cohort="NACC")
    assert filtered["subject_id"].tolist() == ["A"]


def test_string_false_alignment_is_not_kept():
    table = pd.DataFrame(
        [{"subject_id": "A", "field_strength": 1.5, "label": 0, "image_aligned": "False"}]
    )
    with pytest.raises(ValueError, match="retained no rows"):
        filter_scan_manifest(table, cohort="ADNI")


def test_subject_level_split_is_disjoint_and_reproducible():
    table = pd.DataFrame(
        [{"subject_id": f"S{i}", "label": i % 2} for i in range(10) for _ in range(2)]
    )
    first = subject_level_split(table, seed=42)
    second = subject_level_split(table, seed=42)
    assert first == second
    subject_sets = [set(first["subjects"][name]) for name in ("train", "val", "test")]
    assert not (subject_sets[0] & subject_sets[1] or subject_sets[0] & subject_sets[2] or subject_sets[1] & subject_sets[2])
    assert len(sum(first["indices"].values(), [])) == len(table)


def test_filter_requires_field_strength():
    with pytest.raises(ValueError, match="field_strength"):
        filter_scan_manifest(pd.DataFrame({"subject_id": ["A"], "label": [0]}), cohort="ADNI")
