import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.journal_metrics import aggregate_subject_predictions, field_strength_bin
from utils.claim_holdout import (
    assert_index_sets_disjoint,
    assert_no_holdout_leak,
    assert_no_subject_id_overlap,
    eligible_subjects,
    filter_indices_excluding_subjects,
    filter_indices_including_subjects,
    load_holdout_subjects,
)


class ClaimHoldoutTest(unittest.TestCase):
    def test_load_and_filter(self):
        payload = {
            "subjects_le_30d": ["A", "B"],
            "subjects_all": ["A", "B", "C"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdout.json"
            path.write_text(
                __import__("json").dumps(payload), encoding="utf-8"
            )
            excluded = load_holdout_subjects(path, key="subjects_le_30d")
            self.assertEqual(excluded, {"A", "B"})
            subjects = ["A", "C", "B", "D"]
            kept = filter_indices_excluding_subjects(subjects, [0, 1, 2, 3], excluded)
            self.assertEqual(kept, [1, 3])
            paired = filter_indices_including_subjects(subjects, [0, 1, 2, 3], excluded)
            self.assertEqual(paired, [0, 2])
            assert_no_holdout_leak(subjects, kept, excluded, split_name="train")
            with self.assertRaises(AssertionError):
                assert_no_holdout_leak(subjects, [0, 1], excluded, split_name="train")
            assert_index_sets_disjoint(
                subjects, kept, paired, left_name="e1", right_name="e3"
            )
            self.assertEqual(eligible_subjects(subjects, excluded), {"C", "D"})

    def test_source_target_overlap_guard(self):
        assert_no_subject_id_overlap(["A", "B"], ["C"], left_name="s", right_name="t")
        with self.assertRaises(AssertionError):
            assert_no_subject_id_overlap(["A", "B"], ["B", "C"], left_name="s", right_name="t")

    def test_field_strength_bin(self):
        self.assertEqual(field_strength_bin(1.5), "1.5T")
        self.assertEqual(field_strength_bin(3.0), "3T")
        self.assertEqual(field_strength_bin(float("nan")), "unknown")

    def test_earliest_visit_aggregation_rejects_majority_across_labels(self):
        probs = np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 1, 0], dtype=int)
        subjects = np.asarray(["S1", "S1", "S2"], dtype=object)
        folders = np.asarray(
            ["S1-2020_01_01-1", "S1-2021_01_01-3", "S2-2020_06_01-1"],
            dtype=object,
        )
        agg_p, agg_y, _, agg_s = aggregate_subject_predictions(
            probs,
            labels,
            subjects,
            folders=folders,
            label_conflict="earliest_visit",
        )
        self.assertEqual(agg_s.tolist(), ["S1", "S2"])
        self.assertEqual(agg_y.tolist(), [0, 0])
        np.testing.assert_allclose(agg_p[0], [0.9, 0.1])


if __name__ == "__main__":
    unittest.main()
