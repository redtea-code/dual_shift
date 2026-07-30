import tempfile
import unittest
from pathlib import Path

from utils.claim_holdout import (
    assert_no_holdout_leak,
    filter_indices_excluding_subjects,
    load_holdout_subjects,
)
from training.journal_metrics import field_strength_bin


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
            assert_no_holdout_leak(subjects, kept, excluded, split_name="train")
            with self.assertRaises(AssertionError):
                assert_no_holdout_leak(subjects, [0, 1], excluded, split_name="train")

    def test_field_strength_bin(self):
        self.assertEqual(field_strength_bin(1.5), "1.5T")
        self.assertEqual(field_strength_bin(3.0), "3T")
        self.assertEqual(field_strength_bin(float("nan")), "unknown")


if __name__ == "__main__":
    unittest.main()
