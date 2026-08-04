"""Single-class strata must not spam sklearn metric warnings."""
from __future__ import annotations

import unittest
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from training.journal_metrics import compute_journal_metrics


class JournalMetricsSingleClassTests(unittest.TestCase):
    def test_two_class_balanced_accuracy_matches_sklearn(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [0.9, 0.1],
                [0.6, 0.4],
                [0.2, 0.8],
                [0.4, 0.6],
            ]
        )
        metrics = compute_journal_metrics(probs, labels, input_type="probabilities")
        predicted = probs.argmax(axis=1)
        self.assertAlmostEqual(
            metrics["balanced_accuracy"],
            float(balanced_accuracy_score(labels, predicted)),
        )
        self.assertTrue(np.isfinite(metrics["auc"]))

    def test_single_class_group_is_warning_free(self):
        labels = np.array([0, 0, 0, 0])
        probs = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.55, 0.45],
            ]
        )
        envs = np.array(["a", "a", "b", "b"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_journal_metrics(
                probs,
                labels,
                envs,
                input_type="probabilities",
            )
        sklearn_warns = [
            w
            for w in caught
            if "sklearn.metrics" in (w.filename or "")
            or w.category.__name__
            in {"UserWarning", "UndefinedMetricWarning"}
            and (
                "single label" in str(w.message).lower()
                or "only one class" in str(w.message).lower()
                or "classes not in y_true" in str(w.message).lower()
            )
        ]
        self.assertEqual(sklearn_warns, [], [str(w.message) for w in sklearn_warns])
        self.assertTrue(np.isnan(metrics["auc"]))
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertTrue(np.isnan(metrics["per_group"]["a"]["auc"]))
        self.assertTrue(np.isnan(metrics["per_group"]["b"]["auc"]))

    def test_single_class_with_false_positives(self):
        labels = np.array([0, 0, 0])
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_journal_metrics(
                probs, labels, input_type="probabilities"
            )
        bad = [
            w
            for w in caught
            if "single label" in str(w.message).lower()
            or "only one class" in str(w.message).lower()
            or "classes not in y_true" in str(w.message).lower()
        ]
        self.assertEqual(bad, [])
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0 / 3.0)
        self.assertTrue(np.isnan(metrics["auc"]))


if __name__ == "__main__":
    unittest.main()
