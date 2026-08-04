import unittest

from experiments.summarize_apic_v3_diagnostics import summarize_history


class APICV3DiagnosticTests(unittest.TestCase):
    def test_history_summary_detects_intervention_decay(self):
        history = []
        for epoch in range(1, 51):
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "val_loss": float(epoch),
                    "val_auc": float(epoch) / 100.0,
                    "apis_coefficient_l2": 1e-5 if epoch <= 10 else 1e-7,
                }
            )
        summary = summarize_history(
            {
                "variant": "apic_v3_x",
                "history": history,
                "target": {"metrics": {"balanced_accuracy": 0.6}},
                "apic_v3_audit": {"valid_slots": 8},
            }
        )
        self.assertEqual(summary["max_logged_composite_epoch"], 50)
        self.assertAlmostEqual(summary["late_to_early_l2_ratio"], 0.01)
        self.assertEqual(summary["memory_valid_slots"], 8)


if __name__ == "__main__":
    unittest.main()
