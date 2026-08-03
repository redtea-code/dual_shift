import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.apic_v3_protocol import (
    APIC_V3_PRIMARY_VARIANTS,
    APIC_V3_SECONDARY_VARIANTS,
    APIC_V3_VARIANT_SPECS,
)
from experiments.run_apic_v3_screening import _load_and_validate_config, _require_passed_gate
from experiments.report_apic_v3_screening import evaluate_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class APICV3ModalityTest(unittest.TestCase):
    def test_frozen_variants_encode_two_fair_modality_axes(self):
        self.assertEqual(
            set(APIC_V3_VARIANT_SPECS),
            set(APIC_V3_PRIMARY_VARIANTS + APIC_V3_SECONDARY_VARIANTS),
        )
        for variant in APIC_V3_PRIMARY_VARIANTS:
            self.assertEqual(APIC_V3_VARIANT_SPECS[variant]["modalities"], "X")
            self.assertFalse(APIC_V3_VARIANT_SPECS[variant]["use_demographics"])
        for variant in APIC_V3_SECONDARY_VARIANTS:
            self.assertEqual(APIC_V3_VARIANT_SPECS[variant]["modalities"], "X+D")
            self.assertTrue(APIC_V3_VARIANT_SPECS[variant]["use_demographics"])


class APICV3ConfigTest(unittest.TestCase):
    def test_repository_configs_freeze_both_modality_axes(self):
        for name in (
            "journal_dual_shift_apic_v3_screen_cn_ad.yaml",
            "journal_dual_shift_apic_v3_screen_mci_ad.yaml",
        ):
            with self.subTest(config=name):
                payload = _load_and_validate_config(PROJECT_ROOT / name)
                self.assertEqual(
                    tuple(payload["apic_v3_screening"]["primary_variants"]),
                    APIC_V3_PRIMARY_VARIANTS,
                )
                self.assertEqual(
                    tuple(payload["apic_v3_screening"]["secondary_variants"]),
                    APIC_V3_SECONDARY_VARIANTS,
                )

    def test_validator_rejects_acquisition_input(self):
        source = PROJECT_ROOT / "journal_dual_shift_apic_v3_screen_cn_ad.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload = copy.deepcopy(payload)
        payload["apic_v3_screening"]["acquisition_input"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "acquisition_input"):
                _load_and_validate_config(path)

    def test_gate_requires_pre_registered_cross_task_pattern(self):
        rows = []
        for task in ("CN_vs_AD", "MCI_vs_AD"):
            for direction in ("ADNI_to_NACC", "NACC_to_ADNI"):
                for seed in (42, 43):
                    rows.extend(
                        [
                            {
                                "task": task,
                                "direction": direction,
                                "seed": seed,
                                "variant": "ce_x",
                                "balanced_accuracy": 0.60,
                            },
                            {
                                "task": task,
                                "direction": direction,
                                "seed": seed,
                                "variant": "mixstyle_x",
                                "balanced_accuracy": 0.61,
                            },
                            {
                                "task": task,
                                "direction": direction,
                                "seed": seed,
                                "variant": "apic_v3_x",
                                "balanced_accuracy": 0.63,
                                "sensitivity": 0.60,
                                "specificity": 0.66,
                                "valid_memory_slots": 8,
                            },
                        ]
                    )
        report = evaluate_gate(rows)
        self.assertTrue(report["pass"])
        rows[-1]["valid_memory_slots"] = 0
        self.assertFalse(evaluate_gate(rows)["pass"])

    def test_secondary_requires_matching_passed_gate(self):
        config_path = PROJECT_ROOT / "journal_dual_shift_apic_v3_screen_cn_ad.yaml"
        payload = _load_and_validate_config(config_path)
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "gate.json"
            report_path.write_text('{"gate": "APIC-V3-S1", "pass": true, "config_hashes_seed42": {}}')
            with self.assertRaisesRegex(SystemExit, "does not match"):
                _require_passed_gate(report_path, [(config_path, payload)])


if __name__ == "__main__":
    unittest.main()
