import unittest

import numpy as np
import pandas as pd

from experiments.audit_scan_parameters import audit_manifest, field_strength_bin


class ScanParameterAuditTests(unittest.TestCase):
    def test_field_strength_bins(self):
        self.assertEqual(field_strength_bin(1.5), "1.5T")
        self.assertEqual(field_strength_bin("3.0"), "3T")
        self.assertEqual(field_strength_bin(np.nan), "unknown")
        self.assertEqual(field_strength_bin(7), "other")

    def test_audit_reports_missing_and_subject_conflicts(self):
        table = pd.DataFrame(
            {
                "pre_folder": ["a", "b", "c"],
                "subject_id": ["s1", "s1", "s2"],
                "age": [70, 70, 71],
                "sex": ["male", "male", "female"],
                "education": [16, 16, 14],
                "field_strength": [1.5, 3.0, np.nan],
                "manufacturer": ["GE", "GE", pd.NA],
            }
        )
        audit = audit_manifest(table, "ADNI")
        strength = {row["field_strength"]: row["n_rows"] for row in audit["field_strength_summary"]}
        self.assertEqual(strength, {"1.5T": 1, "3T": 1, "unknown": 1})
        manufacturer = next(row for row in audit["field_summary"] if row["field"] == "manufacturer")
        self.assertEqual(manufacturer["n_missing"], 1)
        conflicts = next(row for row in audit["subject_parameter_conflicts"] if row["field"] == "field_strength")
        self.assertEqual(conflicts["n_subjects_multiple_values"], 1)


if __name__ == "__main__":
    unittest.main()
