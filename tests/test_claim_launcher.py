import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_apis_v2_claim_e1 import _config_fingerprint, _job_complete


class ClaimLauncherIdentityTest(unittest.TestCase):
    def test_job_complete_requires_full_identity_match(self):
        config = {"seed": 42, "claim": {"protocol_revision": 2, "split_seed": 7}}
        config_hash = _config_fingerprint(config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in (
                "ce_only",
                "mixstyle",
                "metadata",
                "metadata_xda",
                "apis_v2",
            ):
                variant_dir = root / variant
                variant_dir.mkdir()
                (variant_dir / "journal_metrics.json").write_text(
                    json.dumps(
                        {
                            "claim_protocol_revision": 2,
                            "config_hash": config_hash,
                            "split_seed": 7,
                            "training_seed": 42,
                        }
                    ),
                    encoding="utf-8",
                )

            self.assertTrue(
                _job_complete(
                    root,
                    protocol_revision=2,
                    expected_config_hash=config_hash,
                    split_seed=7,
                    training_seed=42,
                )
            )
            self.assertFalse(
                _job_complete(
                    root,
                    protocol_revision=2,
                    expected_config_hash="stale",
                    split_seed=7,
                    training_seed=42,
                )
            )


if __name__ == "__main__":
    unittest.main()
