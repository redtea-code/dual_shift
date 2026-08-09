import pandas as pd
import yaml

from experiments.run_scan_filtered_e0 import run_e0


def test_e0_writes_frozen_manifests_and_hash_audit(tmp_path):
    config = {
        "claim": {"task_pair": "CN_vs_AD"},
        "scan_filtered_protocol": {
            "enabled": True,
            "version": "scan_filtered_v1_2026-08-08",
            "files": {"ADNI": "adni.csv", "NACC": "nacc.csv"},
        },
    }
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    for name, strength in (("adni_input.csv", "1.5T"), ("nacc_input.csv", "3T")):
        pd.DataFrame([{"subject_id": "S1", "field_strength": strength, "image_aligned": True}]).to_csv(tmp_path / name, index=False)
    report = run_e0(config_path, tmp_path / "adni_input.csv", tmp_path / "nacc_input.csv", tmp_path / "out")
    assert report["pass"] is True
    assert (tmp_path / "out" / "e0_manifest_audit.json").is_file()
