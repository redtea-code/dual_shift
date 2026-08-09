"""Create frozen scan-filtered manifests and a portable E0 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from data.scan_filtered_loader import PROTOCOL_VERSION, write_filtered_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_e0(config_path: str, adni_input: str, nacc_input: str, output_root: str) -> dict:
    config_file, root = Path(config_path), Path(output_root)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    protocol = config.get("scan_filtered_protocol") or {}
    if protocol.get("enabled") is not True or protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("task config must enable the current scan-filtered protocol")
    if (config.get("claim") or {}).get("exclude_subjects_json"):
        raise ValueError("legacy paired-subject holdout is forbidden")
    files = protocol.get("files") or {}
    root.mkdir(parents=True, exist_ok=True)
    audits = {}
    for cohort, input_path in (("ADNI", adni_input), ("NACC", nacc_input)):
        output = root / files.get(cohort, f"{cohort}_scan_filtered_manifest.csv")
        audits[cohort] = write_filtered_manifest(
            input_path, output, cohort=cohort, audit_json=root / f"filter_audit_{cohort}.json"
        )
        table = pd.read_csv(output)
        if not table["protocol_version"].eq(PROTOCOL_VERSION).all():
            raise ValueError(f"{cohort} protocol marker mismatch")
    report = {
        "pass": True,
        "protocol_version": PROTOCOL_VERSION,
        "task_config_sha256": _sha256(config_file),
        "manifests": audits,
        "manifest_sha256": {
            cohort: _sha256(root / files.get(cohort, f"{cohort}_scan_filtered_manifest.csv"))
            for cohort in ("ADNI", "NACC")
        },
    }
    (root / "e0_manifest_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adni-input", required=True)
    parser.add_argument("--nacc-input", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_e0(args.config, args.adni_input, args.nacc_input, args.output_root)


if __name__ == "__main__":
    main()
