"""Launch APIS v2 claim E1 wave-1 jobs (ce_only/mixstyle/metadata/apis_v2).

Outputs are forced under outputs/journal/dual_shift_apis_v2/claim/e1/.
Smoke and legacy dual_shift_* trees are refused.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = (("ADNI_to_NACC", "adni_to_nacc"), ("NACC_to_ADNI", "nacc_to_adni"))
VARIANTS = ("ce_only", "mixstyle", "metadata", "apis_v2")
CLAIM_ROOT = "outputs/journal/dual_shift_apis_v2/claim"


def _validate_claim_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid claim config: {path}")
    root = str(payload.get("output_root") or "")
    if "smoke" in root.replace("\\", "/").lower():
        raise SystemExit("claim launcher refuses a smoke output_root")
    if CLAIM_ROOT not in root.replace("\\", "/"):
        raise SystemExit(
            f"claim launcher requires output_root under {CLAIM_ROOT}, got {root!r}"
        )
    dual = payload.get("dual_shift") or {}
    if float(dual.get("alpha_max", 0.25)) != 0.25:
        raise SystemExit("claim E1 requires dual_shift.alpha_max == 0.25")
    variants = list(payload.get("variants") or [])
    for required in VARIANTS:
        if required not in variants:
            raise SystemExit(f"claim config missing required variant {required!r}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--config_path",
        default="journal_dual_shift_apis_v2_claim.yaml",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    _validate_claim_config(config_path)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    for seed in seeds:
        for direction, slug in DIRECTIONS:
            output_dir = f"{CLAIM_ROOT}/e1/seed{seed}/{slug}"
            if "smoke" in output_dir.lower():
                raise SystemExit("refusing smoke output path")
            cmd = [
                args.python,
                "run_v2.py",
                "--exp",
                "journal",
                "--direction",
                direction,
                "--variants",
                *VARIANTS,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--output-dir",
                output_dir,
                "--config_path",
                str(config_path.relative_to(PROJECT_ROOT)),
            ]
            if args.force:
                cmd.append("--force-variants")
            print("[claim-e1]", " ".join(cmd), flush=True)
            code = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
            if code != 0:
                raise SystemExit(code)


if __name__ == "__main__":
    main()
