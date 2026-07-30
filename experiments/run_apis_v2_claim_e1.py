"""Launch APIS v2 claim E1 wave-1 jobs (ce_only/mixstyle/metadata/apis_v2)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
DIRECTIONS = (("ADNI_to_NACC", "adni_to_nacc"), ("NACC_to_ADNI", "nacc_to_adni"))
VARIANTS = ("ce_only", "mixstyle", "metadata", "apis_v2")


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
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    for seed in seeds:
        for direction, slug in DIRECTIONS:
            output_dir = (
                f"outputs/journal/dual_shift_apis_v2/claim/e1/seed{seed}/{slug}"
            )
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
                args.config_path,
            ]
            if args.force:
                cmd.append("--force-variants")
            print("[claim-e1]", " ".join(cmd), flush=True)
            code = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
            if code != 0:
                raise SystemExit(code)


if __name__ == "__main__":
    main()
