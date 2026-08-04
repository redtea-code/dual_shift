"""Launch the APIC v3_2-A strict image-only primary matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apic_v3_protocol import APIC_V3_2_PRIMARY_VARIANTS, config_fingerprint

DIRECTIONS = (("ADNI_to_NACC", "adni_to_nacc"), ("NACC_to_ADNI", "nacc_to_adni"))


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload.get("apic_v3_2_screening") or {}
    if section.get("method_family") != "APIC_v3_2":
        raise ValueError(f"{path.name}: method_family must be APIC_v3_2")
    if section.get("code_variant") != "v3_2_balanced_style_memory":
        raise ValueError(f"{path.name}: code_variant is not frozen")
    if section.get("implementation_status") != "prototype_not_protocol_compliant":
        raise ValueError(f"{path.name}: implementation status must remain explicit")
    if bool(section.get("formal_run_allowed", True)):
        raise ValueError(f"{path.name}: formal_run_allowed must remain false until review closes")
    if bool(section.get("acquisition_input", True)):
        raise ValueError(f"{path.name}: acquisition_input must be false")
    if tuple(payload.get("variants") or ()) != APIC_V3_2_PRIMARY_VARIANTS:
        raise ValueError(f"{path.name}: variants must be {APIC_V3_2_PRIMARY_VARIANTS}")
    claim = payload.get("claim") or {}
    if int(claim.get("protocol_revision", 0)) != 4:
        raise ValueError(f"{path.name}: claim.protocol_revision must be 4")
    if int(claim.get("split_seed", payload.get("split_seed", -1))) != 42:
        raise ValueError(f"{path.name}: split_seed must be 42")
    root = str(payload.get("output_root") or "").replace("\\", "/")
    if not root.startswith("outputs/journal/apic_v3_2_screening_"):
        raise ValueError(f"{path.name}: output_root must be APIC v3_2 isolated")
    if list((payload.get("study") or {}).get("directions") or []) != [item[0] for item in DIRECTIONS]:
        raise ValueError(f"{path.name}: both directions are required")
    return payload


def _run(job: dict) -> tuple[str, int, Path]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    with job["log"].open("w", encoding="utf-8") as stream:
        result = subprocess.run(job["argv"], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return job["label"], int(result.returncode), job["log"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[
        "journal_dual_shift_apic_v3_2_screen_cn_ad.yaml",
        "journal_dual_shift_apic_v3_2_screen_mci_ad.yaml",
    ])
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fingerprint-only", action="store_true")
    parser.add_argument(
        "--allow-prototype-run",
        action="store_true",
        help="Explicitly allow the reviewed prototype; formal revision-4 runs remain blocked.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if not seeds or not set(seeds).issubset({42, 43}):
        raise SystemExit("APIC v3_2 primary screening seeds must be a subset of {42,43}")
    gpus = tuple(item.strip() for item in args.gpu_ids.split(",") if item.strip())
    if not gpus or args.max_workers < 1 or args.max_workers > len(gpus):
        raise SystemExit("--gpu-ids and --max-workers are inconsistent")
    configs = []
    for raw in args.configs:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        configs.append((path, load_config(path)))
    for path, payload in configs:
        freeze = PROJECT_ROOT / payload["output_root"] / "protocol_freeze"
        freeze.mkdir(parents=True, exist_ok=True)
        (freeze / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        (freeze / "config_fingerprint.json").write_text(
            json.dumps({"config": path.name, "sha256": config_fingerprint(payload), "method_family": "APIC_v3_2", "protocol_revision": 4}, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.fingerprint_only:
        return
    if not args.allow_prototype_run:
        raise SystemExit(
            "APIC v3_2 is currently a reviewed prototype, not a protocol-revision-4 "
            "implementation. Use --fingerprint-only, or pass --allow-prototype-run "
            "only for non-claim mechanism development."
        )
    jobs = []
    slot = 0
    for path, payload in configs:
        for seed in seeds:
            for direction, slug in DIRECTIONS:
                output = PROJECT_ROOT / payload["output_root"] / "r4" / f"seed{seed}" / slug
                jobs.append({
                    "label": f"{path.stem} seed{seed} {slug}",
                    "gpu": gpus[slot % len(gpus)],
                    "log": output.parent / "logs" / f"seed{seed}_{slug}.log",
                    "argv": [sys.executable, "run_v2.py", "--exp", "journal", "--config_path", str(path.relative_to(PROJECT_ROOT)), "--device", "cuda", "--direction", direction, "--seed", str(seed), "--variants", *APIC_V3_2_PRIMARY_VARIANTS, "--output-dir", str(output.relative_to(PROJECT_ROOT)), "--force-variants"],
                })
                slot += 1
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for future in as_completed(futures):
            label, code, log = future.result()
            print(f"[apic-v3_2] {'OK' if code == 0 else 'FAIL'} {label}: {log}", flush=True)
            if code:
                failures.append((label, code, str(log)))
    if failures:
        raise SystemExit(f"APIC v3_2 screening failures: {failures}")


if __name__ == "__main__":
    main()
