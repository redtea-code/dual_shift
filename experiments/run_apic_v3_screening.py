"""Launch the frozen APIC v3 two-seed, multi-task screening protocol."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.apic_v3_protocol import (
    APIC_V3_PRIMARY_VARIANTS,
    APIC_V3_SECONDARY_VARIANTS,
    APIC_V3_VARIANT_SPECS,
    config_fingerprint,
)


DEFAULT_CONFIGS = (
    "journal_dual_shift_apic_v3_screen_cn_ad.yaml",
    "journal_dual_shift_apic_v3_screen_mci_ad.yaml",
)
DIRECTIONS = (("ADNI_to_NACC", "adni_to_nacc"), ("NACC_to_ADNI", "nacc_to_adni"))
REQUIRED_PROTOCOL_REVISION = 3
REQUIRED_SPLIT_SEED = 42


def _load_and_validate_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    claim = payload.get("claim") or {}
    screening = payload.get("apic_v3_screening") or {}
    expected = set(APIC_V3_VARIANT_SPECS)
    configured = set(payload.get("variants") or [])
    if configured != expected:
        raise ValueError(
            f"{path.name}: variants must freeze exactly {sorted(expected)}, "
            f"got {sorted(configured)}"
        )
    if int(claim.get("protocol_revision", 0)) != REQUIRED_PROTOCOL_REVISION:
        raise ValueError(f"{path.name}: claim.protocol_revision must be 3")
    split_seed = int(claim.get("split_seed", payload.get("split_seed", -1)))
    if split_seed != REQUIRED_SPLIT_SEED:
        raise ValueError(f"{path.name}: split_seed must be 42")
    if claim.get("exclude_subjects_key") != "subjects_all_paired":
        raise ValueError(
            f"{path.name}: all 73 paired subjects must be frozen for Phase P"
        )
    if screening.get("method_family") != "APIC_v3":
        raise ValueError(f"{path.name}: method_family must be APIC_v3")
    if screening.get("code_variant") != "v3_style_memory":
        raise ValueError(f"{path.name}: code_variant must be v3_style_memory")
    if bool(screening.get("acquisition_input", True)):
        raise ValueError(f"{path.name}: acquisition_input must be false")
    if tuple(screening.get("primary_variants") or ()) != APIC_V3_PRIMARY_VARIANTS:
        raise ValueError(f"{path.name}: primary variant order is not frozen")
    if tuple(screening.get("secondary_variants") or ()) != APIC_V3_SECONDARY_VARIANTS:
        raise ValueError(f"{path.name}: secondary variant order is not frozen")
    output_root = str(payload.get("output_root") or "").replace("\\", "/")
    if not output_root.startswith("outputs/journal/apic_v3_screening_"):
        raise ValueError(f"{path.name}: output_root is not an APIC v3 screening tree")
    study = payload.get("study") or {}
    if list(study.get("directions") or []) != [item[0] for item in DIRECTIONS]:
        raise ValueError(f"{path.name}: both frozen directions are required")
    if list(study.get("seeds") or []) != [42, 43]:
        raise ValueError(f"{path.name}: study.seeds must be [42, 43]")
    return payload


def _selected_variants(phase: str) -> tuple[str, ...]:
    if phase == "primary":
        return APIC_V3_PRIMARY_VARIANTS
    return APIC_V3_SECONDARY_VARIANTS


def _require_passed_gate(report_path: Path, configs: list[tuple[Path, dict]]) -> None:
    """Reject X+D runs unless the matching frozen X protocol passed Gate S1."""
    if not report_path.exists():
        raise SystemExit(
            "Phase secondary requires a completed Gate S1 report. Run "
            "experiments/report_apic_v3_screening.py after the primary matrix."
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read Gate S1 report {report_path}: {exc}") from exc
    if report.get("gate") != "APIC-V3-S1" or report.get("pass") is not True:
        raise SystemExit("Phase secondary is blocked: Gate S1 did not pass.")
    expected = {
        path.name: config_fingerprint(payload)
        for path, payload in configs
    }
    if report.get("config_hashes_seed42") != expected:
        raise SystemExit(
            "Phase secondary is blocked: Gate S1 report does not match the "
            "currently frozen configuration files."
        )


def _job_complete(
    output_dir: Path,
    *,
    seeded_config: dict,
    variants: tuple[str, ...],
) -> bool:
    expected_hash = config_fingerprint(seeded_config)
    for variant in variants:
        path = output_dir / variant / "journal_metrics.json"
        if not path.exists():
            return False
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        spec = APIC_V3_VARIANT_SPECS[variant]
        if metrics.get("claim_protocol_revision") != REQUIRED_PROTOCOL_REVISION:
            return False
        if metrics.get("config_hash") != expected_hash:
            return False
        if metrics.get("variant") != variant:
            return False
        if metrics.get("input_modalities") != spec["modalities"]:
            return False
        if bool(metrics.get("use_demographics")) != bool(spec["use_demographics"]):
            return False
        if bool(metrics.get("uses_acquisition_metadata", True)):
            return False
    return True


def _write_fingerprint(
    configs: list[tuple[Path, dict]], variants: tuple[str, ...], *, phase: str
) -> None:
    for config_path, payload in configs:
        root = PROJECT_ROOT / str(payload["output_root"])
        freeze = root / "protocol_freeze"
        freeze.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, freeze / config_path.name)
        fingerprint = {
            "method_family": "APIC_v3",
            "code_variant": "v3_style_memory",
            "config": config_path.name,
            "config_hash_seed42": config_fingerprint(payload),
            "variants": list(variants),
            "split_seed": REQUIRED_SPLIT_SEED,
            "training_seeds": [42, 43],
            "directions": [item[0] for item in DIRECTIONS],
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": sys.version,
            "runtime_versions": "recorded by each training job",
        }
        (freeze / f"env_fingerprint_{phase}.json").write_text(
            json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
        )


def _run_job(job: dict) -> tuple[str, int, Path]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    log_path: Path = job["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            job["argv"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return job["label"], int(completed.returncode), log_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--phase", choices=("primary", "secondary"), default="primary")
    parser.add_argument(
        "--gate-report",
        default="outputs/journal/apic_v3_screening_gate_s1.json",
        help="Required passed Gate S1 report for the secondary X+D phase.",
    )
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fingerprint-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not seeds or not set(seeds).issubset({42, 43}):
        raise SystemExit("screening seeds must be a non-empty subset of {42, 43}")
    gpu_ids = tuple(item.strip() for item in args.gpu_ids.split(",") if item.strip())
    if not gpu_ids:
        raise SystemExit("--gpu-ids must contain at least one GPU id")
    max_workers = int(args.max_workers)
    if max_workers < 1 or max_workers > len(gpu_ids):
        raise SystemExit("--max-workers must be between 1 and the number of GPU ids")

    configs: list[tuple[Path, dict]] = []
    for raw in args.configs:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        configs.append((path, _load_and_validate_config(path)))
    if args.phase == "secondary":
        gate_report = Path(args.gate_report)
        if not gate_report.is_absolute():
            gate_report = PROJECT_ROOT / gate_report
        _require_passed_gate(gate_report, configs)
    variants = _selected_variants(args.phase)
    _write_fingerprint(configs, variants, phase=args.phase)
    if args.fingerprint_only:
        return

    jobs = []
    slot = 0
    for config_path, payload in configs:
        output_root = PROJECT_ROOT / str(payload["output_root"]) / "s1"
        for seed in seeds:
            seeded = copy.deepcopy(payload)
            seeded["seed"] = int(seed)
            for direction, slug in DIRECTIONS:
                output_dir = output_root / f"seed{seed}" / slug
                if not args.force and _job_complete(
                    output_dir, seeded_config=seeded, variants=variants
                ):
                    print(f"[apic-v3] SKIP complete {config_path.stem} seed{seed} {slug}")
                    continue
                label = f"{config_path.stem} seed{seed} {slug}"
                argv_job = [
                    sys.executable,
                    "run_v2.py",
                    "--exp",
                    "journal",
                    "--config_path",
                    str(config_path.relative_to(PROJECT_ROOT)),
                    "--device",
                    "cuda",
                    "--direction",
                    direction,
                    "--seed",
                    str(seed),
                    "--variants",
                    *variants,
                    "--output-dir",
                    str(output_dir.relative_to(PROJECT_ROOT)),
                    "--force-variants",
                ]
                jobs.append(
                    {
                        "label": label,
                        "argv": argv_job,
                        "gpu": gpu_ids[slot % len(gpu_ids)],
                        "log_path": output_root
                        / "logs"
                        / args.phase
                        / f"seed{seed}_{slug}.log",
                    }
                )
                slot += 1

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_job, job) for job in jobs]
        for future in as_completed(futures):
            label, returncode, log_path = future.result()
            print(f"[apic-v3] {'OK' if returncode == 0 else 'FAIL'} {label}: {log_path}")
            if returncode != 0:
                failures.append((label, returncode, str(log_path)))
    if failures:
        raise SystemExit(f"APIC v3 screening failures: {failures}")


if __name__ == "__main__":
    main()
