"""Launch APIS v2 claim E1 wave-1 jobs.

Variants: ce_only / mixstyle / metadata / metadata_xda / apis_v2.
Outputs are forced under outputs/journal/dual_shift_apis_v2/claim*/e1/.
Smoke trees are refused. Max workers hard-capped at 3.

Uses a GPU slot queue: each concurrent job pins CUDA_VISIBLE_DEVICES to one
slot; when a job finishes the slot is returned and the next queued job starts
automatically (ThreadPoolExecutor).

Protocol revision 2: exclude hold-out before split, carve E1 target vs E3,
fixed split_seed, fair metadata_xda baseline. Default --force so stale
revision-1 metrics are not reused.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = (("ADNI_to_NACC", "adni_to_nacc"), ("NACC_to_ADNI", "nacc_to_adni"))
VARIANTS = ("ce_only", "mixstyle", "metadata", "metadata_xda", "apis_v2")
EXPLORATORY_VARIANTS = ("apis_v2_shuffle", "film_scan", "apis_scan")
ALLOWED_VARIANTS = frozenset(VARIANTS + EXPLORATORY_VARIANTS)
# Allowed claim trees: .../claim or task-specific .../claim_<task> (e.g. claim_mci_ad).
CLAIM_ROOT_PREFIX = "outputs/journal/dual_shift_apis_v2/claim"
MAX_WORKERS_HARD_CAP = 3
REQUIRED_PROTOCOL_REVISION = 2


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_claim_root(output_root: str) -> str:
    root = str(output_root or "").replace("\\", "/").strip().rstrip("/")
    if not root:
        raise SystemExit("claim config missing output_root")
    if "smoke" in root.lower():
        raise SystemExit("claim launcher refuses a smoke output_root")
    if not (
        root == CLAIM_ROOT_PREFIX or root.startswith(CLAIM_ROOT_PREFIX + "_")
    ):
        raise SystemExit(
            f"claim launcher requires output_root '{CLAIM_ROOT_PREFIX}' "
            f"or '{CLAIM_ROOT_PREFIX}_<task>', got {root!r}"
        )
    return root


def _validate_claim_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid claim config: {path}")
    payload["output_root"] = _normalize_claim_root(str(payload.get("output_root") or ""))
    dual = payload.get("dual_shift") or {}
    if float(dual.get("alpha_max", 0.25)) != 0.25:
        raise SystemExit("claim E1 requires dual_shift.alpha_max == 0.25")
    claim = payload.get("claim") or {}
    if int(claim.get("protocol_revision", 0)) != REQUIRED_PROTOCOL_REVISION:
        raise SystemExit(
            f"claim E1 requires claim.protocol_revision == {REQUIRED_PROTOCOL_REVISION}"
        )
    if "split_seed" not in claim and "split_seed" not in payload:
        raise SystemExit("claim E1 requires split_seed (frozen subject partition)")
    variants = list(payload.get("variants") or [])
    for required in VARIANTS:
        if required not in variants:
            raise SystemExit(f"claim config missing required variant {required!r}")
    return payload


def _parse_variants(raw: str, payload: dict) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not requested:
        raise SystemExit("--variants must contain at least one variant")
    unknown = set(requested).difference(ALLOWED_VARIANTS)
    if unknown:
        raise SystemExit(f"unsupported claim variants: {sorted(unknown)}")
    configured = set(payload.get("variants") or [])
    missing = set(requested).difference(configured)
    if missing:
        raise SystemExit(
            f"requested variants are not frozen in config: {sorted(missing)}"
        )
    return requested


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def write_env_fingerprint(
    config_path: Path, payload: dict, *, variants: tuple[str, ...] = VARIANTS
) -> Path:
    claim_root = _normalize_claim_root(str(payload.get("output_root") or ""))
    claim_dir = PROJECT_ROOT / claim_root
    claim_dir.mkdir(parents=True, exist_ok=True)
    freeze_dir = claim_dir / "protocol_freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    frozen = freeze_dir / config_path.name
    shutil.copy2(config_path, frozen)

    holdout_rel = (
        ((payload.get("claim") or {}).get("exclude_subjects_json"))
        or "data/claim/paired_holdout_subjects.json"
    )
    holdout_path = PROJECT_ROOT / holdout_rel
    holdout_count = None
    holdout_sha = None
    if holdout_path.exists():
        holdout_sha = _sha256_file(holdout_path)
        hold = json.loads(holdout_path.read_text(encoding="utf-8"))
        key = (payload.get("claim") or {}).get("exclude_subjects_key") or "subjects_le_30d"
        subjects = hold.get(key) if isinstance(hold, dict) else None
        if isinstance(subjects, list):
            holdout_count = len(subjects)
        elif isinstance(hold, dict) and isinstance(hold.get("subjects"), list):
            holdout_count = len(hold["subjects"])

    scan_manifest_hashes = {}
    scan_cfg = payload.get("scan_manifest") or {}
    scan_root = Path(str(scan_cfg.get("root") or ""))
    for cohort, filename in (scan_cfg.get("files") or {}).items():
        manifest_path = scan_root / str(filename)
        scan_manifest_hashes[str(cohort)] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path) if manifest_path.exists() else None,
        }

    torch_info = {}
    try:
        import torch

        torch_info = {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:  # pragma: no cover
        torch_info = {"error": str(exc)}

    claim = payload.get("claim") or {}
    fingerprint = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "git_head": _git_head(),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": _sha256_file(config_path),
        "output_root": claim_root,
        "protocol_freeze_copy": str(frozen.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "holdout_json": str(holdout_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "holdout_sha256": holdout_sha,
        "holdout_subject_count": holdout_count,
        "scan_manifests": scan_manifest_hashes,
        "torch": torch_info,
        "claim_protocol": claim.get("protocol"),
        "claim_protocol_revision": claim.get("protocol_revision"),
        "task_mode": (payload.get("task") or {}).get("mode"),
        "label_mapping": (payload.get("task") or {}).get("label_mapping"),
        "split_seed": claim.get("split_seed", payload.get("split_seed")),
        "primary_metric": claim.get("primary_metric"),
        "primary_baselines": claim.get("primary_baselines"),
        "variants": list(variants),
        "seeds_default": [42, 43, 44, 45, 46],
        "max_workers_hard_cap": MAX_WORKERS_HARD_CAP,
        "gpu_slot_queue": True,
    }
    out = claim_dir / "env_fingerprint.json"
    out.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    print(f"[claim-e1] wrote {out}", flush=True)
    return out


def _config_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _job_complete(
    output_dir: Path,
    *,
    protocol_revision: int,
    expected_config_hash: str,
    split_seed: int,
    training_seed: int,
    variants: tuple[str, ...] = VARIANTS,
) -> bool:
    for variant in variants:
        metrics_path = output_dir / variant / "journal_metrics.json"
        if not metrics_path.exists():
            return False
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("claim_protocol_revision") != protocol_revision:
            return False
        if payload.get("config_hash") != expected_config_hash:
            return False
        if payload.get("split_seed") != split_seed:
            return False
        if payload.get("training_seed") != training_seed:
            return False
    return True


def _parse_gpu_ids(raw: str, max_workers: int) -> list[int]:
    ids = [int(x) for x in str(raw).split(",") if str(x).strip() != ""]
    if not ids:
        raise SystemExit("--gpu-ids must list at least one GPU index")
    if len(ids) < max_workers:
        raise SystemExit(
            f"--gpu-ids has {len(ids)} entries but --max-workers={max_workers}"
        )
    return ids[:max_workers]


def _run_one_job(job: dict, gpu_slots: "queue.Queue[int]") -> dict:
    cmd = job["cmd"]
    label = job["label"]
    gpu_id = gpu_slots.get()
    env = {**cmd["env"], "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    print(
        f"[claim-e1] START {label} (CUDA_VISIBLE_DEVICES={gpu_id})",
        flush=True,
    )
    t0 = time.time()
    try:
        code = subprocess.call(cmd["argv"], cwd=cmd["cwd"], env=env)
    finally:
        gpu_slots.put(gpu_id)
    elapsed = time.time() - t0
    status = "ok" if code == 0 else f"exit={code}"
    print(
        f"[claim-e1] DONE  {label} ({status}, {elapsed/60:.1f} min, "
        f"freed gpu={gpu_id})",
        flush=True,
    )
    return {"label": label, "code": code, "elapsed_sec": elapsed, "gpu_id": gpu_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--config_path",
        default="journal_dual_shift_apis_v2_claim.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=True,
        help="Retrain even if metrics exist (default True for protocol r2)",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Allow skipping variants with matching protocol_revision metrics",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated frozen variants. Exploratory options: "
        + ",".join(EXPLORATORY_VARIANTS),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help=f"Concurrent (seed,direction) jobs; hard-capped at {MAX_WORKERS_HARD_CAP}",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2",
        help="Comma-separated physical GPU indices for the worker slots "
        "(length must be >= max-workers; extras ignored)",
    )
    parser.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="Only write env_fingerprint.json / protocol_freeze, then exit",
    )
    args = parser.parse_args()
    force = bool(args.force) and not bool(args.no_force)
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    if args.max_workers > MAX_WORKERS_HARD_CAP:
        print(
            f"[claim-e1] clamping --max-workers {args.max_workers} -> {MAX_WORKERS_HARD_CAP}",
            flush=True,
        )
        args.max_workers = MAX_WORKERS_HARD_CAP
    gpu_ids = _parse_gpu_ids(args.gpu_ids, args.max_workers)

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = _validate_claim_config(config_path)
    variants = _parse_variants(args.variants, payload)
    claim_root = _normalize_claim_root(str(payload.get("output_root") or ""))
    write_env_fingerprint(config_path, payload, variants=variants)
    if args.fingerprint_only:
        return

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    jobs: list[dict] = []
    skipped = 0
    for seed in seeds:
        for direction, slug in DIRECTIONS:
            output_dir = PROJECT_ROOT / claim_root / "e1" / f"seed{seed}" / slug
            label = f"seed{seed}/{slug}"
            seeded_payload = json.loads(json.dumps(payload))
            seeded_payload["seed"] = int(seed)
            split_seed = int(
                (seeded_payload.get("claim") or {}).get(
                    "split_seed", seeded_payload.get("split_seed", seed)
                )
            )
            if (not force) and _job_complete(
                output_dir,
                protocol_revision=REQUIRED_PROTOCOL_REVISION,
                expected_config_hash=_config_fingerprint(seeded_payload),
                split_seed=split_seed,
                training_seed=int(seed),
                variants=variants,
            ):
                print(f"[claim-e1] SKIP complete {label}", flush=True)
                skipped += 1
                continue
            if "smoke" in str(output_dir).replace("\\", "/").lower():
                raise SystemExit("refusing smoke output path")
            rel_out = f"{claim_root}/e1/seed{seed}/{slug}"
            argv = [
                args.python,
                "run_v2.py",
                "--exp",
                "journal",
                "--direction",
                direction,
                "--variants",
                *variants,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--output-dir",
                rel_out,
                "--config_path",
                str(config_path.relative_to(PROJECT_ROOT)),
            ]
            if force:
                argv.append("--force-variants")
            jobs.append(
                {
                    "label": label,
                    "cmd": {"argv": argv, "cwd": str(PROJECT_ROOT), "env": env},
                }
            )

    print(
        f"[claim-e1] queue={len(jobs)} skipped={skipped} "
        f"max_workers={args.max_workers} gpu_ids={gpu_ids} force={force}",
        flush=True,
    )
    if not jobs:
        print("[claim-e1] nothing to run", flush=True)
        return

    gpu_slots: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_slots.put(int(gpu_id))

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(_run_one_job, job, gpu_slots) for job in jobs
        ]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result["code"] != 0:
                failures.append(result)

    if failures:
        for item in failures:
            print(f"[claim-e1] FAILED {item['label']} code={item['code']}", flush=True)
        raise SystemExit(1)
    print("[claim-e1] all jobs finished ok", flush=True)


if __name__ == "__main__":
    main()
