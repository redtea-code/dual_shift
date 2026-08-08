#!/usr/bin/env python3
"""
ADNI MRI Preprocessing -- Step 5 (Batch): Intensity Normalization
=================================================================
Normalizes cropped MNI-space brain images to [-1, 1] using robust
percentile min-max scaling, masked to brain voxels only.

Batch wrapper around the single-subject _5_normalize.py logic.

Input:  MRI_brain_mni152_cropped.nii.gz       (cropped brain image, Step 3/4 output)
        MRI_brain_mask_mni152_cropped.nii.gz   (cropped brain mask, Step 3/4 output)
Output: MRI_brain_mni152_cropped_norm.nii.gz   (intensity-normalized image)
        MRI_brain_mni152_cropped_norm_stats.json (per-subject normalization stats)

Usage:
  python _5_2_normalize.py
  python _5_2_normalize.py --dry-run
  python _5_2_normalize.py --limit 10
  python _5_2_normalize.py --no-resume --workers 4
  python _5_2_normalize.py --low-pct 2 --high-pct 98 --exclude-zero
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# ============================================================
# CONFIG -- defaults, override via CLI
# ============================================================
INPUT_ROOT = r"D:\ADNI_dataset\Gmamba\NACC\NACC_pre"

# Input files (from Step 3/4 -- Resample & Crop)
INPUT_IMAGE = "MRI_N4_brain_mni152_cropped.nii.gz"
INPUT_MASK = "MRI_brain_mask_mni152_cropped.nii.gz"

# Output files
OUTPUT_IMAGE = "MRI_N4_brain_mni152_cropped_norm.nii.gz"
OUTPUT_STATS = "MRI_brain_mni152_cropped_norm_stats.json"


# Normalization defaults
DEFAULT_LOW_PCT = 1.0
DEFAULT_HIGH_PCT = 99.0
DEFAULT_EXCLUDE_ZERO = True
DEFAULT_MASK_THRESHOLD = 0.5

# Worker defaults
DEFAULT_WORKERS = 1  # single-process by default for safety
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADNI Step 5 (Batch): Intensity normalization to [-1, 1]"
    )
    parser.add_argument(
        "--input-root", default=INPUT_ROOT,
        help="Root directory containing subject folders",
    )
    parser.add_argument(
        "--low-pct", type=float, default=DEFAULT_LOW_PCT,
        help="Lower percentile for robust min-max scaling (default: %.1f)" % DEFAULT_LOW_PCT,
    )
    parser.add_argument(
        "--high-pct", type=float, default=DEFAULT_HIGH_PCT,
        help="Upper percentile for robust min-max scaling (default: %.1f)" % DEFAULT_HIGH_PCT,
    )
    parser.add_argument(
        "--exclude-zero", action="store_true", default=DEFAULT_EXCLUDE_ZERO,
        help="Exclude zero-valued voxels from percentile estimation",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD,
        help="Mask threshold; voxels > this are inside mask (default: %.2f)" % DEFAULT_MASK_THRESHOLD,
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Number of parallel workers (default: %d; set to 0 for autodetect)" % DEFAULT_WORKERS,
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip subjects with existing output (default: True)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of subjects (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing files",
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# Core normalization logic (extracted from _5_normalize.py)
# ────────────────────────────────────────────────────────────


def load_mask(
    mask_path: Path | None,
    reference_shape: tuple[int, ...],
    threshold: float,
) -> np.ndarray:
    """Load binary brain mask, with fallback to all-ones."""
    if mask_path is None:
        return np.ones(reference_shape, dtype=bool)
    mask_data = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32)
    mask = mask_data > threshold
    if mask.shape != reference_shape:
        raise ValueError(
            "Mask shape %s does not match input shape %s." % (mask.shape, reference_shape)
        )
    return mask


def normalize_to_minus1_1(
    data: np.ndarray,
    mask: np.ndarray,
    low_pct: float,
    high_pct: float,
    exclude_zero: bool,
) -> tuple[np.ndarray, dict]:
    """
    Robust percentile min-max normalization to [-1, 1].

    Uses mask + optional zero-exclusion for percentile estimation.
    Voxels outside mask are set to 0.
    """
    stats_mask: np.ndarray = mask & np.isfinite(data)
    if exclude_zero:
        stats_mask &= data != 0

    values = data[stats_mask]

    if values.size == 0:
        out = np.zeros_like(data, dtype=np.float32)
        return out, {
            "status": "empty_mask",
            "low_pct": low_pct,
            "high_pct": high_pct,
            "exclude_zero": exclude_zero,
            "num_mask_voxels": int(mask.sum()),
            "num_stat_voxels": 0,
        }

    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi <= lo:
        hi = lo + 1e-6

    scaled01 = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    scaled = scaled01 * 2.0 - 1.0
    scaled[~np.isfinite(scaled)] = 0.0
    scaled[~mask] = 0.0

    return scaled.astype(np.float32), {
        "status": "ok",
        "low_pct": low_pct,
        "high_pct": high_pct,
        "exclude_zero": exclude_zero,
        "p_low": lo,
        "p_high": hi,
        "num_mask_voxels": int(mask.sum()),
        "num_stat_voxels": int(values.size),
        "out_min": float(scaled[mask].min()) if mask.any() else 0.0,
        "out_max": float(scaled[mask].max()) if mask.any() else 0.0,
    }


# ────────────────────────────────────────────────────────────
# Single-subject processing
# ────────────────────────────────────────────────────────────


def process_one_subject(
    subject_name: str,
    subject_dir: str,
    args: argparse.Namespace,
) -> dict:
    """
    Normalize a single subject's cropped brain image to [-1, 1].

    Returns a result dict with status, subject name, and any diagnostics.
    """
    img_path = Path(subject_dir) / INPUT_IMAGE
    mask_path = Path(subject_dir) / INPUT_MASK
    out_path = Path(subject_dir) / OUTPUT_IMAGE
    stats_path = Path(subject_dir) / OUTPUT_STATS

    # --- Resume check ---
    if args.resume and out_path.exists():
        return {"status": "skipped", "reason": "output exists", "subject": subject_name}

    # --- Input validation ---
    if not img_path.exists():
        return {"status": "error", "reason": "missing " + INPUT_IMAGE, "subject": subject_name}
    if not mask_path.exists():
        return {"status": "error", "reason": "missing " + INPUT_MASK, "subject": subject_name}

    # --- Load image ---
    try:
        img = nib.load(str(img_path))
        data = np.asarray(img.get_fdata(), dtype=np.float32)
    except Exception as e:
        return {"status": "error", "reason": "load image failed: " + str(e), "subject": subject_name}

    if data.ndim != 3:
        return {
            "status": "error",
            "reason": "image is not 3D (shape=%s)" % str(data.shape),
            "subject": subject_name,
        }

    # --- Load mask ---
    try:
        mask = load_mask(mask_path, data.shape, args.mask_threshold)
    except Exception as e:
        return {"status": "error", "reason": "load mask failed: " + str(e), "subject": subject_name}

    # --- Normalize ---
    norm_data, stats = normalize_to_minus1_1(
        data=data,
        mask=mask,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        exclude_zero=args.exclude_zero,
    )

    # --- Save ---
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(norm_data, img.affine, img.header),
            str(out_path),
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    result = {
        "status": stats["status"],
        "subject": subject_name,
        "shape": list(data.shape),
        "mask_voxels": stats.get("num_mask_voxels", 0),
    }
    if stats["status"] == "ok":
        result.update({
            "p_low": stats["p_low"],
            "p_high": stats["p_high"],
            "stat_voxels": stats["num_stat_voxels"],
            "out_min": stats["out_min"],
            "out_max": stats["out_max"],
        })
    elif stats["status"] == "empty_mask":
        result["reason"] = "no valid voxels in mask"

    return result


# ────────────────────────────────────────────────────────────
# Worker entry point (multiprocessing)
# ────────────────────────────────────────────────────────────

# Global variable to hold parsed args for workers spawned via fork/spawn.
_worker_args: argparse.Namespace | None = None


def _init_worker(args: argparse.Namespace) -> None:
    """Initialize worker with shared arguments."""
    global _worker_args
    _worker_args = args


def _worker_fn(task: tuple[int, str, str]) -> dict:
    """Process a single subject (called from worker pool)."""
    _, subject_name, subject_dir = task
    if _worker_args is None:
        raise RuntimeError("worker pool was started without _init_worker arguments")
    return process_one_subject(subject_name, subject_dir, _worker_args)


# ────────────────────────────────────────────────────────────
# Main driver
# ────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    if args.no_resume:
        args.resume = False
    if args.workers <= 0:
        args.workers = max(1, multiprocessing.cpu_count() - 1)

    input_root = Path(args.input_root)
    if not input_root.is_dir():
        print("[FATAL] Input root not found: %s" % input_root)
        sys.exit(1)

    # --- Discover subjects ---
    subjects = sorted([
        d for d in input_root.iterdir()
        if d.is_dir()
    ])
    total_available = len(subjects)
    if args.limit:
        subjects = subjects[:args.limit]

    print("[INFO] Input root       : %s" % input_root)
    print("[INFO] Input image       : %s" % INPUT_IMAGE)
    print("[INFO] Input mask        : %s" % INPUT_MASK)
    print("[INFO] Output image      : %s" % OUTPUT_IMAGE)
    print("[INFO] Output stats      : %s" % OUTPUT_STATS)
    print("[INFO] Low percentile    : %.1f" % args.low_pct)
    print("[INFO] High percentile   : %.1f" % args.high_pct)
    print("[INFO] Exclude zero      : %s" % args.exclude_zero)
    print("[INFO] Mask threshold    : %.2f" % args.mask_threshold)
    print("[INFO] Workers           : %d" % args.workers)
    print("[INFO] Resume            : %s" % args.resume)
    print("[INFO] Found %d subjects (processing %d)" % (total_available, len(subjects)))
    if args.dry_run:
        print("[INFO] DRY RUN -- no files will be written")
    print()

    # --- Prepare task list ---
    tasks = [
        (idx, subj.name, str(subj))
        for idx, subj in enumerate(subjects)
    ]

    # --- Process ---
    t0 = time.time()
    results: list[dict] = []

    if args.workers > 1 and not args.dry_run:
        # Multiprocessing path
        with multiprocessing.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args,),
        ) as pool:
            for result in pool.imap_unordered(_worker_fn, tasks):
                results.append(result)
                # Print progress immediately
                _print_result(result, len(subjects))
                sys.stdout.flush()
    else:
        # Single-process path (also used for dry-run)
        for idx, subject_name, subject_dir in tasks:
            result = process_one_subject(subject_name, subject_dir, args)
            results.append(result)
            _print_result(result, len(subjects))
            sys.stdout.flush()

    elapsed = time.time() - t0

    # --- Summary ---
    # Sort results by subject order for consistent reporting
    results.sort(key=lambda r: r.get("subject", ""))

    ok_list = [r for r in results if r["status"] == "ok"]
    skipped_list = [r for r in results if r["status"] == "skipped"]
    error_list = [r for r in results if r["status"] not in ("ok", "skipped")]
    empty_list = [r for r in results if r["status"] == "empty_mask"]

    n_ok = len(ok_list)
    n_skipped = len(skipped_list)
    n_error = len(error_list)
    n_empty = len(empty_list)

    print()
    print("=" * 78)
    print("[DONE] Step 5: Intensity Normalization in %.1fs" % elapsed)
    print("  Total          : %d" % len(tasks))
    print("  OK             : %d" % n_ok)
    print("  Skipped        : %d" % n_skipped)
    print("  Empty mask     : %d" % n_empty)
    print("  Errors         : %d" % n_error)
    print("=" * 78)

    # --- Aggregate normalized-value statistics across all subjects ---
    if ok_list:
        p_lows = [r["p_low"] for r in ok_list]
        p_highs = [r["p_high"] for r in ok_list]
        print()
        print("[Global Stats] Across %d normalized subjects:" % n_ok)
        print("  p_low  (percentile %.1f): median=%.4f  range=[%.4f, %.4f]" % (
            args.low_pct,
            float(np.median(p_lows)),
            float(np.min(p_lows)),
            float(np.max(p_lows)),
        ))
        print("  p_high (percentile %.1f): median=%.4f  range=[%.4f, %.4f]" % (
            args.high_pct,
            float(np.median(p_highs)),
            float(np.min(p_highs)),
            float(np.max(p_highs)),
        ))
        print("  out_min : median=%.4f  range=[%.4f, %.4f]" % (
            float(np.median([r["out_min"] for r in ok_list])),
            float(np.min([r["out_min"] for r in ok_list])),
            float(np.max([r["out_min"] for r in ok_list])),
        ))
        print("  out_max : median=%.4f  range=[%.4f, %.4f]" % (
            float(np.median([r["out_max"] for r in ok_list])),
            float(np.min([r["out_max"] for r in ok_list])),
            float(np.max([r["out_max"] for r in ok_list])),
        ))

    # --- Save global summary JSON ---
    summary = {
        "args": {
            "input_root": str(input_root),
            "low_pct": args.low_pct,
            "high_pct": args.high_pct,
            "exclude_zero": args.exclude_zero,
            "mask_threshold": args.mask_threshold,
            "workers": args.workers,
        },
        "counts": {
            "total": len(tasks),
            "ok": n_ok,
            "skipped": n_skipped,
            "empty_mask": n_empty,
            "error": n_error,
        },
        "elapsed_s": round(elapsed, 1),
        "errors": [
            {
                "subject": r["subject"],
                "status": r["status"],
                "reason": r.get("reason", "?"),
            }
            for r in error_list
        ],
    }
    if ok_list:
        summary["global_stats"] = {
            "p_low_median": float(np.median(p_lows)),
            "p_low_min": float(np.min(p_lows)),
            "p_low_max": float(np.max(p_lows)),
            "p_high_median": float(np.median(p_highs)),
            "p_high_min": float(np.min(p_highs)),
            "p_high_max": float(np.max(p_highs)),
        }

    if not args.dry_run:
        summary_path = input_root / "step5_normalize_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("[INFO] Summary: %s" % summary_path)

    if error_list:
        print()
        print("[ERRORS] %d subjects failed:" % n_error)
        for r in error_list[:30]:
            print("  %s: [%s] %s" % (r["subject"], r["status"], r.get("reason", "?")))
        if len(error_list) > 30:
            print("  ... and %d more" % (len(error_list) - 30))


def _print_result(result: dict, total: int) -> None:
    """Print a single-subject progress line."""
    status = result["status"]
    subj = result["subject"]

    if status == "ok":
        tag = "OK   "
        extra = "  shape=%s  p_low=%.4f  p_high=%.4f  mask=%d vox" % (
            result.get("shape", "?"),
            result.get("p_low", 0.0),
            result.get("p_high", 0.0),
            result.get("mask_voxels", 0),
        )
    elif status == "skipped":
        tag = "SKIP "
        extra = "  (%s)" % result.get("reason", "?")
    elif status == "empty_mask":
        tag = "EMPTY"
        extra = "  mask_voxels=0"
    else:
        tag = "ERR  "
        extra = "  [%s] %s" % (status, result.get("reason", "?"))

    print("[%s] %s  %s" % (tag, subj, extra))


if __name__ == "__main__":
    main()
