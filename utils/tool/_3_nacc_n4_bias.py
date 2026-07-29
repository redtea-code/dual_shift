"""
NACC N4 Bias Field Correction
===============================
Applies N4 bias field correction using ANTsPy on Windows.

Key design decision (from clinical best practice):
  N4 is applied to the RAW T1 with brain mask as weight/constraint,
  NOT on the already skull-stripped image. This is because N4 estimates
  the low-frequency intensity field of the original acquisition.

Fallback: if brain mask is missing, generates one from the skull-stripped brain
image (threshold > 0), then applies N4 on raw T1 with that mask.

Input:  {subject}/MRI.nii.gz (raw T1) + {subject}/MRI_brain_mask.nii.gz
        [fallback: MRI_brain.nii.gz -> generate mask]
Output: {subject}/MRI_N4.nii.gz       (N4-corrected full FOV)
        {subject}/MRI_N4_brain.nii.gz  (N4-corrected + brain-masked)

Requirements:
  pip install antspyx nibabel

Usage:
  python _3_nacc_n4_bias.py                      # process all subjects
  python _3_nacc_n4_bias.py --dry-run             # list pending
  python _3_nacc_n4_bias.py --subject NACC000806-2023_11_14-1  # single
"""

import os
import sys
import argparse
import logging
import traceback
from datetime import datetime

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _0_nacc_config import (
    NACC_PRE_DIR, MRI_RAW, MRI_BRAIN, MRI_BRAIN_MASK,
    MRI_N4, MRI_N4_BRAIN,
    N4_CONVERGENCE, N4_SHRINK_FACTOR, N4_SPLINE_PARAM,
    RESUME, LOG_DIR,
)

os.makedirs(LOG_DIR, exist_ok=True)


# ---- Mask Generation Fallback ----

def ensure_brain_mask(subject_dir):
    """
    Ensure brain mask exists.

    PRIMARY path: SynthStrip's MRI_brain_mask.nii.gz (preferred - cleanest).
    FALLBACK:    generate from MRI_brain.nii.gz via threshold > 0.
                 This fallback is for robustness; SynthStrip's direct mask
                 should always be the primary choice for N4 constraint.

    Returns mask_path or None on failure.
    """
    mask_path = os.path.join(subject_dir, MRI_BRAIN_MASK)

    if os.path.isfile(mask_path) and os.path.getsize(mask_path) > 0:
        return mask_path

    # Fallback: generate from skull-stripped brain
    brain_path = os.path.join(subject_dir, MRI_BRAIN)
    if not os.path.isfile(brain_path):
        return None

    try:
        brain_img = nib.load(brain_path)
        brain_data = brain_img.get_fdata()
        mask_data = (brain_data > 0).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header)
        nib.save(mask_img, mask_path)
        return mask_path
    except Exception:
        return None


# ---- N4 Bias Correction ----

def run_n4_correction(subject_dir):
    """
    N4 bias correction on raw T1 constrained by brain mask.

    Flow:
      1. Load raw T1 + brain mask (generate mask from brain if missing)
      2. Validate that mask matches T1 space
      3. N4 on raw T1 using mask as weight
      4. Save N4-corrected full FOV
      5. Apply mask -> N4-corrected brain-only

    Returns:
        (subject_name, status, message)
    """
    import ants

    subject_name = os.path.basename(subject_dir)
    t1_path = os.path.join(subject_dir, MRI_RAW)
    n4_path = os.path.join(subject_dir, MRI_N4)
    n4_brain_path = os.path.join(subject_dir, MRI_N4_BRAIN)

    # --- Resume check ---
    if RESUME and os.path.isfile(n4_brain_path) and os.path.getsize(n4_brain_path) > 0:
        return subject_name, "skipped", f"exists ({os.path.getsize(n4_brain_path) // 1024}KB)"

    # --- Validate raw T1 ---
    if not os.path.isfile(t1_path):
        return subject_name, "error", "MRI.nii.gz not found"

    # --- Ensure brain mask ---
    mask_path = ensure_brain_mask(subject_dir)
    if mask_path is None:
        return subject_name, "error", (
            "brain mask not found and cannot generate (run _2_nacc_skullstrip.py first)"
        )

    try:
        import numpy as np

        # Load images
        t1_raw = ants.image_read(t1_path)
        mask = ants.image_read(mask_path)

        # Binarize mask explicitly before N4.
        # SynthStrip mask may have soft edges; threshold ensures clean binary.
        mask = ants.threshold_image(mask, 0.5, 1e9, 1, 0)

        # Validate mask is in the same physical space as T1.
        # shape alone is not enough: spacing/origin/direction must also match.
        if t1_raw.shape != mask.shape:
            return subject_name, "error", (
                f"shape mismatch: T1 {t1_raw.shape} vs mask {mask.shape}"
            )
        tol = 1e-4
        if not (
            np.allclose(t1_raw.spacing, mask.spacing, atol=tol)
            and np.allclose(t1_raw.origin, mask.origin, atol=tol)
            and np.allclose(np.array(t1_raw.direction), np.array(mask.direction), atol=tol)
        ):
            return subject_name, "error", (
                f"space mismatch: "
                f"T1(spacing={t1_raw.spacing}, origin={t1_raw.origin}); "
                f"mask(spacing={mask.spacing}, origin={mask.origin})"
            )

        # --- N4 bias field correction ---
        # Key: N4 on RAW T1, mask constrains the bias field estimation.
        # This is clinically correct: bias field is estimated from the
        # original acquisition, not from a skull-stripped image.
        # NOTE: ANTsPy parameter is 'spline_param', NOT 'bspline_params'.
        t1_n4 = ants.n4_bias_field_correction(
            image=t1_raw,
            mask=mask,
            convergence=N4_CONVERGENCE,
            shrink_factor=N4_SHRINK_FACTOR,
            spline_param=N4_SPLINE_PARAM,
            rescale_intensities=False,
            verbose=True,
        )

        # Save N4-corrected full FOV
        ants.image_write(t1_n4, n4_path)

        # Apply brain mask: zero out non-brain voxels.
        # Use new_image_like() instead of from_numpy() to preserve ANTs metadata.
        t1_n4_arr = t1_n4.numpy()
        mask_arr = mask.numpy() > 0.5
        t1_n4_brain_arr = t1_n4_arr.copy()
        t1_n4_brain_arr[~mask_arr] = 0

        t1_n4_brain = t1_n4.new_image_like(t1_n4_brain_arr)
        ants.image_write(t1_n4_brain, n4_brain_path)

        output_size = os.path.getsize(n4_brain_path) // 1024
        return subject_name, "success", f"{output_size}KB"

    except Exception:
        # Clean partial outputs
        for p in [n4_path, n4_brain_path]:
            if os.path.isfile(p):
                os.remove(p)
        return subject_name, "error", traceback.format_exc()[-300:]


# ---- Batch Processing ----

def get_tasks():
    """Discover subjects that need N4 correction."""
    tasks = []
    for entry in sorted(os.listdir(NACC_PRE_DIR)):
        sub_dir = os.path.join(NACC_PRE_DIR, entry)
        if not os.path.isdir(sub_dir):
            continue

        t1_path = os.path.join(sub_dir, MRI_RAW)
        if not os.path.isfile(t1_path):
            continue

        if RESUME:
            n4_brain_path = os.path.join(sub_dir, MRI_N4_BRAIN)
            if os.path.isfile(n4_brain_path) and os.path.getsize(n4_brain_path) > 0:
                continue

        tasks.append(sub_dir)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="NACC N4 Bias Field Correction")
    parser.add_argument("--dry-run", action="store_true", help="List pending")
    parser.add_argument("--subject", type=str, default=None, help="Single subject")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess all")
    args = parser.parse_args()

    global RESUME
    if args.no_resume:
        RESUME = False

    log_file = os.path.join(LOG_DIR, f"n4_bias_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # --- Single subject ---
    if args.subject:
        sub_dir = os.path.join(NACC_PRE_DIR, args.subject)
        name, status, msg = run_n4_correction(sub_dir)
        logger.info(f"[{name}] {status.upper()}: {msg}")
        sys.exit(0 if status in ("success", "skipped") else 1)

    # --- Batch ---
    tasks = get_tasks()

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  Pending N4: {len(tasks)} subjects")
        print(f"  (run _2_nacc_skullstrip.py first if needed)")
        print(f"{'='*60}")
        for t in tasks[:20]:
            print(f"  {os.path.basename(t)}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        print(f"{'='*60}\n")
        sys.exit(0)

    if not tasks:
        logger.info("All subjects already N4-corrected. Nothing to do.")
        sys.exit(0)

    total = len(tasks)
    logger.info(f"=" * 60)
    logger.info(f"N4 Bias Correction Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Total: {total} | Resume: {RESUME}")
    logger.info(f"=" * 60)

    success, skipped, failed = 0, 0, 0

    for i, task_dir in enumerate(tasks, 1):
        subject_name = os.path.basename(task_dir)
        logger.info(f"[{datetime.now():%H:%M:%S}] #{i}/{total}: {subject_name}")

        name, status, msg = run_n4_correction(task_dir)

        if status == "success":
            success += 1
            logger.info(f"  -> OK (done={success}, skip={skipped}, fail={failed})")
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            logger.info(f"  -> FAILED: {msg}")

    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"N4 Bias Correction Complete: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Total:    {total}")
    logger.info(f"Success:  {success}")
    logger.info(f"Skipped:  {skipped}")
    logger.info(f"Failed:   {failed}")
    logger.info(f"=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
