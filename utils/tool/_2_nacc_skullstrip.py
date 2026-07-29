"""
NACC SynthStrip Skull Stripping
================================
Runs SynthStrip via WSL with FreeSurfer, producing brain-extracted images
and brain masks (required for N4 bias correction).

Based on synthstrip_adni_pre.sh pattern:
  - Source $FREESURFER_HOME/SetUpFreeSurfer.sh before running
  - Blacklist for subjects that crash WSL
  - Resume support (skip if output exists)

Input:  {subject}/MRI.nii.gz
Output: {subject}/MRI_brain.nii.gz  (skull-stripped brain)
        {subject}/MRI_brain_mask.nii.gz (binary mask, needed for N4)

Usage:
  python _2_nacc_skullstrip.py                    # process all subjects
  python _2_nacc_skullstrip.py --dry-run           # list pending subjects
  python _2_nacc_skullstrip.py --subject NACC000806-2023_11_14-1  # single
  python _2_nacc_skullstrip.py --no-resume         # reprocess all
"""



import os
import sys
import subprocess
import argparse
import logging
import traceback
from datetime import datetime

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _0_nacc_config import (
    NACC_PRE_DIR, MRI_RAW, MRI_BRAIN, MRI_BRAIN_MASK,
    FREESURFER_HOME, SYNTHSTRIP_CMD, WSL_DISTRO,
    SYNTHSTRIP_BLACKLIST, RESUME, LOG_DIR,
)

os.makedirs(LOG_DIR, exist_ok=True)


# ---- Path Conversion ----

def windows_to_wsl(win_path):
    """Convert Windows path to WSL path: E:\\path -> /mnt/e/path"""
    win_path = os.path.normpath(win_path)
    drive = win_path[0].lower()
    wsl_path = "/mnt/" + drive + win_path[2:].replace("\\", "/")
    return wsl_path


# ---- Build WSL Command ----

def build_synthstrip_cmd(wsl_input, wsl_brain, wsl_mask):
    """
    Build the full WSL command that:
      1. Sets FREESURFER_HOME and sources SetUpFreeSurfer.sh
      2. Runs mri_synthstrip with input -> brain + mask outputs

    IMPORTANT: Uses ; not && because export + source must persist
    in the same shell context for mri_synthstrip to inherit them.

    Returns list of args for subprocess.run.
    """
    # mri_synthstrip flags:
    #   -i <input>   : input T1
    #   -o <brain>   : skull-stripped output
    #   -m <mask>    : binary brain mask (needed for N4 bias correction)
    synthstrip_cmd = (
        f"{SYNTHSTRIP_CMD} -i '{wsl_input}' -o '{wsl_brain}' -m '{wsl_mask}'"
    )

    # Setup shell context: export FREESURFER_HOME, source setup, run cmd.
    # Using ; ensures all run in the same shell where export persists.
    bash_script = (
        f"export FREESURFER_HOME={FREESURFER_HOME}; "
        f"source {FREESURFER_HOME}/SetUpFreeSurfer.sh; "
        f"{synthstrip_cmd}"
    )

    return ["wsl", "-d", WSL_DISTRO, "--", "bash", "-c", bash_script]


# ---- Core Processing ----

def run_synthstrip(subject_dir, logger=None):
    """
    Run SynthStrip on a single subject.

    Args:
        subject_dir: Path to subject directory containing MRI.nii.gz

    Returns:
        (subject_name, status, message)
        status: "success" | "skipped" | "blacklisted" | "error"
    """
    subject_name = os.path.basename(subject_dir)
    input_path = os.path.join(subject_dir, MRI_RAW)
    brain_path = os.path.join(subject_dir, MRI_BRAIN)
    mask_path = os.path.join(subject_dir, MRI_BRAIN_MASK)

    # --- Blacklist check ---
    if subject_name in SYNTHSTRIP_BLACKLIST:
        return subject_name, "blacklisted", "in blacklist"

    # --- Resume check ---
    if RESUME:
        if os.path.isfile(brain_path) and os.path.getsize(brain_path) > 0:
            if os.path.isfile(mask_path) and os.path.getsize(mask_path) > 0:
                b_kb = os.path.getsize(brain_path) // 1024
                m_kb = os.path.getsize(mask_path) // 1024
                return subject_name, "skipped", f"brain={b_kb}KB mask={m_kb}KB"

    # --- Validate input ---
    if not os.path.isfile(input_path):
        return subject_name, "error", f"MRI.nii.gz not found"
    input_size = os.path.getsize(input_path)
    if input_size < 1024:
        return subject_name, "error", f"MRI.nii.gz too small ({input_size} bytes)"

    # --- Convert paths for WSL ---
    wsl_input = windows_to_wsl(input_path)
    wsl_brain = windows_to_wsl(brain_path)
    wsl_mask = windows_to_wsl(mask_path)

    # --- Run SynthStrip ---
    cmd = build_synthstrip_cmd(wsl_input, wsl_brain, wsl_mask)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min per subject
        )

        if result.returncode != 0:
            err_msg = result.stderr.strip()[-300:] if result.stderr else "unknown error"
            # Clean partial outputs on failure
            for p in [brain_path, mask_path]:
                if os.path.isfile(p):
                    os.remove(p)
            return subject_name, "error", f"exit={result.returncode}: {err_msg}"

        # --- Verify outputs ---
        if not os.path.isfile(brain_path) or os.path.getsize(brain_path) == 0:
            return subject_name, "error", "brain output missing/empty"

        # If mask flag not supported by this FreeSurfer version,
        # generate mask from brain image (threshold > 0)
        if not os.path.isfile(mask_path) or os.path.getsize(mask_path) == 0:
            _generate_mask_from_brain(brain_path, mask_path)
            if not os.path.isfile(mask_path):
                return subject_name, "error", "mask generation failed"

        b_kb = os.path.getsize(brain_path) // 1024
        m_kb = os.path.getsize(mask_path) // 1024
        return subject_name, "success", f"brain={b_kb}KB mask={m_kb}KB"

    except subprocess.TimeoutExpired:
        for p in [brain_path, mask_path]:
            if os.path.isfile(p):
                os.remove(p)
        return subject_name, "error", "timeout (>300s)"
    except FileNotFoundError:
        return subject_name, "error", "WSL not found"
    except Exception:
        for p in [brain_path, mask_path]:
            if os.path.isfile(p):
                os.remove(p)
        return subject_name, "error", traceback.format_exc()[-300:]


def _generate_mask_from_brain(brain_path, mask_path):
    """
    Fallback: generate binary brain mask from skull-stripped image.
    Any voxel > 0 in the brain image is considered brain tissue.
    """
    try:
        import nibabel as nib
        import numpy as np

        brain_img = nib.load(brain_path)
        brain_data = brain_img.get_fdata()
        mask_data = (brain_data > 0).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header)
        nib.save(mask_img, mask_path)
    except Exception:
        pass


# ---- Batch Processing ----

def get_tasks():
    """Discover subjects that need skull stripping."""
    tasks = []
    for entry in sorted(os.listdir(NACC_PRE_DIR)):
        sub_dir = os.path.join(NACC_PRE_DIR, entry)
        if not os.path.isdir(sub_dir):
            continue
        input_path = os.path.join(sub_dir, MRI_RAW)
        if not os.path.isfile(input_path):
            continue

        # Check blacklist
        if entry in SYNTHSTRIP_BLACKLIST:
            continue

        # Check resume
        if RESUME:
            brain_path = os.path.join(sub_dir, MRI_BRAIN)
            mask_path = os.path.join(sub_dir, MRI_BRAIN_MASK)
            if os.path.isfile(brain_path) and os.path.getsize(brain_path) > 0:
                if os.path.isfile(mask_path) and os.path.getsize(mask_path) > 0:
                    continue
        tasks.append(sub_dir)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="NACC SynthStrip Skull Stripping")
    parser.add_argument("--dry-run", action="store_true", help="List pending only")
    parser.add_argument("--subject", type=str, default=None, help="Single subject")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess all")
    args = parser.parse_args()

    global RESUME
    if args.no_resume:
        RESUME = False

    # Logging
    log_file = os.path.join(LOG_DIR, f"synthstrip_{datetime.now():%Y%m%d_%H%M%S}.log")
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
        if not os.path.isdir(sub_dir):
            logger.error(f"Subject not found: {args.subject}")
            sys.exit(1)
        name, status, msg = run_synthstrip(sub_dir, logger)
        logger.info(f"[{name}] {status.upper()}: {msg}")
        sys.exit(0 if status in ("success", "skipped") else 1)

    # --- Batch ---
    tasks = get_tasks()

    if args.dry_run:
        blacklisted = sum(1 for b in SYNTHSTRIP_BLACKLIST
                          if os.path.isdir(os.path.join(NACC_PRE_DIR, b)))
        print(f"\n{'='*60}")
        print(f"  Pending: {len(tasks)} subjects")
        print(f"  Blacklisted: {blacklisted} subjects")
        print(f"{'='*60}")
        for t in tasks[:20]:
            print(f"  {os.path.basename(t)}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        print(f"{'='*60}\n")
        sys.exit(0)

    if not tasks:
        logger.info("All subjects already processed. Nothing to do.")
        sys.exit(0)

    total = len(tasks)
    logger.info(f"=" * 60)
    logger.info(f"SynthStrip Batch Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"NACC_PRE_DIR: {NACC_PRE_DIR}")
    logger.info(f"FREESURFER_HOME: {FREESURFER_HOME}")
    logger.info(f"Total: {total} | Blacklist: {len(SYNTHSTRIP_BLACKLIST)} | Resume: {RESUME}")
    logger.info(f"=" * 60)

    # Serial processing (SynthStrip uses GPU, safer one at a time)
    success, skipped, blacklisted, failed = 0, 0, 0, 0

    for i, task_dir in enumerate(tasks, 1):
        subject_name = os.path.basename(task_dir)
        logger.info(f"[{datetime.now():%H:%M:%S}] #{i}/{total}: {subject_name}")

        name, status, msg = run_synthstrip(task_dir, logger)

        if status == "success":
            success += 1
            logger.info(f"  -> OK (done={success}, skip={skipped}, "
                        f"blacklisted={blacklisted}, fail={failed})")
        elif status == "skipped":
            skipped += 1
        elif status == "blacklisted":
            blacklisted += 1
            logger.info(f"  -> BLACKLISTED (skip)")
        else:
            failed += 1
            logger.info(f"  -> FAILED: {msg}")

    # --- Report ---
    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"SynthStrip Batch Complete: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Total:        {total}")
    logger.info(f"Success:      {success}")
    logger.info(f"Skipped:      {skipped}")
    logger.info(f"Blacklisted:  {blacklisted}")
    logger.info(f"Failed:       {failed}")
    logger.info(f"=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
