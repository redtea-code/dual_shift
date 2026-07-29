"""
NACC MRI Full Preprocessing Pipeline
======================================
Master orchestrator that runs the complete NACC preprocessing workflow:

  Step 1: SynthStrip skull stripping (via WSL)
  Step 2: N4 bias field correction (ANTsPy on Windows)
  Step 3: MNI152 registration (ANTsPy on Windows)
  Step 4: Intensity normalization & QC report

Each step produces intermediate outputs and supports resume, so you can
stop and restart without redoing completed work.

Usage:
  # Full pipeline
  python nacc_mri_pipeline.py

  # Specific steps
  python nacc_mri_pipeline.py --steps skullstrip,n4
  python nacc_mri_pipeline.py --steps n4,mni
  python nacc_mri_pipeline.py --steps qc

  # Single subject
  python nacc_mri_pipeline.py --subject NACC000806-2023_11_14-1

  # Dry run (list pending)
  python nacc_mri_pipeline.py --dry-run

Pipeline Overview:
  ┌─────────────────────────────────────────────────────────┐
  │ NACC Raw T1 (Accelerated Sagittal MPRAGE)               │
  │   {subject}/MRI.nii.gz                                  │
  └──────────────────────┬──────────────────────────────────┘
                         │ ① SynthStrip (WSL)
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Skull-stripped + Brain mask                             │
  │   {subject}/MRI_brain.nii.gz                            │
  │   {subject}/MRI_brain_mask.nii.gz                       │
  └──────────────────────┬──────────────────────────────────┘
                         │ ② N4 Bias Correction (ANTsPy Windows)
                         │    on RAW T1, masked by brain mask
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ N4-corrected + Brain-masked                             │
  │   {subject}/MRI_N4.nii.gz                               │
  │   {subject}/MRI_N4_brain.nii.gz                         │
  └──────────────────────┬──────────────────────────────────┘
                         │ ③ MNI152 Registration (ANTsPy)
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ MNI152 registered                                       │
  │   {subject}/MRI_N4_brain_mni152.nii.gz                  │
  └──────────────────────┬──────────────────────────────────┘
                         │ ④ QC Report
                         ▼
                         Done ✓
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _0_nacc_config import (
    NACC_PRE_DIR, LOG_DIR, RESUME,
    MRI_RAW, MRI_BRAIN, MRI_BRAIN_MASK,
    MRI_N4, MRI_N4_BRAIN, MRI_N4_BRAIN_MNI,
)

os.makedirs(LOG_DIR, exist_ok=True)

# Path to companion scripts
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SKULLSTRIP_SCRIPT = os.path.join(UTILS_DIR, "_2_nacc_skullstrip.py")
N4_SCRIPT = os.path.join(UTILS_DIR, "_3_nacc_n4_bias.py")
MNI_SCRIPT = os.path.join(UTILS_DIR, "_4_2_pre_MNI152_mul.py")

# Python executable (D:\Anaconda\python.exe)
PYTHON_EXE = sys.executable

AVAILABLE_STEPS = ["skullstrip", "n4", "mni", "qc"]


# ---- Step Runner ----

def run_step(step_name, extra_args=None):
    """Run a preprocessing step script and return success/failure."""
    step_map = {
        "skullstrip": SKULLSTRIP_SCRIPT,
        "n4": N4_SCRIPT,
        "mni": MNI_SCRIPT,
    }
    if step_name not in step_map:
        return True  # qc is handled separately

    script = step_map[step_name]
    if not os.path.isfile(script):
        if step_name == "mni":
            print(f"  [SKIP] MNI script not found at {script}")
            print(f"         Run _4_2_pre_MNI152_mul.py separately with source pointing to N4 outputs.")
            return True
        print(f"  [ERROR] Script not found: {script}")
        return False

    cmd = [PYTHON_EXE, script]
    if extra_args:
        cmd.extend(extra_args)
    if not RESUME:
        cmd.append("--no-resume")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ---- QC Report ----

def run_qc():
    """Generate QC report: count files per step."""
    print(f"\n{'='*60}")
    print(f"  QC Report: File Inventory")
    print(f"{'='*60}")

    counts = Counter()
    total_subjects = 0
    errors_list = []

    for entry in sorted(os.listdir(NACC_PRE_DIR)):
        sub_dir = os.path.join(NACC_PRE_DIR, entry)
        if not os.path.isdir(sub_dir):
            continue
        total_subjects += 1

        has_raw = os.path.isfile(os.path.join(sub_dir, MRI_RAW))
        has_brain = os.path.isfile(os.path.join(sub_dir, MRI_BRAIN))
        has_mask = os.path.isfile(os.path.join(sub_dir, MRI_BRAIN_MASK))
        has_n4 = os.path.isfile(os.path.join(sub_dir, MRI_N4))
        has_n4_brain = os.path.isfile(os.path.join(sub_dir, MRI_N4_BRAIN))
        has_mni = os.path.isfile(os.path.join(sub_dir, MRI_N4_BRAIN_MNI))

        if has_raw:
            counts["raw"] += 1
        if has_brain:
            counts["brain (skullstrip)"] += 1
        if has_mask:
            counts["brain_mask"] += 1
        if has_n4:
            counts["N4 (full FOV)"] += 1
        if has_n4_brain:
            counts["N4_brain"] += 1
        if has_mni:
            counts["MNI152"] += 1

        # Detect incomplete subjects
        if has_raw and not has_n4_brain:
            missing = []
            if not has_brain:
                missing.append("brain")
            if not has_mask:
                missing.append("mask")
            if not has_n4:
                missing.append("N4")
            if not has_n4_brain:
                missing.append("N4_brain")
            errors_list.append(f"  {entry}: missing {missing}")

    print(f"\n  Total subjects: {total_subjects}")
    print(f"  {'─'*40}")
    for step_name, count in counts.items():
        pct = count * 100.0 / total_subjects if total_subjects else 0
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  {step_name:<22s} {count:>5d}  {bar} {pct:5.1f}%")

    if errors_list:
        print(f"\n  [!] {len(errors_list)} subjects incomplete:")
        for e in errors_list[:20]:
            print(e)
        if len(errors_list) > 20:
            print(f"  ... and {len(errors_list) - 20} more")

    print(f"\n{'='*60}")

    # Pipeline recommendation
    if counts.get("MNI152", 0) < total_subjects:
        if counts.get("N4_brain", 0) < total_subjects:
            print("  NEXT: python nacc_mri_pipeline.py --steps skullstrip,n4")
        else:
            print("  NEXT: python nacc_mri_pipeline.py --steps mni")
    else:
        print("  ✓ All subjects fully processed!")
    print(f"{'='*60}\n")


# ---- Main ----

def main():
    # Ensure UTF-8 output on Windows terminals
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="NACC MRI Full Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nacc_mri_pipeline.py                           # Full pipeline
  python nacc_mri_pipeline.py --steps skullstrip,n4     # Only skullstrip + N4
  python nacc_mri_pipeline.py --steps qc                # QC report only
  python nacc_mri_pipeline.py --subject NACC000806-2023_11_14-1
  python nacc_mri_pipeline.py --dry-run
        """,
    )
    parser.add_argument(
        "--steps", type=str, default="all",
        help=f"Comma-separated steps: {','.join(AVAILABLE_STEPS)} or 'all'",
    )
    parser.add_argument("--subject", type=str, default=None, help="Single subject only")
    parser.add_argument("--dry-run", action="store_true", help="Show pending counts")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess all outputs")
    args = parser.parse_args()

    global RESUME
    if args.no_resume:
        RESUME = False

    log_file = os.path.join(LOG_DIR, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Parse steps
    if args.steps == "all":
        steps = ["skullstrip", "n4", "mni", "qc"]
    else:
        steps = [s.strip().lower() for s in args.steps.split(",")]
        for s in steps:
            if s not in AVAILABLE_STEPS:
                logger.error(f"Unknown step: '{s}'. Available: {AVAILABLE_STEPS}")
                sys.exit(1)

    # Subject-specific args
    subj_args = ["--subject", args.subject] if args.subject else []

    logger.info(f"Pipeline start: steps={steps}, subject={args.subject or 'all'}, resume={RESUME}")
    logger.info(f"NACC_PRE_DIR: {NACC_PRE_DIR}")

    # --- Dry run ---
    if args.dry_run:
        run_qc()
        sys.exit(0)

    # --- Execute steps ---
    failed = False
    for step in steps:
        print(f"\n{'='*60}")
        print(f"  STEP: {step.upper()}")
        print(f"{'='*60}")

        if step == "qc":
            run_qc()
        else:
            ok = run_step(step, extra_args=subj_args)
            if not ok:
                logger.error(f"Step '{step}' failed. Stopping pipeline.")
                failed = True
                break

    if not failed:
        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE ✓")
        print(f"{'='*60}")
        # Always show final QC
        if "qc" not in steps:
            run_qc()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
