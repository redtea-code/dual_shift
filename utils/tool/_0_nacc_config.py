"""
NACC MRI Preprocessing Configuration
=====================================
Centralized paths and parameters shared across all NACC preprocessing scripts.

Edit this file to customize paths for your environment.
"""

import os

# ========================== DATA PATHS ==========================

# NACC raw data directory (contains subject folders with MRI.nii.gz + MRI.json)
NACC_PRE_DIR = r"D:\ADNI_dataset\Gmamba\NACC\NACC_pre"

# MNI152 template for registration
MNI152_TEMPLATE = r"D:\MNI152\MNI152_T1_1mm_Brain.nii.gz"

# ========================== WSL / FREESURFER / SYNTHSTRIP ==========================

# WSL distribution name (from 'wsl --list')
WSL_DISTRO = "Ubuntu"

# FreeSurfer installation in WSL
FREESURFER_HOME = "/home/cyhbad/freesurfer/freesurfer"

# SynthStrip command (available after sourcing SetUpFreeSurfer.sh)
SYNTHSTRIP_CMD = "mri_synthstrip"

# Note: This FreeSurfer version's mri_synthstrip does NOT support --device.
# It auto-uses CPU; for GPU, consider installing the pip synthstrip package.

# Blacklist: subjects known to crash WSL during SynthStrip
SYNTHSTRIP_BLACKLIST = [
    # Add problematic NACC subjects here as you discover them
    # e.g. "NACC000806-2023_11_14-1",
]

# ========================== N4 BIAS CORRECTION ==========================

# N4 convergence parameters (ANTsPy >=0.6.x requires dict, not string)
# {iters: [resolution levels], tol: convergence threshold}
N4_CONVERGENCE = {"iters": [50, 50, 50, 50], "tol": 0.000001}
N4_SHRINK_FACTOR = 4
# spline_param: single int (mm spacing for spline grid).
# ANTsPy wraps it as "[N]" internally; pass the raw int.
N4_SPLINE_PARAM = 200

# ========================== INTENSITY NORMALIZATION ==========================

# Percentile clipping range (within brain mask)
INTENSITY_CLIP_LOW = 1   # 1st percentile
INTENSITY_CLIP_HIGH = 99  # 99th percentile

# ========================== OUTPUT NAMING ==========================

MRI_RAW = "MRI.nii.gz"                    # raw T1 (input)
MRI_JSON = "MRI.json"                     # metadata (input)
MRI_BRAIN = "MRI_brain.nii.gz"            # skull-stripped brain (SynthStrip output)
MRI_BRAIN_MASK = "MRI_brain_mask.nii.gz"  # binary brain mask (SynthStrip -m output)
MRI_N4 = "MRI_N4.nii.gz"                  # N4-corrected full FOV
MRI_N4_BRAIN = "MRI_N4_brain.nii.gz"      # N4-corrected + brain-masked
MRI_N4_BRAIN_MNI = "MRI_N4_brain_mni152.nii.gz"  # MNI152 registered

# ========================== PROCESSING ==========================

# Enable/disable resume (skip if output exists)
RESUME = True

# Number of parallel workers (0 = auto = cpu_count, max 8 for memory safety)
NUM_WORKERS = 0

# ========================== LOGGING ==========================

LOG_DIR = os.path.join(os.path.dirname(NACC_PRE_DIR), "logs")
LOG_LEVEL = "INFO"


def validate_config():
    """Check that all required paths exist."""
    issues = []
    if not os.path.isdir(NACC_PRE_DIR):
        issues.append(f"NACC_PRE_DIR not found: {NACC_PRE_DIR}")
    if not os.path.isfile(MNI152_TEMPLATE):
        issues.append(f"MNI152_TEMPLATE not found: {MNI152_TEMPLATE}")
    if issues:
        print("[CONFIG WARNINGS]")
        for i in issues:
            print(f"  ! {i}")
    else:
        print("[CONFIG] All paths validated OK")
    return len(issues) == 0


if __name__ == "__main__":
    validate_config()
    print(f"\nNACC_PRE_DIR:     {NACC_PRE_DIR}")
    print(f"MNI152_TEMPLATE:   {MNI152_TEMPLATE}")
    print(f"FREESURFER_HOME:   {FREESURFER_HOME}")
    print(f"SYNTHSTRIP_CMD:    {SYNTHSTRIP_CMD}")
    print(f"WSL_DISTRO:        {WSL_DISTRO}")
    print(f"BLACKLIST:         {len(SYNTHSTRIP_BLACKLIST)} subjects")
    print(f"RESUME:            {RESUME}")
