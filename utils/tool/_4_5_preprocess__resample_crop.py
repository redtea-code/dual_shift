#!/usr/bin/env python3
"""
NACC/ADNI MRI Preprocessing -- Step: Resample & Crop (Flexible Crop Dimension)
===============================================================================
A flexible resample & crop script that supports multiple crop modes,
making it easy to customize crop dimensions for different datasets.

┌──────────────────────────────────────────────────────────────────┐
│ CROP MODES                                                       │
├──────────────┬───────────────────────────────────────────────────┤
│ fixed_voxel  │ Fixed crop box in voxel indices (backward compat) │
│ fixed_mm     │ Fixed crop box in MNI mm coordinates              │
│ mask_bbox    │ Auto-crop to brain-mask bounding box + padding    │
│ center_fixed │ Fixed-size crop centered on mask COM (CNN-ready)  │
│ none         │ Skip cropping — resample only                     │
└──────────────┴───────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ RESAMPLE MODES                                                   │
├──────────────┬───────────────────────────────────────────────────┤
│ reference    │ Resample to MNI152 1mm reference grid             │
│ spacing      │ Resample to target voxel spacing                  │
│ none         │ Skip resampling                                   │
└──────────────┴───────────────────────────────────────────────────┘

Key improvements over previous scripts:
  1.  Crop box in MNI mm coordinates → auto-converts to voxel indices
  2.  Auto-crop to mask bounding box with configurable padding (in mm)
  3.  Strict-mode toggle: enforce exact crop or lenient clipping
  4.  Flexible I/O filenames via CLI (works with NACC / ADNI / any dataset)
  5.  Single-subject mode (--single) for debugging
  6.  Resample to MNI152 reference grid for consistent spatial alignment

Input:   MNI-registered brain image (.nii.gz)
         MNI-registered brain mask  (.nii.gz)
Output:  Cropped brain image (.nii.gz)
         Cropped brain mask  (.nii.gz)

Usage examples:
  # Fixed-voxel crop (backward compatible with old scripts)
  python _4_5_preprocess__resample_crop.py                          \\
      --input-root E:/2.causal/NACC_pre                           \\
      --mni-ref D:/MNI152/MNI152_T1_1mm_Brain.nii.gz              \\
      --crop-mode fixed_voxel                                     \\
      --crop-box-voxel 16 166 18 204 12 170

  # Fixed-mm crop (researcher-friendly MNI coordinates)
  python _4_5_preprocess__resample_crop.py                          \\
      --input-root E:/2.causal/NACC_pre                           \\
      --mni-ref D:/MNI152/MNI152_T1_1mm_Brain.nii.gz              \\
      --crop-mode fixed_mm                                        \\
      --crop-box-mm -90 90 -125 90 -70 105

  # Auto crop to mask bounding box with 10 mm padding
  python _4_5_preprocess__resample_crop.py                          \\
      --input-root E:/2.causal/NACC_pre                           \\
      --mni-ref D:/MNI152/MNI152_T1_1mm_Brain.nii.gz              \\
      --crop-mode mask_bbox                                       \\
      --bbox-padding 10

  # Single subject with custom I/O filenames
  python _4_5_preprocess__resample_crop.py                          \\
      --input-root E:/2.causal/NACC_pre                           \\
      --mni-ref D:/MNI152/MNI152_T1_1mm_Brain.nii.gz              \\
      --single I123456                                            \\
      --input-image MRI_N4_brain_mni152.nii.gz                    \\
      --input-mask MRI_brain_mask_mni152.nii.gz                   \\
      --output-image MRI_brain_mni152_cropped.nii.gz              \\
      --output-mask MRI_brain_mask_mni152_cropped.nii.gz

  # Dry-run to preview what will happen
  python _4_5_preprocess__resample_crop.py                          \\
      --input-root E:/2.causal/NACC_pre                           \\
      --mni-ref D:/MNI152/MNI152_T1_1mm_Brain.nii.gz              \\
      --crop-mode mask_bbox --bbox-padding 10                     \\
      --dry-run --limit 5
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output


# ============================================================
# DEFAULT CONFIG
# ============================================================
INPUT_ROOT = r"D:\ADNI_dataset\Gmamba\NACC\NACC_pre"
MNI152_REF = r"D:\MNI152\MNI152_T1_1mm_Brain.nii.gz"

# Default I/O filenames (override via CLI)
INPUT_IMAGE = "MRI_N4_brain_mni152.nii.gz"
INPUT_MASK = "MRI_brain_mask_mni152.nii.gz"
OUTPUT_IMAGE = "MRI_N4_brain_mni152_cropped.nii.gz"
OUTPUT_MASK = "MRI_brain_mask_mni152_cropped.nii.gz"

# Target spacing
TARGET_SPACING = (1.0, 1.0, 1.0)

# Default MNI152 1mm shape: (182, 218, 182)
#   i: 16..166 = 150   j: 18..204 = 186   k: 12..170 = 158
DEFAULT_CROP_VOXEL = (16, 166, 18, 204, 12, 170)
# ============================================================

# ─────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────

def world_to_voxel(xyz_mm, affine):
    """Convert world-coordinate (mm) to voxel index.
    xyz_mm: (x, y, z) in mm world space
    affine: 4x4 NIfTI affine
    Returns (i, j, k) as floats.
    """
    inv = np.linalg.inv(affine)
    return nib.affines.apply_affine(inv, np.asarray(xyz_mm, dtype=float))


def voxel_to_world(ijk, affine):
    """Convert voxel index to world-coordinate (mm).
    ijk: (i, j, k) voxel indices
    affine: 4x4 NIfTI affine
    Returns (x, y, z) in mm.
    """
    return nib.affines.apply_affine(affine, np.asarray(ijk, dtype=float))


def mm_crop_to_voxel_crop(x_min, x_max, y_min, y_max, z_min, z_max, affine):
    """Convert a crop box from MNI mm coordinates to voxel indices.

    Input:  (x_min, x_max, y_min, y_max, z_min, z_max) in mm
    Output: (i_min, i_max, j_min, j_max, k_min, k_max) rounded to int voxel indices
            sorted such that min < max regardless of input order.
    """
    # Convert corners (each corner → voxel)
    corners_mm = np.array([
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
    ])
    corners_vox = np.array([world_to_voxel(c, affine) for c in corners_mm])

    i_min, i_max = corners_vox[:, 0]
    j_min, j_max = corners_vox[:, 1]
    k_min, k_max = corners_vox[:, 2]

    # Ensure min < max and round
    def _ordered(a, b):
        lo, hi = sorted([a, b])
        return int(round(lo)), int(round(hi))

    i0, i1 = _ordered(i_min, i_max)
    j0, j1 = _ordered(j_min, j_max)
    k0, k1 = _ordered(k_min, k_max)

    return i0, i1, j0, j1, k0, k1


# ─────────────────────────────────────────────────────────────
# Crop functions
# ─────────────────────────────────────────────────────────────

def compute_mask_bbox(mask_data, padding_mm, affine):
    """Compute the bounding-box crop from a binary mask.

    Finds the smallest [i:j] range that contains all mask voxels > 0.5
    and expands by padding_mm (converted to voxels).

    Returns (i_min, i_max, j_min, j_max, k_min, k_max) or None if mask is empty.
    """
    indices = np.argwhere(mask_data > 0.5)
    if len(indices) == 0:
        return None

    i_min, i_max = indices[:, 0].min(), indices[:, 0].max()
    j_min, j_max = indices[:, 1].min(), indices[:, 1].max()
    k_min, k_max = indices[:, 2].min(), indices[:, 2].max()

    if padding_mm > 0:
        # Convert padding from mm to approximate voxel units along each axis
        spacings = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        pad_vox = np.ceil(padding_mm / spacings).astype(int)

        shape = mask_data.shape
        i_min = max(0, i_min - pad_vox[0])
        i_max = min(shape[0], i_max + pad_vox[0] + 1)
        j_min = max(0, j_min - pad_vox[1])
        j_max = min(shape[1], j_max + pad_vox[1] + 1)
        k_min = max(0, k_min - pad_vox[2])
        k_max = min(shape[2], k_max + pad_vox[2] + 1)
    else:
        i_max += 1
        j_max += 1
        k_max += 1

    return int(i_min), int(i_max), int(j_min), int(j_max), int(k_min), int(k_max)


def compute_mask_com_center_crop(mask_data, target_shape, strict_center, affine):
    """Compute a fixed-size crop box centered on the brain mask's
    center of mass (COM).

    Steps:
      1. Compute mask COM via weighted average of non-zero voxel coordinates
      2. Center the target_shape box at COM
      3. If out of bounds: shift to fit (default) or error (strict_center=True)

    Parameters
    ----------
    mask_data : ndarray (3D binary)
    target_shape : tuple (si, sj, sk)  -- desired output shape in voxels
    strict_center : bool
        If True, raise ValueError when the centered box goes out of bounds.
        If False, shift the box just enough to stay in bounds.
    affine : 4x4 array -- only used for mm logging in result dict

    Returns
    -------
    (i_min, i_max, j_min, j_max, k_min, k_max) -- voxel crop box

    Raises ValueError if mask is empty or box cannot fit.
    """
    si, sj, sk = target_shape
    vol_shape = mask_data.shape

    # 1. Compute center of mass
    indices = np.argwhere(mask_data > 0.5)
    if len(indices) == 0:
        raise ValueError("mask is empty -- cannot compute COM")

    com_i = indices[:, 0].mean()
    com_j = indices[:, 1].mean()
    com_k = indices[:, 2].mean()

    # 2. Center the box
    half_i = si // 2
    half_j = sj // 2
    half_k = sk // 2

    i_min = int(round(com_i)) - half_i
    j_min = int(round(com_j)) - half_j
    k_min = int(round(com_k)) - half_k

    i_max = i_min + si
    j_max = j_min + sj
    k_max = k_min + sk

    # 3. Handle out-of-bounds
    if (i_min < 0 or j_min < 0 or k_min < 0 or
            i_max > vol_shape[0] or j_max > vol_shape[1] or k_max > vol_shape[2]):
        if strict_center:
            raise ValueError(
                "COM-centered box (size %s) goes out of bounds: "
                "box=[%d:%d, %d:%d, %d:%d] vol=%s. "
                "Use --lenient-center to auto-shift."
                % (target_shape, i_min, i_max, j_min, j_max, k_min, k_max, vol_shape))

        # Shift to fit
        shift_i = max(0, -i_min) if i_min < 0 else min(0, vol_shape[0] - i_max)
        shift_j = max(0, -j_min) if j_min < 0 else min(0, vol_shape[1] - j_max)
        shift_k = max(0, -k_min) if k_min < 0 else min(0, vol_shape[2] - k_max)

        i_min += shift_i
        i_max += shift_i
        j_min += shift_j
        j_max += shift_j
        k_min += shift_k
        k_max += shift_k

        # Double-check after shift
        i_min = max(0, i_min)
        j_min = max(0, j_min)
        k_min = max(0, k_min)
        i_max = min(vol_shape[0], i_max)
        j_max = min(vol_shape[1], j_max)
        k_max = min(vol_shape[2], k_max)

        actual_shape = (i_max - i_min, j_max - j_min, k_max - k_min)
        if actual_shape != target_shape:
            raise ValueError(
                "cannot fit target shape %s into volume %s after shift; "
                "got %s. Mask COM may be too close to the edge."
                % (target_shape, vol_shape, actual_shape))

    return i_min, i_max, j_min, j_max, k_min, k_max


def apply_crop_strict(data, i_min, i_max, j_min, j_max, k_min, k_max):
    """Crop 3D array strictly: error if out of bounds. No silent clipping."""
    shape = data.shape
    if i_min < 0 or j_min < 0 or k_min < 0:
        return None, "negative voxel index: i=%d j=%d k=%d" % (i_min, j_min, k_min)
    if i_max > shape[0] or j_max > shape[1] or k_max > shape[2]:
        return None, "crop exceeds bounds: need max(%d,%d,%d) got shape %s" % (
            i_max, j_max, k_max, shape)
    if i_min >= i_max or j_min >= j_max or k_min >= k_max:
        return None, "empty crop range [%d:%d, %d:%d, %d:%d]" % (
            i_min, i_max, j_min, j_max, k_min, k_max)

    cropped = data[i_min:i_max, j_min:j_max, k_min:k_max].copy()
    return cropped, (i_min, i_max, j_min, j_max, k_min, k_max)


def apply_crop_lenient(data, i_min, i_max, j_min, j_max, k_min, k_max):
    """Crop 3D array leniently: clip to valid bounds, warn if out of range."""
    shape = data.shape
    i0, i1 = max(0, i_min), min(shape[0], i_max)
    j0, j1 = max(0, j_min), min(shape[1], j_max)
    k0, k1 = max(0, k_min), min(shape[2], k_max)

    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return None, "empty crop range after clipping"

    any_clipped = (i0 != i_min or i1 != i_max or
                   j0 != j_min or j1 != j_max or
                   k0 != k_min or k1 != k_max)

    cropped = data[i0:i1, j0:j1, k0:k1].copy()
    return cropped, (i0, i1, j0, j1, k0, k1), any_clipped


def update_affine_after_crop(affine, i0, j0, k0):
    """Shift affine origin to account for crop offset (i, j, k array axes)."""
    new_aff = affine.copy()
    new_aff[:3, 3] = nib.affines.apply_affine(affine, np.array([i0, j0, k0], dtype=float))
    return new_aff


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def validate_same_grid(img_a, img_b, tol=1e-3):
    """Check that two NIfTI images share the same spatial grid."""
    if img_a.shape != img_b.shape:
        return False, "shape mismatch: %s vs %s" % (img_a.shape, img_b.shape)
    sp_a = np.sqrt((img_a.affine[:3, :3] ** 2).sum(axis=0))
    sp_b = np.sqrt((img_b.affine[:3, :3] ** 2).sum(axis=0))
    if not np.allclose(sp_a, sp_b, atol=tol):
        return False, "spacing mismatch: %s vs %s" % (
            ["%.4f" % s for s in sp_a], ["%.4f" % s for s in sp_b])
    if not np.allclose(img_a.affine, img_b.affine, atol=tol):
        return False, "affine mismatch"
    return True, ""


def check_3d(data, label="volume"):
    """Raise if data is not 3D."""
    if data.ndim != 3:
        raise ValueError("%s is not 3D (shape=%s)" % (label, data.shape))


# ─────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────

def resample_image_to_ref(img, ref_img, order=3):
    """Resample image to reference grid using cubic (order=3) interpolation."""
    return resample_from_to(img, ref_img, order=order, mode="constant", cval=0.0)


def resample_mask_to_ref(mask_img, ref_img):
    """Resample mask to reference grid using nearest-neighbor (order=0)."""
    return resample_from_to(mask_img, ref_img, order=0, mode="constant", cval=0.0)


def resample_image_to_spacing(img, voxel_sizes, order=3):
    """Resample image to target voxel spacing."""
    return resample_to_output(img, voxel_sizes=voxel_sizes, order=order)


def resample_mask_to_spacing(mask_img, voxel_sizes):
    """Resample mask to target voxel spacing (nearest-neighbor)."""
    return resample_to_output(mask_img, voxel_sizes=voxel_sizes, order=0)


def needs_resample(img, target_spacing, tol_relative=0.005):
    """Check if image needs resampling to target spacing."""
    current_sp = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    return np.any(np.abs(current_sp - np.array(target_spacing)) / (current_sp + 1e-8) > tol_relative)


# ─────────────────────────────────────────────────────────────
# Per-subject processing
# ─────────────────────────────────────────────────────────────

def resolve_crop_box(args, mask_data, affine, shape, subj_name):
    """Resolve the crop box based on crop_mode.

    Returns (i_min, i_max, j_min, j_max, k_min, k_max) voxel crop box
    with i_max inclusive (used for slicing: data[i_min:i_max]).

    Raises ValueError on invalid configuration.
    """
    mode = args.crop_mode

    if mode == "fixed_voxel":
        box = args.crop_box_voxel
        return (box[0], box[1], box[2], box[3], box[4], box[5])

    elif mode == "fixed_mm":
        if args.crop_box_mm is None:
            raise ValueError("--crop-box-mm is required for crop_mode=fixed_mm")
        x_min, x_max, y_min, y_max, z_min, z_max = args.crop_box_mm
        box = mm_crop_to_voxel_crop(x_min, x_max, y_min, y_max, z_min, z_max, affine)
        return box

    elif mode == "mask_bbox":
        bbox = compute_mask_bbox(mask_data, args.bbox_padding, affine)
        if bbox is None:
            raise ValueError("mask is empty — cannot compute bounding box")
        return bbox

    elif mode == "center_fixed":
        target = tuple(args.crop_size)
        box = compute_mask_com_center_crop(
            mask_data, target, args.strict_center, affine)
        return box

    elif mode == "none":
        return 0, shape[0], 0, shape[1], 0, shape[2]

    else:
        raise ValueError("unknown crop_mode: %s" % mode)


def process_subject(subject_dir, args, ref_img):
    """Process a single subject directory. Returns result dict."""
    img_path = os.path.join(subject_dir, args.input_image)
    mask_path = os.path.join(subject_dir, args.input_mask)
    out_img = os.path.join(subject_dir, args.output_image)
    out_mask = os.path.join(subject_dir, args.output_mask)
    subj_name = os.path.basename(subject_dir)

    # --- Resume check ---
    if args.resume and os.path.exists(out_img) and os.path.exists(out_mask):
        return {"status": "skipped", "reason": "output exists"}

    # --- Check inputs ---
    if not os.path.exists(img_path):
        return {"status": "error", "reason": "missing %s" % args.input_image}
    if not os.path.exists(mask_path) and args.crop_mode != "none":
        return {"status": "error", "reason": "missing %s" % args.input_mask}

    # ============================================================
    #  1. Load image
    # ============================================================
    try:
        img = nib.load(img_path)
        img_data = np.asarray(img.get_fdata(), dtype=np.float32)
    except Exception as e:
        return {"status": "error", "reason": "load image failed: %s" % e}

    if img_data.ndim != 3:
        return {"status": "error",
                "reason": "image not 3D (shape=%s)" % str(img_data.shape)}

    original_shape = img_data.shape
    original_affine = img.affine.copy()

    # ============================================================
    #  2. Load mask
    # ============================================================
    mask_img = None
    mask_data = None
    if os.path.exists(mask_path):
        try:
            mask_img = nib.load(mask_path)
            mask_data = np.asarray(mask_img.get_fdata())
            mask_data = (mask_data > 0.5).astype(np.uint8)
        except Exception as e:
            return {"status": "error", "reason": "load mask failed: %s" % e}
        if mask_data.ndim != 3:
            return {"status": "error",
                    "reason": "mask not 3D (shape=%s)" % str(mask_data.shape)}
    elif args.crop_mode == "none":
        # Create a dummy all-ones mask for "no mask" mode
        mask_data = np.ones(img_data.shape, dtype=np.uint8)
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    else:
        return {"status": "error", "reason": "missing %s" % args.input_mask}

    # Validate same grid
    ok, err = validate_same_grid(img, mask_img)
    if not ok:
        return {"status": "error", "reason": "pre-resample grid mismatch: %s" % err}

    original_mask_voxels = int(mask_data.sum())

    # ============================================================
    #  3. Resample
    # ============================================================
    was_resampled = False
    resample_mode = args.resample_mode

    if resample_mode == "reference":
        ref_shape = ref_img.shape
        ref_affine = ref_img.affine
        same_grid = (img.shape == ref_shape and
                     np.allclose(img.affine, ref_affine, atol=1e-3))
        if not same_grid:
            img_rs = resample_image_to_ref(img, ref_img, order=3)
            mask_rs = resample_mask_to_ref(mask_img, ref_img)
            img_data = np.asarray(img_rs.get_fdata(), dtype=np.float32)
            mask_data = (np.asarray(mask_rs.get_fdata()) > 0.5).astype(np.uint8)
            img = img_rs
            mask_img = mask_rs
            was_resampled = True

            # Re-validate grid
            ok, err = validate_same_grid(img, mask_img)
            if not ok:
                return {"status": "error", "reason": "post-resample grid mismatch: %s" % err}

    elif resample_mode == "spacing":
        if needs_resample(img, args.target_spacing):
            img_rs = resample_image_to_spacing(img, args.target_spacing, order=3)
            mask_rs = resample_mask_to_spacing(mask_img, args.target_spacing)
            img_data = np.asarray(img_rs.get_fdata(), dtype=np.float32)
            mask_data = (np.asarray(mask_rs.get_fdata()) > 0.5).astype(np.uint8)
            img = img_rs
            mask_img = mask_rs
            was_resampled = True

    elif resample_mode != "none":
        return {"status": "error", "reason": "unknown resample_mode: %s" % resample_mode}

    resampled_shape = img_data.shape
    resampled_mask_voxels = int(mask_data.sum())

    # ============================================================
    #  4. Resolve crop box
    # ============================================================
    try:
        crop_ijk = resolve_crop_box(args, mask_data, img.affine,
                                    img_data.shape, subj_name)
    except ValueError as e:
        return {"status": "error", "reason": "resolve crop: %s" % e}

    crop_imin, crop_imax, crop_jmin, crop_jmax, crop_kmin, crop_kmax = crop_ijk

    # ============================================================
    #  5. Apply crop
    # ============================================================
    if args.strict_crop:
        img_cropped, crop_result = apply_crop_strict(
            img_data, crop_imin, crop_imax, crop_jmin, crop_jmax, crop_kmin, crop_kmax)
        clipped = False
    else:
        img_cropped, crop_result = apply_crop_lenient(
            img_data, crop_imin, crop_imax, crop_jmin, crop_jmax, crop_kmin, crop_kmax)
        if isinstance(img_cropped, tuple):
            # lenient returns (data, (i0,i1,j0,j1,k0,k1), clipped)
            img_cropped, crop_result, clipped = img_cropped
        else:
            clipped = False

    if img_cropped is None:
        return {"status": "error", "reason": "crop failed: %s" % crop_result}

    i0, i1, j0, j1, k0, k1 = crop_result

    # Crop mask with same box
    mask_cropped = mask_data[i0:i1, j0:j1, k0:k1].copy()

    # Validate shapes match
    if img_cropped.shape != mask_cropped.shape:
        return {"status": "error",
                "reason": "cropped img/mask shape mismatch: %s vs %s" % (
                    img_cropped.shape, mask_cropped.shape)}

    cropped_affine = update_affine_after_crop(img.affine, i0, j0, k0)

    # ============================================================
    #  6. Zero out non-brain voxels
    # ============================================================
    img_cropped[mask_cropped == 0] = 0.0
    cropped_mask_voxels = int(mask_cropped.sum())

    # ============================================================
    #  7. Save outputs
    # ============================================================
    final_shape = img_cropped.shape

    if not args.dry_run:
        # Image: preserve header metadata
        img_header = img.header.copy()
        img_header.set_data_shape(final_shape)
        img_header.set_zooms(args.target_spacing)
        nib.save(nib.Nifti1Image(img_cropped, cropped_affine, img_header), out_img)

        # Mask: uint8 binary
        mask_header = mask_img.header.copy() if mask_img is not None else img.header.copy()
        mask_header.set_data_dtype(np.uint8)
        mask_header.set_data_shape(final_shape)
        mask_header.set_zooms(args.target_spacing)
        nib.save(nib.Nifti1Image(mask_cropped.astype(np.uint8),
                                 cropped_affine, mask_header), out_mask)

    # Compute MNI mm extent for logging
    mnix, mniy, mniz = voxel_to_world([i0, j0, k0], img.affine)
    mnix2, mniy2, mniz2 = voxel_to_world([i1 - 1, j1 - 1, k1 - 1], img.affine)

    return {
        "status": "ok",
        "original_shape": list(original_shape),
        "resampled_shape": list(resampled_shape),
        "final_shape": list(final_shape),
        "was_resampled": was_resampled,
        "brain_voxels_original": original_mask_voxels,
        "brain_voxels_resampled": resampled_mask_voxels,
        "brain_voxels_cropped": cropped_mask_voxels,
        "crop_ijk": [int(i0), int(i1), int(j0), int(j1), int(k0), int(k1)],
        "crop_mm": [float(round(v, 1)) for v in (
            mnix, mnix2, mniy, mniy2, mniz, mniz2)],
        "crop_clipped": clipped if not args.strict_crop else False,
        "crop_mode": args.crop_mode,
        "resample_mode": args.resample_mode,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Flexible MRI Resample & Crop with custom crop dimension support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Crop mode examples:
  fixed_voxel  --crop-box-voxel 16 166 18 204 12 170
  fixed_mm     --crop-box-mm -90 90 -125 90 -70 105
  mask_bbox    --bbox-padding 10   (auto-crop to mask + 10mm padding)
  center_fixed --crop-size 160 196 160  (CNN-ready, centered on mask COM)
  none                             (skip cropping)
        """)

    # ── Paths ──
    p.add_argument("--input-root", default=INPUT_ROOT,
                   help="Root directory containing subject folders")
    p.add_argument("--mni-ref", default=MNI152_REF,
                   help="MNI152 1mm reference image (used for resample_mode=reference)")

    # ── I/O filenames ──
    p.add_argument("--input-image", default=INPUT_IMAGE,
                   help="Input image filename (within each subject dir)")
    p.add_argument("--input-mask", default=INPUT_MASK,
                   help="Input mask filename (within each subject dir)")
    p.add_argument("--output-image", default=OUTPUT_IMAGE,
                   help="Output cropped image filename")
    p.add_argument("--output-mask", default=OUTPUT_MASK,
                   help="Output cropped mask filename")

    # ── Crop mode ──
    p.add_argument("--crop-mode",
                   choices=["fixed_voxel", "fixed_mm", "mask_bbox", "center_fixed", "none"],
                   default="center_fixed",
                   help="Crop strategy (default: fixed_voxel)")

    g_crop = p.add_argument_group("Crop box arguments")
    g_crop.add_argument("--crop-box-voxel", type=int, nargs=6,
                        default=list(DEFAULT_CROP_VOXEL),
                        metavar=("IMIN", "IMAX", "JMIN", "JMAX", "KMIN", "KMAX"),
                        help="Fixed voxel crop box [crop_mode=fixed_voxel]")
    g_crop.add_argument("--crop-box-mm", type=float, nargs=6,
                        default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Fixed MNI mm crop box [crop_mode=fixed_mm], "
                             "e.g. -90 90 -125 90 -70 105")
    g_crop.add_argument("--bbox-padding", type=float, default=10.0,
                        help="Padding in mm for mask_bbox mode (default: 10)")
    g_crop.add_argument("--strict_crop", action="store_true", default=True,
                        help="Fail on out-of-bounds crop (default: True)")
    g_crop.add_argument("--lenient_crop", action="store_true",
                        help="Clip crop box silently to image bounds")
    g_crop.add_argument("--crop-size", type=int, nargs=3,
                        default=[160, 196, 160],
                        metavar=("SI", "SJ", "SK"),
                        help="Target crop size in voxels [crop_mode=center_fixed] "
                             "(default: 160 196 160)")
    g_crop.add_argument("--strict_center", action="store_true", default=False,
                        help="Error if COM-centered box goes out of bounds "
                             "[crop_mode=center_fixed]")
    g_crop.add_argument("--lenient_center",default=True, action="store_true",
                        help="Auto-shift box to stay in bounds "
                             "[crop_mode=center_fixed, default],default=False ")

    # ── Resample mode ──
    p.add_argument("--resample-mode",
                   choices=["reference", "spacing", "none"],
                   default="reference",
                   help="Resampling strategy (default: reference)")
    p.add_argument("--target-spacing", type=float, nargs=3,
                   default=list(TARGET_SPACING),
                   metavar=("X", "Y", "Z"),
                   help="Target voxel spacing in mm [resample_mode=spacing]")

    # ── Subject selection ──
    p.add_argument("--single", default=None, metavar="SUBJECT_ID",
                   help="Process a single subject (by folder name)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of subjects (for testing)")
    p.add_argument("--subject_pattern", default=None,
                   help="Regex pattern to filter subject folders")

    # ── Execution ──
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip subjects with existing outputs (default: True)")
    p.add_argument("--no_resume", action="store_true",
                   help="Overwrite existing outputs")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview without writing files")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed per-subject info")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.no_resume:
        args.resume = False
    if args.lenient_crop:
        args.strict_crop = False

    # ── Validate crop mode args ──
    if args.crop_mode == "fixed_mm" and args.crop_box_mm is None:
        print("[FATAL] --crop-box-mm is required when crop_mode=fixed_mm")
        sys.exit(1)
    if args.crop_mode == "center_fixed":
        if args.lenient_center:
            args.strict_center = False
        else:
            args.strict_center = True  # default: strict
        args.strict_crop = True  # center_fixed always strict about shape
        args.crop_size = tuple(args.crop_size)
        if len(args.crop_size) != 3:
            print("[FATAL] --crop-size must be exactly 3 integers")
            sys.exit(1)

    # ── Load MNI152 reference ──
    ref_img = None
    if args.resample_mode == "reference":
        if not os.path.exists(args.mni_ref):
            print("[FATAL] MNI152 reference not found: %s" % args.mni_ref)
            sys.exit(1)
        ref_img = nib.load(args.mni_ref)
        if ref_img.get_fdata().ndim != 3:
            print("[FATAL] MNI152 reference is not 3D (shape=%s)" %
                  str(ref_img.shape))
            sys.exit(1)
        print("[INFO] MNI152 reference: %s" % args.mni_ref)
        print("[INFO]   shape : %s" % str(ref_img.shape))
        print("[INFO]   zooms : %s" % str(ref_img.header.get_zooms()[:3]))
    elif args.resample_mode == "spacing":
        print("[INFO] Resample to target spacing: %s mm" % str(args.target_spacing))

    # ── Gather subjects ──
    if args.single:
        subjects = [args.single]
    else:
        import re
        subjects = sorted([
            d for d in os.listdir(args.input_root)
            if os.path.isdir(os.path.join(args.input_root, d))
        ])
        if args.subject_pattern:
            pat = re.compile(args.subject_pattern)
            subjects = [s for s in subjects if pat.search(s)]

    total_available = len(subjects)
    if args.limit:
        subjects = subjects[:args.limit]

    # ── Summary banner ──
    print("[INFO] Input root       : %s" % args.input_root)
    print("[INFO] Crop mode        : %s" % args.crop_mode)
    if args.crop_mode == "fixed_voxel":
        cb = args.crop_box_voxel
        print("[INFO]   voxel box     : i[%d:%d] j[%d:%d] k[%d:%d]" % tuple(cb))
    elif args.crop_mode == "fixed_mm":
        cb = args.crop_box_mm
        print("[INFO]   MNI mm box    : x[%.0f:%.0f] y[%.0f:%.0f] z[%.0f:%.0f]" % tuple(cb))
    elif args.crop_mode == "mask_bbox":
        print("[INFO]   bbox padding  : %.1f mm" % args.bbox_padding)
    elif args.crop_mode == "center_fixed":
        print("[INFO]   target shape  : %s (centered on mask COM)" % str(args.crop_size))
        print("[INFO]   center mode   : %s" % ("strict" if args.strict_center else "lenient (auto-shift)"))
    elif args.crop_mode == "none":
        print("[INFO]   (no cropping)")
    print("[INFO] Resample mode    : %s" % args.resample_mode)
    print("[INFO] Crop validation  : %s" % ("strict" if args.strict_crop else "lenient"))
    print("[INFO] Input image      : %s" % args.input_image)
    print("[INFO] Input mask       : %s" % args.input_mask)
    print("[INFO] Output image     : %s" % args.output_image)
    print("[INFO] Output mask      : %s" % args.output_mask)
    print("[INFO] Found %d subjects (processing %d)" % (total_available, len(subjects)))
    if args.dry_run:
        print("[INFO] *** DRY RUN -- no files will be written ***")
    print()

    # ── Process ──
    stats = {"ok": 0, "skipped": 0, "error": 0, "total": len(subjects)}
    errors_log = []
    crop_shapes = {}
    t0 = time.time()

    for idx, subj in enumerate(subjects):
        subj_dir = os.path.join(args.input_root, subj)
        if not os.path.isdir(subj_dir):
            print("[%5d/%d] SKIP  %s (not a directory)" % (idx + 1, len(subjects), subj))
            stats["skipped"] += 1
            continue

        r = process_subject(subj_dir, args, ref_img)

        if r["status"] == "ok":
            stats["ok"] += 1
            # Track shape distribution
            shape_key = tuple(r["final_shape"])
            crop_shapes[shape_key] = crop_shapes.get(shape_key, 0) + 1

            line = "[%5d/%d] OK    %s  %s -> %s  vox=%d -> %d -> %d" % (
                idx + 1, len(subjects), subj,
                r["original_shape"], r["final_shape"],
                r["brain_voxels_original"],
                r["brain_voxels_resampled"],
                r["brain_voxels_cropped"])
            if r.get("was_resampled"):
                line += " [RESAMPLED]"
            if r.get("crop_clipped"):
                line += " [CLIPPED-CROP]"
            if r["crop_mode"] == "mask_bbox":
                line += " mm=%s" % r.get("crop_mm", "?")
            if args.verbose:
                line += " crop_ijk=%s" % r.get("crop_ijk", "?")
            print(line)

        elif r["status"] == "skipped":
            stats["skipped"] += 1
            print("[%5d/%d] SKIP  %s  (%s)" % (
                idx + 1, len(subjects), subj, r["reason"]))

        else:
            stats["error"] += 1
            errors_log.append({"subject": subj, "reason": r.get("reason", "unknown")})
            print("[%5d/%d] ERR   %s  (%s)" % (
                idx + 1, len(subjects), subj, r.get("reason", "?")))

        sys.stdout.flush()

    elapsed = time.time() - t0

    # ── Final summary ──
    print()
    print("=" * 70)
    print("[DONE] Resample & Crop in %.1fs" % elapsed)
    print("  Total          : %d" % stats["total"])
    print("  OK             : %d" % stats["ok"])
    print("  Skipped        : %d" % stats["skipped"])
    print("  Errors         : %d" % stats["error"])
    print("  Crop mode      : %s" % args.crop_mode)
    print("  Resample mode  : %s" % args.resample_mode)
    if crop_shapes:
        print("  Output shapes  :")
        for shp, cnt in sorted(crop_shapes.items(), key=lambda x: -x[1]):
            print("    %s  (%d subjects, %.1f%%)" % (
                str(shp), cnt, 100.0 * cnt / max(stats["ok"], 1)))
    print("=" * 70)

    # ── Error log ──
    if errors_log:
        log_path = os.path.join(args.input_root, "resample_crop_errors.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(errors_log, f, indent=2, ensure_ascii=False)
        print("[INFO] Error log saved: %s" % log_path)

    # ── Crop summary CSV (useful for mask_bbox mode) ──
    if args.crop_mode == "mask_bbox" and stats["ok"] > 0:
        crop_csv_path = os.path.join(args.input_root, "resample_crop_bbox_summary.json")
        # Already logged via error system; OK subjects have crop info inline


if __name__ == "__main__":
    main()
