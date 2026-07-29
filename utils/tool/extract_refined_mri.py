#!/usr/bin/env python3
"""
ADNI MRI Refined Data Extractor
================================
从 ADNI_pre 多步预处理目录中提取最终预处理结果（MRI_brain.nii.gz），
保留目录结构，生成精简数据集，便于存储与运输。

可选地，在成功提取后清理源目录中的中间处理文件，节省磁盘空间。

Usage:
    python extract_refined_mri.py --dry-run
    python extract_refined_mri.py --dry-run --cleanup
    python extract_refined_mri.py --cleanup
    python extract_refined_mri.py --target E:/backup/ADNI_refined --cleanup
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Tuple


# ── Configuration ────────────────────────────────────────────
DEFAULT_SOURCE = r"D:\ADNI_dataset\Causal\causal_dataset1\ADNI_base"
DEFAULT_TARGET = r"D:\ADNI_dataset\Causal\causal_dataset1\ADNI_dataset"
DEFAULT_PATTERN = "MRI_brain_mni152_cropped_norm.nii.gz"


def setup_logging(log_dir: Path, source: str, target: str, pattern: str, dry_run: bool) -> Tuple[logging.Logger, Path]:
    """Initialize file + console logger, return (logger, log_path)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"extract_refined_{timestamp}.log"

    logger = logging.getLogger("extract_refined")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # Console handler (INFO only)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    logger.info("=== Extraction started ===")
    logger.info(f"Source:  {source}")
    logger.info(f"Target:  {target}")
    logger.info(f"Pattern: {pattern}")
    logger.info(f"DryRun:  {dry_run}")

    return logger, log_path


def md5_hex(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def should_copy(src: Path, dst: Path) -> bool:
    """Return True if src should be copied to dst (dst missing or content differs)."""
    if not src.exists():
        return False
    if not dst.exists():
        return True
    return md5_hex(src) != md5_hex(dst)


def format_size(size_bytes: int) -> str:
    """Human-readable size string."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"


def find_intermediate_files(subj_dir: Path, clean_patterns: list) -> list:
    """
    Identify intermediate files in a subject directory.

    Only files matching any pattern in clean_patterns are considered
    intermediate and returned for deletion (targeted denylist approach).
    Files not matching any pattern are left untouched.

    Returns list of Path objects to be deleted.
    """
    intermediates = []
    for f in subj_dir.iterdir():
        if f.is_file():
            if any(f.match(p) for p in clean_patterns):
                intermediates.append(f)
    return sorted(intermediates)


def print_banner():
    """Print header banner."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  ADNI MRI Refined Data Extractor                 ║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def print_progress(current: int, total: int, label: str):
    """Print a single-line progress bar."""
    pct = round(current / total * 100, 1) if total > 0 else 0.0
    bar_len = int(pct // 5)
    bar = "[" + "#" * bar_len + " " * (20 - bar_len) + "]"
    msg = f"\r{bar} {pct}% ({current}/{total}) - {label}"
    # Pad to clear previous line
    sys.stdout.write(msg.ljust(100))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Extract the most refined ADNI MRI preprocessing results."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"Source directory (default: {DEFAULT_SOURCE})")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"Target directory (default: {DEFAULT_TARGET})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"Filename pattern to extract (default: {DEFAULT_PATTERN})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without copying files")
    parser.add_argument("--cleanup", default=False, action="store_true",
                        help="Delete intermediate files from source after successful extraction")
    parser.add_argument("--clean-patterns", nargs="+",
                        default=["MRI_brain_mask_mni152.nii.gz"],
                        help="Filename patterns to delete during cleanup (default: MRI_brain_mask.nii.gz MRI_brain_mask_mni152.nii.gz)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    pattern = args.pattern
    dry_run = args.dry_run
    cleanup = args.cleanup
    clean_patterns = args.clean_patterns

    print_banner()

    # ── Pre-flight checks ────────────────────────────────────
    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = target_dir.parent
    logger, log_path = setup_logging(log_dir, str(source_dir), str(target_dir),
                                     pattern, dry_run)
    if cleanup:
        logger.info(f"Cleanup: enabled (patterns: {', '.join(clean_patterns)})")

    # ── Discover subject directories ─────────────────────────
    subject_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    total = len(subject_dirs)
    logger.info(f"Found {total} subject directories")

    if total == 0:
        logger.warning(f"No subdirectories found in {source_dir}")
        sys.exit(0)

    # ── Statistics ───────────────────────────────────────────
    stats = {"total": total, "copied": 0, "skipped": 0, "missing": 0, "errors": 0,
             "cleaned": 0, "clean_errors": 0}
    missing_list = []
    error_list = []
    clean_list = []
    clean_error_list = []
    size_total = 0
    size_cleaned = 0

    # ── Main loop ────────────────────────────────────────────
    t_start = time.time()
    for i, subj_dir in enumerate(subject_dirs, 1):
        src_file = subj_dir / pattern
        dst_dir = target_dir / subj_dir.name
        dst_file = dst_dir / pattern

        print_progress(i, total, subj_dir.name)

        if not src_file.exists():
            stats["missing"] += 1
            missing_list.append(subj_dir.name)
            continue

        file_size = src_file.stat().st_size

        if not should_copy(src_file, dst_file):
            stats["skipped"] += 1
            # Cleanup intermediate files even for previously-copied subjects
            if cleanup:
                intermediates = find_intermediate_files(subj_dir, clean_patterns)
                for ifile in intermediates:
                    try:
                        if not dry_run:
                            ifile.unlink()
                        stats["cleaned"] += 1
                        size_cleaned += ifile.stat().st_size
                        clean_list.append(f"{subj_dir.name}/{ifile.name}")
                        logger.debug(f"Deleted intermediate: {subj_dir.name}/{ifile.name}")
                    except Exception as e:
                        stats["clean_errors"] += 1
                        clean_error_list.append(f"{subj_dir.name}/{ifile.name}: {e}")
            continue

        if dry_run:
            stats["copied"] += 1
            size_total += file_size
            # Simulate cleanup in dry-run mode
            if cleanup:
                intermediates = find_intermediate_files(subj_dir, clean_patterns)
                for ifile in intermediates:
                    stats["cleaned"] += 1
                    size_cleaned += ifile.stat().st_size
                    clean_list.append(f"{subj_dir.name}/{ifile.name}")
            continue

        # Actual copy
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Read + write (cross-platform safe)
            with open(src_file, "rb") as f_src, open(dst_file, "wb") as f_dst:
                while True:
                    chunk = f_src.read(1024 * 1024)  # 1MB buffer
                    if not chunk:
                        break
                    f_dst.write(chunk)
            stats["copied"] += 1
            size_total += file_size

            # ── Cleanup intermediate files after successful copy ──
            if cleanup:
                intermediates = find_intermediate_files(subj_dir, clean_patterns)
                for ifile in intermediates:
                    try:
                        if not dry_run:
                            ifile.unlink()
                        stats["cleaned"] += 1
                        size_cleaned += ifile.stat().st_size if ifile.exists() else 0
                        clean_list.append(f"{subj_dir.name}/{ifile.name}")
                        logger.debug(f"Deleted intermediate: {subj_dir.name}/{ifile.name}")
                    except Exception as e:
                        stats["clean_errors"] += 1
                        clean_error_list.append(f"{subj_dir.name}/{ifile.name}: {e}")

        except Exception as e:
            stats["errors"] += 1
            error_list.append(f"{subj_dir.name}: {e}")

    # Clear progress line
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()

    elapsed = time.time() - t_start

    # ── Final report ─────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  Extraction Complete                             ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Total directories scanned : {stats['total']}")
    print(f"  Files copied              : {stats['copied']}")
    print(f"  Skipped (unchanged)       : {stats['skipped']}")
    print(f"  Missing (no target file)  : {stats['missing']}")
    print(f"  Errors                    : {stats['errors']}")
    if stats["copied"] > 0:
        print(f"  Total data (if copied)    : {format_size(size_total)}")
    if cleanup:
        print(f"  Intermediate files deleted: {stats['cleaned']}")
        if stats['cleaned'] > 0:
            print(f"  Space freed               : {format_size(size_cleaned)}")
        print(f"  Cleanup errors            : {stats['clean_errors']}")
    print(f"  Elapsed time              : {elapsed:.1f}s")
    print()
    print(f"  Log file: {log_path}")

    # ── Detail lists ─────────────────────────────────────────
    if missing_list:
        print()
        print(f"--- Missing files (no {pattern} found) ---")
        for name in missing_list:
            print(f"  [!] {name}")

    if error_list:
        print()
        print("--- Copy Errors ---")
        for msg in error_list:
            print(f"  [X] {msg}")

    if cleanup and clean_list:
        print()
        print(f"--- {'Would Delete' if dry_run else 'Deleted'} Intermediate Files ({len(clean_list)}) ---")
        for msg in clean_list[:50]:  # show first 50 only
            print(f"  [D] {msg}")
        if len(clean_list) > 50:
            print(f"  ... and {len(clean_list) - 50} more (see log)")

    if clean_error_list:
        print()
        print(f"--- Cleanup Errors ({len(clean_error_list)}) ---")
        for msg in clean_error_list:
            print(f"  [X] {msg}")

    # ── Log summary ──────────────────────────────────────────
    logger.info("=== Extraction finished ===")
    logger.info(f"Total: {stats['total']} | Copied: {stats['copied']} | "
                f"Skipped: {stats['skipped']} | Missing: {stats['missing']} | "
                f"Errors: {stats['errors']} | "
                f"Cleaned: {stats['cleaned']} | CleanErrors: {stats['clean_errors']} | "
                f"Time: {elapsed:.1f}s")

    if dry_run:
        print()
        print(">>> DRY RUN MODE - No files were actually copied or deleted. <<<")
        print(">>> Remove --dry-run flag to perform actual extraction. <<<")

    print()


if __name__ == "__main__":
    main()
