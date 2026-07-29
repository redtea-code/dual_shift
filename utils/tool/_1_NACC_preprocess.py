"""
NACC MRI Preprocessing Pipeline
================================
1. Walk SCAN directory, discover all DICOM scan directories (I*)
2. Group same-subject + same-day scans, keep the largest
3. Match each scan to the closest CSV visit (PTID + VISDATE) within 90 days
4. Convert matched scans to NIfTI via dcm2niix -> {NACCID}-{date}-{label}/MRI.nii.gz
5. Move unmatched scans to NACC_pre_nomatch
6. Resume support: skip if target MRI.nii.gz already exists

Usage: D:\Anaconda\python.exe _1_NACC_preprocess.py
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ========================== CONFIGURATION ==========================
SOURCE_DIR = r"E:\2.causal\causal_dataset2-2_dataset\SCAN"
CSV_PATH = r"E:\2.causal\NACC.csv"
OUTPUT_DIR = r"E:\2.causal\NACC_pre"
NOMATCH_DIR = r"E:\2.causal\NACC_pre_nomatch"
DCM2NIIX = r"D:\dcm2niix.exe"
MATCH_THRESHOLD_DAYS = 90
DCM2NIIX_TIMEOUT = 300  # seconds per conversion

# ==================================================================


def parse_date_from_dirname(dirname):
    """Parse '2024-04-15_11_54_53.0' -> (datetime, '2024-04-15', '2024_04_15')"""
    try:
        date_str = dirname.split("_")[0]
        date_output = date_str.replace("-", "_")
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt, date_str, date_output
    except (ValueError, IndexError):
        return None, None, None


def get_dir_size(path):
    """Calculate total file size in a directory (non-recursive)."""
    total = 0
    try:
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    except OSError:
        pass
    return total


def move_to_nomatch(src_path, naccid, date_output, serial):
    """Move a DICOM directory to the nomatch area, preserving structure."""
    nomatch_subdir = os.path.join(NOMATCH_DIR, f"{naccid}-{date_output}")
    os.makedirs(nomatch_subdir, exist_ok=True)
    dst = os.path.join(nomatch_subdir, serial)
    if os.path.exists(src_path):
        shutil.move(src_path, dst)
    return nomatch_subdir


def main():
    # ---- Step 1: Read CSV ----
    print("=" * 60)
    print("[1/4] Reading CSV: {}".format(CSV_PATH))
    df = pd.read_csv(CSV_PATH)

    # Build lookup: PTID -> list of (visdate_datetime, label_int)
    visit_lookup = defaultdict(list)
    parse_errors = 0
    for _, row in df.iterrows():
        ptid = str(row["PTID"]).strip()
        try:
            visdate = datetime.strptime(str(row["VISDATE"]).strip(), "%Y-%m-%d")
        except ValueError:
            parse_errors += 1
            continue
        label = int(float(row["label"]))
        visit_lookup[ptid].append((visdate, label))

    print("  {} unique PTIDs, {} total visits ({} parse errors)".format(
        len(visit_lookup), len(df), parse_errors))

    # ---- Step 2: Discover scans ----
    print("[2/4] Discovering DICOM scans...")

    raw_scans = []
    walk_errors = 0

    for dirpath, dirnames, filenames in os.walk(SOURCE_DIR):
        dir_basename = os.path.basename(dirpath)
        # Only process I* serial directories
        if not dir_basename.startswith("I") or not dir_basename[1:].isdigit():
            continue

        parts = os.path.normpath(dirpath).split(os.sep)
        if len(parts) < 5:
            continue

        serial = parts[-1]
        date_time_dir = parts[-2]
        sequence = parts[-3]
        naccid = parts[-4]

        if "MPRAGE" not in sequence:
            continue

        scan_dt, date_str, date_output = parse_date_from_dirname(date_time_dir)
        if scan_dt is None:
            print("  WARNING: Cannot parse date from '{}', skipping".format(date_time_dir))
            walk_errors += 1
            continue

        total_size = get_dir_size(dirpath)

        raw_scans.append({
            "naccid": naccid,
            "scan_date": scan_dt,
            "date_str": date_str,
            "date_output": date_output,
            "full_path": dirpath,
            "total_size": total_size,
            "serial": serial,
        })

    print("  Found {} raw scan directories ({} walk errors)".format(
        len(raw_scans), walk_errors))

    # ---- Step 2b: Dedup same-day scans ----
    raw_scans.sort(key=lambda s: (s["naccid"], s["scan_date"], -s["total_size"]))

    scans = []
    seen_keys = set()
    removed = 0
    for s in raw_scans:
        key = (s["naccid"], s["scan_date"])
        if key in seen_keys:
            removed += 1
            continue
        seen_keys.add(key)
        scans.append(s)

    if removed:
        print("  Removed {} same-day duplicates (kept largest)".format(removed))
    print("  After dedup: {} unique scans".format(len(scans)))

    # ---- Step 3: Match & Convert ----
    print("[3/4] Matching & converting...")
    print("-" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(NOMATCH_DIR, exist_ok=True)

    stats = {
        "total": len(scans),
        "matched": 0,
        "skipped": 0,
        "nomatch": 0,
        "failed": 0,
        "no_csv": 0,
    }

    total = len(scans)
    for idx, scan in enumerate(scans):
        naccid = scan["naccid"]
        scan_date = scan["scan_date"]
        date_output = scan["date_output"]
        full_path = scan["full_path"]
        serial = scan["serial"]
        total_size = scan["total_size"]

        visits = visit_lookup.get(naccid, [])
        if not visits:
            stats["no_csv"] += 1
            move_to_nomatch(full_path, naccid, date_output, serial)
            print("  [{:4d}/{}] NO-CSV:  {} (no CSV entry)".format(idx + 1, total, naccid))
            continue

        # Find closest VISDATE within threshold
        best_visdate = None
        best_label = None
        best_diff = float('inf')

        for visdate, label in visits:
            diff = abs((scan_date - visdate).days)
            if diff < best_diff:
                best_diff = diff
                best_visdate = visdate
                best_label = label
            elif diff == best_diff and best_visdate is not None:
                # Tie: pick earlier VISDATE
                if visdate < best_visdate:
                    best_visdate = visdate
                    best_label = label

        if best_visdate is None or best_diff > MATCH_THRESHOLD_DAYS:
            # No match within threshold -> nomatch
            stats["nomatch"] += 1
            dest = move_to_nomatch(full_path, naccid, date_output, serial)
            print("  [{:4d}/{}] NOMATCH: {}-{} -> {} (closest: {}d)".format(
                idx + 1, total, naccid, date_output, dest,
                best_diff if best_visdate else "N/A"))
            continue

        # Matched — prepare output path
        output_subdir = os.path.join(OUTPUT_DIR, "{}-{}-{}".format(
            naccid, date_output, best_label))
        output_file = os.path.join(output_subdir, "MRI.nii.gz")

        # Checkpoint: skip if already exists and non-empty
        if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
            stats["skipped"] += 1
            print("  [{:4d}/{}] SKIP:    {}-{}-{} (exists, {} MB)".format(
                idx + 1, total, naccid, date_output, best_label,
                round(total_size / 1048576, 1)))
            continue

        # Convert
        os.makedirs(output_subdir, exist_ok=True)
        prefix = "  [{:4d}/{}] CONVERT: {}-{}-{}".format(
            idx + 1, total, naccid, date_output, best_label)
        print("{} ({}d, {} MB)...".format(
            prefix, best_diff, round(total_size / 1048576, 1)), end="", flush=True)

        cmd = [DCM2NIIX, "-f", "MRI", "-z", "y", "-o", output_subdir, full_path]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=DCM2NIIX_TIMEOUT)
            if result.returncode != 0:
                err = result.stderr.strip()[:150] if result.stderr else "unknown error"
                print(" FAIL (dcm2niix: {})".format(err))
                stats["failed"] += 1
                continue
            if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
                out_size = os.path.getsize(output_file)
                print(" OK ({} MB)".format(round(out_size / 1048576, 1)))
                stats["matched"] += 1
            else:
                print(" FAIL (output missing)")
                stats["failed"] += 1
        except subprocess.TimeoutExpired:
            print(" FAIL (timeout >{}s)".format(DCM2NIIX_TIMEOUT))
            stats["failed"] += 1
        except Exception as e:
            print(" FAIL ({})".format(str(e)[:100]))
            stats["failed"] += 1

    # ---- Step 4: Report ----
    print("")
    print("=" * 60)
    print("          NACC MRI Preprocessing Report")
    print("=" * 60)
    print("  Source:          {}".format(SOURCE_DIR))
    print("  CSV:             {}".format(CSV_PATH))
    print("  Threshold:       {} days".format(MATCH_THRESHOLD_DAYS))
    print("-" * 60)
    print("  Total scans:     {}".format(stats["total"]))
    print("  Success matched: {}".format(stats["matched"]))
    print("  Skipped (exist): {}".format(stats["skipped"]))
    print("  No CSV entry:    {}".format(stats["no_csv"]))
    print("  No match (>{}d): {}".format(MATCH_THRESHOLD_DAYS, stats["nomatch"]))
    print("  Convert failed:  {}".format(stats["failed"]))
    print("-" * 60)
    print("  Output dir:      {}".format(OUTPUT_DIR))
    print("  Nomatch dir:     {}".format(NOMATCH_DIR))
    print("=" * 60)

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
