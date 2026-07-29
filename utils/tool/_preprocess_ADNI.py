"""
ADNI MRI Preprocessing Pipeline
================================
Step 1: Explore ADNI directory structure and extract MPR-R__GradWarp__B1_Correction__N3 DICOM sequence
Step 2: Match DICOM scan date with CSV diagnosis labels by closest VISDATE
Step 3: Convert DICOM to NIfTI (.nii.gz) via dcm2niix

Output:
  ADNI_pre/{PTID}-{YYYY_MM_DD}-{label}/MRI.nii.gz           (VISDATE diff <= 90 days)
  ADNI_pre_nomatch/{PTID}-{YYYY_MM_DD}-{label}/MRI.nii.gz   (VISDATE diff > 90 days)
"""

import os
import sys
import re
import gzip
import shutil
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = r"D:\ADNI_dataset\Causal\ADNI.csv"
INPUT_ROOT = r"D:\ADNI_dataset\Causal\causal_dataset1\ADNI"
OUTPUT_NORMAL = r"D:\ADNI_dataset\Causal\causal_dataset1\ADNI_pre"
OUTPUT_NOMATCH = r"D:\ADNI_dataset\Causal\causal_dataset1\ADNI_pre_nomatch"
TARGET_SEQUENCE = "MPR-R__GradWarp__B1_Correction__N3"
MAX_MATCH_DAYS = 90
# ============================================================


def build_csv_lookup(csv_path):
    """
    Read ADNI.csv and build a per-PTID lookup table.

    Returns:
        dict: {PTID: [(visdate_datetime, diagnosis_int), ...]}
    """
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    print(f"[INFO] CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    ptid_records = {}
    for _, row in df.iterrows():
        ptid = str(row['PTID']).strip()
        visdate = pd.to_datetime(row['VISDATE'])
        diagnosis = int(row['DIAGNOSIS'])
        if ptid not in ptid_records:
            ptid_records[ptid] = []
        ptid_records[ptid].append({
            'visdate': visdate,
            'diagnosis': diagnosis
        })

    print(f"[INFO] Unique PTIDs in CSV: {len(ptid_records)}")
    return ptid_records


def find_closest_record(scan_date, records):
    """
    Given a scan date and list of (visdate, diagnosis) records for a PTID,
    return (day_diff, diagnosis) of the closest match.

    Returns:
        tuple: (min_diff_days, diagnosis)
    """
    min_diff = None
    closest_diagnosis = None
    for rec in records:
        diff = abs((rec['visdate'] - scan_date).days)
        if min_diff is None or diff < min_diff:
            min_diff = diff
            closest_diagnosis = rec['diagnosis']
    return min_diff, closest_diagnosis


def resolve_output_folder(out_base, base_name):
    """
    Resolve output folder name, handling collisions properly:
    - No folder exists                     -> use base_name
    - Folder exists WITH  MRI.nii.gz       -> skip (already done)  [returns None]
    - Folder exists WITHOUT MRI.nii.gz     -> reuse it (clean leftover)
    - Name collision with another scan     -> _2, _3 suffix
    
    Returns (folder_name, should_skip) or None if all slots are done.
    """
    def has_nii(name):
        return os.path.exists(os.path.join(out_base, name, "MRI.nii.gz"))

    if not has_nii(base_name):
        # Slot is free — may have empty folder from failed run, that's ok
        return base_name

    # base_name slot is taken — try suffixes
    for suffix in range(2, 100):  # safety cap
        candidate = f"{base_name}_{suffix}"
        if not has_nii(candidate):
            return candidate

    # All 99 suffix slots taken (extremely unlikely)
    return None


def process_ptid(ptid_folder, ptid_path, seq_path, ptid_records, stats):
    """
    Process all scans for a single PTID.

    Args:
        ptid_folder: PTID folder name (e.g. '002_S_0685')
        ptid_path: absolute path to the PTID folder
        seq_path: path to the target sequence folder
        ptid_records: CSV lookup dict
        stats: shared stats dict (mutable)
    """
    records = ptid_records[ptid_folder]

    scan_folders = sorted(os.listdir(seq_path))
    for scan_folder in scan_folders:
        scan_path = os.path.join(seq_path, scan_folder)
        if not os.path.isdir(scan_path):
            continue

        # Parse scan date from folder name: YYYY-MM-DD_HH_MM_SS.S
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})', scan_folder)
        if not date_match:
            print(f"  [SKIP] {scan_folder}: cannot parse date from folder name")
            stats['skipped_no_date'] += 1
            continue

        scan_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')

        # Find closest CSV record
        day_diff, label = find_closest_record(scan_date, records)
        date_formatted = scan_date.strftime('%Y_%m_%d')

        # Decide output root
        if day_diff > MAX_MATCH_DAYS:
            out_base = OUTPUT_NOMATCH
            stats['nomatch'] += 1
            match_tag = "NOMATCH"
        else:
            out_base = OUTPUT_NORMAL
            stats['normal'] += 1
            match_tag = "OK"

        # Resolve output folder (reuse empty leftovers, skip completed)
        base_name = f"{ptid_folder}-{date_formatted}-{label}"
        final_name = resolve_output_folder(out_base, base_name)
        if final_name is None:
            print(f"  [SKIP] all slots taken: {base_name}/MRI.nii.gz")
            stats['skipped_exists'] += 1
            continue
        out_folder = os.path.join(out_base, final_name)

        os.makedirs(out_folder, exist_ok=True)

        # Find .nii file in scan folder (nested one level deep)
        src_nii = None
        for root, dirs, files in os.walk(scan_path):
            for f in files:
                if f.endswith('.nii') and not f.endswith('.nii.gz'):
                    src_nii = os.path.join(root, f)
                    break
            if src_nii:
                break

        if src_nii is None:
            print(f"  [SKIP] {scan_folder}: no .nii file found")
            stats['errors'] += 1
            continue

        dst_gz = os.path.join(out_folder, "MRI.nii.gz")

        print(f"  [{match_tag}] {os.path.basename(src_nii)} -> {final_name}/MRI.nii.gz  "
              f"(diff={day_diff}d, label={label})")
        sys.stdout.flush()

        try:
            with open(src_nii, 'rb') as f_in:
                with gzip.open(dst_gz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            stats['processed'] += 1
        except Exception as e:
            print(f"  [ERROR] gzip failed: {e}")
            stats['errors'] += 1


def main():
    # Initialization
    os.makedirs(OUTPUT_NORMAL, exist_ok=True)
    os.makedirs(OUTPUT_NOMATCH, exist_ok=True)

    # Load CSV
    ptid_records = build_csv_lookup(CSV_PATH)

    # Stats
    stats = {
        'processed': 0,
        'normal': 0,
        'nomatch': 0,
        'skipped_no_seq': 0,
        'skipped_no_csv': 0,
        'skipped_no_date': 0,
        'skipped_exists': 0,
        'errors': 0,
    }

    # Traverse input directory
    ptid_dirs = sorted(os.listdir(INPUT_ROOT))
    total_ptid = 0

    for ptid_folder in ptid_dirs:
        ptid_path = os.path.join(INPUT_ROOT, ptid_folder)
        if not os.path.isdir(ptid_path):
            continue

        total_ptid += 1
        seq_path = os.path.join(ptid_path, TARGET_SEQUENCE)

        if not os.path.isdir(seq_path):
            stats['skipped_no_seq'] += 1
            continue

        if ptid_folder not in ptid_records:
            print(f"[SKIP] {ptid_folder}: PTID not in CSV")
            stats['skipped_no_csv'] += 1
            continue

        process_ptid(ptid_folder, ptid_path, seq_path, ptid_records, stats)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"[DONE] Pipeline finished")
    print(f"  Total PTID folders scanned     : {total_ptid}")
    print(f"  Scans processed successfully   : {stats['processed']}")
    print(f"    Normal   (<= {MAX_MATCH_DAYS}d)        : {stats['normal']}")
    print(f"    NoMatch  (>  {MAX_MATCH_DAYS}d)        : {stats['nomatch']}")
    print(f"  Skipped (no target sequence)   : {stats['skipped_no_seq']}")
    print(f"  Skipped (PTID not in CSV)      : {stats['skipped_no_csv']}")
    print(f"  Skipped (can't parse date)     : {stats['skipped_no_date']}")
    print(f"  Skipped (already converted)    : {stats['skipped_exists']}")
    print(f"  Conversion errors              : {stats['errors']}")
    print(f"  Output normal  : {OUTPUT_NORMAL}")
    print(f"  Output nomatch : {OUTPUT_NOMATCH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
