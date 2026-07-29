"""Append dictionary experiment results to central TSV tables."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from os.path import join as j

import numpy as np

MAIN_COLUMNS = [
    'run_id', 'preset', 'timestamp', 'cache_root', 'run_root',
    'encoder_type', 'k_d', 'k_c', 'num_epochs',
    'test_auc_mean', 'test_auc_std',
    'test_acc_mean', 'test_acc_std',
    'partial_r2_age_mean', 'partial_r2_age_std',
    'delta_auc_diag_mean', 'delta_auc_diag_std',
    'Y_morph_auc_mean', 'Y_age_calibrated_auc_mean',
    'n_folds',
]

FOLD_COLUMNS = [
    'run_id', 'preset', 'fold',
    'val_auc', 'test_auc', 'test_accuracy',
    'partial_r2_age', 'delta_auc_diag',
    'Y_morph_auc', 'Y_age_calibrated_auc',
]


def _mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _load_fold_report(run_root: str, fold: int) -> dict:
    json_path = j(run_root, f'fold_{fold}', 'test', 'dictionary_summary.json')
    if os.path.isfile(json_path):
        with open(json_path, encoding='utf-8') as f:
            return json.load(f)
    txt_path = j(run_root, f'fold_{fold}', 'test', 'dictionary_summary.txt')
    if not os.path.isfile(txt_path):
        return {}
    report = {}
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' not in line or line.startswith('Dictionary'):
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            try:
                report[key] = float(val)
            except ValueError:
                report[key] = val
    return report


def _append_tsv(path: str, row: dict, columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in columns})


def _write_tsv(path: str, rows: list[dict], columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in columns})


def collect_run_results(
        run_root: str,
        fold_results: list[dict] | None = None,
        preset: str = 'dual_dict_core',
) -> tuple[list[dict], dict]:
    """Collect per-fold and aggregate metrics from a dictionary run."""
    per_fold = []
    if fold_results:
        for r in fold_results:
            fold_idx = r['fold']
            report = _load_fold_report(run_root, fold_idx)
            per_fold.append({
                'fold': fold_idx,
                'val_auc': r.get('val_auc'),
                'test_auc': r.get('test_auc') or report.get('test_auc'),
                'test_accuracy': report.get('test_accuracy'),
                'partial_r2_age': r.get('partial_r2_age') or report.get('partial_r2_age'),
                'delta_auc_diag': report.get('delta_auc_diag'),
                'Y_morph_auc': report.get('Y_morph_auc'),
                'Y_age_calibrated_auc': report.get('Y_age_calibrated_auc'),
            })
    else:
        for fold_dir in sorted(os.listdir(run_root)):
            if not fold_dir.startswith('fold_'):
                continue
            fold_idx = int(fold_dir.split('_')[1])
            report = _load_fold_report(run_root, fold_idx)
            if not report:
                continue
            per_fold.append({
                'fold': fold_idx,
                'val_auc': None,
                'test_auc': report.get('test_auc'),
                'test_accuracy': report.get('test_accuracy'),
                'partial_r2_age': report.get('partial_r2_age'),
                'delta_auc_diag': report.get('delta_auc_diag'),
                'Y_morph_auc': report.get('Y_morph_auc'),
                'Y_age_calibrated_auc': report.get('Y_age_calibrated_auc'),
            })

    def _col(key):
        return [r.get(key) for r in per_fold]

    auc_mean, auc_std = _mean_std(_col('test_auc'))
    acc_mean, acc_std = _mean_std(_col('test_accuracy'))
    pr2_mean, pr2_std = _mean_std(_col('partial_r2_age'))
    da_mean, da_std = _mean_std(_col('delta_auc_diag'))
    ym_mean, _ = _mean_std(_col('Y_morph_auc'))
    yc_mean, _ = _mean_std(_col('Y_age_calibrated_auc'))

    aggregate = {
        'test_auc_mean': auc_mean,
        'test_auc_std': auc_std,
        'test_acc_mean': acc_mean,
        'test_acc_std': acc_std,
        'partial_r2_age_mean': pr2_mean,
        'partial_r2_age_std': pr2_std,
        'delta_auc_diag_mean': da_mean,
        'delta_auc_diag_std': da_std,
        'Y_morph_auc_mean': ym_mean,
        'Y_age_calibrated_auc_mean': yc_mean,
        'n_folds': len(per_fold),
    }
    return per_fold, aggregate


def register_dictionary_run(
        *,
        run_root: str,
        cache_root: str,
        project_dir: str = 'weights/classifier',
        preset: str = 'dual_dict_core',
        dict_cfg: dict | None = None,
        fold_results: list[dict] | None = None,
        skip_if_exists: bool = True,
) -> dict:
    """Register a dictionary run into central and per-run TSV tables."""
    run_id = os.path.basename(os.path.normpath(run_root))
    main_table = j(project_dir, 'dictionary_experiments.tsv')
    fold_table = j(project_dir, 'dictionary_experiments_per_fold.tsv')

    if skip_if_exists and os.path.isfile(main_table):
        with open(main_table, encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row.get('run_id') == run_id:
                    print(f"  [INFO] Run already registered: {run_id}")
                    return row

    per_fold, agg = collect_run_results(run_root, fold_results, preset)
    dict_cfg = dict_cfg or {}
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    summary_row = {
        'run_id': run_id,
        'preset': preset,
        'timestamp': timestamp,
        'cache_root': cache_root,
        'run_root': run_root,
        'encoder_type': dict_cfg.get('encoder_type', ''),
        'k_d': dict_cfg.get('k_d', ''),
        'k_c': dict_cfg.get('k_c', ''),
        'num_epochs': dict_cfg.get('num_epochs', ''),
        **agg,
    }

    fold_rows = []
    for r in per_fold:
        fold_rows.append({
            'run_id': run_id,
            'preset': preset,
            **r,
        })

    _append_tsv(main_table, summary_row, MAIN_COLUMNS)
    for row in fold_rows:
        _append_tsv(fold_table, row, FOLD_COLUMNS)

    summary_dir = j(run_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    _write_tsv(j(summary_dir, 'test_per_fold.tsv'), fold_rows, FOLD_COLUMNS)
    _write_tsv(j(summary_dir, 'experiments_per_run.tsv'), [summary_row], MAIN_COLUMNS)

    print(f"  Registered experiment → {main_table}")
    return summary_row
