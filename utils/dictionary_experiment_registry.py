"""Registry TSV for dictionary experiment tracking."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from os.path import join as j
from typing import Iterable, Optional

import numpy as np

FOLD_FIELDNAMES = [
    'run_id',
    'preset',
    'timestamp',
    'fold',
    'encoder_type',
    'k_d',
    'k_c',
    'val_auc',
    'val_acc',
    'selection_mode',
    'n_feasible',
    'test_acc',
    'test_sen',
    'test_spe',
    'test_f1',
    'test_auc',
    'A0_r2',
    'A1_r2',
    'partial_r2_age',
    'delta_mae',
    'B0_auc',
    'B1_auc',
    'delta_auc_diag',
    'Y_morph_auc',
    'Y_age_calibrated_auc',
    'delta_auc_age_calibration',
    'e_d',
    'e_c',
    'cos_mean',
    'dead_atoms_d',
    'dead_atoms_c',
    'run_root',
    'cache_root',
    'notes',
]

_MEAN_STD_METRICS = [
    'test_acc',
    'test_sen',
    'test_spe',
    'test_f1',
    'test_auc',
    'A0_r2',
    'A1_r2',
    'partial_r2_age',
    'delta_mae',
    'B0_auc',
    'B1_auc',
    'delta_auc_diag',
    'Y_morph_auc',
    'Y_age_calibrated_auc',
    'delta_auc_age_calibration',
    'e_d',
    'e_c',
    'cos_mean',
]

RUN_FIELDNAMES = [
    'run_id',
    'preset',
    'timestamp',
    'encoder_type',
    'k_d',
    'k_c',
    'n_folds',
    'n_folds_with_feasible',
    'selection_modes',
] + [
    item
    for metric in _MEAN_STD_METRICS
    for item in (f'{metric}_mean', f'{metric}_std')
] + [
    'run_root',
    'cache_root',
    'notes',
]


def _mean_std(values: Iterable) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and v == v]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_tsv(path: str, fieldnames: list[str], rows: list[dict]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_or_update_master(
        master_path: str,
        fieldnames: list[str],
        rows: list[dict],
        key_fields: tuple[str, ...] = ('run_id',),
):
    existing = []
    if os.path.isfile(master_path):
        with open(master_path, 'r', encoding='utf-8', newline='') as f:
            existing = list(csv.DictReader(f, delimiter='\t'))

    def _key(row: dict) -> tuple:
        return tuple(str(row.get(k, '')) for k in key_fields)

    by_key = {_key(r): r for r in existing if all(r.get(k) for k in key_fields)}
    for row in rows:
        if all(row.get(k) for k in key_fields):
            by_key[_key(row)] = row
    merged = list(by_key.values())
    _write_tsv(master_path, fieldnames, merged)


def write_master_tables(
        project_dir: str,
        fold_rows: list[dict],
        run_rows: list[dict],
        *,
        fold_name: str = 'dictionary_experiments_per_fold.tsv',
        run_name: str = 'dictionary_experiments.tsv',
):
    """Overwrite master registry TSVs with the provided rows."""
    _write_tsv(j(project_dir, fold_name), FOLD_FIELDNAMES, fold_rows)
    _write_tsv(j(project_dir, run_name), RUN_FIELDNAMES, run_rows)


def parse_selection_summary(path: str) -> dict:
    """Parse fold_*/selection_summary.txt key: value lines."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if key in ('selection_mode',):
                out[key] = val
            elif key in ('n_feasible', 'n_candidates', 'epoch'):
                try:
                    out[key] = int(float(val))
                except ValueError:
                    out[key] = val
            elif key in ('val_auc', 'val_acc', 'partial_r2_age', 'delta_auc_diag',
                         'test_accuracy', 'test_auc', 'baseline_auc'):
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def _report_scalar(report: dict, *keys, default=None):
    for key in keys:
        if key in report and report[key] is not None:
            return report[key]
    # fallback: nested probes
    age = report.get('age_incremental_probe') or {}
    diag = report.get('diag_incremental_probe') or {}
    age_cal = report.get('age_calibrated_classifier') or {}
    branch = report.get('branch_energy') or {}
    nested = {
        'A0_r2': (age.get('A0_Y_to_age') or {}).get('test_r2'),
        'A1_r2': (age.get('A1_YQd_to_age') or {}).get('test_r2'),
        'delta_mae': age.get('delta_mae'),
        'B0_auc': (diag.get('B0_age_to_y') or {}).get('test_auc'),
        'B1_auc': (diag.get('B1_ageQc_to_y') or {}).get('test_auc'),
        'delta_auc_age_calibration': age_cal.get('delta_auc_age_calibration'),
        'e_d': branch.get('e_d'),
        'e_c': branch.get('e_c'),
        'cos_mean': branch.get('cos_mean'),
    }
    for key in keys:
        if key in nested and nested[key] is not None:
            return nested[key]
    return default


def collect_fold_rows_from_run(
        run_root: str,
        *,
        run_id: str = None,
        preset: str = 'dual_dict_core',
        cache_root: str = '',
        encoder_type: str = '',
        notes: str = '',
) -> list[dict]:
    """Read fold_*/test/dictionary_summary.json and build per-fold rows."""
    run_id = run_id or os.path.basename(run_root.rstrip('/\\'))
    rows = []
    fold_dirs = sorted(
        d for d in os.listdir(run_root)
        if d.startswith('fold_') and os.path.isdir(j(run_root, d))
    )
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.split('_', 1)[1])
        report_path = j(run_root, fold_dir, 'test', 'dictionary_summary.json')
        if not os.path.isfile(report_path):
            continue
        report = _read_json(report_path)
        dead = report.get('dead_atoms') or {}
        selection = parse_selection_summary(j(run_root, fold_dir, 'selection_summary.txt'))
        rows.append({
            'run_id': run_id,
            'preset': preset,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fold': fold_idx,
            'encoder_type': encoder_type,
            'k_d': dead.get('k_d'),
            'k_c': dead.get('k_c'),
            'val_auc': selection.get('val_auc'),
            'val_acc': selection.get('val_acc'),
            'selection_mode': selection.get('selection_mode'),
            'n_feasible': selection.get('n_feasible'),
            'test_acc': report.get('test_accuracy'),
            'test_sen': report.get('test_sen'),
            'test_spe': report.get('test_spe'),
            'test_f1': report.get('test_f1'),
            'test_auc': report.get('test_auc'),
            'A0_r2': _report_scalar(report, 'A0_r2'),
            'A1_r2': _report_scalar(report, 'A1_r2'),
            'partial_r2_age': report.get('partial_r2_age'),
            'delta_mae': _report_scalar(report, 'delta_mae'),
            'B0_auc': _report_scalar(report, 'B0_auc'),
            'B1_auc': _report_scalar(report, 'B1_auc'),
            'delta_auc_diag': report.get('delta_auc_diag'),
            'Y_morph_auc': report.get('Y_morph_auc'),
            'Y_age_calibrated_auc': report.get('Y_age_calibrated_auc'),
            'delta_auc_age_calibration': _report_scalar(report, 'delta_auc_age_calibration'),
            'e_d': _report_scalar(report, 'e_d'),
            'e_c': _report_scalar(report, 'e_c'),
            'cos_mean': _report_scalar(report, 'cos_mean'),
            'dead_atoms_d': dead.get('Q_d'),
            'dead_atoms_c': dead.get('Q_c'),
            'run_root': run_root,
            'cache_root': cache_root,
            'notes': notes,
        })
    return rows


def merge_training_val_metrics(fold_rows: list[dict], training_results: list[dict]) -> list[dict]:
    """Attach val_auc/val_acc from trainer CV results if available."""
    by_fold = {int(r['fold']): r for r in training_results if r.get('fold') is not None}
    for row in fold_rows:
        tr = by_fold.get(int(row['fold']), {})
        if tr.get('val_auc') is not None:
            row['val_auc'] = tr.get('val_auc')
        if tr.get('val_acc') is not None:
            row['val_acc'] = tr.get('val_acc')
        if tr.get('selection_mode') is not None:
            row['selection_mode'] = tr.get('selection_mode')
        if tr.get('n_feasible') is not None:
            row['n_feasible'] = tr.get('n_feasible')
    return fold_rows


def summarize_run_rows(fold_rows: list[dict], *, encoder_type: str = '') -> dict:
    if not fold_rows:
        return {}
    first = fold_rows[0]
    out = {
        'run_id': first.get('run_id'),
        'preset': first.get('preset'),
        'timestamp': first.get('timestamp'),
        'encoder_type': encoder_type or first.get('encoder_type', ''),
        'k_d': first.get('k_d'),
        'k_c': first.get('k_c'),
        'n_folds': len(fold_rows),
        'run_root': first.get('run_root'),
        'cache_root': first.get('cache_root'),
        'notes': first.get('notes', ''),
    }
    for metric in _MEAN_STD_METRICS:
        mean, std = _mean_std(r.get(metric) for r in fold_rows)
        out[f'{metric}_mean'] = mean
        out[f'{metric}_std'] = std

    feasible_flags = []
    modes = []
    for r in fold_rows:
        n_f = r.get('n_feasible')
        if n_f is not None and n_f == n_f:
            try:
                feasible_flags.append(int(n_f) > 0)
            except (TypeError, ValueError):
                pass
        mode = r.get('selection_mode')
        if mode:
            modes.append(str(mode))
    out['n_folds_with_feasible'] = int(sum(feasible_flags)) if feasible_flags else 0
    out['selection_modes'] = ';'.join(modes) if modes else ''
    return out


def save_dictionary_experiment_tables(
        run_root: str,
        fold_rows: list[dict],
        *,
        project_dir: str = 'weights/classifier',
        encoder_type: str = '',
        update_master: bool = True,
):
    """Write per-run TSVs and optionally update master registry under project_dir."""
    summary_dir = j(run_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    _write_tsv(j(summary_dir, 'test_per_fold.tsv'), FOLD_FIELDNAMES, fold_rows)

    run_row = summarize_run_rows(fold_rows, encoder_type=encoder_type)
    if run_row:
        _write_tsv(j(summary_dir, 'experiments_per_run.tsv'), RUN_FIELDNAMES, [run_row])

    if update_master:
        master_fold = j(project_dir, 'dictionary_experiments_per_fold.tsv')
        master_run = j(project_dir, 'dictionary_experiments.tsv')
        _append_or_update_master(
            master_fold, FOLD_FIELDNAMES, fold_rows,
            key_fields=('run_id', 'fold'),
        )
        _append_or_update_master(
            master_run, RUN_FIELDNAMES, [run_row] if run_row else [],
            key_fields=('run_id',),
        )

    return {
        'test_per_fold': j(summary_dir, 'test_per_fold.tsv'),
        'experiments_per_run': j(summary_dir, 'experiments_per_run.tsv'),
        'run_row': run_row,
    }


def register_dictionary_run(
        run_root: str,
        *,
        preset: str = 'dual_dict_core',
        cache_root: str = '',
        encoder_type: str = 'joint_lista',
        training_results: list[dict] = None,
        project_dir: str = 'weights/classifier',
        notes: str = '',
):
    """Collect diagnostics from run_root and persist experiment tables."""
    run_id = os.path.basename(run_root.rstrip('/\\'))
    fold_rows = collect_fold_rows_from_run(
        run_root,
        run_id=run_id,
        preset=preset,
        cache_root=cache_root,
        encoder_type=encoder_type,
        notes=notes,
    )
    for row in fold_rows:
        row['encoder_type'] = encoder_type
    if training_results:
        fold_rows = merge_training_val_metrics(fold_rows, training_results)
    return save_dictionary_experiment_tables(
        run_root, fold_rows, project_dir=project_dir, encoder_type=encoder_type,
    )
