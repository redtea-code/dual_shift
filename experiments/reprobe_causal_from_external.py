"""
Batch-reprobe causal CV checkpoints from an external weights tree
(e.g. D:/cyh/Causal_Infer3/weights/classifier) and write master TSVs into
this repo's weights/classifier/.

Usage:
  python -m experiments.reprobe_causal_from_external \\
      --runs_root D:/cyh/Causal_Infer3/weights/classifier/causal_cv_summary \\
      --name_filter task13 \\
      --skip_baseline --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from os.path import join as j

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.reprobe_decomposition import (
    _discover_phase_dirs,
    _infer_model_name,
    _to_float,
    reprobe_single_run,
)

MEAN_STD_KEYS = [
    'eval_acc', 'eval_sen', 'eval_spe', 'eval_f1', 'eval_auc', 'eval_loss',
    'r2_age_fd', 'r2_age_fc', 'r2_age_residual_y_fd',
    'auc_y_from_fc', 'auc_y_from_age', 'auc_y_from_fc_age',
    'delta_auc_fc_given_age', 'delta_mi_proxy',
    'fd_effective_rank', 'fc_effective_rank',
]

FOLD_EXTRA = [
    'run_id', 'dataset', 'phase', 'model', 'timestamp', 'run_dir', 'notes',
]


def _mean_std(vals):
    arr = [v for v in vals if v is not None and v == v]
    if not arr:
        return None, None
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std())


def _parse_run_id(run_dir: str) -> tuple[str, str, int]:
    """Return (dataset, run_id, phase) from .../causal_cv_summary/{DS}/{run}/phaseN."""
    run_dir = os.path.normpath(run_dir)
    phase_name = os.path.basename(run_dir)
    parent = os.path.dirname(run_dir)
    run_id = os.path.basename(parent)
    dataset = os.path.basename(os.path.dirname(parent))
    phase = int(phase_name.replace('phase', '')) if phase_name.startswith('phase') else 0
    return dataset, run_id, phase


def _write_tsv(path: str, fieldnames: list[str], rows: list[dict]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize_run(fold_rows: list[dict]) -> dict:
    if not fold_rows:
        return {}
    first = fold_rows[0]
    out = {
        'run_id': first.get('run_id'),
        'dataset': first.get('dataset'),
        'phase': first.get('phase'),
        'model': first.get('model'),
        'timestamp': first.get('timestamp'),
        'n_folds': len(fold_rows),
        'n_heuristic_pass': sum(
            1 for r in fold_rows
            if str(r.get('phase1_heuristic_pass')).lower() in ('1', 'true')
        ),
        'run_dir': first.get('run_dir'),
        'notes': first.get('notes', ''),
    }
    for key in MEAN_STD_KEYS:
        m, s = _mean_std(_to_float(r.get(key)) for r in fold_rows)
        out[f'{key}_mean'] = m
        out[f'{key}_std'] = s
    return out


def main():
    parser = argparse.ArgumentParser(description='Batch causal reprobe + master TSV')
    parser.add_argument(
        '--runs_root', type=str,
        default=r'D:\cyh\Causal_Infer3\weights\classifier\causal_cv_summary',
    )
    parser.add_argument('--name_filter', type=str, default='task13',
                        help='Substring filter on run path (default: task13)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_baseline', action='store_true', default=True)
    parser.add_argument('--no_skip_baseline', action='store_true')
    parser.add_argument('--inplace', action='store_true', default=True)
    parser.add_argument('--out_suffix', type=str, default='ridge')
    parser.add_argument(
        '--project_dir', type=str, default='weights/classifier',
        help='Where to write causal_experiments*.tsv (this repo)',
    )
    parser.add_argument('--max_runs', type=int, default=None)
    args = parser.parse_args()

    skip_baseline = False if args.no_skip_baseline else args.skip_baseline
    inplace = args.inplace
    out_suffix = 'inplace' if inplace else args.out_suffix

    phase_dirs = _discover_phase_dirs(args.runs_root)
    if args.name_filter:
        phase_dirs = [d for d in phase_dirs if args.name_filter in d]
    # Prefer ADNI then NACC; skip obvious junk
    phase_dirs = [
        d for d in phase_dirs
        if 'bad_design' not in d
    ]
    phase_dirs = sorted(
        phase_dirs,
        key=lambda d: (0 if '\\ADNI\\' in d or '/ADNI/' in d else 1, d),
    )
    if args.max_runs:
        phase_dirs = phase_dirs[: args.max_runs]

    print(f'Found {len(phase_dirs)} phase dirs under {args.runs_root}')
    all_fold_rows = []
    all_run_rows = []
    failures = []

    for run_dir in phase_dirs:
        dataset, run_id, phase = _parse_run_id(run_dir)
        model_name = _infer_model_name(os.path.dirname(run_dir))
        if model_name is None:
            failures.append(f'infer model failed: {run_dir}')
            continue
        print(f'\n>>> {dataset}/{run_id}/{os.path.basename(run_dir)}')
        try:
            result = reprobe_single_run(
                run_dir=run_dir,
                model_name=model_name,
                device=args.device,
                out_suffix=args.out_suffix,
                inplace=inplace,
                skip_baseline=skip_baseline,
            )
        except Exception as exc:
            print(f'[ERROR] {run_dir}: {exc}')
            failures.append(f'{run_dir}: {exc}')
            continue

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fold_rows = []
        for row in result.get('tsv_rows') or []:
            enriched = dict(row)
            enriched.update({
                'run_id': run_id,
                'dataset': dataset,
                'phase': phase,
                'model': model_name,
                'timestamp': ts,
                'run_dir': run_dir.replace('\\', '/'),
                'notes': out_suffix,
            })
            fold_rows.append(enriched)
            all_fold_rows.append(enriched)
        run_row = summarize_run(fold_rows)
        if run_row:
            all_run_rows.append(run_row)

    # Fieldnames: extras first, then union of fold keys
    fold_keys = []
    seen = set()
    for key in FOLD_EXTRA:
        fold_keys.append(key)
        seen.add(key)
    for row in all_fold_rows:
        for key in row:
            if key not in seen:
                fold_keys.append(key)
                seen.add(key)

    run_keys = [
        'run_id', 'dataset', 'phase', 'model', 'timestamp', 'n_folds',
        'n_heuristic_pass', 'run_dir', 'notes',
    ]
    for key in MEAN_STD_KEYS:
        run_keys.extend([f'{key}_mean', f'{key}_std'])

    project_dir = args.project_dir
    fold_path = j(project_dir, 'causal_experiments_per_fold.tsv')
    run_path = j(project_dir, 'causal_experiments.tsv')
    _write_tsv(fold_path, fold_keys, all_fold_rows)
    _write_tsv(run_path, run_keys, all_run_rows)

    print(f'\nWrote {len(all_run_rows)} runs / {len(all_fold_rows)} folds')
    print(f'  {run_path}')
    print(f'  {fold_path}')
    if failures:
        print(f'Failures ({len(failures)}):')
        for msg in failures:
            print(' ', msg)
        sys.exit(1)
    print('Done.')


if __name__ == '__main__':
    main()
