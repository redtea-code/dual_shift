"""Summarize gamma-vs-fusion std5cv runs into a TSV.

Usage:
  python -m experiments.analyze_gamma_fusion \\
    --summary_roots weights/classifier/age_cv_summary/ADNI \\
    --out_tsv weights/classifier/gamma_fusion_5fold_summary.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from os.path import join as j

import numpy as np

import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TASK_RE = re.compile(r'_task(\d+)_')


def _parse_class_task(tag: str) -> str:
    m = TASK_RE.search(tag or '')
    return m.group(1) if m else '13'

ID_KEYS = [
    ('B0', 'gamma_fusion_B0'),
    ('B_gamma', 'gamma_fusion_B_gamma'),
    ('B_film', 'gamma_fusion_B_film'),
    ('B_daft', 'gamma_fusion_B_daft'),
    ('B_hyper', 'gamma_fusion_B_hyper'),
    ('B_concat', 'gamma_fusion_B_concat'),
    ('L0', 'gamma_fusion_L0'),
    ('L1', 'gamma_fusion_L1'),
    ('L2', 'gamma_fusion_L2'),
    ('L3', 'gamma_fusion_L3'),
    ('A_global', 'gamma_fusion_A_global'),
    ('A_noshare', 'gamma_fusion_A_noshare'),
    ('A_shuffle', 'gamma_fusion_A_shuffle'),
]


def _read_fold_metrics(fold_dir):
    metrics_path = j(fold_dir, 'test', 'metrics.txt')
    if not os.path.isfile(metrics_path):
        return None
    out = {}
    with open(metrics_path, 'r', encoding='utf-8') as f:
        text = f.read()
    for tok in text.replace(',', ' ').split():
        if ':' not in tok:
            continue
        kk, vv = tok.split(':', 1)
        try:
            out[kk] = float(vv)
        except ValueError:
            pass
    gamma_path = j(fold_dir, 'test', 'gamma_stats.json')
    if os.path.isfile(gamma_path):
        with open(gamma_path, 'r', encoding='utf-8') as f:
            out['gamma_stats'] = json.load(f)
    return out


def _read_summary_tsv(run_dir: str):
    tsv = j(run_dir, 'summary', 'test_per_fold.tsv')
    if not os.path.isfile(tsv):
        return None
    rows = []
    with open(tsv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rows.append(row)
    if len(rows) < 1:
        return None
    try:
        aucs = [float(r['test_auc']) for r in rows]
        accs = [float(r['test_acc']) for r in rows]
    except (KeyError, ValueError):
        return None
    return {
        'n_folds': len(rows),
        'test_auc_mean': float(np.mean(aucs)),
        'test_auc_std': float(np.std(aucs)),
        'test_acc_mean': float(np.mean(accs)),
        'test_acc_std': float(np.std(accs)),
    }


def summarize_run(run_dir: str) -> dict:
    # Prefer authoritative CV table when present
    base = _read_summary_tsv(run_dir) or {}
    folds = []
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith('fold_'):
            continue
        m = _read_fold_metrics(j(run_dir, name))
        if m:
            folds.append(m)

    def col(keys):
        vals = []
        for f in folds:
            for k in keys:
                if k in f and isinstance(f[k], (int, float)):
                    vals.append(float(f[k]))
                    break
        if not vals:
            return None, None
        a = np.asarray(vals, dtype=np.float64)
        return float(a.mean()), float(a.std())

    auc_m, auc_s = col(['AUC', 'auc', 'test_auc'])
    acc_m, acc_s = col(['Acc', 'accuracy', 'test_acc'])
    if base.get('test_auc_mean') is not None:
        auc_m, auc_s = base['test_auc_mean'], base['test_auc_std']
        acc_m, acc_s = base['test_acc_mean'], base['test_acc_std']
    if auc_m is None and not folds:
        return {}

    gamma_means, corrs = [], []
    for f in folds:
        gs = f.get('gamma_stats') or {}
        if gs.get('gamma_mean') is not None:
            gamma_means.append(gs['gamma_mean'])
        if gs.get('corr_gamma_age') is not None:
            corrs.append(gs['corr_gamma_age'])
    return {
        'run_dir': run_dir,
        'n_folds': base.get('n_folds') or len(folds),
        'test_auc_mean': auc_m,
        'test_auc_std': auc_s,
        'test_acc_mean': acc_m,
        'test_acc_std': acc_s,
        'gamma_mean_mean': float(np.mean(gamma_means)) if gamma_means else None,
        'corr_gamma_age_mean': float(np.mean(corrs)) if corrs else None,
    }


def discover(roots):
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = j(root, name)
            if not os.path.isdir(path):
                continue
            if 'gamma_fusion' not in name:
                continue
            row = summarize_run(path)
            if not row:
                continue
            exp_id = None
            for eid, key in ID_KEYS:
                if key in name:
                    exp_id = eid
                    break
            row['exp_id'] = exp_id or name
            row['tag'] = name
            row['class_task'] = _parse_class_task(name)
            rows.append(row)
    return rows


def _alias_l3(rows):
    by_task = {}
    for r in rows:
        by_task.setdefault(r.get('class_task', '13'), {})[r.get('exp_id')] = r
    out = list(rows)
    for task, by_id in by_task.items():
        if 'B_gamma' in by_id and 'L3' not in by_id:
            alias = dict(by_id['B_gamma'])
            alias['exp_id'] = 'L3'
            alias['tag'] = (alias.get('tag') or '') + '_alias_L3'
            out.append(alias)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--summary_roots', nargs='*',
        default=['weights/classifier/age_cv_summary/ADNI'],
    )
    parser.add_argument(
        '--out_tsv',
        default='weights/classifier/gamma_fusion_5fold_summary.tsv',
    )
    args = parser.parse_args()
    rows = _alias_l3(discover(args.summary_roots))
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
    fields = [
        'class_task', 'exp_id', 'tag', 'n_folds', 'test_auc_mean', 'test_auc_std',
        'test_acc_mean', 'test_acc_std', 'gamma_mean_mean',
        'corr_gamma_age_mean', 'run_dir',
    ]
    with open(args.out_tsv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get('class_task', ''), x.get('exp_id', ''))):
            w.writerow({k: r.get(k) for k in fields})
    print(f'Wrote {len(rows)} runs -> {args.out_tsv}')
    # Per-task TSVs for paper tables
    out_dir = os.path.dirname(args.out_tsv) or '.'
    for task in sorted({r.get('class_task', '13') for r in rows}):
        sub = [r for r in rows if r.get('class_task') == task]
        path = j(out_dir, f'gamma_fusion_5fold_summary_task{task}.tsv')
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
            w.writeheader()
            for r in sorted(sub, key=lambda x: x.get('exp_id', '')):
                w.writerow({k: r.get(k) for k in fields})
        print(f'  task{task}: {len(sub)} runs -> {path}')


if __name__ == '__main__':
    main()
