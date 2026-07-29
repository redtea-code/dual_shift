"""Rebuild age_cv overall_summary / test_per_fold.tsv from per-fold test metrics."""
from __future__ import annotations

import argparse
import os
import re
import sys
from os.path import join as j

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _parse_metrics_file(path: str) -> dict:
    text = open(path, encoding='utf-8').read().strip()
    out = {}
    m = re.search(r'val_best_acc:\s*([0-9.]+)', text)
    if m:
        out['val_best_acc'] = float(m.group(1))
    # free-form metrics line: loss:.. Acc:.. recall:.. F1:.. AUC:.. Spe:..
    for key, pat in [
        ('loss', r'loss:([0-9.]+)'),
        ('accuracy', r'Acc:([0-9.]+)'),
        ('recall', r'recall:([0-9.]+)'),
        ('f1', r'F1:([0-9.]+)'),
        ('auc', r'AUC:([0-9.]+)'),
        ('specificity', r'Spe:([0-9.]+)'),
    ]:
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def repair_run(run_root: str) -> bool:
    folds = []
    for name in sorted(os.listdir(run_root)):
        if not name.startswith('fold_'):
            continue
        mp = j(run_root, name, 'test', 'metrics.txt')
        if not os.path.isfile(mp):
            continue
        folds.append((name, _parse_metrics_file(mp)))
    if not folds:
        print(f'  no folds in {run_root}')
        return False

    summary_dir = j(run_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    val_accs = [f.get('val_best_acc', float('nan')) for _, f in folds]
    test_accs = [f.get('accuracy', float('nan')) for _, f in folds]
    lines = [
        'Age CV Summary (repaired from fold test/metrics.txt)',
        f'  run: {run_root}',
        f'  folds: {len(folds)}',
        '',
        f'Val Acc per fold: {[f"{a:.4f}" for a in val_accs]}',
        f'Val mean +/- std: {np.nanmean(val_accs):.4f} +/- {np.nanstd(val_accs):.4f}',
        '',
        f'Test Acc per fold: {[f"{a:.4f}" for a in test_accs]}',
        f'Test mean +/- std: {np.nanmean(test_accs):.4f} +/- {np.nanstd(test_accs):.4f}',
    ]
    with open(j(summary_dir, 'overall_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    tsv = j(summary_dir, 'test_per_fold.tsv')
    with open(tsv, 'w', encoding='utf-8') as f:
        f.write('fold\tval_acc\ttest_acc\ttest_sen\ttest_f1\ttest_auc\ttest_spe\ttest_loss\n')
        for name, m in folds:
            f.write(
                f"{name}\t{m.get('val_best_acc', float('nan')):.4f}\t"
                f"{m.get('accuracy', float('nan')):.4f}\t"
                f"{m.get('recall', float('nan')):.4f}\t"
                f"{m.get('f1', float('nan')):.4f}\t"
                f"{m.get('auc', float('nan')):.4f}\t"
                f"{m.get('specificity', float('nan')):.4f}\t"
                f"{m.get('loss', float('nan')):.6f}\n"
            )
    print(f'repaired {run_root} ({len(folds)} folds)')
    return True


def _tsv_has_rows(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, encoding='utf-8') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return len(lines) >= 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--roots', nargs='*',
        default=['weights/classifier/age_cv_summary/ADNI'],
    )
    parser.add_argument('--only_broken', action='store_true', default=True)
    args = parser.parse_args()
    for root in args.roots:
        root = j(PROJECT_ROOT, root) if not os.path.isabs(root) else root
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if 'gamma_fusion' not in name:
                continue
            run = j(root, name)
            tsv = j(run, 'summary', 'test_per_fold.tsv')
            if args.only_broken and _tsv_has_rows(tsv):
                continue
            # Need at least fold_1/test/metrics.txt
            if os.path.isfile(j(run, 'fold_1', 'test', 'metrics.txt')):
                repair_run(run)


if __name__ == '__main__':
    main()
