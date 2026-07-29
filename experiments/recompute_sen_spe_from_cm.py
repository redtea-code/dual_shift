"""Recompute Sen/Spe from saved confusion matrices (AD-positive binary defs).

Does not retrain. Updates each run's:
  fold_*/test/metrics.txt  (Sen/Spe fields)
  summary/test_per_fold.tsv

Then refreshes readable tables via export_gamma_fusion_table.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from os.path import join as j

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.evaluator import resolve_positive_class_index, sen_spe_from_cm, default_positive_class


def parse_cm_file(path: str):
    """Parse format_confusion_matrix text → (class_names, cm ndarray)."""
    lines = [ln.rstrip() for ln in open(path, encoding='utf-8') if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f'insufficient CM lines in {path}')
    # First non-separator line that looks like a header of class names
    header = None
    data_lines = []
    for ln in lines:
        if set(ln.strip()) <= set('-'):
            continue
        toks = ln.split()
        if header is None:
            # Header may be indented class names only
            header = toks
            continue
        # Row: name + counts
        if len(toks) < 2:
            continue
        data_lines.append(toks)
    if not header or not data_lines:
        raise ValueError(f'cannot parse CM {path}')
    class_names = header
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.float64)
    row_names = []
    for toks in data_lines:
        name, vals = toks[0], toks[1:]
        row_names.append(name)
        if len(vals) != n:
            raise ValueError(f'row {name} has {len(vals)} cols, expected {n} in {path}')
        i = class_names.index(name) if name in class_names else len(row_names) - 1
        cm[i, :] = [float(v) for v in vals]
    return class_names, cm


def _parse_metrics_txt(path: str) -> dict:
    text = open(path, encoding='utf-8').read()
    out = {}
    m = re.search(r'val_best_acc:\s*([0-9.]+)', text)
    if m:
        out['val_best_acc'] = float(m.group(1))
    for key, pat in [
        ('loss', r'loss:([0-9.]+)'),
        ('accuracy', r'Acc:([0-9.]+)'),
        ('f1', r'F1:([0-9.]+)'),
        ('auc', r'AUC:([0-9.]+)'),
        # legacy macro recall key
        ('recall_old', r'(?:recall|Sen):([0-9.]+)'),
        ('spe_old', r'Spe:([0-9.]+)'),
    ]:
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def update_fold(fold_dir: str, positive_class=None) -> dict:
    cm_path = j(fold_dir, 'test', 'confusion_matrix.txt')
    metrics_path = j(fold_dir, 'test', 'metrics.txt')
    if not os.path.isfile(cm_path) or not os.path.isfile(metrics_path):
        return {}
    class_names, cm = parse_cm_file(cm_path)
    pos_name = positive_class or default_positive_class(class_names)
    pos = resolve_positive_class_index(len(class_names), class_names, pos_name)
    sen, spe = sen_spe_from_cm(cm, pos)
    old = _parse_metrics_txt(metrics_path)
    # Rewrite metrics.txt preserving Acc/F1/AUC/Loss; update Sen/Spe (+ pos tag)
    line = (
        f"loss:{old.get('loss', float('nan')):.6f} "
        f"Acc:{old.get('accuracy', float('nan')):.4f} "
        f"Sen:{sen:.4f} F1:{old.get('f1', float('nan')):.4f} "
        f"AUC:{old.get('auc', float('nan')):.4f} Spe:{spe:.4f} "
        f"pos={pos}({class_names[pos]})"
    )
    with open(metrics_path, 'w', encoding='utf-8') as f:
        if 'val_best_acc' in old:
            f.write(f"val_best_acc: {old['val_best_acc']:.4f}\n")
        f.write(line + '\n')
        f.write(
            f"# Sen/Spe: {class_names[pos]}-positive binary "
            f"(TP/(TP+FN), TN/(TN+FP)); was macro-eq "
            f"recall={old.get('recall_old')} spe={old.get('spe_old')}\n"
        )
    return {
        'val_acc': old.get('val_best_acc'),
        'test_acc': old.get('accuracy'),
        'test_sen': sen,
        'test_f1': old.get('f1'),
        'test_auc': old.get('auc'),
        'test_spe': spe,
        'test_loss': old.get('loss'),
    }


def update_run(run_dir: str, positive_class=None) -> int:
    folds = []
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith('fold_'):
            continue
        row = update_fold(j(run_dir, name), positive_class=positive_class)
        if row:
            folds.append((name, row))
    if not folds:
        return 0
    summary_dir = j(run_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    tsv = j(summary_dir, 'test_per_fold.tsv')
    with open(tsv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([
            'fold', 'val_acc', 'test_acc', 'test_sen', 'test_f1',
            'test_auc', 'test_spe', 'test_loss',
        ])
        for name, r in folds:
            w.writerow([
                name,
                f"{r['val_acc']:.4f}" if r['val_acc'] is not None else '',
                f"{r['test_acc']:.4f}" if r['test_acc'] is not None else '',
                f"{r['test_sen']:.4f}",
                f"{r['test_f1']:.4f}" if r['test_f1'] is not None else '',
                f"{r['test_auc']:.4f}" if r['test_auc'] is not None else '',
                f"{r['test_spe']:.4f}",
                f"{r['test_loss']:.6f}" if r['test_loss'] is not None else '',
            ])
    return len(folds)


def main():
    root = j(PROJECT_ROOT, 'weights', 'classifier', 'age_cv_summary', 'ADNI')
    n_runs = 0
    for name in sorted(os.listdir(root)):
        if 'gamma_fusion' not in name:
            continue
        path = j(root, name)
        if not os.path.isdir(path):
            continue
        n = update_run(path, positive_class=None)
        if n:
            print(f'updated {name}: {n} folds')
            n_runs += 1
    print(f'Done: {n_runs} runs')
    # Refresh human tables
    from experiments.export_gamma_fusion_table import main as export_main
    export_main()


if __name__ == '__main__':
    main()
