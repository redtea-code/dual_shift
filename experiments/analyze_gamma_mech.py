"""Summarize gamma mechanism 5-fold runs and scan constant mix ratios on checkpoints.

Usage:
  python -m experiments.analyze_gamma_mech --summary_roots weights/classifier/causal_cv_summary/ADNI
  python -m experiments.analyze_gamma_mech --scan_ckpt <run_phase2_dir> --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from os.path import join as j

import numpy as np
import torch
from torch.utils import data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config
from training.trainer_age import load_age_dataset
from training.evaluator import evaluate_on_indices
from utils.cv_splitter_v2 import resolve_subject_standard_cv_folds
from experiments.model_registry import build_causal_model


def _metric_scalar(v):
    return v.item() if torch.is_tensor(v) else float(v)


def _read_fold_metrics(fold_dir):
    metrics_path = j(fold_dir, 'test', 'metrics.txt')
    if not os.path.isfile(metrics_path):
        return None
    out = {}
    with open(metrics_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            k, v = k.strip(), v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    gamma_path = j(fold_dir, 'test', 'gamma_stats.json')
    if os.path.isfile(gamma_path):
        with open(gamma_path, 'r', encoding='utf-8') as f:
            out['gamma_stats'] = json.load(f)
    return out


def summarize_run(run_phase_dir: str) -> dict:
    """Aggregate test metrics across fold_* under a phase directory."""
    folds = []
    for name in sorted(os.listdir(run_phase_dir)):
        if not name.startswith('fold_'):
            continue
        m = _read_fold_metrics(j(run_phase_dir, name))
        if m is None:
            continue
        folds.append(m)
    if not folds:
        return {}

    def _col(key):
        vals = [f.get(key) for f in folds if isinstance(f.get(key), (int, float))]
        if not vals:
            return None, None
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    auc_m, auc_s = _col('auc')
    acc_m, acc_s = _col('accuracy')
    gamma_means = []
    gamma_corrs = []
    for f in folds:
        gs = f.get('gamma_stats') or {}
        if gs.get('gamma_mean') is not None:
            gamma_means.append(gs['gamma_mean'])
        if gs.get('corr_gamma_age') is not None:
            gamma_corrs.append(gs['corr_gamma_age'])

    return {
        'run_dir': run_phase_dir,
        'n_folds': len(folds),
        'test_auc_mean': auc_m,
        'test_auc_std': auc_s,
        'test_acc_mean': acc_m,
        'test_acc_std': acc_s,
        'gamma_mean_mean': float(np.mean(gamma_means)) if gamma_means else None,
        'corr_gamma_age_mean': float(np.mean(gamma_corrs)) if gamma_corrs else None,
    }


def discover_and_write_summary(roots, out_tsv):
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if 'overall_summary.txt' in filenames or any(
                n.startswith('fold_') for n in dirnames
            ):
                # phase dir: contains fold_*
                if any(n.startswith('fold_') for n in os.listdir(dirpath)):
                    row = summarize_run(dirpath)
                    if row:
                        # tag from parent names
                        parts = os.path.normpath(dirpath).split(os.sep)
                        row['tag'] = '/'.join(parts[-3:])
                        rows.append(row)
    os.makedirs(os.path.dirname(out_tsv) or '.', exist_ok=True)
    fields = [
        'tag', 'n_folds', 'test_auc_mean', 'test_auc_std',
        'test_acc_mean', 'test_acc_std', 'gamma_mean_mean',
        'corr_gamma_age_mean', 'run_dir',
    ]
    with open(out_tsv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f'Wrote {len(rows)} runs → {out_tsv}')
    return rows


@torch.no_grad()
def scan_constant_mix(run_phase_dir, config_path, device='cuda',
                      mix_values=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Override gamma with constant c, report mix m=1-c → AUC on each fold test.

    mix m means F = F_d + m * F_c, i.e. gamma = 1 - m.
    """
    cf = load_config(config_path)
    # Force constant mode evaluation via temporary mode override on loaded model
    full_dataset = load_age_dataset(cf)
    age_cfg = cf.get('age_split') or {}
    n_splits = int(age_cfg.get('num_folds', 5))
    random_seed = age_cfg.get('random_seed', 42)
    stratify = age_cfg.get('stratify', True)
    folds = resolve_subject_standard_cv_folds(
        full_dataset.subject_ids,
        full_dataset.all_labels,
        n_splits=n_splits,
        random_seed=random_seed,
        stratify=stratify,
        verbose=False,
    )
    phase = int((cf.get('causal') or {}).get('phase', 2))
    model_cls, model_kwargs = build_causal_model(
        'resnet10_disentangled', cf, causal_phase=phase,
    )
    model_kwargs = dict(model_kwargs)
    model_kwargs['num_classes'] = full_dataset.num_classes
    model_kwargs['gamma_mech_mode'] = 'constant'

    rows = []
    for fold in folds:
        ckpt = j(run_phase_dir, f'fold_{fold.fold_idx}', 'model_best', 'best_model.pth')
        if not os.path.isfile(ckpt):
            print(f'[WARN] missing {ckpt}')
            continue
        for m in mix_values:
            gamma_c = 1.0 - float(m)
            model_kwargs['gamma_constant_value'] = gamma_c
            model = model_cls(**model_kwargs).to(device)
            state = torch.load(ckpt, map_location=device)
            # Drop mismatched fc when scanning; load with non-strict for gate buffers
            model.load_state_dict(state, strict=False)
            # Force constant gamma buffer
            model.gamma_mech_mode = 'constant'
            c = min(max(gamma_c, 1e-4), 1 - 1e-4)
            logit = float(np.log(c / (1 - c)))
            if hasattr(model, '_fixed_gamma_logit'):
                model._fixed_gamma_logit.fill_(logit)
            else:
                model.register_buffer(
                    '_fixed_gamma_logit',
                    torch.tensor([logit], dtype=torch.float32, device=device),
                )
            model.gamma_logit = None
            results = evaluate_on_indices(
                model, full_dataset, fold.test_idx, device,
                eval_bc=cf.get('eval_bc', 8),
                num_classes=full_dataset.num_classes,
            )
            rows.append({
                'fold': fold.fold_idx,
                'mix_m': m,
                'gamma': gamma_c,
                'test_auc': _metric_scalar(results['auc']),
                'test_acc': _metric_scalar(results['accuracy']),
            })
            print(
                f"  fold={fold.fold_idx} m={m:.2f} gamma={gamma_c:.2f} "
                f"auc={rows[-1]['test_auc']:.4f}"
            )
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

    out_path = j(run_phase_dir, 'summary', 'constant_mix_scan.tsv')
    os.makedirs(j(run_phase_dir, 'summary'), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['fold', 'mix_m', 'gamma', 'test_auc', 'test_acc'],
            delimiter='\t',
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote mix scan → {out_path}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Gamma mechanism analysis')
    parser.add_argument(
        '--summary_roots', nargs='*',
        default=['weights/classifier/causal_cv_summary/ADNI'],
    )
    parser.add_argument(
        '--out_tsv',
        default='weights/classifier/gamma_mech_5fold_summary.tsv',
    )
    parser.add_argument('--scan_ckpt', type=str, default=None,
                        help='Phase-2 run dir to scan constant mix ratios')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    discover_and_write_summary(args.summary_roots, args.out_tsv)
    if args.scan_ckpt:
        scan_constant_mix(args.scan_ckpt, args.config_path, device=args.device)


if __name__ == '__main__':
    main()
