"""Compare dictionary MVP against feature-level and causal baselines."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from os.path import join as j

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Model.dictionary.baselines import (
    aggregate_causal_baseline_rows,
    feature_probe_baseline,
    load_causal_decomposition_tsv,
)
from Model.dictionary.dual_dictionary import DualDictionaryModel
from training.dictionary_diagnostics import run_dictionary_diagnostics, save_dictionary_report
from training.dictionary_data import CachedFeatureDataset
from utils.config import load_config
from utils.dictionary_splits import load_split_manifest
from utils.feature_cache import fold_cache_dir


def _load_split_arrays(cache_fold_dir: str) -> dict:
    out = {}
    for split in ('train', 'val', 'test'):
        data = CachedFeatureDataset(j(cache_fold_dir, f'features_{split}.npz'))
        out[split] = {
            'H': data.H.numpy(),
            'y': data.y.numpy(),
            'ages': data.ages.numpy(),
        }
    return out


def _row_from_report(name: str, fold: int, report: dict) -> dict:
    return {
        'baseline': name,
        'fold': fold,
        'test_accuracy': report.get('test_accuracy'),
        'test_auc': report.get('test_auc'),
        'partial_r2_age': report.get('partial_r2_age'),
        'delta_auc_diag': report.get('delta_auc_diag'),
        'Y_morph_auc': report.get('Y_morph_auc'),
        'Y_age_calibrated_auc': report.get('Y_age_calibrated_auc'),
    }


def compare_dictionary_baselines(
        *,
        config_path: str = 'config/default.yaml',
        cache_root: str,
        run_root: str = None,
        device: str = 'cuda',
        fold: int = None,
        causal_phase1_tsv: str = None,
        causal_phase2_tsv: str = None,
        out_dir: str = None,
):
    cf = load_config(config_path)
    dict_cfg = cf.get('dictionary') or {}
    baseline_cfg = dict_cfg.get('baselines') or {}
    manifest = load_split_manifest(cache_root)
    fold_ids = [fold] if fold else [f['fold_idx'] for f in manifest['folds']]

    rows = []
    for fold_idx in fold_ids:
        fold_dir = fold_cache_dir(cache_root, fold_idx)
        splits = _load_split_arrays(fold_dir)

        ce_probe = feature_probe_baseline(
            splits['train']['H'], splits['train']['y'],
            splits['val']['H'], splits['val']['y'],
            splits['test']['H'], splits['test']['y'],
        )
        rows.append({
            'baseline': 'ce_only_feature_probe',
            'fold': fold_idx,
            'test_accuracy': ce_probe.get('test_accuracy'),
            'test_auc': ce_probe.get('test_auc'),
        })

        ce_age_probe = feature_probe_baseline(
            splits['train']['H'], splits['train']['y'],
            splits['val']['H'], splits['val']['y'],
            splits['test']['H'], splits['test']['y'],
            ages_train=splits['train']['ages'],
            ages_val=splits['val']['ages'],
            ages_test=splits['test']['ages'],
            with_age=True,
        )
        rows.append({
            'baseline': 'ce_age_feature_probe',
            'fold': fold_idx,
            'test_accuracy': ce_age_probe.get('test_accuracy'),
            'test_auc': ce_age_probe.get('test_auc'),
        })

        if run_root:
            ckpt = j(run_root, f'fold_{fold_idx}', 'model_best', 'best_model.pth')
            if os.path.isfile(ckpt):
                train_npz = splits['train']
                num_classes = int(np.max(train_npz['y'])) + 1
                model = DualDictionaryModel(
                    feature_dim=int(dict_cfg.get('feature_dim', 64)),
                    k_d=int(dict_cfg.get('k_d', 16)),
                    k_c=int(dict_cfg.get('k_c', 16)),
                    num_classes=num_classes,
                    encoder_type=dict_cfg.get('encoder_type', 'joint_lista'),
                    lista_steps=int(dict_cfg.get('lista_steps', 5)),
                )
                model.load_state_dict(torch.load(ckpt, map_location=device))
                report = run_dictionary_diagnostics(
                    model, fold_dir, device=device, num_classes=num_classes,
                )
                rows.append(_row_from_report('dual_dictionary', fold_idx, report))
                boot = report.get('bootstrap_age_calibration') or {}
                rows[-1]['delta_auc_bootstrap'] = boot.get('delta_auc')
                rows[-1]['delta_auc_ci_low'] = boot.get('ci_low')
                rows[-1]['delta_auc_ci_high'] = boot.get('ci_high')
                rows[-1]['delta_auc_age_calibration'] = report.get(
                    'age_calibrated_classifier', {},
                ).get('delta_auc_age_calibration')

    p1_tsv = causal_phase1_tsv or baseline_cfg.get('phase1_tsv')
    p2_tsv = causal_phase2_tsv or baseline_cfg.get('phase2_tsv')
    if p1_tsv:
        p1_rows = load_causal_decomposition_tsv(p1_tsv)
        agg = aggregate_causal_baseline_rows(p1_rows)
        rows.append({'baseline': 'phase1_causal', 'fold': 'all', **agg})
    if p2_tsv:
        p2_rows = load_causal_decomposition_tsv(p2_tsv)
        agg = aggregate_causal_baseline_rows(p2_rows)
        rows.append({'baseline': 'phase2_causal', 'fold': 'all', **agg})

    if out_dir is None:
        out_dir = j(run_root or cache_root, 'baseline_comparison')
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = j(out_dir, 'comparison_table.tsv')
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    json_path = j(out_dir, 'comparison_table.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"Saved baseline comparison → {tsv_path}")
    return rows, tsv_path


def main():
    parser = argparse.ArgumentParser(description='Compare dictionary MVP baselines')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--cache_root', type=str, required=True)
    parser.add_argument('--run_root', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--phase1_tsv', type=str, default=None)
    parser.add_argument('--phase2_tsv', type=str, default=None)
    args = parser.parse_args()

    compare_dictionary_baselines(
        config_path=args.config_path,
        cache_root=args.cache_root,
        run_root=args.run_root,
        device=args.device,
        fold=args.fold,
        out_dir=args.out_dir,
        causal_phase1_tsv=args.phase1_tsv,
        causal_phase2_tsv=args.phase2_tsv,
    )


if __name__ == '__main__':
    main()
