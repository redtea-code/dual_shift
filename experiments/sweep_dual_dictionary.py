"""Grid sweep for dictionary MVP ablations (encoder, k_d/k_c, loss presets)."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime
from os.path import join as j

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.trainer_dictionary import train_dictionary_cv
from utils.config import load_config


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    import numpy as np
    return float(np.mean(vals)), float(np.std(vals))


def run_dictionary_sweep(
        config_path: str = 'config/default.yaml',
        device: str = 'cuda',
        cache_root: str = None,
        preset_name: str = None,
        fold: int = None,
):
    cf = load_config(config_path)
    dict_cfg = cf.get('dictionary') or {}
    sweep_cfg = dict_cfg.get('sweep') or {}
    presets = dict_cfg.get('ablation_presets') or {}

    if preset_name:
        preset_list = [presets[preset_name]]
        names = [preset_name]
    else:
        names = list(presets.keys())
        preset_list = [presets[n] for n in names]

    if not preset_list:
        # default minimal sweep if no presets configured
        names = ['default']
        preset_list = [{}]

    all_rows = []
    sweep_root = j(
        cf['project_dir'], 'dictionary_sweep',
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(sweep_root, exist_ok=True)

    for name, preset in zip(names, preset_list):
        override = copy.deepcopy(preset)
        override['run_suffix'] = name
        print(f"\n=== Sweep preset: {name} ===")
        run_root, results = train_dictionary_cv(
            config_path=config_path,
            device=device,
            cache_root=cache_root,
            fold=fold,
            dict_cfg_override=override,
        )
        aucs = [r.get('test_auc') for r in results]
        pr2 = [r.get('partial_r2_age') for r in results]
        auc_mean, auc_std = _mean_std(aucs)
        pr2_mean, pr2_std = _mean_std(pr2)
        row = {
            'preset': name,
            'run_root': run_root,
            'test_auc_mean': auc_mean,
            'test_auc_std': auc_std,
            'partial_r2_age_mean': pr2_mean,
            'partial_r2_age_std': pr2_std,
            'n_folds': len(results),
        }
        row.update({k: v for k, v in override.items() if k != 'run_suffix'})
        all_rows.append(row)

    tsv_path = j(sweep_root, 'comparison_table.tsv')
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)

    json_path = j(sweep_root, 'comparison_table.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_rows, f, indent=2, default=str)

    print(f"\nSweep complete → {sweep_root}")
    return sweep_root, all_rows


def main():
    parser = argparse.ArgumentParser(description='Dictionary MVP ablation sweep')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache_root', type=str, default=None)
    parser.add_argument('--preset', type=str, default=None,
                        help='Run a single ablation preset by name')
    parser.add_argument('--fold', type=int, default=None)
    args = parser.parse_args()

    run_dictionary_sweep(
        config_path=args.config_path,
        device=args.device,
        cache_root=args.cache_root,
        preset_name=args.preset,
        fold=args.fold,
    )


if __name__ == '__main__':
    main()
