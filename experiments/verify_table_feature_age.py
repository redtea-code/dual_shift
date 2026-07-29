"""Verify table_feature=1 exposes only AGE (not full cognitive table)."""
from __future__ import annotations

import sys
from os.path import join as j

import numpy as np

ROOT = j(__file__, '..', '..')
sys.path.insert(0, ROOT)

from data.dataset_v2 import parse_class_task, get_class_names
from training.trainer_age import load_age_dataset
from utils.config import load_config


def audit_task(class_task: str, config_path: str):
    cf = load_config(config_path)
    cf['class_task'] = class_task
    ds = load_age_dataset(cf)
    cols = list(ds.table_df_prepared['conti_x'].columns)
    b = ds[0]
    cx = b['conti_x']
    keep = ds._conti_keep_idx
    picked = cols[keep[0]] if keep else None
    age_col = ds.table_df_prepared['conti_x'][picked] if picked else None
    ok = (
        int(cf.get('table_feature', 0)) == 1
        and cx.numel() == 1
        and picked in ('AGE_YEARS', 'AGE', 'Age', 'age')
    )
    ac = parse_class_task(class_task)
    dist = dict(zip(*np.unique(ds.all_labels, return_counts=True)))
    return {
        'class_task': class_task,
        'class_names': get_class_names(ac),
        'n_samples': len(ds),
        'n_subjects': len(ds.unique_subjects),
        'label_dist': dist,
        'n_cont_cols_raw': len(cols),
        'cont_cols': cols,
        'picked_col': picked,
        'conti_dim': int(cx.numel()),
        'age_only_ok': ok,
    }


def main():
    cfg = j(ROOT, 'config', 'gamma_fusion', 'B_gamma.yaml')
    print('=== table_feature=1 age audit ===\n')
    all_ok = True
    for task in ('13', '23', '12'):
        r = audit_task(task, cfg)
        print(f"task {task} ({' vs '.join(r['class_names'])})")
        print(f"  samples={r['n_samples']} subjects={r['n_subjects']} dist={r['label_dist']}")
        print(f"  raw cont cols ({r['n_cont_cols_raw']}): {r['cont_cols']}")
        print(f"  table_feature=1 -> {r['picked_col']!r}, conti_x dim={r['conti_dim']}")
        print(f"  PASS age-only: {r['age_only_ok']}\n")
        all_ok &= r['age_only_ok']
    if not all_ok:
        raise SystemExit('FAIL: table_feature=1 did not resolve to age column')
    print('ALL PASS: table_feature=1 is single age for tasks 12/13/23')


if __name__ == '__main__':
    main()
