"""Build and run ResNet+γ vs FiLM/DAFT/HyperFusion/Concat suite (std5cv, no Phase1).

Usage:
  python -m experiments.run_gamma_fusion_suite --write_configs_only
  python -m experiments.run_gamma_fusion_suite --device cuda
  python -m experiments.run_gamma_fusion_suite --only B0 B_gamma B_film --device cuda
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from os.path import join as j

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config

CONFIG_DIR = j(PROJECT_ROOT, 'config', 'gamma_fusion')

TASK_LABELS = {
    '13': 'CN vs AD',
    '23': 'MCI vs AD',
    '12': 'CN vs MCI (NC vs MCI)',
}


def _config_dir_for_task(class_task: str) -> str:
    return j(CONFIG_DIR, f'task{class_task}')


def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _sanitize(obj):
    if isinstance(obj, tuple):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def _default_backdoor_kwargs(**extra):
    kw = dict(
        return_gamma=True,
        gamma_range=True,
        gamma_mode='patch',
        shuffle_tabular=False,
        shuffle_seed=0,
        gamma_dropout=0.2,
        gamma_dropout_rescale=True,
        group_sharing=200,
        spatial_smooth_mode='both',
        spatial_smooth_lambda=0.01,
        spatial_filter_alpha=0.4,
    )
    kw.update(extra)
    return kw


def experiment_specs(class_task: str = '13'):
    """Ordered (exp_id, model_name, overrides, notes)."""
    tag = 'gamma_fusion'
    common = {
        'age_split': {'test_ratio': 0},
        'film_stages': 'last',
        'class_task': str(class_task),
    }

    def desc(suffix):
        return f'{tag}_{suffix}'

    specs = [
        (
            'B0',
            'resnet10_ce_only',
            {
                **common,
                'descripe': desc('B0_ce'),
                'table_feature': 0,
                'loss_weights': {},
            },
            'Imaging-only CE lower bound',
        ),
        (
            'B_gamma',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('B_gamma'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(),
                'loss_weights': {
                    'backdoor_sparsity': 0.1,
                    'backdoor_smoothness': 0.2,
                },
            },
            'Main: ResNet+patch-gamma (default regs)',
        ),
        (
            'B_film',
            'resnet10_film',
            {
                **common,
                'descripe': desc('B_film'),
                'table_feature': 1,
                'film_stages': 'last',
                'loss_weights': {},
            },
            'FiLM channel affine',
        ),
        (
            'B_daft',
            'resnet10_daft',
            {
                **common,
                'descripe': desc('B_daft'),
                'table_feature': 1,
                'loss_weights': {},
            },
            'DAFT affine',
        ),
        (
            'B_hyper',
            'hyperfusion',
            {
                **common,
                'descripe': desc('B_hyper'),
                'table_feature': 1,
                'loss_weights': {},
            },
            'HyperFusion',
        ),
        (
            'B_concat',
            'concat_fusion',
            {
                **common,
                'descripe': desc('B_concat'),
                'table_feature': 1,
                'loss_weights': {},
            },
            'Concat late fusion',
        ),
        # Wave 1 — loss ablations on gamma
        (
            'L0',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('L0_ce_only'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(
                    gamma_dropout=0.0,
                    spatial_smooth_mode='none',
                    spatial_smooth_lambda=0.0,
                ),
                'loss_weights': {
                    'backdoor_sparsity': 0.0,
                    'backdoor_smoothness': 0.0,
                },
            },
            'Gamma CE-only (no sparsity/smooth/dropout)',
        ),
        (
            'L1',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('L1_sparsity'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(
                    gamma_dropout=0.0,
                    spatial_smooth_mode='none',
                    spatial_smooth_lambda=0.0,
                ),
                'loss_weights': {
                    'backdoor_sparsity': 0.1,
                    'backdoor_smoothness': 0.0,
                },
            },
            'Gamma + sparsity only',
        ),
        (
            'L2',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('L2_smooth'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(gamma_dropout=0.0),
                'loss_weights': {
                    'backdoor_sparsity': 0.0,
                    'backdoor_smoothness': 0.2,
                },
            },
            'Gamma + smoothness only',
        ),
        (
            'L3',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('L3_full'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(),
                'loss_weights': {
                    'backdoor_sparsity': 0.1,
                    'backdoor_smoothness': 0.2,
                },
            },
            'Alias of B_gamma (full regs)',
        ),
        # Wave 2 — architecture
        (
            'A_global',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('A_global'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(
                    gamma_mode='global',
                    group_sharing=0,
                    spatial_smooth_mode='none',
                    spatial_smooth_lambda=0.0,
                ),
                'loss_weights': {
                    'backdoor_sparsity': 0.1,
                    'backdoor_smoothness': 0.0,
                },
            },
            'Global scalar gamma',
        ),
        (
            'A_noshare',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('A_noshare'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(group_sharing=0),
                'loss_weights': {
                    'backdoor_sparsity': 0.1,
                    'backdoor_smoothness': 0.2,
                },
            },
            'Patch gamma without group sharing',
        ),
        (
            'A_shuffle',
            'resnet10_backdoor',
            {
                **common,
                'descripe': desc('A_shuffle'),
                'table_feature': 1,
                'backdoor_kwargs': _default_backdoor_kwargs(shuffle_tabular=True),
                'loss_weights': {
                    'backdoor_sparsity': 0.1,
                    'backdoor_smoothness': 0.2,
                },
            },
            'Shuffle tabular within batch',
        ),
    ]
    return specs


def write_configs(base_config_path: str, tasks=None):
    tasks = tasks or ['13']
    written = []
    base = load_config(base_config_path)
    for class_task in tasks:
        out_dir = _config_dir_for_task(class_task)
        os.makedirs(out_dir, exist_ok=True)
        for exp_id, _model, overrides, notes in experiment_specs(class_task):
            cf = _sanitize(_deep_update(base, overrides))
            if 'causal' in cf:
                cf['causal']['phase'] = 1
                cf['causal']['phase_init'] = {'enabled': False}
                cf['causal']['run_c2_before_train'] = False
            path = j(out_dir, f'{exp_id}.yaml')
            label = TASK_LABELS.get(str(class_task), class_task)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(
                    f'# Gamma-vs-fusion std5cv | task{class_task} ({label}) | '
                    f'{exp_id}: {notes}\n'
                )
                yaml.safe_dump(cf, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            written.append(path)
            print(f'  wrote {path}')
    return written


def _run_root_for(exp_id, model_name, config_path):
    """Mirror trainer_age run_root naming (incl. std5cv suffix)."""
    from utils.config import load_config as _lc
    from utils.cv_splitter_v2 import append_std5cv_descripe

    cf = _lc(config_path)
    age_cfg = cf.get('age_split') or {}
    test_ratio = age_cfg.get('test_ratio', 0.2)
    use_std = test_ratio is not None and float(test_ratio) <= 0
    descripe = cf.get('descripe', exp_id)
    if use_std:
        descripe = append_std5cv_descripe(descripe, test_ratio)
    else:
        descripe = cf.get('descripe')
    task = str(cf.get('class_task', '13')).replace("'", '')
    dataset = cf.get('dataset', 'ADNI')
    tag = f'{model_name}_task{task}'
    if descripe:
        tag = f'{tag}_{descripe}'
    return j(
        PROJECT_ROOT,
        cf.get('project_dir', 'weights/classifier'),
        'age_cv_summary',
        dataset,
        tag,
    )


def _is_complete(run_root: str) -> bool:
    summary = j(run_root, 'summary', 'overall_summary.txt')
    tsv = j(run_root, 'summary', 'test_per_fold.tsv')
    if not (os.path.isfile(summary) and os.path.isfile(tsv)):
        return False
    # Header-only TSV means summary write failed mid-run (e.g. metric key mismatch)
    with open(tsv, encoding='utf-8') as f:
        rows = [ln for ln in f.read().splitlines() if ln.strip()]
    return len(rows) >= 2


def _run_one(exp_id, model_name, config_path, device, skip_done=True):
    from experiments.model_registry import build_model
    from training.trainer_age import train_age_cv
    from utils.config import load_config as _lc

    cf = _lc(config_path)
    if skip_done:
        run_root = _run_root_for(exp_id, model_name, config_path)
        if _is_complete(run_root):
            print(f'\n### skip {exp_id} (already complete): {run_root}')
            return
    model_cls, model_kwargs, forward_fn, use_score_prior = build_model(model_name, cf)
    print(f'\n===== RUN {exp_id} | {model_name} =====\n')
    train_age_cv(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        config_path=config_path,
        device=device,
        use_score_prior=False,
        forward_fn=forward_fn,
        loss_weights=cf.get('loss_weights') or None,
        exp_type='age',
        model_name=model_name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/default.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--write_configs_only', action='store_true')
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument(
        '--wave', choices=['0', '1', '2', 'all'], default='all',
        help='0=head-to-head, 1=loss, 2=arch, all=everything',
    )
    parser.add_argument(
        '--tasks', nargs='*', default=['13'],
        help='class_task ids: 13=CN/AD, 23=MCI/AD, 12=CN/MCI. Example: --tasks 12 13 23',
    )
    parser.add_argument('--force', action='store_true', help='re-run even if summary exists')
    args = parser.parse_args()
    tasks = [str(t).replace("'", '') for t in args.tasks]

    print('Writing gamma_fusion configs...')
    write_configs(args.config_path, tasks=tasks)
    if args.write_configs_only:
        return

    wave0 = {'B0', 'B_gamma', 'B_film', 'B_daft', 'B_hyper', 'B_concat'}
    wave1 = {'L0', 'L1', 'L2', 'L3'}
    wave2 = {'A_global', 'A_noshare', 'A_shuffle'}
    if args.wave == '0':
        allow = wave0
    elif args.wave == '1':
        allow = wave1
    elif args.wave == '2':
        allow = wave2
    else:
        allow = wave0 | wave1 | wave2

    only = set(args.only) if args.only else None
    for class_task in tasks:
        label = TASK_LABELS.get(class_task, class_task)
        print(f'\n{"=" * 60}\n  TASK {class_task}: {label}\n{"=" * 60}')
        cfg_dir = _config_dir_for_task(class_task)
        scheduled = []
        for exp_id, model_name, _ov, notes in experiment_specs(class_task):
            if exp_id not in allow:
                continue
            if only is not None and exp_id not in only:
                continue
            scheduled.append((exp_id, model_name, notes))

        scheduled_ids = {e[0] for e in scheduled}
        for exp_id, model_name, notes in scheduled:
            if exp_id == 'L3' and 'B_gamma' in scheduled_ids and 'L3' not in (only or set()):
                print('\n### skip L3 (same as B_gamma; analyze will alias)')
                continue
            print(f'\n### {exp_id}: {notes}')
            try:
                _run_one(
                    exp_id, model_name, j(cfg_dir, f'{exp_id}.yaml'),
                    args.device, skip_done=not args.force,
                )
            except Exception as exc:
                print(f'\n!!! FAILED task{class_task} {exp_id}: {exc!r}')
                raise


if __name__ == '__main__':
    main()
