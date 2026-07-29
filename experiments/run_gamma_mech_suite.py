"""Build gamma-mechanism experiment configs and run the std5cv suite.

Usage:
  # Write all YAML overlays under config/gamma_mech/
  python -m experiments.run_gamma_mech_suite --write_configs_only

  # Run Wave 0 then Wave 1 (full 5-fold)
  python -m experiments.run_gamma_mech_suite --device cuda

  # Run a single experiment id
  python -m experiments.run_gamma_mech_suite --only W0_P1 --device cuda
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


CONFIG_DIR = j(PROJECT_ROOT, 'config', 'gamma_mech')


def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def experiment_specs(phase1_run_dir: str = None):
    """Return ordered list of (exp_id, model_entry, overrides, notes).

    model_entry: 'causal:resnet10_disentangled' | 'age:resnet10_film' | 'age:resnet10_ce_only'
    """
    base_descripe = 'dim=64_orth=0.05_age=0.01'
    # After W0_P1, phase1 lives under this pattern (descripe gets _std5cv appended by trainer)
    if phase1_run_dir is None:
        phase1_run_dir = (
            'weights/classifier/causal_cv_summary/ADNI/'
            'resnet10_disentangled_task13_dim=64_orth=0.05_age=0.01_mech_W0_P1_std5cv/phase1'
        )

    freeze_init = {
        'enabled': True,
        'source_run_dir': phase1_run_dir,
        'strict': False,
        'lr': 1.0e-5,
        'freeze_loaded': True,
        'trainable_prefixes': [
            'confounder_encoder', 'confounder_mapper', 'gamma_head',
            'gamma_logit', 'linear_age_gate', 'fc',
        ],
    }

    def p2(desc_suffix, extra_causal=None, cs=0.01):
        causal = {
            'phase': 2,
            'phase_init': freeze_init,
            'loss_weights': {
                'orthogonality': 0.05,
                'age_prediction': 0.01,
                'age_adversarial': 0.0,
                'confounder_supervision': cs,
                'intervention_consistency': 0.0,
                'invariance': 0.0,
            },
            'gamma_mech': {'mode': 'learned', 'constant_value': 0.5, 'shuffle_seed': 0},
            'fusion_mode': 'additive_gate',
        }
        if extra_causal:
            causal = _deep_update(causal, extra_causal)
        return {
            'age_split': {'test_ratio': 0},
            'descripe': f'{base_descripe}_{desc_suffix}',
            'causal': causal,
            'run_c2_before_train': False,
        }

    specs = [
        (
            'W0_P1',
            'causal:resnet10_disentangled',
            {
                'age_split': {'test_ratio': 0},
                'descripe': f'{base_descripe}_mech_W0_P1',
                'causal': {
                    'phase': 1,
                    'phase_init': {'enabled': False},
                    'loss_weights': {
                        'orthogonality': 0.05,
                        'age_prediction': 0.01,
                        'age_adversarial': 0.0,
                        'confounder_supervision': 0.0,
                        'intervention_consistency': 0.0,
                        'invariance': 0.0,
                    },
                },
            },
            'Phase1 init source',
        ),
        (
            'W0_CE',
            'age:resnet10_ce_only',
            {
                'age_split': {'test_ratio': 0},
                'descripe': f'{base_descripe}_mech_W0_CE',
                'table_feature': 0,
            },
            'Imaging-only CE baseline',
        ),
        (
            'W0_FiLM',
            'age:resnet10_film',
            {
                'age_split': {'test_ratio': 0},
                'descripe': f'{base_descripe}_mech_W0_FiLM',
                'table_feature': 1,
            },
            'FiLM tabular baseline',
        ),
        (
            'W0_Main',
            'causal:resnet10_disentangled',
            p2('mech_W0_Main', cs=0.01),
            'Freeze-gamma main method',
        ),
        (
            'A1',
            'causal:resnet10_disentangled',
            p2('mech_A1_constant', extra_causal={
                'gamma_mech': {'mode': 'constant', 'constant_value': 0.5},
            }),
            'Constant gamma=0.5',
        ),
        (
            'A3',
            'causal:resnet10_disentangled',
            p2('mech_A3_shuffle', extra_causal={
                'gamma_mech': {'mode': 'shuffle', 'shuffle_seed': 0},
            }),
            'Batch shuffle age/txt',
        ),
        (
            'A5a',
            'causal:resnet10_disentangled',
            p2('mech_A5a_zeros', extra_causal={
                'gamma_mech': {'mode': 'zeros'},
            }),
            'gamma=0 → Fd+Fc',
        ),
        (
            'A5b',
            'causal:resnet10_disentangled',
            p2('mech_A5b_ones', extra_causal={
                'gamma_mech': {'mode': 'ones'},
            }),
            'gamma=1 → Fd only',
        ),
        (
            'A6',
            'causal:resnet10_disentangled',
            p2('mech_A6_concat', extra_causal={
                'fusion_mode': 'concat_fc',
                'gamma_mech': {'mode': 'learned'},
            }),
            'Late concat Fd,Fc,age',
        ),
        (
            'A8',
            'causal:resnet10_disentangled',
            p2('mech_A8_nocs', cs=0.0),
            'confounder_supervision=0',
        ),
        (
            'A10',
            'causal:resnet10_disentangled',
            _deep_update(
                p2('mech_A10_scratch', cs=0.01),
                {'causal': {'phase_init': {'enabled': False, 'freeze_loaded': False}}},
            ),
            'Phase2 scratch no P1 init',
        ),
    ]
    return specs


def write_configs(base_config_path: str, phase1_run_dir: str = None):
    base = load_config(base_config_path)
    # yaml round-trip: convert tuples back to lists for dump
    def _sanitize(obj):
        if isinstance(obj, tuple):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        return obj

    os.makedirs(CONFIG_DIR, exist_ok=True)
    written = []
    for exp_id, _model, overrides, notes in experiment_specs(phase1_run_dir):
        cf = _sanitize(_deep_update(base, overrides))
        # Disable C2 preflight for speed on ablations (P1 still useful once)
        if 'causal' in cf and exp_id != 'W0_P1':
            cf['causal']['run_c2_before_train'] = False
        path = j(CONFIG_DIR, f'{exp_id}.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'# Gamma mech experiment {exp_id}: {notes}\n')
            yaml.safe_dump(cf, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        written.append(path)
        print(f'  wrote {path}')
    return written


def _run_one(exp_id, model_entry, config_path, device):
    kind, model_name = model_entry.split(':', 1)
    if kind == 'causal':
        from experiments.train_causal import main as _unused  # noqa: F401
        from experiments.model_registry import build_causal_model
        from training.trainer_causal import train_causal_cv
        from utils.config import load_config as _lc
        cf = _lc(config_path)
        phase = int((cf.get('causal') or {}).get('phase', 1))
        model_cls, model_kwargs = build_causal_model(model_name, cf, causal_phase=phase)
        print(f'\n===== RUN {exp_id} | causal phase={phase} | {model_name} =====\n')
        train_causal_cv(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            config_path=config_path,
            device=device,
            model_name=model_name,
            causal_phase=phase,
        )
    elif kind == 'age':
        from experiments.model_registry import build_model
        from training.trainer_age import train_age_cv
        from utils.config import load_config as _lc
        cf = _lc(config_path)
        model_cls, model_kwargs, _, _ = build_model(model_name, cf)
        print(f'\n===== RUN {exp_id} | age-cv | {model_name} =====\n')
        train_age_cv(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            config_path=config_path,
            device=device,
            model_name=model_name,
        )
    else:
        raise ValueError(model_entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/default.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--write_configs_only', action='store_true')
    parser.add_argument('--only', nargs='*', default=None,
                        help='Experiment ids to run, e.g. W0_P1 W0_Main A3')
    parser.add_argument('--phase1_run_dir', default=None,
                        help='Override Phase1 source_run_dir for freeze experiments')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip if overall_summary already exists')
    args = parser.parse_args()

    print('Writing gamma_mech configs…')
    write_configs(args.config_path, phase1_run_dir=args.phase1_run_dir)
    if args.write_configs_only:
        return

    specs = experiment_specs(args.phase1_run_dir)
    only = set(args.only) if args.only else None
    for exp_id, model_entry, _overrides, notes in specs:
        if only is not None and exp_id not in only:
            continue
        config_path = j(CONFIG_DIR, f'{exp_id}.yaml')
        print(f'\n### {exp_id}: {notes}')
        _run_one(exp_id, model_entry, config_path, args.device)


if __name__ == '__main__':
    main()
