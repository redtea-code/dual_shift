"""
C2 — Age confound validation (independent of causal trainer).

Uses registered backbone models to probe whether age is a meaningful confounder
before / alongside disentangled training.

Usage:
  python -m experiments.analyze_age_confound
  python -m experiments.analyze_age_confound --baseline resnet18_film_feature
  python -m experiments.analyze_age_confound --list
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config
from training.trainer_age import load_age_dataset
from experiments.model_registry import (
    build_age_confound_model, print_age_confound_model_list, AGE_CONFOUND_MODELS,
)
from Model.causal.age_confound import (
    run_age_confound_analysis, save_age_confound_report, format_age_confound_report,
)


def main():
    parser = argparse.ArgumentParser(description='Age confound validation (C2)')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument(
        '--baseline', type=str, default='resnet18_film_feature',
        help='Registered model for age←feature probe (see --list)',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip baseline feature probe (dataset-only stats)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output dir (default: project_dir/age_confound_summary/...)')
    parser.add_argument('--list', action='store_true', help='List registered analysis models')

    args = parser.parse_args()

    if args.list:
        print_age_confound_model_list()
        return

    cf = load_config(args.config_path)
    dataset = load_age_dataset(cf)
    dataset_name = cf.get('dataset', 'NACC')
    class_task_str = dataset.class_task_str
    causal_cfg = cf.get('causal') or {}

    baseline_model = None
    if not args.no_baseline:
        try:
            model_cls, model_kwargs = build_age_confound_model(args.baseline, cf)
            model_kwargs = dict(model_kwargs)
            model_kwargs['num_classes'] = dataset.num_classes
            baseline_model = model_cls(**model_kwargs).to(args.device)
            print(f"  Baseline probe model: {args.baseline}")
        except KeyError:
            print(f"Unknown baseline model: {args.baseline}")
            print(f"Registered: {sorted(AGE_CONFOUND_MODELS)}")
            sys.exit(1)

    report = run_age_confound_analysis(
        dataset,
        device=args.device,
        indices=list(range(len(dataset))),
        baseline_model=baseline_model,
        eval_bc=cf.get('eval_bc', 1),
        n_age_bins=cf.get('causal', {}).get('age_confound_bins', 3),
        probe_cfg=causal_cfg,
    )

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            cf['project_dir'], 'age_confound_summary',
            dataset_name, f'task{class_task_str}',
        )

    path = save_age_confound_report(report, out_dir)
    print(format_age_confound_report(report))
    print(f"\nSaved → {path}")


if __name__ == '__main__':
    main()
