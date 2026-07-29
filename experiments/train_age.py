"""
Age-Stratified Generalization Training Entry Point.

结果统一写入 {project_dir}/age_cv_summary/{dataset}/{model}_task{task}/：
  summary/          汇总（overall_summary.txt, test_per_fold.tsv）
  logs/             运行日志（age_cv_log.txt）
  fold_N/           该折训练权重 + test/metrics.txt + test/confusion_matrix.txt

Usage:
  python -m experiments.train_age --model resnet18_backdoor
  python -m experiments.train_age --model hyperfusion
  python -m experiments.train_age --list
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config
from experiments.model_registry import build_model, ALL_MODELS, print_model_list
from training.trainer_age import train_age_cv


def main():
    parser = argparse.ArgumentParser(
        description='Age-Stratified Generalization Training (K-fold CV + age hold-out test)')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='resnet18_backdoor')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--score_prior', action='store_true',
                        help='Enable patch score prior (backbone models only)')
    parser.add_argument('--list', action='store_true', help='List available models')

    args = parser.parse_args()

    if args.list:
        print_model_list()
        return

    if args.model not in ALL_MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {sorted(ALL_MODELS)}")
        sys.exit(1)

    cf = load_config(args.config_path)
    model_cls, model_kwargs, forward_fn, use_score_prior = build_model(args.model, cf)
    use_score_prior = use_score_prior and args.score_prior

    print(f"\n{'='*60}")
    print(f"  Age Generalization | {args.model}"
          f" ({'comparison' if forward_fn else 'backbone'})")
    print(f"  Config: {args.config_path}")
    print(f"{'='*60}")

    train_age_cv(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        config_path=args.config_path,
        device=args.device,
        use_score_prior=use_score_prior,
        forward_fn=forward_fn,
        loss_weights=cf.get('loss_weights'),
        exp_type='age',
        model_name=args.model,
        num_folds=args.num_folds,
    )


if __name__ == '__main__':
    main()
