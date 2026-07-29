"""
Causal Representation Learning — Phase 1+ entry point.

结果写入 {project_dir}/causal_cv_summary/{dataset}/{model}_task{task}_{descripe}/phase{N}/：
  summary/              汇总 + decomposition_summary.txt
  logs/                 causal_cv_log.txt
  fold_N/
    test/               metrics + decomposition.txt + gamma_stats.json

切分模式（由 age_split.test_ratio 控制）：
  test_ratio > 0  年龄留出测试集 + 年轻池 K 折 CV（与历史 train_age 相同）
  test_ratio = 0  全量 subject-aware 5 折，每折 ~60/20/20 train/val/test；
                  选模用 val，正式指标只报 test（std5cv 范式）

Usage:
  python -m experiments.train_causal --model resnet10_disentangled --phase 1
  python -m experiments.run_gamma_mech_suite --only W0_P1 W0_Main
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config
from experiments.model_registry import build_causal_model, CAUSAL_MODELS, print_causal_model_list
from training.trainer_causal import train_causal_cv


def main():
    parser = argparse.ArgumentParser(
        description='Causal Representation Learning (phased construction)')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='resnet18_disentangled')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--phase', type=int, default=None,
                        help='Causal construction phase (1–5). Default: config causal.phase')
    parser.add_argument('--list', action='store_true', help='List causal models')

    args = parser.parse_args()

    if args.list:
        print_causal_model_list()
        return

    if args.model not in CAUSAL_MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {sorted(CAUSAL_MODELS)}")
        sys.exit(1)

    cf = load_config(args.config_path)
    causal_cfg = cf.get('causal') or {}
    phase = args.phase if args.phase is not None else causal_cfg.get('phase', 1)

    if phase > 1:
        print(f"[train_causal] Phase {phase}: gamma/confounder modules are implemented.")
        print("  Use causal.phase_init to load Phase 1 checkpoints for stable Phase 2 runs.")

    model_cls, model_kwargs = build_causal_model(args.model, cf, causal_phase=phase)

    print(f"\n{'='*60}")
    print(f"  Causal Rep Learning | Phase {phase} | {args.model}")
    print(f"  Config: {args.config_path}")
    print(f"{'='*60}")

    train_causal_cv(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        config_path=args.config_path,
        device=args.device,
        model_name=args.model,
        num_folds=args.num_folds,
        causal_phase=phase,
    )


if __name__ == '__main__':
    main()
