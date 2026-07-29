"""
V4 主入口 — 统一调度各训练脚本。

Usage:
  # Subject-aware 标准训练（train_norm_v2）
  python run_v2.py --exp norm                          # 默认：按 config 选择 CV / single
  python run_v2.py --exp norm --cv                   # Subject-aware K 折 CV
  python run_v2.py --exp norm --single --fold 3      # 指定 fold 单次实验

  # 年龄泛化验证（train_age）：最年长 20% 测试 + 年轻受试者 K 折 CV
  python run_v2.py --exp age --model resnet18_backdoor
  python run_v2.py --exp age --model resnet18_backdoor --num_folds 5

  # 因果表示学习（train_causal）：Phase 1 分解实验
  python run_v2.py --exp causal --model resnet18_disentangled
  python run_v2.py --exp causal --model resnet18_disentangled --phase 1

  # 对比方法（train_comparison）
  python run_v2.py --exp comparison --model concat_fusion --cv
  python run_v2.py --exp comparison --list

  # 监督双字典 MVP（需先缓存特征）
  python run_v2.py --exp dictionary --action extract
  python run_v2.py --exp dictionary --action train
  python run_v2.py --exp dictionary --action eval --run_root ... --cache_root ...
  python run_v2.py --exp dictionary --action compare --cache_root ... --run_root ...
  python run_v2.py --exp dictionary --action sweep --cache_root ...

  # 期刊稳健诊断实验（train_journal）
  python run_v2.py --exp journal --smoke-test --device cpu
  python run_v2.py --exp journal --direction ADNI_to_NACC --variants ce_only groupdro
  python run_v2.py --exp journal --study --directions ADNI_to_NACC NACC_to_ADNI
"""
import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.debug_log import agent_log


def _build_argv(args, extra_flags=(), *, include_model=True):
    """重建 sys.argv 供子模块 argparse 消费。"""
    argv = [sys.argv[0]]
    if args.config_path != 'config/default.yaml':
        argv.extend(['--config_path', args.config_path])
    if include_model and args.model:
        argv.extend(['--model', args.model])
    argv.extend(['--device', args.device])
    for flag in extra_flags:
        argv.append(flag)
    if args.fold is not None:
        argv.extend(['--fold', str(args.fold)])
    if args.num_folds is not None:
        argv.extend(['--num_folds', str(args.num_folds)])
    argv.extend(args.extra)
    return argv


def main():
    parser = argparse.ArgumentParser(description='Multi-modal Classification V4')
    parser.add_argument(
        '--exp', type=str, default='dictionary',
        choices=['norm', 'age', 'comparison', 'causal', 'dictionary', 'journal'],
        help='Experiment type: norm | age | comparison | causal | dictionary | journal',
    )
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default=None,
                        help='Model variant (default depends on --exp)')
    parser.add_argument('--device', type=str, default='cuda')

    # norm / comparison 共用
    parser.add_argument('--cv', action='store_true',
                        help='Cross-validation (norm: --cross-val; comparison: --cv)')
    parser.add_argument('--single', action='store_true',
                        help='Single train/val experiment')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index, 1-based (with --single)')

    # age / causal 共用
    parser.add_argument('--num_folds', type=int, default=None,
                        help='K-fold count for --exp age or causal')
    parser.add_argument('--phase', type=int, default=None,
                        help='Causal phase (1–5) for --exp causal')

    # dictionary MVP
    parser.add_argument('--action', type=str, default='train',
                        choices=['extract', 'train', 'eval', 'compare', 'sweep'],
                        help='dictionary pipeline: extract | train | eval | compare | sweep')

    # comparison 专用
    parser.add_argument('--list', action='store_true',
                        help='List comparison models and exit')

    args, unknown = parser.parse_known_args()
    args.extra = unknown

    # #region agent log
    agent_log(
        'run_v2.py:main',
        'parsed CLI args',
        {'exp': args.exp, 'action': getattr(args, 'action', None), 'fold': args.fold, 'model': args.model},
        hypothesis_id='H2',
    )
    # #endregion

    if args.model is None:
        defaults = {
            'norm': 'resnet18_daft',
            'age': 'resnet18_backdoor',
            'comparison': 'madformer',
            'causal': 'resnet18_disentangled',
            'dictionary': 'resnet10_ce_only',
            'journal': None,
        }
        args.model = defaults.get(args.exp)

    if args.exp == 'norm':
        extra = []
        if args.cv:
            extra.append('--cross-val')
        if args.single:
            extra.append('--single')
        sys.argv = _build_argv(args, extra_flags=extra)
        from experiments.train_norm_v2 import main as entry_main

    elif args.exp == 'age':
        if args.cv or args.single or args.fold is not None:
            print("[run_v2] --exp age ignores --cv / --single / --fold "
                  "(uses fixed age test split + K-fold on younger subjects).")
        sys.argv = _build_argv(args)
        from experiments.train_age import main as entry_main

    elif args.exp == 'causal':
        if args.cv or args.single or args.fold is not None:
            print("[run_v2] --exp causal ignores --cv / --single / --fold "
                  "(uses age-stratified test split + K-fold CV).")
        argv = _build_argv(args)
        if args.phase is not None:
            argv.extend(['--phase', str(args.phase)])
        sys.argv = argv
        from experiments.train_causal import main as entry_main

    elif args.exp == 'dictionary':
        if args.action == 'extract':
            sys.argv = _build_argv(args)
            from experiments.extract_dictionary_features import main as entry_main
        elif args.action == 'eval':
            sys.argv = _build_argv(args, include_model=False)
            from experiments.eval_dual_dictionary import main as entry_main
        elif args.action == 'compare':
            sys.argv = _build_argv(args, include_model=False)
            from experiments.compare_dictionary_baselines import main as entry_main
        elif args.action == 'sweep':
            sys.argv = _build_argv(args, include_model=False)
            from experiments.sweep_dual_dictionary import main as entry_main
        else:
            sys.argv = _build_argv(args, include_model=False)
            from experiments.train_dual_dictionary import main as entry_main

    elif args.exp == 'comparison':
        extra = []
        if args.cv:
            extra.append('--cv')
        if args.single:
            extra.append('--single')
        if args.list:
            extra.append('--list')
        sys.argv = _build_argv(args, extra_flags=extra)
        from experiments.train_comparison import main as entry_main

    elif args.exp == 'journal':
        # Journal owns its argparse; forward unknown flags and default to journal config.
        if args.config_path in {'config/default.yaml', 'config/journal.yaml'}:
            argv = [sys.argv[0], '--config_path', 'config/journal.yaml']
        else:
            argv = [sys.argv[0], '--config_path', args.config_path]
        argv.extend(['--device', args.device])
        argv.extend(args.extra)
        sys.argv = argv
        from experiments.train_journal import main as entry_main

    else:
        parser.error(f"Unknown exp: {args.exp}")

    entry_main()


if __name__ == '__main__':
    main()
