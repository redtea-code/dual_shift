"""
Comparison Methods — Training Entry Point.

Usage:
  python -m experiments.train_comparison --model hyperfusion           # CV (default)
  python -m experiments.train_comparison --model madformer --single    # Single exp
  python -m experiments.train_comparison --model cgcn --single --fold 3
  python -m experiments.train_comparison --model mlp                   # Tabular only
  python -m experiments.train_comparison --list                        # List available models
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Model.comparison.factories import MODEL_REGISTRY
from training.trainer_comparison import (
    TrainComparison, train_comparison_cv, train_comparison_single,
)


def main():
    parser = argparse.ArgumentParser(
        description='Comparison Methods Training (V4 Subject-Aware)')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='madformer',
                        help='Model name from registry')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--cv', dest='cross_val', action='store_true',
                        help='Subject-aware cross-validation')
    parser.add_argument('--single', action='store_true',
                        help='Single train/test experiment')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index (1-based, with --single)')

    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override train batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--list', action='store_true',
                        help='List available models and exit')

    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: 2 epochs, no checkpoint saving')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable comparison models:")
        print(f"{'Model':<22} {'Modalities':<18} {'Factory params'}")
        print("-" * 70)
        for name, (factory_fn, kwargs) in MODEL_REGISTRY.items():
            # Infer modalities from factory name
            if 'vit3d' in name or name in ('madformer', 'msgcfnet', 'cgcn',
                    'causal_gcn_pdpa', 'cfgnn', 'causal_mix_net',
                    'causal_patch_3', 'causal_patch_6'):
                mods = 'MRI'
            elif name in ('tabformer', 'mlp'):
                mods = 'TABULAR'
            elif name in ('hyperfusion', 'top_san', 'concat_fusion', 'cross_attention_fusion'):
                mods = 'MRI + TABULAR'
            else:
                mods = '?'
            print(f"  {name:<20} {mods:<18} {kwargs}")
        print()
        return

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        print("Use --list to see all models with details.")
        sys.exit(1)

    factory_fn, model_kwargs = MODEL_REGISTRY[args.model]
    model_kwargs = dict(model_kwargs)  # copy to avoid mutating registry

    # ---- Config overrides ----
    from utils.config import load_config
    cf = load_config(args.config_path)

    # Update table_feature from config if available
    table_feature = cf.get('table_feature', 0)
    if table_feature > 0:
        if 'num_features' in model_kwargs:
            model_kwargs['num_features'] = table_feature
        if 'num_tabular' in model_kwargs:
            model_kwargs['num_tabular'] = table_feature
        if 'in_features' in model_kwargs:
            model_kwargs['in_features'] = table_feature

    # Debug mode
    if args.debug:
        cf['num_epochs'] = 2
        cf['val_inter'] = 1
        cf['save_inter'] = 10
        cf['project_dir'] = 'weights/comparison_debug'
        print(f"\n[DEBUG MODE] 2 epochs, no full training")

    # Epochs override
    if args.epochs is not None:
        cf['num_epochs'] = args.epochs
    if args.batch_size is not None:
        cf['train_bc'] = args.batch_size

    # ---- Build model ----
    print(f"\n{'='*60}")
    print(f"  Comparison Method: {args.model}")
    print(f"  Factory kwargs: {model_kwargs}")
    print(f"{'='*60}")

    # 实例化模型（先创建一次获取 model_cls 的引用）
    model_instance = factory_fn(**model_kwargs)

    # ---- 选择训练模式 ----
    if args.cross_val:
        train_comparison_cv(
            model_cls=type(model_instance),
            model_kwargs=model_kwargs,
            config_path=args.config_path,
            device=args.device,
            exp_type='comparison',
            model_name=args.model,
        )
    elif args.single:
        train_comparison_single(
            model_cls=type(model_instance),
            model_kwargs=model_kwargs,
            config_path=args.config_path,
            device=args.device,
            exp_type='comparison',
            model_name=args.model,
            fold=args.fold,
        )
    else:
        TrainComparison(
            model_cls=type(model_instance),
            model_kwargs=model_kwargs,
            config_path=args.config_path,
            device=args.device,
            exp_type='comparison',
            model_name=args.model,
        )


if __name__ == '__main__':
    main()
