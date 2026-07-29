"""Supervised dual-dictionary MVP training entry point."""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.trainer_dictionary import train_dictionary_cv


def main():
    parser = argparse.ArgumentParser(description='Train dual-dictionary model on cached features')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache_root', type=str, default=None,
                        help='Path to dictionary_feature_cache run directory')
    parser.add_argument('--fold', type=int, default=None, help='Train a single outer fold')
    args = parser.parse_args()

    train_dictionary_cv(
        config_path=args.config_path,
        device=args.device,
        cache_root=args.cache_root,
        fold=args.fold,
    )


if __name__ == '__main__':
    main()
