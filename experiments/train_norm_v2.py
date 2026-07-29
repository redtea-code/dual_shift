"""
V4 — Subject-Aware 训练入口（NACC 数据集）。

Usage:
  python -m experiments.train_norm_v2                          # 默认 CV
  python -m experiments.train_norm_v2 --single                 # 单次实验
  python -m experiments.train_norm_v2 --single --fold 3        # 第 3 折
"""
import argparse
import os
import sys

from Model.backbone.preact_resnet import preact_resnet_t, preact_resnet_ut

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Model import resnet18_film, resnet18_daft
from Model.adpc import ADPC6_2
from Model.backbone.film_backbone import resnet18_e1, resnet10_film, resnet_light_film
from Model import (
    vit_tiny_film, vit_small_film,
    vit_tiny_daft, vit_small_daft,
    vit_tiny_backdoor, vit_small_backdoor,
    resnet10_backdoor, resnet18_backdoor, resnet34_backdoor,
)
from training.trainer_v2 import TrainV4, train_single_v4, train_cross_val_v4

MODELS = {
    'resnet18_film': (resnet18_film, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=True, film_stages='last')),
    'resnet10_film': (resnet10_film, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=True, film_stages='none')),
    'resnet_light_film': (resnet_light_film, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=False, feature=False, film_stages='none')),
    'preact_resnet_t': (preact_resnet_t, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=False, feature=False, film_stages='none')),
    'preact_resnet_ut': (preact_resnet_ut, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=False, feature=False, film_stages='none')),
    'resnet18_daft': (resnet18_daft, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=True, feature=False,film_stages='last')),
    'resnet18_e1': (resnet18_e1, dict(txt_dim=0, num_classes=3,
                                      pretrained_weights=True, feature=False,film_stages='last')),
    'adpc62': (ADPC6_2, dict(txt_dim=0, dim=256, num_classes=3)),
    # ── ViT backbones (new) ──
    'vit_tiny_film': (vit_tiny_film, dict(txt_dim=0, num_classes=3,img_size=(160, 196, 160),
                                           pretrained_weights=False, get_feature=False)),
    'vit_small_film': (vit_small_film, dict(txt_dim=0, num_classes=3,img_size=(160, 196, 160),
                                             pretrained_weights=False, get_feature=False)),
    'vit_tiny_daft': (vit_tiny_daft, dict(txt_dim=0, num_classes=3,img_size=(160, 196, 160),
                                           pretrained_weights=False, get_feature=False)),
    'vit_small_daft': (vit_small_daft, dict(txt_dim=0, num_classes=3,img_size=(160, 196, 160),
                                             pretrained_weights=False, get_feature=False)),
    # ── ResNet Backdoor（img_size 由 config img_sz 在 main 中注入）──
    'resnet18_backdoor': (resnet18_backdoor, dict(txt_dim=0, num_classes=3,
                                                    pretrained_weights=True)),
    'resnet10_backdoor': (resnet10_backdoor, dict(txt_dim=0, num_classes=3,
                                                    pretrained_weights=True)),
    'resnet34_backdoor': (resnet34_backdoor, dict(txt_dim=0, num_classes=3,
                                                    pretrained_weights=True)),
    # ── ViT Backdoor ──
    'vit_tiny_backdoor': (vit_tiny_backdoor, dict(txt_dim=0, num_classes=3,img_size=(160, 196, 160),
                                                   pretrained_weights=False, get_feature=False)),
    'vit_small_backdoor': (vit_small_backdoor, dict(txt_dim=0, num_classes=3,img_size=(160, 196, 160),
                                                     pretrained_weights=False, get_feature=False)),
}


def main():
    parser = argparse.ArgumentParser(description='V4 Subject-Aware Training')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='resnet_light_film')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_score_prior', default=True, action='store_true')

    parser.add_argument('--cross-val', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--fold', type=int, default=None)

    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)

    model_cls, model_kwargs = MODELS[args.model]

    from utils.config import load_config
    cf = load_config(args.config_path)
    model_kwargs['txt_dim'] = cf.get('table_feature', 0)
    model_kwargs['descripe'] = cf.get('descripe', 'none')

    img_sz = tuple(cf.get('img_sz', (160, 160, 96)))
    if 'backdoor' in args.model:
        model_kwargs['img_size'] = img_sz
    elif any(k in args.model for k in ('vit_', 'ViT')):
        model_kwargs['img_size'] = img_sz

    # ── backdoor_kwargs：从 YAML config 读取，仅对 backdoor 系列模型注入 ──
    if 'backdoor_kwargs' in cf and 'backdoor' in args.model:
        model_kwargs['backdoor_kwargs'] = cf['backdoor_kwargs']

    # ── use_class_head + class_head_kwargs（backdoor 系列：ResNet / ViT）──
    if 'backdoor' in args.model:
        model_kwargs['use_class_head'] = cf.get('use_class_head', False)
        model_kwargs['class_head_kwargs'] = cf.get('class_head_kwargs', None)

    # ── loss_weights：从 YAML config 读取 ──
    loss_weights = cf.get('loss_weights', None)

    if args.cross_val:
        train_cross_val_v4(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=not args.no_score_prior,
            loss_weights=loss_weights,
            exp_type='norm', model_name=args.model,
        )
    elif args.single:
        train_single_v4(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=not args.no_score_prior,
            loss_weights=loss_weights,
            fold=args.fold, exp_type='norm', model_name=args.model,
        )
    else:
        TrainV4(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=not args.no_score_prior,
            loss_weights=loss_weights,
            exp_type='norm', model_name=args.model,
        )


if __name__ == '__main__':
    main()
