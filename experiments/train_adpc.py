"""
V3 — ADPC 训练入口。
Usage:
  python -m experiments.train_adpc                          # 从 config 自动检测 CV/单实验
  python -m experiments.train_adpc --cross-val              # 强制 CV
  python -m experiments.train_adpc --single                 # 强制单实验
  python -m experiments.train_adpc --single --fold 3        # 仅跑第 3 折
"""
import argparse

from Model.adpc import ADPC6_2, ADPC6_4, ADPC6_2_VIT
from training.trainer import Train, train_single, train_cross_val

MODELS = {
    'adpc62': (ADPC6_2, dict(txt_dim=9, dim=256, num_classes=3)),
    'adpc64': (ADPC6_4, dict(txt_dim=9, dim=512, num_classes=3, patch_size=(4, 4, 4))),
    'adpc62_vit': (ADPC6_2_VIT, dict(txt_dim=7, dim=256, num_classes=3)),
}


def main():
    parser = argparse.ArgumentParser(description='ADPC Training V3')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='adpc64',
                        choices=['adpc62', 'adpc64', 'adpc62_vit'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_score_prior',default=True, action='store_true')

    # V3: CV / single control
    parser.add_argument('--cross-val', action='store_true',
                        help='强制交叉验证（覆盖 config）')
    parser.add_argument('--single',default=True, action='store_true',
                        help='强制单实验（覆盖 config）')
    parser.add_argument('--fold', type=int, default=None,
                        help='仅跑指定 fold（1-based，与 --single 共用）')

    args = parser.parse_args()

    model_cls, model_kwargs = MODELS[args.model]

    from utils.config import load_config
    cf = load_config(args.config_path)
    table_feat = cf.get('table_feature', 9)
    model_kwargs['txt_dim'] = table_feat

    if args.model == 'adpc64':
        patch = cf.get('patch_size', [4, 4, 4])
        if isinstance(patch, (list, tuple)):
            model_kwargs['patch_size'] = tuple(int(x) for x in patch)

    if args.single:
        train_single(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=not args.no_score_prior,
            loss_weights=None,
            fold=args.fold,
            exp_type='adpc',
            model_name=args.model,
        )
    elif args.cross_val:
        train_cross_val(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=not args.no_score_prior,
            loss_weights=None,
            exp_type='adpc',
            model_name=args.model,
        )
    else:
        Train(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=not args.no_score_prior,
            loss_weights=None,
            exp_type='adpc',
            model_name=args.model,
        )


if __name__ == '__main__':
    main()
