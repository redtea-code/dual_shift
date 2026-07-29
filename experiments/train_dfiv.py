"""
V3 — DFIV 训练入口。
Usage:
  python -m experiments.train_dfiv                          # 从 config 自动检测
  python -m experiments.train_dfiv --cross-val              # 强制 CV
  python -m experiments.train_dfiv --single --fold 3        # 仅第 3 折
"""
import argparse
import torch
from torch import optim

from Model.dfiv import DFIV, DFIV_train
from training.trainer import Train, train_single, train_cross_val


def main():
    parser = argparse.ArgumentParser(description='DFIV Training V3')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='dfiv',
                        choices=['dfiv', 'dfiv_train'])
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--cross-val', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--fold', type=int, default=None)

    args = parser.parse_args()

    from utils.config import load_config
    cf = load_config(args.config_path)
    table_feat = cf.get('table_feature', 9)

    if args.model == 'dfiv':
        model_cls = DFIV
        model_kwargs = dict(txt_dim=table_feat, dim=512, num_classes=3)

        def optimizer_fn(params):
            return torch.optim.Adam(params, lr=5e-4)

        def scheduler_fn(opt):
            return optim.lr_scheduler.MultiStepLR(
                opt, milestones=[30, 60, 100], gamma=0.2
            )
    else:
        model_cls = DFIV_train
        model_kwargs = dict(txt_dim=(table_feat - 2, 2), dim=512, num_classes=3)

        def optimizer_fn(params):
            return torch.optim.Adam(params, lr=5e-4)

        def scheduler_fn(opt):
            return optim.lr_scheduler.MultiStepLR(
                opt, milestones=[30, 60, 100], gamma=0.2
            )

    if args.single:
        train_single(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=False,
            optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
            fold=args.fold,
            exp_type='dfiv',
            model_name=args.model,
        )
    elif args.cross_val:
        train_cross_val(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=False,
            optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
            exp_type='dfiv',
            model_name=args.model,
        )
    else:
        Train(
            model_cls=model_cls, model_kwargs=model_kwargs,
            config_path=args.config_path, device=args.device,
            use_score_prior=False,
            optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
            exp_type='dfiv',
            model_name=args.model,
        )


if __name__ == '__main__':
    main()
