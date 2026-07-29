"""
Table-only 训练入口 — 复刻 train_norm_v2.py 结构。
仅修改：MODELS 字典 + 数据加载用 load_table_dataset (MRI→dummy zeros)。

Usage:
  # 单折（默认）
  python -m experiments.train_table_v2 --model mlp
  python -m experiments.train_table_v2 --model tabformer

  # 5折交叉验证
  python -m experiments.train_table_v2 --cv --model mlp --epochs 2
  python -m experiments.train_table_v2 --cv --model tabformer

  # 指定第k折
  python -m experiments.train_table_v2 --single --fold 3 --model mlp_bn
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np

from training.trainer_v2 import SubjectGroupKFoldSplitter, train_one_fold
from data.table_dataset import load_table_dataset


# ============================================================
# Table-only model builders — all accept (x, x_num=None, **kw)
# ============================================================

class MLP(nn.Module):
    """极简 MLP — 2 隐藏层"""
    def __init__(self, txt_dim=7, num_classes=3, film_stages='none',
                 feature=False, pretrained_weights=False, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(txt_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, x_num=None, **kwargs):
        return self.net(x_num)


class MLP_BN(nn.Module):
    """带 BatchNorm 的 3 层 MLP — 适配 MLPClassifier 结构"""
    def __init__(self, txt_dim=7, hidden=128, dropout=0.3, num_classes=3,
                 film_stages='none', feature=False, pretrained_weights=False, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(txt_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.bn2 = nn.BatchNorm1d(hidden // 2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden // 2, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, x_num=None, **kwargs):
        out = self.fc1(x_num)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)
        return self.fc3(out)


class TabFormerWrapper(nn.Module):
    """TabFormer 的 trainer 兼容包装器。
    原始接口 forward(B, F) → (B, C)，trainer 调用 model(x, x_num)。
    """
    def __init__(self, txt_dim=7, embed_dim=64, num_heads=4, num_layers=2,
                 dropout=0.1, num_classes=3,
                 film_stages='none', feature=False, pretrained_weights=False, **kwargs):
        super().__init__()
        self.num_features = txt_dim
        self.embed_dim = embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, txt_dim, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, x, x_num=None, **kwargs):
        xf = x_num.unsqueeze(-1)                        # (B,F) → (B,F,1)
        xf = self.feature_embed(xf)                     # (B,F,E)
        xf = xf + self.pos_embed[:, :xf.size(1), :]     # 按实际特征数裁剪
        xf = self.encoder(xf)
        xf = xf.mean(dim=1)
        xf = self.norm(xf)
        return self.head(xf)


# ============================================================
# Model registry — 与 trainer 的 model_cls(**model_kwargs) 兼容
# ============================================================

DEFAULT_KWARGS = {'num_classes': 3, 'film_stages': 'none',
                  'feature': False, 'pretrained_weights': False}

MODELS = {
    'mlp':        (MLP,             dict(DEFAULT_KWARGS)),
    'mlp_bn':     (MLP_BN,          dict(DEFAULT_KWARGS, hidden=128, dropout=0.3)),
    'tabformer':  (TabFormerWrapper, dict(DEFAULT_KWARGS, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1)),
}


# ============================================================
# Single-fold runner
# ============================================================

def _run_single(cf, full_dataset, model_cls, model_kwargs,
                device, fold, use_score_prior, loss_weights,
                exp_type, model_name):
    patch_size = tuple(cf.get('patch_size', (5, 5, 3))) if isinstance(cf.get('patch_size', (5, 5, 3)), list) else cf.get('patch_size', (5, 5, 3))

    splitter = SubjectGroupKFoldSplitter(
        n_splits=5,
        random_seed=cf.get('cross_val', {}).get('random_seed', 42),
    )

    if fold is not None:
        folds = splitter.split(full_dataset.subject_ids, full_dataset.all_labels)
        if fold > len(folds):
            raise ValueError(f"fold={fold} > n_splits={len(folds)}")
        train_idx, val_idx = folds[fold - 1]
        print(f"  Using CV fold #{fold}: train={len(train_idx)}, val={len(val_idx)}")
    else:
        single_cfg = cf.get('single_experiment', {})
        test_ratio = 1.0 - single_cfg.get('train_ratio', 0.8)
        train_idx, val_idx = splitter.split_train_test(
            full_dataset.subject_ids,
            full_dataset.all_labels,
            test_ratio=test_ratio,
            test_random_seed=cf.get('cross_val', {}).get('random_seed', 42),
        )
        tr_s = len(np.unique(full_dataset.subject_ids[train_idx]))
        va_s = len(np.unique(full_dataset.subject_ids[val_idx]))
        leaked = len(
            set(full_dataset.subject_ids[train_idx]) &
            set(full_dataset.subject_ids[val_idx])
        )
        print(f"  Subject-aware split: train={len(train_idx)} ({tr_s} subjects), "
              f"val={len(val_idx)} ({va_s} subjects), leak={leaked}")

    from torch.utils import data
    train_ds = data.Subset(full_dataset, train_idx)
    val_ds = data.Subset(full_dataset, val_idx)

    return train_one_fold(
        model_cls=model_cls, model_kwargs=model_kwargs,
        train_dataset=train_ds, val_dataset=val_ds,
        train_bc=cf['train_bc'], eval_bc=cf['eval_bc'],
        num_epochs=cf['num_epochs'],
        val_interval=cf['val_inter'],
        save_interval=cf['save_inter'],
        project_dir=cf['project_dir'],
        device=device, use_score_prior=use_score_prior,
        forward_fn=None, optimizer_fn=None, scheduler_fn=None,
        loss_weights=loss_weights, fold_idx=fold or 1,
        patch_size=patch_size, exp_type=exp_type,
        model_name=model_name, dataset=cf.get('dataset', 'NACC'),
    )


# ============================================================
# Training entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Table-Only Training')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_score_prior', default=True, action='store_true')
    parser.add_argument('--cv', action='store_true', help='5-fold cross validation')
    parser.add_argument('--single', action='store_true', default=True)
    parser.add_argument('--fold', type=int, default=None, help='specific CV fold (1-5)')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)

    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)

    from utils.config import load_config
    cf = load_config(args.config_path)

    if args.batch_size:
        cf['train_bc'] = args.batch_size
    if args.epochs:
        cf['num_epochs'] = args.epochs
        cf['val_inter'] = max(1, args.epochs)

    # ---- Load dataset ONCE ----
    print(f"\n{'='*60}")
    print(f"  TABLE-ONLY: Loading + pairing NACC table <-> MRI")
    print(f"{'='*60}")
    full_dataset = load_table_dataset(cf)

    # Auto-detect table feature count
    sample = full_dataset[0]
    n_feat = sample.get('conti_x', torch.zeros(1)).numel()
    print(f"  [Auto-detect] table_feature = {n_feat}")

    # Build model info
    model_cls, model_kwargs = MODELS[args.model]
    model_kwargs = dict(model_kwargs)
    model_kwargs['txt_dim'] = n_feat
    print(f"  Model: {args.model} ({sum(p.numel() for p in model_cls(**model_kwargs).parameters()):,} params)")
    print(f"  Samples: {len(full_dataset)}, Features: {n_feat}")

    # ---- Dispatch ----
    loss_weights = None
    use_score_prior = not args.no_score_prior

    if args.cv:
        # 5-fold cross validation
        print(f"\n{'='*60}")
        print(f"  CROSS-VAL: 5-fold | model={args.model}")
        print(f"{'='*60}")
        results = []
        for k in range(1, 6):
            print(f"\n  >>> Fold {k}/5")
            best_acc, _ = _run_single(
                cf=cf, full_dataset=full_dataset,
                model_cls=model_cls, model_kwargs=model_kwargs,
                device=args.device, fold=k,
                use_score_prior=use_score_prior,
                loss_weights=loss_weights,
                exp_type='table_only_cv', model_name=args.model,
            )
            results.append(best_acc)
            print(f"\n  Fold {k} complete. Best accuracy: {best_acc:.4f}")

        print(f"\n{'='*60}")
        print(f"  CV Results: {[f'{r:.4f}' for r in results]}")
        print(f"  CV Mean +/- Std: {np.mean(results):.4f} +/- {np.std(results):.4f}")
        print(f"{'='*60}")
    else:
        # Single fold (default)
        _run_single(
            cf=cf, full_dataset=full_dataset,
            model_cls=model_cls, model_kwargs=model_kwargs,
            device=args.device, fold=args.fold,
            use_score_prior=use_score_prior,
            loss_weights=loss_weights,
            exp_type='table_only', model_name=args.model,
        )


if __name__ == '__main__':
    main()
