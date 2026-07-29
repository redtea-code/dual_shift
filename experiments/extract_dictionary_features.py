"""Train CE-only encoder per outer fold and cache frozen GAP features."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from os.path import join as j

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Model.backbone.film_backbone import resnet10_ce_only
from Model.causal.age_confound import extract_backbone_features
from data.augmentation import make_augmentation_collate_fn
from training.evaluator import evaluate, MetricsTracker, compute_loss
from training.trainer_age import load_age_dataset
from training.trainer_v2 import init_weights_adpc, _prepare_model_kwargs
from utils.config import load_config
from utils.dictionary_splits import (
    dictionary_cache_descripe,
    resolve_dictionary_splits,
    resolve_dictionary_test_ratio,
    save_split_manifest,
    save_age_support_report,
)
from utils.feature_cache import (
    FeatureScaler,
    assert_feature_dim,
    assert_scaler_not_fit_on_splits,
    fold_cache_dir,
    load_scaler,
    save_feature_split,
    save_scaler,
)
from utils.io_util import write_config


def _build_ce_encoder(num_classes: int, cf: dict):
    mk = dict(
        txt_dim=0,
        num_classes=num_classes,
        pretrained_weights=cf.get('dictionary', {}).get('pretrained_weights'),
        feature=False,
    )
    return resnet10_ce_only(**mk)


@torch.no_grad()
def _extract_fold_features(model, dataset, indices, device, eval_bc, expected_dim):
    payload = extract_backbone_features(model, dataset, indices, device, eval_bc)
    if payload is None:
        raise RuntimeError(f"Feature extraction failed for {len(indices)} samples")
    assert_feature_dim(payload['features'], expected_dim)
    subject_ids = dataset.subject_ids[np.asarray(indices)]
    return {
        'features': payload['features'],
        'labels': payload['labels'],
        'ages': payload['ages'],
        'subject_ids': subject_ids,
        'sample_indices': np.asarray(indices, dtype=np.int64),
    }


def train_ce_encoder_one_fold(
        model,
        train_dataset,
        val_dataset,
        *,
        device,
        train_bc,
        eval_bc,
        num_epochs,
        val_interval,
        learning_rate,
        num_classes,
        proj_dir,
        augmentation_config=None,
):
    """Minimal CE-only training on MRI only (txt ignored)."""
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(j(proj_dir, 'model_best'), exist_ok=True)

    train_collate = make_augmentation_collate_fn(augmentation_config)
    train_dl = data.DataLoader(
        train_dataset, batch_size=train_bc, shuffle=True,
        drop_last=True, num_workers=0, collate_fn=train_collate,
    )
    val_dl = data.DataLoader(
        val_dataset, batch_size=eval_bc, shuffle=False, num_workers=0,
    )

    model = model.to(device)
    model.apply(init_weights_adpc)
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60], gamma=0.5,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    log_path = j(proj_dir, 'encoder_train_loss.txt')

    for epoch in range(num_epochs):
        model.train()
        loss_total = 0.0
        pbar = tqdm(total=len(train_dl), desc=f'CE-only E{epoch + 1}', leave=False)
        for batch in train_dl:
            x = batch['image'].to(device)
            y = (batch['label'].to(device) - 1)
            x = x[:, None, ...]
            logits = model(x, txt=None)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += loss.item()
            pbar.update(1)
        pbar.close()

        if epoch % val_interval == 0:
            model.eval()
            val_pred, val_labels = evaluate(model, val_dl, device)
            val_mt = MetricsTracker(num_classes=num_classes, device=device)
            val_mt.update(val_pred, val_labels)
            val_results = val_mt.compute()
            acc = float(val_results['accuracy'])
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(
                    f'Epoch {epoch + 1}: loss={loss_total:.4f} '
                    f'val_acc={acc:.4f}\n'
                )
            if acc > best_acc:
                best_acc = acc
                torch.save(
                    model.state_dict(),
                    j(proj_dir, 'model_best', 'best_model.pth'),
                )
        scheduler.step()

    return best_acc


def extract_dictionary_features(
        config_path: str = 'config/default.yaml',
        device: str = 'cuda',
        model_name: str = 'resnet10_ce_only',
        num_folds: int = None,
        fold: int = None,
        skip_encoder_train: bool = False,
):
    """Per outer fold: train CE-only encoder, cache standardized H for train/val/test."""
    cf = load_config(config_path)
    dict_cfg = cf.get('dictionary') or {}
    age_cfg = cf.get('age_split') or {}
    cv_cfg = cf.get('cross_val') or {}

    feature_dim = int(dict_cfg.get('feature_dim', 64))
    n_splits = num_folds or age_cfg.get('num_folds', cv_cfg.get('num_folds', 5))
    test_ratio = resolve_dictionary_test_ratio(cf)
    random_seed = age_cfg.get('random_seed', cv_cfg.get('random_seed', 42))
    stratify = age_cfg.get('stratify', cv_cfg.get('stratify', True))
    encoder_epochs = int(dict_cfg.get('encoder_epochs', cf.get('num_epochs', 50)))
    encoder_lr = float(dict_cfg.get('encoder_lr', cf.get('lr', 1e-4)))
    cache_tag = dictionary_cache_descripe(cf)

    full_dataset = load_age_dataset(cf)
    num_classes = full_dataset.num_classes
    class_task_str = full_dataset.class_task_str
    dataset_name = cf.get('dataset', 'ADNI')

    run_root = j(
        cf['project_dir'], 'dictionary_feature_cache', dataset_name,
        f'{model_name}_task{class_task_str}_{cache_tag}',
    )
    os.makedirs(run_root, exist_ok=True)
    write_config(config_path, j(run_root, os.path.basename(config_path)))

    test_idx, folds, split_mode = resolve_dictionary_splits(
        full_dataset, n_splits=n_splits, test_ratio=test_ratio,
        random_seed=random_seed, stratify=stratify,
    )
    meta = {
        'dataset': dataset_name,
        'model_name': model_name,
        'class_task': class_task_str,
        'feature_dim': feature_dim,
        'n_splits': n_splits,
        'test_ratio': test_ratio,
        'split_mode': split_mode,
        'random_seed': random_seed,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_split_manifest(
        run_root, meta, test_idx, folds, full_dataset.subject_ids,
    )
    if hasattr(full_dataset, 'sample_ages') and full_dataset.sample_ages is not None:
        save_age_support_report(
            run_root,
            full_dataset.sample_ages,
            np.array([full_dataset[i]['label'] for i in range(len(full_dataset))]),
        )

    fold_list = folds if fold is None else [f for f in folds if f.fold_idx == fold]
    if fold is not None and not fold_list:
        raise ValueError(f"Fold {fold} not found (n_splits={n_splits})")

    for split in fold_list:
        fold_num = split.fold_idx
        fold_dir = fold_cache_dir(run_root, fold_num)
        os.makedirs(fold_dir, exist_ok=True)

        ckpt_path = j(fold_dir, 'encoder', 'model_best', 'best_model.pth')
        if skip_encoder_train and os.path.isfile(ckpt_path):
            print(f"  [Fold {fold_num}] Reusing encoder checkpoint")
        else:
            print(f"\n--- Fold {fold_num}: train CE-only encoder ---")
            encoder = _build_ce_encoder(num_classes, cf)
            train_ce_encoder_one_fold(
                encoder,
                data.Subset(full_dataset, split.train_idx),
                data.Subset(full_dataset, split.val_idx),
                device=device,
                train_bc=cf['train_bc'],
                eval_bc=cf['eval_bc'],
                num_epochs=encoder_epochs,
                val_interval=cf.get('val_inter', 4),
                learning_rate=encoder_lr,
                num_classes=num_classes,
                proj_dir=j(fold_dir, 'encoder'),
                augmentation_config=cf.get('augmentation'),
            )

        # Load best encoder in feature-extraction mode
        encoder = _build_ce_encoder(num_classes, cf)
        encoder.load_state_dict(
            torch.load(ckpt_path, map_location=device),
        )
        encoder.get_feature = True
        encoder.eval()
        encoder = encoder.to(device)

        raw = {}
        for split_name, idx in (
            ('train', split.train_idx),
            ('val', split.val_idx),
            ('test', split.test_idx),
        ):
            raw[split_name] = _extract_fold_features(
                encoder, full_dataset, idx, device,
                cf['eval_bc'], feature_dim,
            )

        assert_scaler_not_fit_on_splits(
            split.train_idx, split.val_idx, context='scaler vs val',
        )
        assert_scaler_not_fit_on_splits(
            split.train_idx, split.test_idx, context='scaler vs test',
        )

        scaler = FeatureScaler.fit(raw['train']['features'])
        save_scaler(scaler, j(fold_dir, 'feature_scaler.json'))

        for split_name, idx in (
            ('train', split.train_idx),
            ('val', split.val_idx),
            ('test', split.test_idx),
        ):
            feats = scaler.transform(raw[split_name]['features'])
            save_feature_split(
                j(fold_dir, f'features_{split_name}.npz'),
                features=feats,
                labels=raw[split_name]['labels'],
                ages=raw[split_name]['ages'],
                subject_ids=raw[split_name]['subject_ids'],
                sample_indices=raw[split_name]['sample_indices'],
                standardized=True,
            )

        print(f"  [Fold {fold_num}] Cached features → {fold_dir}")
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    print(f"\nFeature cache complete: {run_root}")
    return run_root


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train per-fold CE-only encoder and cache dictionary features',
    )
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='resnet10_ce_only')
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--fold', type=int, default=None,
                        help='Extract a single outer fold (1-based)')
    parser.add_argument('--skip_encoder_train', action='store_true')
    args = parser.parse_args()

    extract_dictionary_features(
        config_path=args.config_path,
        device=args.device,
        model_name=args.model,
        num_folds=args.num_folds,
        fold=args.fold,
        skip_encoder_train=args.skip_encoder_train,
    )


if __name__ == '__main__':
    main()
