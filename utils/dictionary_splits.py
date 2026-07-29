"""Subject-aware outer splits for dictionary MVP — manifest + leak checks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from os.path import join as j
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DictionaryFoldSplit:
    """One outer CV fold with train / val / test indices."""

    fold_idx: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def to_dict(self):
        return {
            'fold_idx': int(self.fold_idx),
            'train_idx': self.train_idx.tolist(),
            'val_idx': self.val_idx.tolist(),
            'test_idx': self.test_idx.tolist(),
        }


def resolve_dictionary_test_ratio(cf: dict) -> float:
    """Dictionary pipeline may override ``age_split.test_ratio``.

    Prefer ``dictionary.test_ratio`` when set (including 0); otherwise inherit
    ``age_split.test_ratio`` (default 0.2).
    """
    dict_cfg = cf.get('dictionary') or {}
    age_cfg = cf.get('age_split') or {}
    if 'test_ratio' in dict_cfg and dict_cfg['test_ratio'] is not None:
        return float(dict_cfg['test_ratio'])
    return float(age_cfg.get('test_ratio', 0.2))


def resolve_dictionary_splits(
        dataset,
        n_splits: int = 5,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        stratify: bool = True,
) -> Tuple[np.ndarray, List[DictionaryFoldSplit], str]:
    """Resolve outer folds for dictionary MVP.

    Returns:
        fixed_test_idx: age-holdout test indices, or empty for standard_cv
        folds: per-fold train/val/test
        split_mode: 'age_holdout' | 'standard_cv'
    """
    if test_ratio is not None and float(test_ratio) <= 0:
        return _resolve_standard_cv_splits(
            dataset,
            n_splits=n_splits,
            random_seed=random_seed,
            stratify=stratify,
        )

    test_idx, folds = dataset.get_age_cv_folds(
        n_splits=n_splits,
        test_ratio=test_ratio,
        random_seed=random_seed,
        stratify=stratify,
    )
    test_idx = np.asarray(test_idx, dtype=np.int64)
    split_folds = []
    for i, (train_idx, val_idx) in enumerate(folds):
        split_folds.append(DictionaryFoldSplit(
            fold_idx=i + 1,
            train_idx=np.asarray(train_idx, dtype=np.int64),
            val_idx=np.asarray(val_idx, dtype=np.int64),
            test_idx=test_idx.copy(),
        ))
    return test_idx, split_folds, 'age_holdout'


def _resolve_standard_cv_splits(
        dataset,
        n_splits: int = 5,
        random_seed: int = 42,
        stratify: bool = True,
) -> Tuple[np.ndarray, List[DictionaryFoldSplit], str]:
    """Full-pool subject-aware CV: rotating train / val / test (~60/20/20)."""
    from utils.cv_splitter_v2 import resolve_subject_standard_cv_folds

    raw = resolve_subject_standard_cv_folds(
        dataset.subject_ids,
        dataset.all_labels,
        n_splits=n_splits,
        random_seed=random_seed,
        stratify=stratify,
        verbose=True,
    )
    split_folds = [
        DictionaryFoldSplit(
            fold_idx=f.fold_idx,
            train_idx=f.train_idx,
            val_idx=f.val_idx,
            test_idx=f.test_idx,
        )
        for f in raw
    ]
    return np.asarray([], dtype=np.int64), split_folds, 'standard_cv'


def dictionary_cache_descripe(cf: dict) -> str:
    """Cache/run tag: append ``_full_cv`` when dictionary uses standard_cv."""
    base = str(cf.get('descripe', 'default'))
    test_ratio = resolve_dictionary_test_ratio(cf)
    if test_ratio <= 0 and 'full_cv' not in base:
        return f'{base}_full_cv'
    return base


def assert_no_subject_leak(
        subject_ids: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        fold_idx: int = 0,
):
    """Raise if any subject appears in more than one split."""
    def _subs(idxs):
        return set(subject_ids[np.asarray(idxs)]) if len(idxs) else set()

    tr, va, te = _subs(train_idx), _subs(val_idx), _subs(test_idx)
    leaks = {
        'train_val': tr & va,
        'train_test': tr & te,
        'val_test': va & te,
    }
    bad = {k: v for k, v in leaks.items() if v}
    if bad:
        raise ValueError(
            f"Subject leak in fold {fold_idx}: "
            + ', '.join(f"{k}={len(v)}" for k, v in bad.items())
        )


def save_split_manifest(
        run_dir: str,
        meta: dict,
        test_idx: np.ndarray,
        folds: List[DictionaryFoldSplit],
        subject_ids: np.ndarray,
):
    """Persist reproducible split indices and run metadata."""
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        'meta': meta,
        'test_idx': np.asarray(test_idx, dtype=np.int64).tolist(),
        'folds': [f.to_dict() for f in folds],
    }
    path = j(run_dir, 'split_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    for fold in folds:
        assert_no_subject_leak(
            subject_ids, fold.train_idx, fold.val_idx, fold.test_idx,
            fold_idx=fold.fold_idx,
        )
    return path


def load_split_manifest(run_dir: str) -> dict:
    path = j(run_dir, 'split_manifest.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def age_support_report(
        ages: np.ndarray,
        labels: np.ndarray,
        *,
        num_bins: int = 10,
) -> dict:
    """Check per-class age distribution and overlapping support for decomposition."""
    ages = np.asarray(ages, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    per_class = {}
    global_min, global_max = float(ages.min()), float(ages.max())
    for cls in classes:
        mask = labels == cls
        cls_ages = ages[mask]
        per_class[str(int(cls))] = {
            'n': int(mask.sum()),
            'age_min': float(cls_ages.min()) if len(cls_ages) else None,
            'age_max': float(cls_ages.max()) if len(cls_ages) else None,
            'age_mean': float(cls_ages.mean()) if len(cls_ages) else None,
            'age_std': float(cls_ages.std()) if len(cls_ages) else None,
        }

    overlap_min = global_min
    overlap_max = global_max
    for stats in per_class.values():
        if stats['age_min'] is not None:
            overlap_min = max(overlap_min, stats['age_min'])
            overlap_max = min(overlap_max, stats['age_max'])
    has_overlap = overlap_min < overlap_max

    hist, edges = np.histogram(ages, bins=num_bins)
    return {
        'n_samples': int(len(ages)),
        'n_classes': int(len(classes)),
        'per_class': per_class,
        'global_age_range': [global_min, global_max],
        'common_support_range': [float(overlap_min), float(overlap_max)] if has_overlap else None,
        'has_common_age_support': bool(has_overlap),
        'age_histogram_counts': hist.tolist(),
        'age_histogram_edges': edges.tolist(),
    }


def save_age_support_report(run_dir: str, ages: np.ndarray, labels: np.ndarray) -> str:
    report = age_support_report(ages, labels)
    path = j(run_dir, 'age_support_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return path
