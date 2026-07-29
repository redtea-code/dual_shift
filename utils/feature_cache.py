"""Frozen feature cache for dictionary learning — fold-local scaler + NPZ I/O."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from os.path import join as j
from typing import Optional

import numpy as np


@dataclass
class FeatureScaler:
    """Z-score scaler fit on train-fold features only."""

    mean: np.ndarray
    std: np.ndarray
    feature_dim: int

    @classmethod
    def fit(cls, features: np.ndarray, eps: float = 1e-8) -> 'FeatureScaler':
        X = np.asarray(features, dtype=np.float64)
        mean = X.mean(axis=0)
        std = np.maximum(X.std(axis=0), eps)
        return cls(mean=mean, std=std, feature_dim=int(X.shape[1]))

    def transform(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=np.float64)
        return (X - self.mean) / self.std

    def to_dict(self):
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'feature_dim': self.feature_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FeatureScaler':
        return cls(
            mean=np.asarray(d['mean'], dtype=np.float64),
            std=np.asarray(d['std'], dtype=np.float64),
            feature_dim=int(d['feature_dim']),
        )


def save_scaler(scaler: FeatureScaler, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scaler.to_dict(), f, indent=2)


def load_scaler(path: str) -> FeatureScaler:
    with open(path, 'r', encoding='utf-8') as f:
        return FeatureScaler.from_dict(json.load(f))


def save_feature_split(
        path: str,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        ages: np.ndarray,
        subject_ids: np.ndarray,
        sample_indices: np.ndarray,
        standardized: bool = False,
):
    """Save one split's cached features and metadata."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        ages=np.asarray(ages, dtype=np.float64),
        subject_ids=np.asarray(subject_ids),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        standardized=np.array([standardized]),
    )


def load_feature_split(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        'features': data['features'],
        'labels': data['labels'],
        'ages': data['ages'],
        'subject_ids': data['subject_ids'],
        'sample_indices': data['sample_indices'],
        'standardized': bool(data['standardized'][0]) if 'standardized' in data else False,
    }


def fold_cache_dir(run_root: str, fold_idx: int) -> str:
    return j(run_root, f'fold_{fold_idx}')


def assert_scaler_not_fit_on_splits(
        scaler_fit_indices: np.ndarray,
        forbidden_indices: np.ndarray,
        context: str = '',
):
    """Ensure scaler was not fit using validation/test samples."""
    fit_set = set(np.asarray(scaler_fit_indices).tolist())
    forbidden = set(np.asarray(forbidden_indices).tolist())
    overlap = fit_set & forbidden
    if overlap:
        raise ValueError(
            f"Scaler fit indices overlap forbidden set ({context}): "
            f"{len(overlap)} samples"
        )


def assert_feature_dim(features: np.ndarray, expected_dim: int):
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Expected features [N, {expected_dim}], got {features.shape}"
        )
