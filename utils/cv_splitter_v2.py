"""
V4 — 受试者级别（Subject-Aware）的 K-Fold 交叉验证切分器。

与 V3 cv_splitter.py 的差异：
  - 新增 SubjectGroupKFoldSplitter：基于 subject 分组切分
  - 确保同一 subject 的所有样本处于同一 fold
  - 支持分层（按 subject 的主标签进行 stratified split）
  - 保留 KFoldSplitter 用于向后兼容
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold


class SubjectGroupKFoldSplitter:
    """
    受试者级别的 K-Fold 切分器。

    每个 subject 的所有样本必须分到同一个 fold，
    避免同一受试者的不同 scan 出现在 train 和 val 中导致数据泄露。

    用法:
        # dataset 准备好后
        subject_ids = np.array(['S001', 'S001', 'S002', 'S003', 'S003', 'S004'])
        labels = np.array([0, 1, 0, 2, 2, 1])

        splitter = SubjectGroupKFoldSplitter(n_splits=5, random_seed=42, stratify=True)
        folds = splitter.split(subject_ids, labels)
        # folds[0] = (train_indices, val_indices)  ← 样本级索引，但 subject 不跨 fold
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_seed: int = 42,
        stratify: bool = True,
    ):
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.stratify = stratify

    def split(
        self,
        subject_ids: np.ndarray,
        labels: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        按 subject 分组做 K-fold 切分。

        Args:
            subject_ids: shape (N,) — 每个样本对应的 subject ID
            labels:      shape (N,) — 每个样本的标签（0-based）

        Returns:
            [(train_idx_fold0, val_idx_fold0), ..., (train_idx_foldK, val_idx_foldK)]
            每个 idx 是 np.ndarray of int（样本级索引）
        """
        n_samples = len(subject_ids)
        if n_samples != len(labels):
            raise ValueError(
                f"subject_ids length ({len(subject_ids)}) != labels length ({len(labels)})"
            )

        # 1. 按 subject 分组
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        if self.n_splits > n_subjects:
            raise ValueError(
                f"n_splits ({self.n_splits}) > n_subjects ({n_subjects}). "
                f"无法做 subject-level 切分。"
            )

        # 2. 计算每个 subject 的样本索引众数和主标签
        subject_indices = []       # list of np.array — 每个 subject 的所有样本索引
        subject_labels = []        # 每个 subject 的主标签（众数）
        subject_sizes = []         # 每个 subject 的样本数

        for sid in unique_subjects:
            mask = subject_ids == sid
            indices = np.where(mask)[0]
            subject_indices.append(indices)

            sub_labels = labels[indices]
            # 取众数作为该 subject 的代表标签
            counts = Counter(sub_labels.tolist())
            majority_label = counts.most_common(1)[0][0]
            subject_labels.append(majority_label)
            subject_sizes.append(len(indices))

        subject_indices = np.array(subject_indices, dtype=object)
        subject_labels = np.array(subject_labels, dtype=np.int64)
        subject_sizes = np.array(subject_sizes, dtype=np.int64)

        # 3. 对 subject 做 K-fold 切分
        if self.stratify and n_subjects >= self.n_splits:
            try:
                skf = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_seed,
                )
                subject_folds = list(skf.split(np.zeros(n_subjects), subject_labels))
            except Exception as e:
                print(f"  [WARNING] Stratified split 失败 ({e})，回退至随机 split")
                subject_folds = self._random_subject_split(n_subjects)
        else:
            subject_folds = self._random_subject_split(n_subjects)

        # 4. 将 subject 级别 fold 映射回样本级别索引
        sample_folds = []
        for train_subj_idx, val_subj_idx in subject_folds:
            train_sample_idx = np.concatenate(subject_indices[train_subj_idx].tolist())
            val_sample_idx = np.concatenate(subject_indices[val_subj_idx].tolist())
            sample_folds.append((train_sample_idx, val_sample_idx))

        return sample_folds

    def _random_subject_split(
        self, n_subjects: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """非分层随机切分 subject。"""
        rng = np.random.RandomState(self.random_seed)
        indices = rng.permutation(n_subjects)

        folds = []
        fold_size = n_subjects // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_subjects
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_idx, val_idx))
        return folds

    def split_train_test(
        self,
        subject_ids: np.ndarray,
        labels: np.ndarray,
        test_ratio: float = 0.2,
        test_random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        按 subject 分组做单次 train/test split。

        Args:
            subject_ids:  shape (N,)
            labels:       shape (N,)
            test_ratio:   测试集比例 (0 < test_ratio < 1)
            test_random_seed: 随机种子

        Returns:
            (train_indices, test_indices) — 样本级索引
        """
        n_samples = len(subject_ids)
        if n_samples != len(labels):
            raise ValueError("subject_ids and labels must have same length")

        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        rng = np.random.RandomState(test_random_seed)
        perm = rng.permutation(n_subjects)
        n_test = max(1, int(n_subjects * test_ratio))
        test_subj_idx = perm[:n_test]
        train_subj_idx = perm[n_test:]

        test_mask = np.isin(subject_ids, unique_subjects[test_subj_idx])
        train_mask = np.isin(subject_ids, unique_subjects[train_subj_idx])

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        return train_idx, test_idx


@dataclass
class SubjectCVFold:
    """One subject-aware CV fold with disjoint train / val / test (~60/20/20)."""

    fold_idx: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def resolve_subject_standard_cv_folds(
        subject_ids: np.ndarray,
        labels: np.ndarray,
        n_splits: int = 5,
        random_seed: int = 42,
        stratify: bool = True,
        verbose: bool = True,
) -> List[SubjectCVFold]:
    """Full-pool subject-aware CV with rotating train / val / test (~60/20/20).

    Partition subjects into ``n_splits`` folds via SubjectGroupKFoldSplitter; for
    fold ``i``:
      - test = partition i
      - val  = partition (i + 1) % n_splits
      - train = remaining partitions
    """
    splitter = SubjectGroupKFoldSplitter(
        n_splits=n_splits,
        random_seed=random_seed,
        stratify=stratify,
    )
    raw_folds = splitter.split(subject_ids, labels)
    parts = [np.asarray(va, dtype=np.int64) for _, va in raw_folds]

    folds: List[SubjectCVFold] = []
    for i in range(n_splits):
        test_idx = parts[i]
        val_idx = parts[(i + 1) % n_splits]
        train_parts = [
            parts[j] for j in range(n_splits)
            if j != i and j != (i + 1) % n_splits
        ]
        train_idx = np.concatenate(train_parts)

        if verbose:
            tr_subs = len(np.unique(subject_ids[train_idx]))
            va_subs = len(np.unique(subject_ids[val_idx]))
            te_subs = len(np.unique(subject_ids[test_idx]))
            print(
                f"    Fold {i + 1}: train={len(train_idx)} ({tr_subs} subjects), "
                f"val={len(val_idx)} ({va_subs} subjects), "
                f"test={len(test_idx)} ({te_subs} subjects)"
            )
            assert_no_subject_leak(
                subject_ids, train_idx, val_idx, test_idx, fold_name=f'fold_{i + 1}',
            )
        folds.append(SubjectCVFold(
            fold_idx=i + 1,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        ))

    if verbose:
        print(
            f"\n  [Standard CV] Subject-aware K-fold "
            f"(~60/20/20 train/val/test, no age holdout)"
        )
        print(f"    Folds: {n_splits} | seed={random_seed} | stratify={stratify}")

    return folds


def assert_no_subject_leak(subject_ids, train_idx, val_idx, test_idx=None, fold_name=''):
    """Raise if any subject ID appears in more than one of train/val/test."""
    sets = [
        ('train', set(np.asarray(subject_ids)[np.asarray(train_idx)].tolist())),
        ('val', set(np.asarray(subject_ids)[np.asarray(val_idx)].tolist())),
    ]
    if test_idx is not None:
        sets.append(
            ('test', set(np.asarray(subject_ids)[np.asarray(test_idx)].tolist()))
        )
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            leak = sets[i][1] & sets[j][1]
            if leak:
                sample = sorted(leak)
                shown = sample[:5]
                extra = '...' if len(sample) > 5 else ''
                raise AssertionError(
                    f'{fold_name}: subject leak between {sets[i][0]} and '
                    f'{sets[j][0]}: {shown}{extra}'
                )


def append_std5cv_descripe(descripe: str, test_ratio: float) -> str:
    """Append ``_std5cv`` when using standard CV so runs do not overwrite age-holdout."""
    base = str(descripe or 'default')
    if test_ratio is not None and float(test_ratio) <= 0:
        if 'std5cv' not in base and 'full_cv' not in base:
            return f'{base}_std5cv'
    return base


# ---- 向后兼容：重新导出 ----

from utils.cv_splitter import KFoldSplitter, extract_labels_from_paths, get_cv_splitter  # noqa: E402, F401
