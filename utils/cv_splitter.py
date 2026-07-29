"""
V3 — 基于随机种子的 K-Fold 交叉验证切分器。

不依赖预分 Kfold 目录，从单个数据目录加载全部训练样本后，
根据 random_seed 进行分层 K-fold 切分。
"""
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold


class KFoldSplitter:
    """
    K-fold 数据切分器。

    - 支持分层（stratify）与非分层
    - 固定随机种子保证可复现
    - 输出每 fold 的 (train_indices, val_indices)

    Usage:
        labels = np.array([0, 0, 1, 1, 2, 2, ...])
        splitter = KFoldSplitter(n_splits=5, random_seed=42, stratify=True)
        folds = splitter.split(labels)
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_ds = Subset(full_dataset, train_idx)
            val_ds = Subset(full_dataset, val_idx)
            train_one_fold(...)
    """

    def __init__(self, n_splits: int = 5, random_seed: int = 42, stratify: bool = True):
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.stratify = stratify

    def split(self, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        对 labels 做 K-fold 切分。

        Args:
            labels: shape (N,) 的标签数组（0-based 整数）

        Returns:
            [(train_idx_0, val_idx_0), ..., (train_idx_K-1, val_idx_K-1)]
            每个 idx 是 np.ndarray
        """
        n = len(labels)
        if self.n_splits > n:
            raise ValueError(
                f"n_splits ({self.n_splits}) > n_samples ({n})，无法切分"
            )

        if self.stratify:
            try:
                skf = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=True,
                    random_state=self.random_seed
                )
                folds = list(skf.split(np.zeros(n), labels))
            except Exception as e:
                print(f"[WARNING] 分层切分失败 ({e})，回退至随机切分")
                folds = self._random_split(n)
        else:
            folds = self._random_split(n)

        return folds

    def _random_split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """非分层随机切分。"""
        rng = np.random.RandomState(self.random_seed)
        indices = rng.permutation(n)

        folds = []
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_idx, val_idx))
        return folds


def extract_labels_from_paths(pt_paths: List[str]) -> np.ndarray:
    """
    从 .pt 文件路径中提取标签。

    标签在目录名末尾：subject-date-diag-2/  → label=2（1-based）
    返回 0-based 标签数组。

    Args:
        pt_paths: glob 得到的 .pt 文件绝对路径列表

    Returns:
        np.ndarray of int labels (0-based)
    """
    labels = []
    for p in pt_paths:
        try:
            # 目录名最后一位是标签 (1-3)
            folder = p.replace('\\', '/').split('/')[-2]
            label = int(folder[-1])
            labels.append(label - 1)  # 1-based → 0-based
        except (IndexError, ValueError):
            print(f"[WARNING] 无法从路径解析标签: {p}")
            labels.append(0)  # fallback
    return np.array(labels, dtype=np.int64)


def get_cv_splitter(config: dict) -> KFoldSplitter:
    """
    从 config 创建 KFoldSplitter。

    Args:
        config: 加载的 YAML 配置字典

    Returns:
        KFoldSplitter 实例
    """
    cv_config = config.get('cross_val', {})
    return KFoldSplitter(
        n_splits=cv_config.get('num_folds', 5),
        random_seed=cv_config.get('random_seed', 42),
        stratify=cv_config.get('stratify', True),
    )
