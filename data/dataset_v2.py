"""
V4 Dataset — 支持 Full Dataset 模式，自动检测 flat/split 目录结构，
并提取 subject ID，用于受试者级别（subject-aware）的 train/test 切分。

与 V3 dataset.py 的差异：
  - 新增 detect_data_layout / collect_file_list / create_dataset 工厂函数
  - FullDatasetClassifier：不区分 train/test 子目录，全量加载
  - 每个样本返回 subject_id，供 splitter 做 subject-level 分组
  - 兼容 V3 的 PET_classify 回退

依赖：torch + pandas 仅 FullDatasetClassifier 实例化时需要，
    纯工具函数（detect/collect/extract）无任何外部依赖。
"""
import os
import re
import sys
import numpy as np
from glob import glob


# ============================================================
# Subject ID 提取工具（无外部依赖）
# ============================================================

def extract_subject_id(folder_name: str, pattern: str = 'auto') -> str:
    """
    从文件夹名中提取 subject ID。

    格式支持：
      - ADNI: 001_S_0001-2020_01_01-1  → '001_S_0001'
      - NACC: NACC000806-2023_11_14-1  → 'NACC000806'

    Args:
        folder_name: 文件夹名
        pattern:
            'auto'          → 取第一个 '-' 前的部分
            'first_segment' → 取第一个 '-' 或 '_' 前的部分
            regex           → 用 group(1) 提取
    """
    if pattern == 'auto':
        if '-' in folder_name:
            return folder_name.split('-')[0]
        elif '_' in folder_name:
            return folder_name.split('_')[0]
        return folder_name
    elif pattern == 'first_segment':
        parts = re.split(r'[-_]', folder_name, maxsplit=1)
        return parts[0]
    else:
        match = re.search(pattern, folder_name)
        if match and match.lastindex:
            return match.group(1)
        return match.group(0) if match else folder_name


# ============================================================
# 数据布局自动检测（无外部依赖）
# ============================================================

def detect_data_layout(data_root: str) -> dict:
    """
    自动检测数据目录结构：flat 还是 split（有 train/test 子目录）。

    Returns:
        {'layout': 'flat'|'split', 'search_paths': [list of dirs]}
    """
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    search_paths = []
    if os.path.isdir(train_dir):
        search_paths.append(train_dir)
    if os.path.isdir(test_dir):
        search_paths.append(test_dir)

    if search_paths:
        return {'layout': 'split', 'search_paths': search_paths}
    else:
        return {'layout': 'flat', 'search_paths': [data_root]}


def collect_file_list(data_root: str,
                      file_pattern: str = '*.pt') -> list:
    """
    收集所有数据文件，自动处理 split/flat 两种目录结构。

    Args:
        data_root: 数据根目录
        file_pattern: glob 模式，如 '*.pt' 或 '*_norm.nii.gz'

    Returns:
        list of str — 所有匹配文件的绝对路径
    """
    layout = detect_data_layout(data_root)
    all_files = []
    for search_path in layout['search_paths']:
        found = sorted(glob(os.path.join(search_path, '*', file_pattern)))
        all_files.extend(found)
    return all_files


# ============================================================
# 文件夹名解析（无外部依赖）
# ============================================================

def parse_folder_name(folder_name: str, naming: str = 'auto') -> dict:
    """
    从文件夹名提取 subject_id、日期、标签。

    支持: ADNI (001_S_0001-2020_01_01-1), NACC (NACC000806-2023_11_14-1)

    Returns:
        {'subject_id': str, 'date': str, 'label': int (1-based)}
    """
    if naming == 'auto':
        parts = folder_name.split('-')
        if len(parts) >= 3:
            subject_id = parts[0]
            date = parts[1]
            raw_label = parts[-1]
        elif len(parts) == 2:
            subject_id = parts[0]
            date = ''
            raw_label = parts[1]
        else:
            subject_id = folder_name
            date = ''
            raw_label = '1'

        label = 1
        try:
            label = int(raw_label)
        except ValueError:
            pass
        return {'subject_id': subject_id, 'date': date, 'label': label}
    else:
        raise ValueError(f"Unknown naming: {naming}")


# ============================================================
# 分类任务（Class Task）配置
# ============================================================

# 原始诊断标签（1-based，来自文件夹名末尾数字）
DEFAULT_DIAGNOSIS_NAMES = {1: 'CN', 2: 'MCI', 3: 'AD'}

# 预设别名 → 参与训练的原始类别（1-based，有序）
CLASS_TASK_ALIASES = {
    'all': (1, 2, 3),
    '123': (1, 2, 3),
    'multiclass': (1, 2, 3),
    '3class': (1, 2, 3),
    '12': (1, 2),
    '23': (2, 3),
    '13': (1, 3),
    'cnmci': (1, 2),
    'mciad': (2, 3),
    'cnad': (1, 3),
    '1v2': (1, 2),
    '2v3': (2, 3),
    '1v3': (1, 3),
}


def parse_class_task(class_task=None):
    """
    解析分类任务配置，返回参与训练的原始类别（1-based）。

    Args:
        class_task:
            None              → 三分类 (1, 2, 3)，向后兼容
            str               → '123' | '12' | '23' | '13' | 'cn_mci' | 'mci_ad' | 'cn_ad' | ...
            list/tuple        → [1, 2] | [2, 3] | [1, 3] | [1, 2, 3]

    Returns:
        tuple[int, ...]: 有序、去重后的原始类别，如 (1, 2) 或 (1, 2, 3)

    Raises:
        ValueError: 非法类别或任务字符串
    """
    if class_task is None:
        return (1, 2, 3)

    if isinstance(class_task, (list, tuple)):
        if len(class_task) < 2:
            raise ValueError(
                f"class_task must include at least 2 classes, got {class_task}"
            )
        classes = tuple(sorted({int(c) for c in class_task}))
    elif isinstance(class_task, str):
        key = class_task.strip().lower()
        norm_key = key.replace('-', '').replace(' ', '')
        if norm_key in CLASS_TASK_ALIASES:
            classes = CLASS_TASK_ALIASES[norm_key]
        elif norm_key.replace('_', '') in CLASS_TASK_ALIASES:
            classes = CLASS_TASK_ALIASES[norm_key.replace('_', '')]
        elif norm_key.isdigit() and len(norm_key) >= 2:
            classes = tuple(sorted({int(ch) for ch in norm_key}))
        else:
            raise ValueError(
                f"Unknown class_task '{class_task}'. "
                f"Use '123', '12', '23', '13', 'cn_mci', 'mci_ad', 'cn_ad', "
                f"or a list like [1, 2]."
            )
    else:
        raise TypeError(
            f"class_task must be None, str, list, or tuple, got {type(class_task)}"
        )

    for c in classes:
        if c not in (1, 2, 3):
            raise ValueError(
                f"Invalid class id {c} in class_task {classes}. Expected labels in {{1, 2, 3}}."
            )

    if len(classes) < 2:
        raise ValueError(
            f"class_task must include at least 2 classes, got {classes}"
        )

    return classes


def class_task_to_string(active_classes):
    """将 active_classes 转为紧凑字符串，如 (1, 2) → '12'。"""
    return ''.join(str(c) for c in active_classes)


def normalize_class_task_value(task):
    """将 YAML null / 'null' / 空串 规范为 None。"""
    if task is None:
        return None
    if isinstance(task, str) and task.strip().lower() in ('null', 'none', ''):
        return None
    return task


def resolve_class_task(cf: dict, *, use_age_split_override: bool = False):
    """
    从 config 解析 class_task。

    use_age_split_override=False（norm 等）：仅顶层 ``class_task``。
    use_age_split_override=True（age 实验）：
      - ``age_split.class_task`` 为有效值时覆盖顶层；
      - ``age_split.class_task: null`` 或未设置时回退顶层 ``class_task``。
    """
    global_task = normalize_class_task_value(cf.get('class_task', None))

    if not use_age_split_override:
        return global_task

    age_cfg = cf.get('age_split') or {}
    if not isinstance(age_cfg, dict) or 'class_task' not in age_cfg:
        return global_task

    age_task = normalize_class_task_value(age_cfg['class_task'])
    return age_task if age_task is not None else global_task


def get_class_names(active_classes):
    """返回 active_classes 对应的诊断名称列表。"""
    return [DEFAULT_DIAGNOSIS_NAMES[c] for c in active_classes]


class FullDatasetClassifier:
    """
    V4 全量数据集类。

    从数据目录加载所有文件（自动处理 flat/split），自动提取 subject ID，
    返回 sample-level 的 DataLoader 配合 SubjectGroupKFoldSplitter 使用。

    用法:
        ds = FullDatasetClassifier(
            data_path='E:/2.causal/NACC_dataset',
            table_path='',                   # optional CSV
            subject_pattern='auto',
            save_load=False,                 # True = 首次 .nii.gz → .pt 转换
        )
        subjects = ds.subject_ids    # np.array of str
        labels = ds.all_labels       # np.array of int (0-based, remapped)
        original_labels = ds.original_labels  # np.array of int (1-based diagnosis)
    """

    def __init__(
        self,
        data_path: str,
        table_path: str = '',
        desired_shape=(160, 160, 96),
        days_threshold=90,
        dataset: str = 'NACC',
        subject_pattern: str = 'auto',
        save_load: bool = False,
        class_task=None,
        table_feature=None,
    ):
        import torch
        import pandas as pd

        self.data_path = data_path
        self.desired_shape = desired_shape
        self.dataset_name = dataset
        self.subject_pattern = subject_pattern
        self.class_task = class_task
        # None = keep all continuous columns (legacy).
        # 0 = zeros(1); 1 = age-only (AGE/AGE_YEARS); N>1 = first N cols.
        self.table_feature = table_feature
        self._conti_keep_idx = None

        # --- nii.gz → .pt 转换（可选） ---
        if save_load:
            from data.preprocessing import preprocess_one
            nii_list = glob(os.path.join(data_path, '*',
                                         'MRI_N4_brain_mni152_cropped_norm.nii.gz'))
            if not nii_list:
                nii_list = glob(os.path.join(data_path, '*', '*_norm.nii.gz'))
            if not nii_list:
                nii_list = glob(os.path.join(data_path, '*', '*.nii.gz'))

            self.PET_nii = nii_list
            self.min_diff = days_threshold

            # 表格过滤（如果有）
            self.import_table = len(table_path) > 0
            if self.import_table:
                from data.preprocessing import prepare_table_n
                self.table_df = pd.read_csv(table_path).dropna().reset_index()
                print(f"  [FullDataset] Before table filter: {len(self.PET_nii)}")
                to_remove = []
                for i, path in enumerate(self.PET_nii):
                    folder = path.replace('\\', '/').split('/')[-2]
                    status, _ = self._find_index(folder, self.table_df)
                    if not status:
                        to_remove.append(i)
                for i in reversed(to_remove):
                    self.PET_nii.pop(i)
                self.table_df = prepare_table_n(self.table_df, dataset=dataset)
                print(f"  [FullDataset] After table filter: {len(self.PET_nii)}")

            self._save_pt_files(data_path, preprocess_one, torch)
            print(f"  [FullDataset] .pt files saved in {data_path}")

        # --- 加载 .pt 文件列表（自动处理 flat/split） ---
        self.PET_pt = collect_file_list(data_path, '*.pt')
        if len(self.PET_pt) == 0:
            raise FileNotFoundError(
                f"No .pt files found under {data_path}. "
                f"Run with save_load=True to convert .nii.gz → .pt first."
            )

        # 提取文件夹名 + subject_id
        self._folder_names = []
        self._subject_ids = []
        for pt_path in self.PET_pt:
            folder = pt_path.replace('\\', '/').split('/')[-2]
            self._folder_names.append(folder)
            self._subject_ids.append(extract_subject_id(folder, self.subject_pattern))

        self.subject_ids = np.array(self._subject_ids)
        self.unique_subjects = np.unique(self.subject_ids)

        # 标签（文件夹名末尾数字，1-based → 0-based）
        self.all_labels = self._extract_labels()
        self.original_labels = self.all_labels + 1  # 1-based 原始诊断，过滤前

        # 按 class_task 过滤样本并重映射标签（None = 三分类，向后兼容）
        self._apply_class_task(class_task)

        # 表格
        self.import_table = len(table_path) > 0
        if self.import_table:
            from data.preprocessing import prepare_table_n
            self.table_df = pd.read_csv(table_path).dropna().reset_index()
            self.table_df_prepared = prepare_table_n(self.table_df, dataset=dataset)
            print(f"  [FullDataset] Table: "
                  f"{self.table_df_prepared['num_cat']} cat + "
                  f"{self.table_df_prepared['num_cont']} cont")
            self._configure_table_feature()
        else:
            self.table_df = None
            self.table_df_prepared = None

        print(f"  [FullDataset] Samples: {len(self.PET_pt)}")
        print(f"  [FullDataset] Unique subjects: {len(self.unique_subjects)}")
        print(f"  [FullDataset] Class task: {self.class_task_str} "
              f"({self.num_classes}-class: {', '.join(self.class_names)})")
        label_dist = dict(zip(*np.unique(self.all_labels, return_counts=True)))
        remapped_names = {
            i: self.class_names[i] for i in range(self.num_classes)
        }
        named_dist = {remapped_names.get(k, k): v for k, v in label_dist.items()}
        print(f"  [FullDataset] Label distribution (remapped): {named_dist}")
        if self.num_classes < 3:
            orig_dist = dict(zip(*np.unique(self.original_labels, return_counts=True)))
            orig_named = {
                DEFAULT_DIAGNOSIS_NAMES.get(k, k): v for k, v in orig_dist.items()
            }
            print(f"  [FullDataset] Original diagnosis distribution: {orig_named}")

    def _apply_class_task(self, class_task):
        """按 class_task 过滤样本，并将标签重映射为连续的 0-based 索引。"""
        active_classes = parse_class_task(class_task)
        self.active_classes = active_classes
        self.num_classes = len(active_classes)
        self.class_names = get_class_names(active_classes)
        self.class_task_str = class_task_to_string(active_classes)

        if len(active_classes) == 3:
            return

        mask = np.isin(self.original_labels, active_classes)
        n_kept = int(mask.sum())
        if n_kept == 0:
            raise ValueError(
                f"class_task={active_classes} matched 0 samples under {self.data_path}. "
                f"Check label distribution or class_task setting."
            )

        n_dropped = len(mask) - n_kept
        if n_dropped > 0:
            print(f"  [FullDataset] class_task filter: kept {n_kept}, "
                  f"dropped {n_dropped} samples outside {active_classes}")

        keep_idx = np.where(mask)[0]
        self.PET_pt = [self.PET_pt[i] for i in keep_idx]
        self._folder_names = [self._folder_names[i] for i in keep_idx]
        self._subject_ids = [self._subject_ids[i] for i in keep_idx]
        self.subject_ids = np.array(self._subject_ids)
        self.unique_subjects = np.unique(self.subject_ids)

        kept_original = self.original_labels[mask]
        self.original_labels = kept_original
        label_remap = {orig: idx for idx, orig in enumerate(active_classes)}
        self.all_labels = np.array(
            [label_remap[int(o)] for o in kept_original], dtype=np.int64
        )

    def _extract_labels(self) -> np.ndarray:
        labels = []
        for folder in self._folder_names:
            try:
                label = int(folder[-1])
                if label < 1 or label > 3:
                    raise ValueError(
                        f"Unexpected label value {label} (expected 1-3) "
                        f"parsed from folder: '{folder}'. "
                        f"Check folder naming convention."
                    )
                labels.append(label - 1)
            except ValueError as e:
                if "Unexpected label value" in str(e):
                    raise
                raise ValueError(
                    f"Cannot parse label from folder name: '{folder}'. "
                    f"Expected format: '<subject>-<date>-<label>' where label is 1-3. "
                    f"Got last character: '{folder[-1]}' which is not a valid integer."
                )
            except IndexError:
                raise ValueError(
                    f"Cannot parse label from folder name: '{folder}'. "
                    f"Folder name is empty or too short to extract the last character."
                )
        return np.array(labels, dtype=np.int64)

    def _find_row(self, ptid, current_datetime, diagnosis, df):
        from utils.io_util import date_difference
        subset = df[df['PTID'] == ptid]
        limit = self.min_diff if hasattr(self, 'min_diff') else 90
        min_index = -1
        for idx, row in subset.iterrows():
            diff = date_difference(row['VISDATE'], current_datetime)
            if limit > diff:
                limit = diff
                min_index = idx
            if limit == 0:
                break
        if limit != (self.min_diff if hasattr(self, 'min_diff') else 90):
            return True, min_index
        print(f"  [WARN] No table row: PTID={ptid}, date={current_datetime}")
        return False, min_index

    def _find_index(self, folder_name, df):
        parts = folder_name.split('-')
        ptid = parts[0]
        date_str = parts[1] if len(parts) > 1 else '1900_01_01'
        diagnosis = parts[2] if len(parts) > 2 else '1'
        return self._find_row(ptid, date_str, diagnosis, df)

    def _save_pt_files(self, save_dir, preprocess_one_fn, torch):
        from tqdm import tqdm
        for nii_path in tqdm(self.PET_nii, desc='  Converting .nii.gz -> .pt'):
            folder = nii_path.replace('\\', '/').split('/')[-2]
            out_path = os.path.join(save_dir, folder, 'MRI.pt')
            if os.path.exists(out_path):
                continue
            tensor = preprocess_one_fn(nii_path)
            torch.save(tensor, out_path)

    def __getitem__(self, index: int):
        import torch
        pt_path = self.PET_pt[index]
        image = torch.load(pt_path, map_location='cpu')
        folder = self._folder_names[index]

        batch = {
            'image': image,
            'label': int(self.all_labels[index] + 1),  # 1-based remapped，兼容 trainer
            'original_label': int(self.original_labels[index]),  # 1-based 原始诊断
            'subject_id': self._subject_ids[index],
            'folder': folder,
        }

        if self.import_table and self.table_df_prepared is not None:
            _, date_index = self._find_index(folder, self.table_df)
            if date_index >= 0:
                batch['cate_x'] = torch.tensor(
                    self.table_df_prepared['cate_x'].iloc[date_index].values,
                    dtype=torch.int64)
                conti = torch.tensor(
                    self.table_df_prepared['conti_x'].iloc[date_index].values,
                    dtype=torch.float32)
                batch['conti_x'] = self._select_conti_features(conti)
            else:
                batch['cate_x'] = torch.zeros(1, dtype=torch.int64)
                batch['conti_x'] = torch.zeros(1, dtype=torch.float32)

        return batch

    def _configure_table_feature(self):
        """Resolve which continuous columns to expose as conti_x."""
        if self.table_df_prepared is None:
            return
        cols = list(self.table_df_prepared['conti_x'].columns)
        n = len(cols)
        tf = self.table_feature
        if tf is None:
            self._conti_keep_idx = list(range(n))
            return
        try:
            tf = int(tf)
        except (TypeError, ValueError):
            tf = None
            self._conti_keep_idx = list(range(n))
            return
        if tf <= 0:
            self._conti_keep_idx = []
            print('  [FullDataset] table_feature=0 -> conti_x = zeros(1)')
            return
        if tf == 1:
            age_names = ('AGE_YEARS', 'AGE', 'Age', 'age')
            idx = None
            for name in age_names:
                if name in cols:
                    idx = cols.index(name)
                    break
            if idx is None:
                idx = n - 1  # fallback: last continuous column
            self._conti_keep_idx = [idx]
            print(f'  [FullDataset] table_feature=1 -> age-only column '
                  f'{cols[idx]!r} (idx={idx})')
            return
        keep = min(tf, n)
        self._conti_keep_idx = list(range(keep))
        print(f'  [FullDataset] table_feature={tf} -> first {keep} cont cols')

    def _select_conti_features(self, conti):
        import torch
        if self.table_feature is not None and int(self.table_feature) <= 0:
            return torch.zeros(1, dtype=torch.float32)
        if self._conti_keep_idx is None:
            return conti
        if len(self._conti_keep_idx) == 0:
            return torch.zeros(1, dtype=torch.float32)
        if conti.dim() == 0:
            return conti.unsqueeze(0)
        return conti[self._conti_keep_idx]

    def __len__(self):
        return len(self.PET_pt)

    def get_subject_info(self) -> dict:
        """返回 subject 级别的聚合信息。"""
        sample_to_subject = np.zeros(len(self), dtype=np.int64)
        subject_labels = []
        subject_counts = []

        for i, sid in enumerate(self.unique_subjects):
            mask = self.subject_ids == sid
            sample_to_subject[mask] = i
            sub_labels = self.all_labels[mask]
            counts = np.bincount(sub_labels)
            subject_labels.append(counts.argmax())
            subject_counts.append(mask.sum())

        return {
            'subject_ids': self.unique_subjects,
            'subject_labels': np.array(subject_labels, dtype=np.int64),
            'subject_counts': np.array(subject_counts, dtype=np.int64),
            'sample_to_subject': sample_to_subject,
        }


# ============================================================
# AgeSplitDataset — 按受试者年龄划分 train / val / test
# ============================================================

class AgeSplitDataset(FullDatasetClassifier):
    """按受试者年龄进行泛化性验证的数据集。

    划分规则（受试者级别，同一受试者所有 scan 同组）：
      - 测试集：年龄最大的 top ``test_ratio``（默认 20%）受试者
      - 其余受试者构成 CV 池，在其中做 K 折交叉验证
      - 若不使用 K 折，CV 池按 70% / 10%（占总受试者）划分为训练 / 验证

    需要提供 table_path（CSV 中含 AGE / AGE_YEARS 列）以读取年龄。

    支持 ``class_task``（与 config ``default.yaml`` 中一致）调整分类任务：
      None / '123' → 三分类；'12' / '23' / '13' 等为二分类，并过滤掉
      不参与任务的样本。

    用法::

        ds = AgeSplitDataset(data_path=..., table_path=...)
        split = ds.get_age_holdout_split()          # 固定 test + 可选 train/val
        test_idx, folds = ds.get_age_cv_folds(n_splits=5)
    """

    AGE_COLUMN_CANDIDATES = ('AGE', 'AGE_YEARS', 'Age', 'age')

    def __init__(
        self,
        data_path: str,
        table_path: str = '',
        desired_shape=(160, 160, 96),
        days_threshold=90,
        dataset: str = 'NACC',
        subject_pattern: str = 'auto',
        save_load: bool = False,
        class_task=None,
        age_column: str = None,
        age_agg: str = 'max',
        table_feature=None,
    ):
        if not table_path or not os.path.isfile(table_path):
            raise ValueError(
                "AgeSplitDataset requires a valid table_path with age information."
            )

        super().__init__(
            data_path=data_path,
            table_path=table_path,
            desired_shape=desired_shape,
            days_threshold=days_threshold,
            dataset=dataset,
            subject_pattern=subject_pattern,
            save_load=save_load,
            class_task=class_task,
            table_feature=table_feature,
        )

        self.age_column = age_column or self._detect_age_column()
        self.age_agg = age_agg
        self.sample_ages = self._build_sample_ages()
        self.subject_ages = self._build_subject_ages()

        ages = list(self.subject_ages.values())
        print(f"  [AgeSplitDataset] Age column: {self.age_column}")
        print(f"  [AgeSplitDataset] Subject age range: "
              f"{min(ages):.1f} – {max(ages):.1f} (agg={self.age_agg})")

        self.check_and_print_test_subject_labels()

    def _remapped_label_name_map(self):
        """重映射后的 0-based 标签 → 诊断名。"""
        return {i: name for i, name in enumerate(self.class_names)}

    def _select_oldest_test_subjects(self, test_ratio: float = 0.2):
        """返回年龄最大的 top test_ratio 受试者集合。"""
        n_subjects = len(self.unique_subjects)
        sorted_subjects = sorted(
            self.unique_subjects,
            key=lambda s: self.subject_ages[s],
            reverse=True,
        )
        n_test = max(1, int(round(n_subjects * test_ratio)))
        test_subjects = set(sorted_subjects[:n_test])
        return test_subjects, n_test, sorted_subjects

    def _label_count_str(self, counts: dict, class_names: dict) -> str:
        parts = []
        for lbl in sorted(counts.keys()):
            name = class_names.get(lbl, f'C{lbl}')
            parts.append(f"{name}={counts[lbl]}")
        return ', '.join(parts) if parts else '(empty)'

    def check_and_print_test_subject_labels(
        self,
        test_ratio: float = 0.2,
        class_names=None,
    ):
        """检查并打印最年长 test_ratio 受试者（默认 20%）的标签分布。

        同时输出受试者级与样本级（scan）计数，便于发现测试集类别失衡。
        标签为 class_task 重映射后的 0-based 类别（见 self.class_names）。
        """
        class_name_map = class_names if class_names is not None else self._remapped_label_name_map()

        test_subjects, n_test, _ = self._select_oldest_test_subjects(test_ratio)
        test_idx = self._subjects_to_sample_indices(test_subjects)

        subject_info = self.get_subject_info()
        sid_to_label = dict(zip(subject_info['subject_ids'], subject_info['subject_labels']))

        subject_label_counts = {}
        for sid in test_subjects:
            lbl = int(sid_to_label[sid])
            subject_label_counts[lbl] = subject_label_counts.get(lbl, 0) + 1

        sample_label_counts = {}
        for lbl in self.all_labels[test_idx]:
            lbl = int(lbl)
            sample_label_counts[lbl] = sample_label_counts.get(lbl, 0) + 1

        test_ages = [self.subject_ages[s] for s in test_subjects]
        age_lo, age_hi = min(test_ages), max(test_ages)

        print(f"\n  [AgeSplitDataset] Oldest {test_ratio:.0%} test subjects "
              f"(n={n_test}/{len(self.unique_subjects)}, age {age_lo:.1f}–{age_hi:.1f}) "
              f"| task={self.class_task_str} ({self.num_classes}-class)")
        print(f"    Subject-level labels: {self._label_count_str(subject_label_counts, class_name_map)}")
        print(f"    Sample-level labels:  {self._label_count_str(sample_label_counts, class_name_map)} "
              f"(total scans={len(test_idx)})")

        missing_classes = set(range(self.num_classes)) - set(subject_label_counts.keys())
        if missing_classes:
            missing_names = [class_name_map[c] for c in sorted(missing_classes)]
            print(f"    [WARN] No test subjects for class(es): {', '.join(missing_names)}")

        return {
            'test_subjects': test_subjects,
            'test_idx': test_idx,
            'subject_label_counts': subject_label_counts,
            'sample_label_counts': sample_label_counts,
        }

    def _detect_age_column(self) -> str:
        for col in self.AGE_COLUMN_CANDIDATES:
            if col in self.table_df.columns:
                return col
        raise ValueError(
            f"No age column found in table. Tried: {self.AGE_COLUMN_CANDIDATES}. "
            f"Available: {list(self.table_df.columns)}"
        )

    def _build_sample_ages(self) -> np.ndarray:
        ages = np.full(len(self), np.nan, dtype=np.float64)
        for i, folder in enumerate(self._folder_names):
            ok, row_idx = self._find_index(folder, self.table_df)
            if ok and row_idx >= 0:
                val = self.table_df.loc[row_idx, self.age_column]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    ages[i] = float(val)
        missing = int(np.isnan(ages).sum())
        if missing > 0:
            raise ValueError(
                f"Missing age for {missing}/{len(self)} samples. "
                f"Check table_path and VISDATE matching."
            )
        return ages

    def _build_subject_ages(self) -> dict:
        subject_ages = {}
        for sid in self.unique_subjects:
            mask = self.subject_ids == sid
            vals = self.sample_ages[mask]
            if self.age_agg == 'max':
                subject_ages[sid] = float(np.max(vals))
            elif self.age_agg == 'mean':
                subject_ages[sid] = float(np.mean(vals))
            else:
                raise ValueError(f"Unknown age_agg: {self.age_agg}")
        return subject_ages

    def _subjects_to_sample_indices(self, subject_set) -> np.ndarray:
        mask = np.isin(self.subject_ids, list(subject_set))
        return np.where(mask)[0]

    def get_age_holdout_split(
        self,
        test_ratio: float = 0.2,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> dict:
        """固定年龄划分：test = 最年长 20%；其余按 70% / 10% 分 train / val。

        Returns:
            dict with keys:
              test_idx, train_idx, val_idx  — 样本级索引
              test_subjects, train_subjects, val_subjects
              cv_pool_idx  — train + val 的并集（供 K 折使用）
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, "
                f"got {train_ratio + val_ratio + test_ratio}"
            )

        n_subjects = len(self.unique_subjects)
        test_subjects, n_test, sorted_subjects = self._select_oldest_test_subjects(test_ratio)
        remaining = sorted_subjects[n_test:]

        n_train = int(round(n_subjects * train_ratio))
        n_val = int(round(n_subjects * val_ratio))

        if len(remaining) < n_train + n_val:
            raise ValueError(
                f"Not enough non-test subjects ({len(remaining)}) for "
                f"train={n_train} + val={n_val}"
            )

        rng = np.random.RandomState(random_seed)
        remaining = list(remaining)
        rng.shuffle(remaining)

        train_subjects = set(remaining[:n_train])
        val_subjects = set(remaining[n_train:n_train + n_val])

        test_idx = self._subjects_to_sample_indices(test_subjects)
        train_idx = self._subjects_to_sample_indices(train_subjects)
        val_idx = self._subjects_to_sample_indices(val_subjects)
        cv_pool_idx = np.concatenate([train_idx, val_idx])

        self._print_split_summary(
            test_subjects, train_subjects, val_subjects,
            title="Age hold-out split (70% / 10% / 20%)",
        )

        return {
            'test_idx': test_idx,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'cv_pool_idx': cv_pool_idx,
            'test_subjects': test_subjects,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
        }

    def get_age_cv_folds(
        self,
        n_splits: int = 5,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        stratify: bool = True,
    ):
        """年龄留出测试集 + 在较年轻受试者池上做 K 折 CV。

        Returns:
            test_idx:  固定测试集样本索引（年龄最大 20% 受试者）
            folds:     [(train_idx, val_idx), ...] 仅在 CV 池内的样本索引
        """
        from utils.cv_splitter_v2 import SubjectGroupKFoldSplitter

        test_subjects, n_test, sorted_subjects = self._select_oldest_test_subjects(test_ratio)
        cv_subjects = set(sorted_subjects[n_test:])

        test_idx = self._subjects_to_sample_indices(test_subjects)
        cv_pool_idx = self._subjects_to_sample_indices(cv_subjects)

        cv_subject_ids = self.subject_ids[cv_pool_idx]
        cv_labels = self.all_labels[cv_pool_idx]

        splitter = SubjectGroupKFoldSplitter(
            n_splits=n_splits,
            random_seed=random_seed,
            stratify=stratify,
        )
        local_folds = splitter.split(cv_subject_ids, cv_labels)

        folds = [
            (cv_pool_idx[tr], cv_pool_idx[va])
            for tr, va in local_folds
        ]

        self._print_split_summary(
            test_subjects, cv_subjects, set(),
            title=f"Age split + {n_splits}-fold CV (test={test_ratio:.0%})",
            cv_pool_subjects=cv_subjects,
            n_splits=n_splits,
        )

        return test_idx, folds

    def _print_split_summary(
        self,
        test_subjects,
        train_subjects,
        val_subjects,
        title: str = "",
        cv_pool_subjects=None,
        n_splits: int = None,
    ):
        def _age_range(subs):
            if not subs:
                return "n/a"
            ages = [self.subject_ages[s] for s in subs]
            return f"{min(ages):.1f}–{max(ages):.1f}"

        print(f"\n  [AgeSplitDataset] {title}")
        print(f"    Test:  {len(test_subjects):3d} subjects, "
              f"age {_age_range(test_subjects)}")
        if cv_pool_subjects is not None:
            print(f"    CV pool: {len(cv_pool_subjects):3d} subjects, "
                  f"age {_age_range(cv_pool_subjects)} ({n_splits}-fold)")
        else:
            print(f"    Train: {len(train_subjects):3d} subjects, "
                  f"age {_age_range(train_subjects)}")
            print(f"    Val:   {len(val_subjects):3d} subjects, "
                  f"age {_age_range(val_subjects)}")

        leaked = test_subjects & (train_subjects | val_subjects | (cv_pool_subjects or set()))
        if leaked:
            print(f"    [WARN] Subject leak between splits: {len(leaked)}")


def create_age_dataset(
    data_root: str,
    table_path: str,
    file_pattern: str = '*.pt',
    desired_shape=(160, 160, 96),
    days_threshold=90,
    dataset: str = 'NACC',
    subject_pattern: str = 'auto',
    class_task=None,
    age_column: str = None,
    age_agg: str = 'max',
    table_feature=None,
) -> AgeSplitDataset:
    """工厂函数：创建 AgeSplitDataset。

    Args:
        class_task: 分类任务，None/'123'=三分类，'12'/'23'/'13' 等为二分类。
                    与 config ``class_task`` 字段一致。
        table_feature: None=全部连续列；0=zeros；1=仅年龄；N=前 N 列。
    """
    layout = detect_data_layout(data_root)
    print(f"  [create_age_dataset] Layout: {layout['layout']} "
          f"({len(layout['search_paths'])} dir(s))")

    nii_files = collect_file_list(
        data_root,
        file_pattern.replace('*.pt', 'MRI_N4_brain_mni152_cropped_norm.nii.gz'),
    )
    pt_files = collect_file_list(data_root, file_pattern)

    if not pt_files and nii_files:
        save_load = True
    elif pt_files:
        save_load = False
    else:
        raise FileNotFoundError(
            f"No .pt or .nii.gz files found under {data_root}"
        )

    return AgeSplitDataset(
        data_path=data_root,
        table_path=table_path,
        desired_shape=desired_shape,
        days_threshold=days_threshold,
        dataset=dataset,
        subject_pattern=subject_pattern,
        save_load=save_load,
        class_task=class_task,
        age_column=age_column,
        age_agg=age_agg,
        table_feature=table_feature,
    )


# ============================================================
# 工厂函数：自动检测 + 创建 dataset
# ============================================================

def create_dataset(data_root: str,
                   table_path: str = '',
                   file_pattern: str = '*.pt',
                   desired_shape=(160, 160, 96),
                   days_threshold=90,
                   dataset: str = 'NACC',
                   subject_pattern: str = 'auto',
                   class_task=None) -> FullDatasetClassifier:
    """
    工厂函数：自动检测数据目录结构并创建 FullDatasetClassifier。

    两种目录结构均支持：
      1. Flat:  data_root/subject-date-label/MRI.pt          (NACC 默认)
      2. Split: data_root/{train,test}/subject-date-label/*.pt (ADNI 风格)
    无论哪种，全部数据统一加载。

    Args:
        data_root:      数据根目录
        table_path:     表格 CSV 路径（可选）
        file_pattern:   文件匹配模式
        desired_shape:  目标图像尺寸
        dataset:        数据集名称标签
        subject_pattern: subject ID 提取模式
        class_task:     分类任务，None/'123'=三分类，'12'/'23'/'13' 等为二分类

    Returns:
        FullDatasetClassifier 实例
    """
    layout = detect_data_layout(data_root)
    print(f"  [create_dataset] Layout: {layout['layout']} "
          f"({len(layout['search_paths'])} dir(s))")

    # 收集文件列表
    nii_files = collect_file_list(
        data_root,
        file_pattern.replace('*.pt', 'MRI_N4_brain_mni152_cropped_norm.nii.gz')
    )
    pt_files = collect_file_list(data_root, file_pattern)

    # 确定是否需要 nii.gz → .pt 转换
    if not pt_files and nii_files:
        save_load = True
        print(f"  [create_dataset] Found {len(nii_files)} .nii.gz files, "
              f"will convert to .pt")
    elif pt_files:
        save_load = False
        print(f"  [create_dataset] Found {len(pt_files)} .pt files")
    else:
        raise FileNotFoundError(
            f"No .pt or .nii.gz files found under {data_root}"
        )

    return FullDatasetClassifier(
        data_path=data_root,
        table_path=table_path,
        desired_shape=desired_shape,
        days_threshold=days_threshold,
        dataset=dataset,
        subject_pattern=subject_pattern,
        save_load=save_load,
        class_task=class_task,
    )


# ============================================================
# AugmentedDataset — 数据增强包装器
# ============================================================

class AugmentedDataset:
    """
    Wraps a dataset (or Subset) to apply image augmentations on-the-fly.

    Only the 'image' key is transformed; all other keys (label, cate_x,
    conti_x, subject_id, folder) pass through unchanged.

    Usage:
        base_ds = FullDatasetClassifier(...)
        train_subset = Subset(base_ds, train_indices)
        augmented_train_ds = AugmentedDataset(train_subset, aug_transforms)

    This ensures augmentations are only applied to the training split
    while leaving the validation split untouched.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['image'] = self.transform(sample['image'])
        return sample

    def __len__(self):
        return len(self.dataset)


# ============================================================
# 向后兼容：V3 类（条件导入，仅训练环境需要）
# ============================================================

try:
    from data.dataset import PET_classify, PET_classify_wo_table, PET_classify_test  # noqa
except ImportError:
    PET_classify = None        # type: ignore
    PET_classify_wo_table = None   # type: ignore
    PET_classify_test = None   # type: ignore
