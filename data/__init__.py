"""Data module — lazy imports for non-training environments."""
# V4 优先：确保 V4 模块直接可用（无 torch 依赖）
from .dataset_v2 import (  # noqa: F401
    detect_data_layout, collect_file_list, extract_subject_id,
    parse_folder_name, FullDatasetClassifier, create_dataset, AugmentedDataset,
)

# Augmentation (requires torch)
try:
    from .augmentation import (  # noqa: F401
        RandomAffine3D, RandomIntensityScale, RandomIntensityShift,
        GaussianNoise, RandomBiasField, Compose, build_train_augmentation,
        BatchAugmentation, make_augmentation_collate_fn,
    )
except ImportError:
    RandomAffine3D = None
    RandomIntensityScale = None
    RandomIntensityShift = None
    GaussianNoise = None
    RandomBiasField = None
    Compose = None
    build_train_augmentation = None

# V3 类（需要 torch，仅训练环境可用）
try:
    from .dataset import PET_classify  # noqa: F401
    from .preprocessing import adaptive_normal, preprocess_one  # noqa: F401
    from .transforms import register_transform, register_transform2  # noqa: F401
except ImportError:
    PET_classify = None           # type: ignore
    adaptive_normal = None        # type: ignore
    preprocess_one = None         # type: ignore
    register_transform = None     # type: ignore
    register_transform2 = None    # type: ignore
