from .adpc import ADPC6_2, ADPC6_4, ADPC6_2_VIT
from .dfiv import DFIV, DFIV_train
from .freq_mask import (
    FAMC3DWrapper,
    create_famc3d_adpc2, create_famc3d_adpc4,
    DualBranchClassifier, MSFAM3DWrapper, PatchFreqClassifier3D,
    LearnableDirectionalMask, adversarial_artifact_augment,
    feature_consistency_loss3d,
    # New ADNI-adapted losses
    mask_budget_loss3d, structural_protection_loss3d,
    directional_tv_loss3d, ADNIFrequencyLoss,
)
from .backbone import (
    ResNetFilmBackbone, ResNetBackdoorBackbone, ResNetDAFTBackbone,
    ResNetSCACAPMBackbone,
    resnet10_film, resnet18_film, resnet34_film,
    resnet10_backdoor, resnet18_backdoor, resnet34_backdoor,
    resnet10_daft, resnet18_daft, resnet34_daft,
    resnet10_disentangled, resnet18_disentangled,
    resnet18_sca, resnet18_sc, resnet18_csra, resnet18_var,
    ViTFiLMBackbone, ViTDAFTBackbone, ViTBackdoorBackbone,
    vit_tiny_film, vit_small_film,
    vit_tiny_daft, vit_small_daft,
    vit_tiny_backdoor, vit_small_backdoor,
)
from .backbone.journal_resnet import (
    JournalResNet3D,
    LateStageSpatialModulation,
    journal_resnet10,
    journal_resnet18,
)
