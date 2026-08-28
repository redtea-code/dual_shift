"""Ablation models for IE-CAPM mechanism studies."""

from .scale_table_transformer import (
    ABLATION_PRESETS,
    OriginalPatchwiseCAPM,
    ScaleTableInteractionAblation3D,
    TransformerCalibratedCAPM,
    build_scale_table_ablation,
    demographic_var_specs,
)
from .target_style_transport import TargetStyleCAPM, TargetStyleFeatureTransport3D
from .residual_adaptation import (
    CAPMResidualAdaptation3D,
    CAPMResidualAdapter3D,
    CAPMResidualStats,
    FeatureMomentAccumulator,
    RESIDUAL_STATS_SCHEMA,
    TARGET_SPLIT_SCHEMA,
    build_capm_residual_model,
    build_capm_residual_stats_from_loaders,
    save_capm_residual_target_split,
)
from .capm_frequency_grl import (
    CAPMFrequencyGRL3D,
    FrequencyBatch,
    PROJECTOR_SCHEMA,
    TaskSupportProjector,
    compute_capm_frequency_losses,
    make_frequency_batch,
)
from .capm_residual_distribution_alignment import (
    CAPMResidualDistributionAlignment3D,
    RESIDUAL_DISTRIBUTION_STATS_SCHEMA,
    RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA,
    ResidualDistributionStats,
    ResidualDistributionTransport3D,
    SubjectResidualAccumulator,
    apply_bounded_intensity_perturbation,
    apply_fourier_amplitude_perturbation,
    audit_synthetic_perturbation,
    build_residual_distribution_stats_from_loaders,
    load_task_support_projector_artifact,
    save_residual_distribution_target_split,
)

from .frequency_uda import (
    DomainGuidedFrequencyGate3D,
    FeatureSpectrumAccumulator,
    FrequencyGuidedScaleTable3D,
    FrequencyPrior,
    build_frequency_guided_model,
)

from .frequency_mixstyle import (
    MIX_MODES,
    FrequencyMixStyle3D,
    FrequencyMixStyleScaleTable3D,
    build_frequency_mixstyle_model,
)

from .raw_frequency_skip import (
    RAW_SKIP_STAGES,
    RawFrequencySkip3D,
    RawFrequencySkipScaleTable3D,
    build_raw_frequency_skip_model,
    extract_raw_frequency_features,
)

__all__ = [
    "ABLATION_PRESETS",
    "OriginalPatchwiseCAPM",
    "ScaleTableInteractionAblation3D",
    "TransformerCalibratedCAPM",
    "build_scale_table_ablation",
    "demographic_var_specs",
    "TargetStyleCAPM",
    "TargetStyleFeatureTransport3D",
    "CAPMResidualAdaptation3D",
    "CAPMResidualAdapter3D",
    "CAPMResidualStats",
    "FeatureMomentAccumulator",
    "RESIDUAL_STATS_SCHEMA",
    "TARGET_SPLIT_SCHEMA",
    "build_capm_residual_model",
    "build_capm_residual_stats_from_loaders",
    "save_capm_residual_target_split",
    "CAPMFrequencyGRL3D",
    "FrequencyBatch",
    "PROJECTOR_SCHEMA",
    "TaskSupportProjector",
    "compute_capm_frequency_losses",
    "make_frequency_batch",
    "CAPMResidualDistributionAlignment3D",
    "RESIDUAL_DISTRIBUTION_STATS_SCHEMA",
    "RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA",
    "ResidualDistributionStats",
    "ResidualDistributionTransport3D",
    "SubjectResidualAccumulator",
    "apply_bounded_intensity_perturbation",
    "apply_fourier_amplitude_perturbation",
    "audit_synthetic_perturbation",
    "build_residual_distribution_stats_from_loaders",
    "load_task_support_projector_artifact",
    "save_residual_distribution_target_split",
]
