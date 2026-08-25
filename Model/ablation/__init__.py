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
]
