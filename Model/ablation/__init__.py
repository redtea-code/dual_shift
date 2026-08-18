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

__all__ = [
    "ABLATION_PRESETS",
    "OriginalPatchwiseCAPM",
    "ScaleTableInteractionAblation3D",
    "TransformerCalibratedCAPM",
    "build_scale_table_ablation",
    "demographic_var_specs",
    "TargetStyleCAPM",
    "TargetStyleFeatureTransport3D",
]
