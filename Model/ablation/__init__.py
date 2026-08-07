"""Ablation models for IE-CAPM mechanism studies."""

from .scale_table_transformer import (
    ABLATION_PRESETS,
    OriginalPatchwiseCAPM,
    ScaleTableInteractionAblation3D,
    TransformerCalibratedCAPM,
    build_scale_table_ablation,
    demographic_var_specs,
)

__all__ = [
    "ABLATION_PRESETS",
    "OriginalPatchwiseCAPM",
    "ScaleTableInteractionAblation3D",
    "TransformerCalibratedCAPM",
    "build_scale_table_ablation",
    "demographic_var_specs",
]
