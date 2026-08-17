"""Ablation models for IE-CAPM mechanism studies."""

from .scale_table_transformer import (
    ABLATION_PRESETS,
    OriginalPatchwiseCAPM,
    ScaleTableInteractionAblation3D,
    TransformerCalibratedCAPM,
    build_scale_table_ablation,
    demographic_var_specs,
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

    "DomainGuidedFrequencyGate3D",
    "FeatureSpectrumAccumulator",
    "FrequencyGuidedScaleTable3D",
    "FrequencyPrior",
    "build_frequency_guided_model",

    "MIX_MODES",
    "FrequencyMixStyle3D",
    "FrequencyMixStyleScaleTable3D",
    "build_frequency_mixstyle_model",

    "RAW_SKIP_STAGES",
    "RawFrequencySkip3D",
    "RawFrequencySkipScaleTable3D",
    "build_raw_frequency_skip_model",
    "extract_raw_frequency_features",
]
