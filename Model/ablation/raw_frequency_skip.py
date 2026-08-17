"""Raw-volume Fourier skip features for source-only frequency robustness tests.

The branch computes a compact representation from the MRI volume before the
CNN stem, then injects it as a bounded residual at a later spatial stage.  It
does not consume a target sample, target statistic, or domain identifier.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .frequency_uda import BAND_NAMES, band_power_fractions
from .scale_table_transformer import ScaleTableInteractionAblation3D


RAW_SKIP_STAGES = ("layer2", "layer3", "layer4")
_STAGE_CHANNELS = {"layer2": 128, "layer3": 256, "layer4": 512}


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_grid(grid: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in grid)
    if len(values) != 3 or any(value < 1 for value in values):
        raise ValueError("spectral_grid must contain three positive integers")
    return values


def extract_raw_frequency_features(
    images: Tensor,
    *,
    spectral_grid: Sequence[int] = (8, 8, 8),
) -> Tensor:
    """Return compact log-amplitude and phase features before CNN processing.

    For every input channel, the descriptor preserves log amplitude plus the
    real/imaginary unit-phase components.  Adaptive pooling keeps its memory
    bounded for full MRI volumes while retaining a fixed rFFT-grid contract.
    """
    if images.ndim != 5:
        raise ValueError("images must have shape [B,C,D,H,W]")
    grid = _validate_grid(spectral_grid)
    spectrum = torch.fft.rfftn(images.float(), dim=(-3, -2, -1), norm="ortho")
    amplitude = torch.log1p(spectrum.abs())
    phase = torch.angle(spectrum)
    features = torch.cat((amplitude, torch.cos(phase), torch.sin(phase)), dim=1)
    return F.adaptive_avg_pool3d(features, grid).to(dtype=images.dtype)


class RawFrequencySkip3D(nn.Module):
    """Encode a pre-CNN rFFT descriptor and add it to a CNN feature map."""

    def __init__(
        self,
        *,
        image_channels: int = 1,
        target_channels: int,
        spectral_grid: Sequence[int] = (8, 8, 8),
        hidden_channels: int = 32,
        max_residual: float = 0.15,
        gate_init: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if image_channels < 1 or target_channels < 1 or hidden_channels < 1:
            raise ValueError("image_channels, target_channels, and hidden_channels must be positive")
        if not 0.0 < max_residual <= 1.0:
            raise ValueError("max_residual must lie in (0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.image_channels = int(image_channels)
        self.target_channels = int(target_channels)
        self.spectral_grid = _validate_grid(spectral_grid)
        self.hidden_channels = int(hidden_channels)
        self.max_residual = float(max_residual)
        self.eps = float(eps)
        norm_groups = _group_count(hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv3d(image_channels * 3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Conv3d(hidden_channels, target_channels, kernel_size=1, bias=False)
        self.raw_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_audit: dict[str, Tensor] | None = None

    @property
    def effective_strength(self) -> Tensor:
        return self.max_residual * torch.tanh(self.raw_gate)

    def forward(
        self,
        images: Tensor,
        features: Tensor,
        *,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if images.ndim != 5 or features.ndim != 5:
            raise ValueError("images and features must have shape [B,C,D,H,W]")
        if images.shape[0] != features.shape[0]:
            raise ValueError("image and feature batch sizes must match")
        if images.shape[1] != self.image_channels or features.shape[1] != self.target_channels:
            raise ValueError("raw-frequency skip channel contract does not match its inputs")
        descriptor = extract_raw_frequency_features(images, spectral_grid=self.spectral_grid)
        encoded = self.encoder(descriptor)
        residual = self.project(encoded)
        residual = F.interpolate(
            residual,
            size=tuple(features.shape[-3:]),
            mode="trilinear",
            align_corners=False,
        )
        feature_rms = features.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
        residual_rms = residual.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
        normalized_residual = residual * feature_rms / residual_rms.clamp_min(self.eps)
        output = features + self.effective_strength * normalized_residual
        raw_fractions, _, _ = band_power_fractions(images)
        relative_delta = (
            (output - features).square().mean(dim=(1, 2, 3, 4)).sqrt()
            / features.square().mean(dim=(1, 2, 3, 4)).sqrt().clamp_min(self.eps)
        )
        audit = {
            "raw_band_fractions": raw_fractions,
            "descriptor_rms": descriptor.square().mean(dim=(1, 2, 3, 4)).sqrt(),
            "residual_relative_rms": relative_delta,
            "effective_strength": self.effective_strength.reshape(1),
            "nonidentity_fraction": relative_delta.gt(self.eps).float().mean().reshape(1),
        }
        self.last_audit = audit
        return (output, audit) if return_audit else output


class RawFrequencySkipScaleTable3D(ScaleTableInteractionAblation3D):
    """``raw MRI rFFT -> bounded skip -> layerN -> layer5 -> original CAPM``."""

    def __init__(
        self,
        *,
        preset: str = "layer5_pixel",
        skip_stage: str = "layer3",
        spectral_grid: Sequence[int] = (8, 8, 8),
        hidden_channels: int = 32,
        max_residual: float = 0.15,
        raw_gate_init: float = 0.1,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        if preset != "layer5_pixel":
            raise ValueError("RawFrequencySkipScaleTable3D requires preset='layer5_pixel'")
        if skip_stage not in RAW_SKIP_STAGES:
            raise ValueError(f"skip_stage must be one of {RAW_SKIP_STAGES}")
        super().__init__(preset=preset, interaction="original_capm", num_classes=num_classes, **kwargs)
        self.skip_stage = str(skip_stage)
        self.raw_frequency_skip = RawFrequencySkip3D(
            image_channels=1,
            target_channels=_STAGE_CHANNELS[self.skip_stage],
            spectral_grid=spectral_grid,
            hidden_channels=hidden_channels,
            max_residual=max_residual,
            gate_init=raw_gate_init,
        )

    def extract_features(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if force_capm:
            raise ValueError("original_capm has no force_capm path")
        features = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        audit: dict[str, Tensor] = {}
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(self, name)(features)
            if name == self.skip_stage:
                features, skip_audit = self.raw_frequency_skip(
                    image, features, return_audit=True
                )
                audit.update(
                    {f"raw_frequency_skip_{key}": value for key, value in skip_audit.items()}
                )
        features = self.layer5(features)
        features, capm_audit = self._apply_calibrator(features, table, False, return_audit)
        audit.update(capm_audit)
        return features, audit

    def forward(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, audit = self.extract_features(
            image, table, force_capm=force_capm, return_audit=return_audit
        )
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audit
        return logits

    def experiment_signature(self) -> dict[str, Any]:
        signature = super().experiment_signature()
        signature.update(
            {
                "model_family": "raw_frequency_skip",
                "raw_frequency_skip_stage": self.skip_stage,
                "raw_frequency_spectral_grid": self.raw_frequency_skip.spectral_grid,
                "raw_frequency_hidden_channels": self.raw_frequency_skip.hidden_channels,
                "raw_frequency_max_residual": self.raw_frequency_skip.max_residual,
                "raw_frequency_representation": "log_amplitude_cos_phase_sin_phase",
                "raw_frequency_band_names": BAND_NAMES,
            }
        )
        return signature


def build_raw_frequency_skip_model(**kwargs: Any) -> RawFrequencySkipScaleTable3D:
    """Build the source-only raw-frequency residual-skip candidate."""
    return RawFrequencySkipScaleTable3D(**kwargs)


__all__ = [
    "RAW_SKIP_STAGES",
    "RawFrequencySkip3D",
    "RawFrequencySkipScaleTable3D",
    "build_raw_frequency_skip_model",
    "extract_raw_frequency_features",
]
