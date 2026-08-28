"""Source-side 3-D Fourier MixStyle for ScaleTable CAPM models.

The module deliberately does not consume a target prior, domain identifier, or
target label.  During training it mixes source feature amplitudes from a
same-class donor while retaining the recipient phase; evaluation is an exact
identity path.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .frequency_uda import BAND_NAMES, DEFAULT_BAND_EDGES, radial_band_masks_rfft
from .scale_table_transformer import ScaleTableInteractionAblation3D


MIX_MODES = ("full_amplitude", "bandwise_statistics")


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _validate_alpha(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("alpha must be positive")
    return value


def _same_class_partners(labels: Tensor | None, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return a non-self donor for each class group with at least two samples."""
    partners = torch.arange(batch_size, device=device)
    eligible = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if labels is None:
        return partners, eligible
    labels = labels.to(device=device).reshape(-1)
    if labels.shape != (batch_size,):
        raise ValueError("style_labels must have shape [B]")
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        shuffled = indices[torch.randperm(indices.numel(), device=device)]
        partners[shuffled] = torch.roll(shuffled, shifts=1)
        eligible[indices] = True
    return partners, eligible


def _unconditional_partners(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    partners = torch.arange(batch_size, device=device)
    eligible = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if batch_size < 2:
        return partners, eligible
    shuffled = torch.randperm(batch_size, device=device)
    partners[shuffled] = torch.roll(shuffled, shifts=1)
    eligible.fill_(True)
    return partners, eligible


class FrequencyMixStyle3D(nn.Module):
    """Mix source 3-D feature amplitudes while retaining recipient phase.

    ``full_amplitude`` is a FACT-like whole-spectrum control.  In
    ``bandwise_statistics`` mode, low/mid/high amplitude means and standard
    deviations are mixed independently.  Both modes use only same-label source
    donors and return an identity result outside training.
    """

    def __init__(
        self,
        *,
        mode: str = "bandwise_statistics",
        probability: float = 0.5,
        alpha: float = 0.1,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
        class_conditional: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in MIX_MODES:
            raise ValueError(f"mode must be one of {MIX_MODES}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.mode = str(mode)
        self.probability = _validate_probability(probability, name="probability")
        self.alpha = _validate_alpha(alpha)
        self.band_edges = tuple(float(value) for value in band_edges)
        self.class_conditional = bool(class_conditional)
        self.eps = float(eps)
        self.last_audit: dict[str, Tensor] | None = None

    def _sample_lambda(self, batch_size: int, bands: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        concentration = torch.full((batch_size, bands), self.alpha, device=device, dtype=dtype)
        return torch.distributions.Beta(concentration, concentration).sample()

    def _bandwise_statistics_mix(
        self,
        amplitude: Tensor,
        partners: Tensor,
        mix_lambda: Tensor,
        masks: Tensor,
    ) -> Tensor:
        # amplitude: [B,C,D,H,W], masks: [K,D,H,W], statistics: [B,K,C].
        band_count = masks.sum(dim=(-3, -2, -1)).clamp_min(1.0).reshape(1, -1, 1)
        values = amplitude.unsqueeze(1)
        weighted = values * masks[None, :, None]
        mean = weighted.sum(dim=(-3, -2, -1)) / band_count
        centered = values - mean[..., None, None, None]
        variance = (centered.square() * masks[None, :, None]).sum(dim=(-3, -2, -1)) / band_count
        std = variance.clamp_min(self.eps).sqrt()
        donor_mean = mean[partners]
        donor_std = std[partners]
        weight = mix_lambda[..., None]
        mixed_mean = weight * mean + (1.0 - weight) * donor_mean
        mixed_std = weight * std + (1.0 - weight) * donor_std
        standardized = centered / std[..., None, None, None]
        restyled = (standardized * mixed_std[..., None, None, None] + mixed_mean[..., None, None, None]).clamp_min(0.0)
        return (restyled.permute(0, 2, 1, 3, 4, 5) * masks[None, None]).sum(dim=2)

    def _audit(
        self,
        original: Tensor,
        mixed: Tensor,
        amplitude: Tensor,
        mixed_amplitude: Tensor,
        masks: Tensor,
        partners: Tensor,
        donor_eligible: Tensor,
        applied: Tensor,
        mix_lambda: Tensor,
    ) -> dict[str, Tensor]:
        amplitude_delta = mixed_amplitude - amplitude
        amplitude_relative_delta = amplitude_delta.square().mean(dim=(1, 2, 3, 4)).sqrt() / amplitude.square().mean(dim=(1, 2, 3, 4)).sqrt().clamp_min(self.eps)
        feature_relative_delta = (mixed - original).square().mean(dim=(1, 2, 3, 4)).sqrt() / original.square().mean(dim=(1, 2, 3, 4)).sqrt().clamp_min(self.eps)
        band_delta = (amplitude_delta.abs().unsqueeze(1) * masks[None, :, None]).sum(dim=(2, 3, 4, 5))
        band_scale = (amplitude.abs().unsqueeze(1) * masks[None, :, None]).sum(dim=(2, 3, 4, 5)).clamp_min(self.eps)
        return {
            "applied_mask": applied,
            "applied_fraction": applied.float().mean().reshape(1),
            "partner_indices": partners,
            "donor_eligible_mask": donor_eligible,
            "donor_eligible_fraction": donor_eligible.float().mean().reshape(1),
            "identity_fallback_fraction": (1.0 - applied.float().mean()).reshape(1),
            "mix_lambda": mix_lambda,
            "amplitude_relative_delta": amplitude_relative_delta,
            "band_amplitude_relative_delta": band_delta / band_scale,
            "feature_relative_delta": feature_relative_delta,
            "phase_preserved": torch.ones(original.shape[0], dtype=torch.bool, device=original.device),
        }

    def forward(self, features: Tensor, *, style_labels: Tensor | None = None, return_audit: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,C,D,H,W]")
        batch_size = int(features.shape[0])
        if not self.training or self.probability == 0.0 or batch_size < 2:
            audit = self._identity_audit(features)
            self.last_audit = audit
            return (features, audit) if return_audit else features
        if self.class_conditional and style_labels is None:
            raise ValueError("class-conditional FrequencyMixStyle3D requires source labels during training")
        if self.class_conditional:
            partners, donor_eligible = _same_class_partners(style_labels, batch_size, features.device)
        else:
            partners, donor_eligible = _unconditional_partners(batch_size, features.device)
        applied = donor_eligible & (torch.rand(batch_size, device=features.device) < self.probability)
        if not applied.any():
            audit = self._identity_audit(
                features,
                partners=partners,
                donor_eligible=donor_eligible,
            )
            self.last_audit = audit
            return (features, audit) if return_audit else features

        spectrum = torch.fft.rfftn(features.float(), dim=(-3, -2, -1), norm="ortho")
        amplitude = spectrum.abs()
        masks = radial_band_masks_rfft(features.shape[-3:], band_edges=self.band_edges, device=features.device, dtype=amplitude.dtype)
        n_bands = 1 if self.mode == "full_amplitude" else len(BAND_NAMES)
        mix_lambda = self._sample_lambda(batch_size, n_bands, features.device, amplitude.dtype)
        if self.mode == "full_amplitude":
            weight = mix_lambda[:, 0, None, None, None, None]
            candidate_amplitude = weight * amplitude + (1.0 - weight) * amplitude[partners]
        else:
            candidate_amplitude = self._bandwise_statistics_mix(amplitude, partners, mix_lambda, masks)
        mixed_amplitude = torch.where(applied[:, None, None, None, None], candidate_amplitude, amplitude)
        mixed_spectrum = torch.polar(mixed_amplitude, torch.angle(spectrum))
        mixed = torch.fft.irfftn(mixed_spectrum, s=tuple(features.shape[-3:]), dim=(-3, -2, -1), norm="ortho").to(dtype=features.dtype)
        # rFFT/irFFT round-off would otherwise perturb ineligible samples.
        mixed = torch.where(applied[:, None, None, None, None], mixed, features)
        audit = self._audit(
            features,
            mixed,
            amplitude,
            mixed_amplitude,
            masks,
            partners,
            donor_eligible,
            applied,
            mix_lambda,
        )
        self.last_audit = audit
        return (mixed, audit) if return_audit else mixed

    def _identity_audit(
        self,
        features: Tensor,
        *,
        partners: Tensor | None = None,
        donor_eligible: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size = int(features.shape[0])
        if partners is None:
            partners = torch.arange(batch_size, device=features.device)
        if donor_eligible is None:
            donor_eligible = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        bands = 1 if self.mode == "full_amplitude" else len(BAND_NAMES)
        return {
            "applied_mask": torch.zeros(batch_size, dtype=torch.bool, device=features.device),
            "applied_fraction": features.new_zeros(1),
            "partner_indices": partners,
            "donor_eligible_mask": donor_eligible,
            "donor_eligible_fraction": donor_eligible.float().mean().reshape(1),
            "identity_fallback_fraction": features.new_ones(1),
            "mix_lambda": features.new_ones((batch_size, bands)),
            "amplitude_relative_delta": features.new_zeros(batch_size),
            "band_amplitude_relative_delta": features.new_zeros((batch_size, len(BAND_NAMES))),
            "feature_relative_delta": features.new_zeros(batch_size),
            "phase_preserved": torch.ones(batch_size, dtype=torch.bool, device=features.device),
        }


class FrequencyMixStyleScaleTable3D(ScaleTableInteractionAblation3D):
    """``layerN -> FrequencyMixStyle3D -> ... -> layer5 -> original CAPM``."""

    def __init__(
        self,
        *,
        preset: str = "layer5_pixel",
        mix_stage: str = "layer3",
        mix_mode: str = "bandwise_statistics",
        probability: float = 0.5,
        alpha: float = 0.1,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
        class_conditional: bool = True,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        if preset != "layer5_pixel":
            raise ValueError("FrequencyMixStyleScaleTable3D requires preset='layer5_pixel'")
        if mix_stage not in {"layer2", "layer3", "layer4"}:
            raise ValueError("mix_stage must be one of layer2, layer3, layer4")
        super().__init__(preset=preset, interaction="original_capm", num_classes=num_classes, **kwargs)
        self.mix_stage = str(mix_stage)
        self.frequency_mixstyle = FrequencyMixStyle3D(
            mode=mix_mode,
            probability=probability,
            alpha=alpha,
            band_edges=band_edges,
            class_conditional=class_conditional,
        )

    def extract_features(self, image: Tensor, table: Tensor | None = None, *, style_labels: Tensor | None = None, force_capm: bool = False, return_audit: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        if force_capm:
            raise ValueError("original_capm has no force_capm path")
        features = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        audit: dict[str, Tensor] = {}
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(self, name)(features)
            if name == self.mix_stage:
                features, mix_audit = self.frequency_mixstyle(features, style_labels=style_labels, return_audit=True)
                audit.update({f"frequency_mixstyle_{key}": value for key, value in mix_audit.items()})
        features = self.layer5(features)
        features, capm_audit = self._apply_calibrator(features, table, False, return_audit)
        audit.update(capm_audit)
        return features, audit

    def forward(self, image: Tensor, table: Tensor | None = None, *, style_labels: Tensor | None = None, force_capm: bool = False, return_audit: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, audit = self.extract_features(image, table, style_labels=style_labels, force_capm=force_capm, return_audit=return_audit)
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audit
        return logits

    def experiment_signature(self) -> dict[str, Any]:
        signature = super().experiment_signature()
        signature.update(
            {
                "model_family": "frequency_mixstyle",
                "frequency_mixstyle_insert_stage": self.mix_stage,
                "frequency_mixstyle_mode": self.frequency_mixstyle.mode,
                "frequency_mixstyle_probability": self.frequency_mixstyle.probability,
                "frequency_mixstyle_alpha": self.frequency_mixstyle.alpha,
                "frequency_mixstyle_band_names": BAND_NAMES,
                "frequency_mixstyle_band_edges": self.frequency_mixstyle.band_edges,
                "frequency_mixstyle_class_conditional": self.frequency_mixstyle.class_conditional,
            }
        )
        return signature


def build_frequency_mixstyle_model(*, mix_mode: str = "bandwise_statistics", **kwargs: Any) -> FrequencyMixStyleScaleTable3D:
    """Build either the complete-amplitude or band-wise Frequency MixStyle model."""
    return FrequencyMixStyleScaleTable3D(mix_mode=mix_mode, **kwargs)


__all__ = [
    "MIX_MODES",
    "FrequencyMixStyle3D",
    "FrequencyMixStyleScaleTable3D",
    "build_frequency_mixstyle_model",
]
