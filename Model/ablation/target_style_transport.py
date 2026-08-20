"""Target-style feature transport for a reproducible UDA probe.

This module is deliberately a small experimental adapter, not a new claim
about domain adaptation.  It transports only the spatial frequency amplitude
of an unlabeled target feature map and keeps the source feature phase.  The
source label is reused for the transported feature, while target labels and
target covariates are never consumed by the forward pass.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TargetStyleFeatureTransport3D(nn.Module):
    """Mix source and target feature amplitudes in the spatial Fourier domain.

    ``strength=0`` is an exact identity control.  The target feature is
    detached by default: target samples provide a style statistic, but the
    adaptation loss cannot update a second target branch through the shared
    encoder.  Source phase is preserved to retain source semantic layout.
    """

    def __init__(
        self,
        strength: float = 0.5,
        *,
        eps: float = 1e-6,
        detach_target: bool = True,
        phase_mode: str = "source",
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.eps = float(eps)
        self.detach_target = bool(detach_target)
        self.phase_mode = str(phase_mode)
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.phase_mode not in {"source", "target"}:
            raise ValueError("phase_mode must be 'source' or 'target'")

    @staticmethod
    def _validate(source: Tensor, target: Tensor) -> None:
        if source.ndim != 5 or target.ndim != 5:
            raise ValueError("source and target features must have shape [B,C,D,H,W]")
        if source.shape[:2] != target.shape[:2]:
            raise ValueError(
                "source and target must have equal batch and channel dimensions; "
                f"got {tuple(source.shape)} and {tuple(target.shape)}"
            )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        *,
        strength: float | None = None,
        phase_mode: str | None = None,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
        self._validate(source, target)
        alpha = self.strength if strength is None else float(strength)
        selected_phase_mode = self.phase_mode if phase_mode is None else str(phase_mode)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        if selected_phase_mode not in {"source", "target"}:
            raise ValueError("phase_mode must be 'source' or 'target'")
        if target.shape[-3:] != source.shape[-3:]:
            target = F.interpolate(
                target,
                size=source.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        if self.detach_target:
            target = target.detach()
        if alpha == 0.0:
            output = source
            audit = {
                "strength": source.new_zeros(()),
                "amplitude_l1": source.new_zeros(()),
                "target_detached": source.new_tensor(float(self.detach_target)),
                "phase_mode": selected_phase_mode,
            }
            return (output, audit) if return_audit else output

        dims = (-3, -2, -1)
        source_fft = torch.fft.rfftn(source, dim=dims, norm="ortho")
        target_fft = torch.fft.rfftn(target, dim=dims, norm="ortho")
        source_amp = source_fft.abs()
        target_amp = target_fft.abs()
        mixed_amp = torch.lerp(source_amp, target_amp, alpha)
        # Unit complex phase is more stable than differentiating angle().
        phase_fft = source_fft if selected_phase_mode == "source" else target_fft
        phase_amplitude = source_amp if selected_phase_mode == "source" else target_amp
        selected_phase = phase_fft / phase_amplitude.clamp_min(self.eps)
        mixed_fft = mixed_amp * selected_phase
        transported = torch.fft.irfftn(
            mixed_fft,
            s=tuple(int(size) for size in source.shape[-3:]),
            dim=dims,
            norm="ortho",
        )
        # The inverse transform can have a small numerical scale drift.  Match
        # source mean/std so the adapter cannot become an intensity re-scaler.
        reduce_dims = (-3, -2, -1)
        source_mean = source.mean(dim=reduce_dims, keepdim=True)
        source_std = source.std(dim=reduce_dims, keepdim=True, unbiased=False)
        transported_mean = transported.mean(dim=reduce_dims, keepdim=True)
        transported_std = transported.std(dim=reduce_dims, keepdim=True, unbiased=False)
        output = (transported - transported_mean) / transported_std.clamp_min(self.eps)
        output = output * source_std + source_mean
        audit = {
            "strength": source.new_tensor(alpha),
            "amplitude_l1": (source_amp - target_amp).abs().mean().detach(),
            "target_detached": source.new_tensor(float(self.detach_target)),
            "phase_mode": selected_phase_mode,
            "source_feature_mean": source.mean().detach(),
            "source_feature_std": source.std(unbiased=False).detach(),
            "source_feature_norm": source.norm().detach(),
            "transported_feature_mean": output.mean().detach(),
            "transported_feature_std": output.std(unbiased=False).detach(),
            "transported_feature_norm": output.norm().detach(),
            "transported_feature_finite": torch.isfinite(output).all().to(source.dtype).detach(),
        }
        return (output, audit) if return_audit else output


class TargetStyleCAPM(nn.Module):
    """Wrap an existing layer4 CAPM model with source-label transport training.

    The wrapped model must be the existing ``ScaleTableInteractionAblation3D``
    contract.  Feature extraction is table-free; CAPM is applied only after
    transport, using the source covariates for both clean and transported
    source branches.  ``predict`` is target-inference only and never adapts on
    target labels.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        transport_strength: float = 0.5,
        transport_phase_mode: str = "source",
        force_capm: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "_apply_calibrator"):
            raise TypeError("backbone must expose the scale-table CAPM contract")
        preset = getattr(backbone, "preset", None)
        if getattr(preset, "selected_stage", None) != "layer4":
            raise ValueError("TargetStyleCAPM currently requires a layer4 preset")
        if getattr(backbone, "interaction", None) != "capm":
            raise ValueError("TargetStyleCAPM requires interaction='capm'")
        self.backbone = backbone
        self.transport = TargetStyleFeatureTransport3D(transport_strength, phase_mode=transport_phase_mode)
        self.force_capm = bool(force_capm)

    def extract_features(self, image: Tensor) -> Tensor:
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(image))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(self.backbone, name)(x)
        return x

    def classify_features(
        self, features: Tensor, covariates: Tensor, *, return_audit: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        conditioned, capm_audit = self.backbone._apply_calibrator(
            features,
            covariates,
            force_capm=self.force_capm,
            return_audit=return_audit,
        )
        logits = self.backbone.fc(self.backbone.dropout(self.backbone.pool(conditioned).flatten(1)))
        return (logits, capm_audit) if return_audit else logits

    def predict(self, image: Tensor, covariates: Tensor) -> Tensor:
        return self.classify_features(self.extract_features(image), covariates)

    def forward(
        self,
        image: Tensor,
        covariates: Tensor,
        *,
        target_image: Tensor | None = None,
        return_audit: bool = False,
    ) -> dict[str, Any]:
        source_features = self.extract_features(image)
        clean_result = self.classify_features(source_features, covariates, return_audit=return_audit)
        clean_logits, clean_capm_audit = (
            clean_result if return_audit else (clean_result, None)
        )
        result: dict[str, Any] = {
            "clean_logits": clean_logits,
            "source_features": source_features,
        }
        if return_audit:
            result["clean_capm_audit"] = clean_capm_audit
        if target_image is None:
            return result
        target_features = self.extract_features(target_image)
        mixed_features, audit = self.transport(
            source_features,
            target_features,
            return_audit=True,
        )
        result["mixed_features"] = mixed_features
        mixed_result = self.classify_features(mixed_features, covariates, return_audit=return_audit)
        mixed_logits, mixed_capm_audit = (
            mixed_result if return_audit else (mixed_result, None)
        )
        result["mixed_logits"] = mixed_logits
        if return_audit:
            result["transport_audit"] = audit
            result["mixed_capm_audit"] = mixed_capm_audit
        return result


__all__ = ["TargetStyleCAPM", "TargetStyleFeatureTransport3D"]
