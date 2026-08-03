"""Model-only APIS v3 candidates for scan-aware domain generalization.

The candidates share the APIS v2 public surface so they can be selected by
``DualShiftResNet3D(apis_variant=...)`` without changing data or training code.
None of the candidates consumes target labels or fits target-domain statistics.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.dual_shift.apis import ProtocolResidualOperator, channel_stats


APIS_V3_VARIANTS = (
    "v3_distance_residual",
    "v3_latent_prompt",
    "v3_monotonic_intensity",
    "v3_hybrid",
)


class ShallowStylePrompt(nn.Module):
    """Compress shallow feature statistics into a bounded latent-domain prompt."""

    def __init__(self, channels: int, prompt_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels) * 2)
        self.projection = nn.Sequential(
            nn.Linear(int(channels) * 2, int(prompt_dim)),
            nn.GELU(),
            nn.Linear(int(prompt_dim), int(prompt_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean, std = channel_stats(features)
        stats = torch.cat([mean.flatten(1), std.flatten(1)], dim=1)
        return torch.tanh(self.projection(self.norm(stats)))


class MonotonicIntensityIntervention(nn.Module):
    """Conditioned piecewise-linear intensity mapping with positive slopes.

    The mapping is monotonic and remains inside each image's factual intensity
    range. Zero-initialized controller weights yield an exact identity map.
    """

    def __init__(self, condition_dim: int, segments: int = 8) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.segments = int(segments)
        if self.segments < 2:
            raise ValueError("segments must be >= 2")
        self.controller = nn.Linear(self.condition_dim, self.segments)
        nn.init.zeros_(self.controller.weight)
        nn.init.zeros_(self.controller.bias)

    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        strength: float,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if image.ndim != 5:
            raise ValueError("monotonic APIS expects [B, C, D, H, W]")
        if condition.shape != (image.shape[0], self.condition_dim):
            raise ValueError(
                f"condition must be [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if strength <= 0:
            zeros = image.new_zeros((image.shape[0],))
            return image, {"strength": zeros.mean(), "realized_per_sample": zeros}

        work = image.float()
        minimum = work.flatten(1).amin(dim=1).reshape(-1, 1, 1, 1, 1)
        maximum = work.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1, 1)
        span = (maximum - minimum).clamp_min(1e-6)
        normalized = ((work - minimum) / span).clamp(0.0, 1.0)

        # Positive slopes guarantee monotonicity. Normalization makes the
        # cumulative map start at zero and end at one.
        slopes = F.softplus(self.controller(condition).float()) + 1e-4
        slope_sum = slopes.sum(dim=1, keepdim=True)
        segment_mass = slopes / slope_sum
        cumulative = torch.cumsum(segment_mass, dim=1)
        cumulative_before = F.pad(cumulative[:, :-1], (1, 0), value=0.0)

        scaled = normalized * float(self.segments)
        index = scaled.floor().long().clamp(max=self.segments - 1)
        fraction = (scaled - index.to(scaled.dtype)).clamp(0.0, 1.0)
        gather_index = index.reshape(image.shape[0], -1)
        before = cumulative_before.gather(1, gather_index).reshape_as(normalized)
        mass = segment_mass.gather(1, gather_index).reshape_as(normalized)
        mapped = before + fraction * mass
        alpha = float(max(0.0, min(strength, 1.0)))
        blended = normalized + alpha * (mapped - normalized)
        shifted = (minimum + span * blended).to(image.dtype)

        if valid_mask is not None:
            mask = valid_mask.to(image.device, image.dtype).reshape(-1, 1, 1, 1, 1)
            shifted = image + mask * (shifted - image)
        realized = (
            (shifted.float() - work).flatten(1).square().mean(1).sqrt()
            / span.flatten().clamp_min(1e-6)
        )
        return shifted, {
            "strength": realized.mean(),
            "realized_per_sample": realized,
        }


class APISV3Module(nn.Module):
    """Collection of bounded APIS v3 interventions behind one stable API."""

    def __init__(
        self,
        *,
        variant: str,
        layer1_channels: int,
        layer2_channels: int,
        acquisition_dim: int,
        alpha_max: float = 0.25,
        basis_count: int = 4,
        rank: int = 8,
        intensity_segments: int = 8,
    ) -> None:
        super().__init__()
        if variant not in APIS_V3_VARIANTS:
            raise ValueError(
                f"Unknown APIS v3 variant {variant!r}; expected {APIS_V3_VARIANTS}"
            )
        self.variant = str(variant)
        self.alpha_max = float(alpha_max)
        self.acquisition_dim = int(acquisition_dim)
        self.enabled = True
        self.current_alpha = 0.0
        self.uses_latent_prompt = self.variant in {"v3_latent_prompt", "v3_hybrid"}
        self.uses_monotonic_intensity = self.variant in {
            "v3_monotonic_intensity",
            "v3_hybrid",
        }
        self.uses_feature_residual = self.variant in {
            "v3_distance_residual",
            "v3_latent_prompt",
            "v3_hybrid",
        }

        self.scan_condition_dim = self.acquisition_dim * 3
        self.style_prompt = ShallowStylePrompt(layer1_channels, self.acquisition_dim)
        self.condition_dim = self.scan_condition_dim + (
            self.acquisition_dim if self.uses_latent_prompt else 0
        )
        self.reliability = nn.Linear(self.condition_dim, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.layer1_operator = ProtocolResidualOperator(
            layer1_channels,
            self.condition_dim,
            basis_count=basis_count,
            rank=rank,
        )
        self.layer2_operator = ProtocolResidualOperator(
            layer2_channels,
            self.condition_dim,
            basis_count=basis_count,
            rank=rank,
        )
        self.intensity = MonotonicIntensityIntervention(
            self.condition_dim, segments=intensity_segments
        )
        self._condition_gate: Optional[torch.Tensor] = None
        self._protocol_distance: Optional[torch.Tensor] = None
        self._last_audit: Dict[str, Dict[str, torch.Tensor]] = {}
        self._last_intensity_audit: Dict[str, torch.Tensor] = {}

    @property
    def mode_name(self) -> str:
        return self.variant

    def set_alpha(self, alpha: float) -> None:
        self.current_alpha = float(max(0.0, min(alpha, self.alpha_max)))

    def protocol_condition(
        self,
        factual: torch.Tensor,
        target: torch.Tensor,
        *,
        layer1_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if factual.shape != target.shape:
            raise ValueError("factual and target acquisition embeddings must match")
        delta = target - factual
        scan_condition = torch.cat([factual, target, delta], dim=1)
        distance = delta.float().square().mean(dim=1).sqrt()
        self._protocol_distance = distance

        if self.uses_latent_prompt:
            if layer1_features is None:
                raise ValueError(f"{self.variant} requires clean layer1 features")
            prompt = self.style_prompt(layer1_features)
            condition = torch.cat([scan_condition, prompt], dim=1)
            # Learned reliability is bounded away from both zero and one. This
            # prevents a noisy latent prompt from fully disabling or dominating
            # an observed scan-parameter intervention.
            learned = torch.sigmoid(self.reliability(condition)).flatten()
            gate = 0.25 + 0.75 * learned
        else:
            condition = scan_condition
            # Same-protocol pairs are identity; increasingly distant observed
            # protocols receive smoothly larger, but always bounded, weight.
            gate = torch.tanh(distance / math.sqrt(float(self.acquisition_dim)))
        self._condition_gate = gate.to(factual.dtype)
        return condition

    def make_shifted_image(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._last_intensity_audit = {}
        if not self.uses_monotonic_intensity:
            return image
        alpha = self.current_alpha if self.enabled else 0.0
        shifted, audit = self.intensity(
            image,
            condition,
            alpha,
            valid_mask=valid_mask,
        )
        self._last_intensity_audit = audit
        return shifted

    def make_shift_fns(
        self,
        condition: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        alpha = self.current_alpha if self.enabled else 0.0
        self._last_audit = {}
        valid = None if valid_mask is None else valid_mask.to(dtype=torch.bool).reshape(-1)
        gate = self._condition_gate
        if gate is None:
            gate = condition.new_ones((condition.shape[0],))
        if valid is not None:
            gate = gate * valid.to(gate.device, gate.dtype)

        def apply(
            features: torch.Tensor,
            operator: ProtocolResidualOperator,
            key: str,
        ) -> torch.Tensor:
            if alpha <= 0 or not self.uses_feature_residual:
                return features
            shifted, audit = operator(features, condition, alpha)
            keep = gate.to(features.device, features.dtype).reshape(
                -1, *([1] * (features.ndim - 1))
            )
            shifted = features + keep * (shifted - features)
            coeff = audit["coefficient_l2"] * gate.to(
                audit["coefficient_l2"].device,
                audit["coefficient_l2"].dtype,
            )
            realized = audit["realized_per_sample"] * gate.to(
                audit["realized_per_sample"].device,
                audit["realized_per_sample"].dtype,
            )
            self._last_audit[key] = {
                "strength": realized.mean(),
                "coefficient_l2": coeff,
                "realized_per_sample": realized,
            }
            return shifted

        def shift1(features: torch.Tensor) -> torch.Tensor:
            return apply(features, self.layer1_operator, "layer1")

        def shift2(features: torch.Tensor) -> torch.Tensor:
            return apply(features, self.layer2_operator, "layer2")

        return shift1, shift2

    def audit_tensors(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        per_sample_zero = reference.new_zeros((reference.shape[0],))
        feature_strengths = [entry["strength"] for entry in self._last_audit.values()]
        coefficients = [entry["coefficient_l2"] for entry in self._last_audit.values()]
        per_sample = [entry["realized_per_sample"] for entry in self._last_audit.values()]
        feature_strength = (
            torch.stack(feature_strengths).mean() if feature_strengths else zero
        )
        intensity_strength = self._last_intensity_audit.get("strength", zero)
        total_strength = torch.maximum(feature_strength, intensity_strength)
        coefficient_l2 = (
            torch.stack([item.mean() for item in coefficients]).mean()
            if coefficients
            else zero
        )
        coefficient_per_sample = (
            torch.stack(coefficients).mean(dim=0) if coefficients else per_sample_zero
        )
        realized_per_sample = (
            torch.stack(per_sample).mean(dim=0) if per_sample else per_sample_zero
        )
        return {
            "strength": total_strength,
            "feature_strength": feature_strength,
            "intensity_strength": intensity_strength,
            "coefficient_l2": coefficient_l2,
            "coefficient_l2_per_sample": coefficient_per_sample,
            "realized_per_sample": realized_per_sample,
            "protocol_distance": (
                self._protocol_distance.mean().to(reference.device)
                if self._protocol_distance is not None
                else zero
            ),
            "condition_gate": (
                self._condition_gate.mean().to(reference.device)
                if self._condition_gate is not None
                else zero
            ),
        }


def build_apis_v3(
    variant: str,
    *,
    layer1_channels: int,
    layer2_channels: int,
    acquisition_dim: int,
    alpha_max: float,
    basis_count: int,
    rank: int,
) -> APISV3Module:
    return APISV3Module(
        variant=variant,
        layer1_channels=layer1_channels,
        layer2_channels=layer2_channels,
        acquisition_dim=acquisition_dim,
        alpha_max=alpha_max,
        basis_count=basis_count,
        rank=rank,
    )
