"""APIS v3 candidates for acquisition-robust domain generalization.

The image-only style-memory candidate does not consume scanner metadata, target
labels, or target-domain statistics. Legacy scan-conditioned candidates remain
available as ablations behind ``DualShiftResNet3D(apis_variant=...)``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.dual_shift.apis import ProtocolResidualOperator, channel_stats


APIS_V3_SCAN_VARIANTS = (
    "v3_distance_residual",
    "v3_latent_prompt",
    "v3_monotonic_intensity",
    "v3_hybrid",
)

APIS_V3_VARIANTS = APIS_V3_SCAN_VARIANTS + (
    "v3_style_memory",
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
        if variant not in APIS_V3_SCAN_VARIANTS:
            raise ValueError(
                f"Unknown scan-conditioned APIS v3 variant {variant!r}; "
                f"expected {APIS_V3_SCAN_VARIANTS}"
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
class StyleMemoryAPISV3(nn.Module):
    """Image-only APIS v3 using latent style memory and content gating.

    The module discovers nuisance/style directions from source images rather
    than consuming scanner metadata.  Memory entries are EMA statistics and
    are never differentiated through, while the residual operators remain
    trainable.  A confidence/content gate keeps uncertain style transfers on
    the clean path.
    """

    def __init__(
        self,
        *,
        layer1_channels: int,
        layer2_channels: int,
        layer3_channels: int,
        alpha_max: float = 0.25,
        basis_count: int = 4,
        rank: int = 8,
        style_dim: int = 16,
        memory_size: int = 8,
        memory_beta: float = 0.95,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.variant = "v3_style_memory"
        self.alpha_max = float(alpha_max)
        self.current_alpha = 0.0
        self.enabled = True
        self.style_dim = int(style_dim)
        self.memory_size = int(memory_size)
        self.memory_beta = float(memory_beta)
        self.temperature = float(max(temperature, 1e-3))
        if self.memory_size < 2:
            raise ValueError("memory_size must be >= 2")

        # Mean/std alone would collapse back toward MixStyle.  Multi-scale
        # low/high-frequency energy adds image-derived degradation cues while
        # remaining independent of scanner metadata.
        style_input_dim = 4 * (int(layer1_channels) + int(layer2_channels))
        anatomy_input_dim = 2 * int(layer3_channels)
        self.style_encoder = nn.Sequential(
            nn.LayerNorm(style_input_dim),
            nn.Linear(style_input_dim, max(32, self.style_dim * 2)),
            nn.GELU(),
            nn.Linear(max(32, self.style_dim * 2), self.style_dim),
        )
        self.anatomy_encoder = nn.Sequential(
            nn.LayerNorm(anatomy_input_dim),
            nn.Linear(anatomy_input_dim, self.style_dim),
            nn.Tanh(),
        )

        # The buffers are deliberately source-image statistics, not metadata
        # prototypes.  They are included in checkpoints for reproducibility.
        self.register_buffer(
            "style_prototypes", torch.zeros(self.memory_size, self.style_dim)
        )
        self.register_buffer("style_counts", torch.zeros(self.memory_size))
        self.register_buffer(
            "style_valid", torch.zeros(self.memory_size, dtype=torch.bool)
        )

        condition_dim = self.style_dim * 3 + 2
        self.condition_dim = condition_dim
        self.gate = nn.Sequential(
            nn.Linear(self.style_dim * 2 + 2, self.style_dim),
            nn.GELU(),
            nn.Linear(self.style_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.layer1_operator = ProtocolResidualOperator(
            layer1_channels, condition_dim, basis_count=basis_count, rank=rank
        )
        self.layer2_operator = ProtocolResidualOperator(
            layer2_channels, condition_dim, basis_count=basis_count, rank=rank
        )
        self._gate: Optional[torch.Tensor] = None
        self._confidence: Optional[torch.Tensor] = None
        self._style_entropy: Optional[torch.Tensor] = None
        self._style_delta: Optional[torch.Tensor] = None
        self._last_audit: Dict[str, Dict[str, torch.Tensor]] = {}

    @property
    def mode_name(self) -> str:
        return self.variant

    def set_alpha(self, alpha: float) -> None:
        self.current_alpha = float(max(0.0, min(alpha, self.alpha_max)))

    def _style_stats(self, layer1: torch.Tensor, layer2: torch.Tensor) -> torch.Tensor:
        def summarize(features: torch.Tensor) -> torch.Tensor:
            mean, std = channel_stats(features)
            padded = F.pad(features, (1, 1, 1, 1, 1, 1), mode="replicate")
            low = F.avg_pool3d(padded, kernel_size=3, stride=1)
            high = features - low
            low_energy = (
                low.float().square().mean(dim=(2, 3, 4)).sqrt().to(mean.dtype)
            )
            high_energy = (
                high.float().square().mean(dim=(2, 3, 4)).sqrt().to(mean.dtype)
            )
            return torch.cat(
                [mean.flatten(1), std.flatten(1), low_energy, high_energy], dim=1
            )

        return torch.cat([summarize(layer1), summarize(layer2)], dim=1)

    def _anatomy_stats(self, layer3: torch.Tensor) -> torch.Tensor:
        mean, std = channel_stats(layer3)
        return torch.cat([mean.flatten(1), std.flatten(1)], dim=1)

    @torch.no_grad()
    def _update_memory(self, style: torch.Tensor) -> None:
        """Update source style memory without letting it become a gradient path."""
        style_values = style.detach()
        for value in style_values:
            valid = torch.nonzero(self.style_valid, as_tuple=False).flatten()
            if valid.numel() == 0:
                slot = 0
            elif valid.numel() < self.memory_size:
                distances = (self.style_prototypes[valid] - value).square().sum(dim=1)
                slot = int(valid[distances.argmin()].item())
                if float(self.style_counts[slot]) >= 4.0:
                    empty = torch.nonzero(~self.style_valid, as_tuple=False).flatten()
                    slot = int(empty[0].item())
            else:
                distances = (self.style_prototypes - value).square().sum(dim=1)
                slot = int(distances.argmin().item())
            if not bool(self.style_valid[slot]):
                self.style_prototypes[slot].copy_(value)
                self.style_counts[slot] = 1.0
                self.style_valid[slot] = True
            else:
                beta = self.memory_beta
                self.style_prototypes[slot].mul_(beta).add_(value, alpha=1.0 - beta)
                self.style_counts[slot].add_(1.0)

    def _target_style(
        self, style: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = torch.nonzero(self.style_valid, as_tuple=False).flatten()
        if valid.numel() < 2:
            if style.shape[0] > 1:
                target = torch.roll(style.detach(), shifts=1, dims=0)
                confidence = style.new_full((style.shape[0],), 0.25)
                entropy = style.new_ones((style.shape[0],))
            else:
                target = style.detach()
                confidence = style.new_zeros((style.shape[0],))
                entropy = style.new_zeros((style.shape[0],))
            return target, confidence, entropy

        prototypes = self.style_prototypes[valid].detach()
        distances = torch.cdist(style.float(), prototypes.float(), p=2)
        assignment = torch.softmax(-distances / self.temperature, dim=1)
        own = assignment.argmax(dim=1)
        weights = assignment.clone()
        weights[torch.arange(style.shape[0], device=style.device), own] = 0.0
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target = (weights @ prototypes) / denominator
        confidence = assignment.max(dim=1).values
        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(float(valid.numel()))
        return target, confidence, entropy

    def prepare_style_condition(
        self,
        layer1: torch.Tensor,
        layer2: torch.Tensor,
        layer3: torch.Tensor,
        *,
        update_memory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        style = torch.tanh(self.style_encoder(self._style_stats(layer1, layer2)))
        anatomy = self.anatomy_encoder(self._anatomy_stats(layer3))
        target, confidence, entropy = self._target_style(style)
        delta = target - style
        delta_norm = delta.float().square().mean(dim=1).sqrt()
        gate_input = torch.cat(
            [anatomy, delta, confidence.unsqueeze(1), (1.0 - entropy).unsqueeze(1)],
            dim=1,
        )
        learned_gate = 0.25 + 0.75 * torch.sigmoid(self.gate(gate_input)).flatten()
        # High-entropy assignments and tiny style changes should stay close to
        # the clean path, while preserving a differentiable signal for learning.
        certainty = 0.25 + 0.75 * (1.0 - entropy).clamp(0.0, 1.0)
        gate = learned_gate * confidence * certainty
        gate = gate * (delta_norm > 1e-4).to(gate.dtype)
        condition = torch.cat(
            [style, target.detach(), delta, confidence.unsqueeze(1), gate.unsqueeze(1)],
            dim=1,
        )
        self._gate = gate
        self._confidence = confidence
        self._style_entropy = entropy
        self._style_delta = delta_norm
        if update_memory:
            self._update_memory(style)
        return condition, gate > 1e-4

    def make_shift_fns(self, condition: torch.Tensor, *, valid_mask=None):
        alpha = self.current_alpha if self.enabled else 0.0
        self._last_audit = {}
        valid = None if valid_mask is None else valid_mask.to(dtype=torch.bool).reshape(-1)
        gate = self._gate
        if gate is None:
            gate = condition.new_zeros((condition.shape[0],))
        if valid is not None:
            gate = gate * valid.to(gate.device, gate.dtype)

        def apply(features: torch.Tensor, operator: ProtocolResidualOperator, key: str):
            if alpha <= 0:
                return features
            shifted, audit = operator(features, condition, alpha)
            keep = gate.to(features.device, features.dtype).reshape(
                -1, *([1] * (features.ndim - 1))
            )
            shifted = features + keep * (shifted - features)
            coeff = audit["coefficient_l2"] * gate.to(
                audit["coefficient_l2"].device, audit["coefficient_l2"].dtype
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

        return (
            lambda features: apply(features, self.layer1_operator, "layer1"),
            lambda features: apply(features, self.layer2_operator, "layer2"),
        )

    def audit_tensors(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        per_sample_zero = reference.new_zeros((reference.shape[0],))
        strengths = [entry["strength"] for entry in self._last_audit.values()]
        coeffs = [entry["coefficient_l2"] for entry in self._last_audit.values()]
        per_sample = [entry["realized_per_sample"] for entry in self._last_audit.values()]
        return {
            "strength": torch.stack(strengths).mean() if strengths else zero,
            "feature_strength": torch.stack(strengths).mean() if strengths else zero,
            "intensity_strength": zero,
            "coefficient_l2": (
                torch.stack([item.mean() for item in coeffs]).mean() if coeffs else zero
            ),
            "coefficient_l2_per_sample": (
                torch.stack(coeffs).mean(dim=0) if coeffs else per_sample_zero
            ),
            "realized_per_sample": (
                torch.stack(per_sample).mean(dim=0) if per_sample else per_sample_zero
            ),
            "style_confidence": (
                self._confidence.mean().to(reference.device)
                if self._confidence is not None
                else zero
            ),
            "style_entropy": (
                self._style_entropy.mean().to(reference.device)
                if self._style_entropy is not None
                else zero
            ),
            "condition_gate": (
                self._gate.mean().to(reference.device) if self._gate is not None else zero
            ),
            "style_delta": (
                self._style_delta.mean().to(reference.device)
                if self._style_delta is not None
                else zero
            ),
        }


def build_apis_v3(
    variant: str,
    *,
    layer1_channels: int,
    layer2_channels: int,
    layer3_channels: Optional[int] = None,
    acquisition_dim: int,
    alpha_max: float,
    basis_count: int,
    rank: int,
    style_dim: int = 16,
    memory_size: int = 8,
    memory_beta: float = 0.95,
    temperature: float = 0.5,
) -> nn.Module:
    if variant == "v3_style_memory":
        if layer3_channels is None:
            raise ValueError("v3_style_memory requires layer3_channels")
        return StyleMemoryAPISV3(
            layer1_channels=layer1_channels,
            layer2_channels=layer2_channels,
            layer3_channels=layer3_channels,
            alpha_max=alpha_max,
            basis_count=basis_count,
            rank=rank,
            style_dim=style_dim,
            memory_size=memory_size,
            memory_beta=memory_beta,
            temperature=temperature,
        )
    return APISV3Module(
        variant=variant,
        layer1_channels=layer1_channels,
        layer2_channels=layer2_channels,
        acquisition_dim=acquisition_dim,
        alpha_max=alpha_max,
        basis_count=basis_count,
        rank=rank,
    )
