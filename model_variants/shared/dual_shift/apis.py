"""Data-constrained acquisition-protocol interventions for 3D CNN features."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_stats(
    features: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return channel statistics retained for prototype-bank compatibility."""
    if features.ndim != 5:
        raise ValueError("channel_stats expects [B, C, D, H, W] features")
    reduce_dims = (2, 3, 4)
    mean = features.mean(dim=reduce_dims, keepdim=True)
    variance = features.var(dim=reduce_dims, unbiased=False, keepdim=True)
    return mean, torch.sqrt(variance + eps)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ProtocolResidualOperator(nn.Module):
    """Mix fixed residual bases using a low-dimensional protocol condition.

    The condition controls only basis coefficients. It never generates CNN
    weights, so the module is a constrained residual intervention rather than
    a hypernetwork or channel-statistic replacement.
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        *,
        basis_count: int = 4,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.condition_dim = int(condition_dim)
        self.basis_count = int(basis_count)
        self.rank = int(min(rank, channels))
        if self.basis_count < 1 or self.rank < 1:
            raise ValueError("basis_count and rank must be positive")

        self.condition_norm = nn.LayerNorm(self.condition_dim)
        self.controller = nn.Linear(self.condition_dim, self.basis_count)
        self.feature_norm = nn.GroupNorm(
            _group_count(self.channels), self.channels, affine=False
        )
        self.down = nn.Conv3d(self.channels, self.rank, kernel_size=1, bias=False)
        # Each low-rank channel owns basis_count spatial response kernels.
        self.spatial_bases = nn.Conv3d(
            self.rank,
            self.rank * self.basis_count,
            kernel_size=3,
            padding=1,
            groups=self.rank,
            bias=False,
        )
        self.up = nn.Conv3d(self.rank, self.channels, kernel_size=1, bias=False)
        nn.init.normal_(self.controller.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.controller.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
        strength: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if features.ndim != 5:
            raise ValueError("APIS expects [B, C, D, H, W] features")
        if condition.ndim != 2 or condition.shape != (
            features.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                f"condition must be [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if strength <= 0:
            zero = features.new_zeros(())
            zeros = features.new_zeros((features.shape[0],))
            return features, {
                "strength": zero,
                "coefficient_l2": zeros,
                "realized_per_sample": zeros,
            }

        coefficients = torch.tanh(self.controller(self.condition_norm(condition)))
        coefficients = coefficients / math.sqrt(float(self.basis_count))
        latent = self.down(self.feature_norm(features))
        raw_bases = self.spatial_bases(latent)
        batch, _, depth, height, width = raw_bases.shape
        bases = raw_bases.reshape(
            batch, self.rank, self.basis_count, depth, height, width
        ).permute(0, 2, 1, 3, 4, 5)
        mixed = (
            bases
            * coefficients.reshape(batch, self.basis_count, 1, 1, 1, 1)
        ).sum(dim=1)
        residual = self.up(F.gelu(mixed))

        # Bound the intervention RMS relative to the factual feature RMS.
        feature_rms = features.float().flatten(1).square().mean(1).sqrt()
        residual_rms = residual.float().flatten(1).square().mean(1).sqrt()
        rms_scale = torch.minimum(
            torch.ones_like(residual_rms),
            feature_rms / residual_rms.clamp_min(1e-6),
        ).to(residual.dtype)
        bounded = residual * rms_scale.reshape(-1, 1, 1, 1, 1)
        shifted = features + float(strength) * bounded
        realized = (
            (float(strength) * residual_rms * rms_scale.float())
            / feature_rms.clamp_min(1e-6)
        )
        return shifted, {
            "strength": realized.mean(),
            "coefficient_l2": coefficients.square().mean(dim=-1),
            "realized_per_sample": realized,
        }


class APISModule(nn.Module):
    """Anatomy-Preserving Interventional Steering over observed protocols."""

    def __init__(
        self,
        *,
        layer1_channels: int,
        layer2_channels: int,
        acquisition_dim: int,
        alpha_max: float = 0.25,
        basis_count: int = 4,
        rank: int = 8,
        use_layer1: bool = True,
        use_layer2: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.use_layer1 = bool(use_layer1)
        self.use_layer2 = bool(use_layer2)
        # factual, target, and directed target-minus-factual descriptors.
        self.condition_dim = int(acquisition_dim) * 3
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
        self.enabled = True
        self.current_alpha = 0.0
        self._last_audit: Dict[str, Dict[str, torch.Tensor]] = {}

    def set_alpha(self, alpha: float) -> None:
        self.current_alpha = float(max(0.0, min(alpha, self.alpha_max)))

    def protocol_condition(
        self, factual: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if factual.shape != target.shape:
            raise ValueError("factual and target acquisition embeddings must match")
        return torch.cat([factual, target, target - factual], dim=1)

    def make_shift_fns(
        self,
        condition: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ):
        alpha = self.current_alpha if self.enabled else 0.0
        self._last_audit = {}
        mask = None
        if valid_mask is not None:
            mask = valid_mask.to(dtype=torch.bool).reshape(-1)

        def _apply(features: torch.Tensor, operator: ProtocolResidualOperator, key: str):
            if alpha <= 0:
                return features
            shifted, audit = operator(features, condition, alpha)
            coeff = audit["coefficient_l2"]
            if mask is not None:
                keep = mask.to(device=features.device, dtype=features.dtype).reshape(
                    -1, *([1] * (features.ndim - 1))
                )
                shifted = features + keep * (shifted - features)
                coeff = coeff.to(features.device) * mask.to(
                    device=features.device, dtype=coeff.dtype
                )
            self._last_audit[key] = {
                "strength": audit["strength"],
                "coefficient_l2": coeff,
            }
            return shifted

        def shift1(features: torch.Tensor) -> torch.Tensor:
            if not self.use_layer1:
                return features
            return _apply(features, self.layer1_operator, "layer1")

        def shift2(features: torch.Tensor) -> torch.Tensor:
            if not self.use_layer2:
                return features
            return _apply(features, self.layer2_operator, "layer2")

        return shift1, shift2

    def audit_tensors(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._last_audit:
            zero = reference.new_zeros(())
            return {"strength": zero, "coefficient_l2": zero}
        strengths = []
        coeffs = []
        for entry in self._last_audit.values():
            strengths.append(entry["strength"])
            coeff = entry["coefficient_l2"]
            if coeff.ndim == 0:
                coeffs.append(coeff)
            else:
                # Mean over samples that retained a non-zero masked penalty mass.
                active = coeff.detach() != 0
                if bool(active.any().item()):
                    coeffs.append(coeff[active].mean())
                else:
                    coeffs.append(coeff.new_zeros(()))
        return {
            "strength": torch.stack(strengths).mean(),
            "coefficient_l2": torch.stack(coeffs).mean(),
            "coefficient_l2_per_sample": torch.stack(
                [
                    entry["coefficient_l2"]
                    if entry["coefficient_l2"].ndim > 0
                    else entry["coefficient_l2"].expand(reference.shape[0])
                    for entry in self._last_audit.values()
                ]
            ).mean(dim=0),
        }
