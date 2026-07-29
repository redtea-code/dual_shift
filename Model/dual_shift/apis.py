"""Anatomy-preserving imaging shift via constrained channel-statistic transport."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def channel_stats(features: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    # features: [B, C, D, H, W]
    reduce_dims = (2, 3, 4)
    mean = features.mean(dim=reduce_dims, keepdim=True)
    var = features.var(dim=reduce_dims, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    return mean, std


def apply_adain(
    features: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    alpha: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    if alpha <= 0:
        return features
    mean, std = channel_stats(features, eps=eps)
    normalized = (features - mean) / std
    # target_mean/std: [C] or [B, C, 1, 1, 1]
    if target_mean.ndim == 1:
        target_mean = target_mean.view(1, -1, 1, 1, 1)
        target_std = target_std.view(1, -1, 1, 1, 1)
    elif target_mean.ndim == 2:
        target_mean = target_mean.view(target_mean.shape[0], -1, 1, 1, 1)
        target_std = target_std.view(target_std.shape[0], -1, 1, 1, 1)
    shifted = normalized * target_std + target_mean
    return (1.0 - alpha) * features + alpha * shifted


class APISModule(nn.Module):
    """Apply optional AdaIN shifts at layer1 and/or layer2."""

    def __init__(self, alpha_max: float = 0.5, use_layer1: bool = True, use_layer2: bool = True):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.use_layer1 = bool(use_layer1)
        self.use_layer2 = bool(use_layer2)
        self.enabled = True
        self.current_alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        self.current_alpha = float(max(0.0, min(alpha, self.alpha_max)))

    def make_shift_fns(
        self,
        *,
        target_mean1: Optional[torch.Tensor],
        target_std1: Optional[torch.Tensor],
        target_mean2: Optional[torch.Tensor],
        target_std2: Optional[torch.Tensor],
    ):
        alpha = self.current_alpha if self.enabled else 0.0

        def shift1(features: torch.Tensor) -> torch.Tensor:
            if (
                not self.use_layer1
                or alpha <= 0
                or target_mean1 is None
                or target_std1 is None
            ):
                return features
            return apply_adain(features, target_mean1, target_std1, alpha)

        def shift2(features: torch.Tensor) -> torch.Tensor:
            if (
                not self.use_layer2
                or alpha <= 0
                or target_mean2 is None
                or target_std2 is None
            ):
                return features
            return apply_adain(features, target_mean2, target_std2, alpha)

        return shift1, shift2
