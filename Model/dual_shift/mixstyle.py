"""Random MixStyle baseline (protocol-unaware channel-statistic mixing)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Batch MixStyle on channel statistics (Zhou et al.).

    Unlike APIS, the target style is a random in-batch partner, not a real
    scanner/protocol prototype.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._activated = True

    def set_activated(self, activated: bool) -> None:
        self._activated = bool(activated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (not self._activated) or self.p <= 0:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        batch = x.shape[0]
        if batch < 2:
            return x
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        perm = torch.randperm(batch, device=x.device)
        mean2 = mean[perm]
        std2 = std[perm]
        # Beta sample per instance
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch, 1, 1, 1, 1))
        lam = lam.to(device=x.device, dtype=x.dtype)
        mixed_mean = lam * mean + (1.0 - lam) * mean2
        mixed_std = lam * std + (1.0 - lam) * std2
        return x_norm * mixed_std + mixed_mean
