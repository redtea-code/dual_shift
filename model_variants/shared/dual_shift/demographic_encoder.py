"""Demographic context encoder and low-rank image–table fusion."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _RBFEncoder(nn.Module):
    def __init__(self, n_centers: int = 8, min_val: float = -5.0, max_val: float = 5.0):
        super().__init__()
        centers = torch.linspace(float(min_val), float(max_val), int(n_centers))
        self.register_buffer("centers", centers)
        span = max(float(max_val) - float(min_val), 1e-6)
        self.register_buffer(
            "sigma",
            torch.tensor(span / max(n_centers - 1, 1), dtype=torch.float32),
        )

    @property
    def out_dim(self) -> int:
        return int(self.centers.numel())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, 1)
        return torch.exp(-((values - self.centers.view(1, -1)) / self.sigma).square())


class DemographicEncoder(nn.Module):
    """Encode preprocessed covariates ``[age, sex, education]`` (+ optional masks)."""

    def __init__(
        self,
        age_dim: int = 16,
        sex_dim: int = 8,
        education_dim: int = 16,
        n_centers: int = 8,
    ):
        super().__init__()
        self.age_rbf = _RBFEncoder(n_centers=n_centers)
        self.edu_rbf = _RBFEncoder(n_centers=n_centers)
        self.age_mlp = nn.Sequential(
            nn.Linear(self.age_rbf.out_dim + 1, age_dim),
            nn.ReLU(inplace=True),
            nn.Linear(age_dim, age_dim),
        )
        self.sex_emb = nn.Embedding(3, sex_dim)  # female, male, missing
        self.edu_mlp = nn.Sequential(
            nn.Linear(self.edu_rbf.out_dim + 1, education_dim),
            nn.ReLU(inplace=True),
            nn.Linear(education_dim, education_dim),
        )
        self.out_dim = int(age_dim + sex_dim + education_dim)

    def forward(
        self,
        covariates: torch.Tensor,
        age_missing: Optional[torch.Tensor] = None,
        sex_missing: Optional[torch.Tensor] = None,
        education_missing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        age = covariates[:, 0]
        sex = covariates[:, 1]
        education = covariates[:, 2]
        batch = covariates.shape[0]
        device = covariates.device
        if age_missing is None:
            age_missing = torch.zeros(batch, device=device, dtype=covariates.dtype)
        if education_missing is None:
            education_missing = torch.zeros(batch, device=device, dtype=covariates.dtype)
        if sex_missing is None:
            sex_missing = torch.zeros(batch, device=device, dtype=torch.long)
        age_feat = self.age_mlp(
            torch.cat(
                [self.age_rbf(age), age_missing.reshape(-1, 1).to(covariates.dtype)],
                dim=1,
            )
        )
        # sex codes from CovariatePreprocessor: 0=female, 1=male; missing->2
        sex_idx = sex.round().long().clamp(0, 1)
        sex_idx = torch.where(
            sex_missing.reshape(-1).bool(),
            torch.full_like(sex_idx, 2),
            sex_idx,
        )
        sex_feat = self.sex_emb(sex_idx)
        edu_feat = self.edu_mlp(
            torch.cat(
                [
                    self.edu_rbf(education),
                    education_missing.reshape(-1, 1).to(covariates.dtype),
                ],
                dim=1,
            )
        )
        return torch.cat([age_feat, sex_feat, edu_feat], dim=1)


class LowRankDemographicFusion(nn.Module):
    """Identity-initialized residual fusion: h = h_x + U[(V h_x) ⊙ g]."""

    def __init__(self, feature_dim: int, demographic_dim: int, rank: int = 16):
        super().__init__()
        self.project_z = nn.Linear(demographic_dim, rank)
        self.project_v = nn.Linear(feature_dim, rank, bias=False)
        self.project_u = nn.Linear(rank, feature_dim, bias=False)
        nn.init.zeros_(self.project_u.weight)

    def forward(
        self, image_features: torch.Tensor, demographic_embedding: torch.Tensor
    ) -> torch.Tensor:
        g = torch.tanh(self.project_z(demographic_embedding))
        interaction = self.project_v(image_features) * g
        return image_features + self.project_u(interaction)
