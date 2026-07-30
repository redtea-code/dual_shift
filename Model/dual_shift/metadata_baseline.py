"""Fair acquisition-metadata baseline for APIS v2 claim comparisons.

Matched to DualShiftResNet3D / apis_v2 on:
  - DualShiftBackbone
  - DemographicEncoder + LowRankDemographicFusion (age/sex/education)

Differing axis only:
  - metadata: concatenate AcquisitionDescriptorEncoder output at the head
  - apis_v2: residual protocol intervention in early CNN layers (no CDT)
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from Model.dual_shift.acquisition_encoder import AcquisitionDescriptorEncoder
from Model.dual_shift.backbone import DualShiftBackbone
from Model.dual_shift.demographic_encoder import (
    DemographicEncoder,
    LowRankDemographicFusion,
)


class MetadataConcatBaseline(nn.Module):
    """Image + demographic fusion + direct acquisition-protocol concatenation."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        base_channels: int = 32,
        acquisition_out_dim: int = 32,
        fusion_rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = DualShiftBackbone(base_channels=base_channels)
        self.demographic_encoder = DemographicEncoder()
        self.fusion = LowRankDemographicFusion(
            feature_dim=self.backbone.out_channels,
            demographic_dim=self.demographic_encoder.out_dim,
            rank=int(fusion_rank),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.acquisition_encoder = AcquisitionDescriptorEncoder(
            out_dim=int(acquisition_out_dim)
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(
            self.backbone.out_channels + int(acquisition_out_dim),
            int(num_classes),
        )

    def fit_acquisition_encoder(
        self, acquisitions: Sequence[Mapping[str, Any]]
    ) -> None:
        self.acquisition_encoder.fit(acquisitions)
        device = next(self.backbone.parameters()).device
        self.acquisition_encoder.to(device)

    def forward(
        self,
        image: torch.Tensor,
        covariates: torch.Tensor,
        acquisitions: Sequence[Mapping[str, Any]],
        *,
        age_missing: Optional[torch.Tensor] = None,
        sex_missing: Optional[torch.Tensor] = None,
        education_missing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.acquisition_encoder.is_fitted_:
            raise RuntimeError(
                "MetadataConcatBaseline requires fit_acquisition_encoder() first"
            )
        feats = self.backbone(image)["layer4"]
        pooled = self.pool(feats).flatten(1)
        demo = self.demographic_encoder(
            covariates,
            age_missing=age_missing,
            sex_missing=sex_missing,
            education_missing=education_missing,
        )
        fused = self.fusion(pooled, demo)
        acq_feat = self.acquisition_encoder(acquisitions)
        return self.classifier(self.dropout(torch.cat([fused, acq_feat], dim=-1)))
