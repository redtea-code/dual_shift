"""Acquisition-protocol metadata conditioning baseline for APIS v2 claim.

Uses the same AcquisitionDescriptorEncoder fields as APIS (manufacturer,
field strength, scanner model, sequence, TR/TE/TI, etc.), not demographics.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from Model.dual_shift.acquisition_encoder import AcquisitionDescriptorEncoder
from Model.dual_shift.backbone import DualShiftBackbone


class MetadataConcatBaseline(nn.Module):
    """Late fusion of image GAP features with acquisition-protocol embeddings."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        base_channels: int = 32,
        acquisition_out_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = DualShiftBackbone(base_channels=base_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.acquisition_encoder = AcquisitionDescriptorEncoder(
            out_dim=int(acquisition_out_dim)
        )
        img_dim = int(self.backbone.out_channels)
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + int(acquisition_out_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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
        acquisitions: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if not self.acquisition_encoder.is_fitted_:
            raise RuntimeError(
                "MetadataConcatBaseline requires fit_acquisition_encoder() first"
            )
        feats = self.backbone(image)["layer4"]
        image_feat = self.pool(feats).flatten(1)
        acq_feat = self.acquisition_encoder(acquisitions)
        return self.classifier(torch.cat([image_feat, acq_feat], dim=-1))
