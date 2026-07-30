"""Acquisition-metadata baseline for APIS v2 claim (table: X + A, no D).

Image GAP features are concatenated with AcquisitionDescriptorEncoder
embeddings (manufacturer, field strength, model, sequence, TR/TE/TI, ...).
Demographics are intentionally not used — that axis is reserved for
mixstyle / apis_v2 / cdt / film comparisons.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from Model.dual_shift.acquisition_encoder import AcquisitionDescriptorEncoder
from Model.dual_shift.backbone import DualShiftBackbone


class MetadataConcatBaseline(nn.Module):
    """Pure image + direct acquisition-protocol concatenation (no demographics)."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        base_channels: int = 32,
        acquisition_out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = DualShiftBackbone(base_channels=base_channels)
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
        acquisitions: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if not self.acquisition_encoder.is_fitted_:
            raise RuntimeError(
                "MetadataConcatBaseline requires fit_acquisition_encoder() first"
            )
        feats = self.backbone(image)["layer4"]
        image_feat = self.pool(feats).flatten(1)
        acq_feat = self.acquisition_encoder(acquisitions)
        return self.classifier(self.dropout(torch.cat([image_feat, acq_feat], dim=-1)))
