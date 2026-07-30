"""Minimal image+tabular concat baseline for APIS v2 claim metadata conditioning.

Uses the same DualShiftBackbone family as APIS so the claim comparison does not
depend on Model.comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from Model.dual_shift.backbone import DualShiftBackbone


class MetadataConcatBaseline(nn.Module):
    """Late-fusion metadata baseline: image GAP features ∥ tabular MLP."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        tabular_dim: int = 3,
        base_channels: int = 32,
        tab_feat_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = DualShiftBackbone(base_channels=base_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        img_dim = int(self.backbone.out_channels)
        self.table_encoder = nn.Sequential(
            nn.Linear(tabular_dim, tab_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tab_feat_dim, tab_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + tab_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)["layer4"]
        image_feat = self.pool(feats).flatten(1)
        table_feat = self.table_encoder(table)
        return self.classifier(torch.cat([image_feat, table_feat], dim=-1))
