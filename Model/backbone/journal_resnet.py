"""Late-stage covariate-conditioned spatial modulation for journal ResNet3D."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.backbone.film_backbone import BasicBlock


def default_journal_var_specs() -> List[dict]:
    return [
        {
            "name": "age",
            "type": "continuous",
            "min_val": -5.0,
            "max_val": 5.0,
            "n_centers": 8,
            "n_bases": 4,
        },
        {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 2},
        {
            "name": "education",
            "type": "continuous",
            "min_val": -5.0,
            "max_val": 5.0,
            "n_centers": 8,
            "n_bases": 4,
        },
    ]


class _RBFEncoder(nn.Module):
    def __init__(self, min_val: float, max_val: float, n_centers: int):
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


class _CategoricalEncoder(nn.Module):
    def __init__(self, n_cats: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(int(n_cats), int(embedding_dim))
        self.out_dim = int(embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        indices = values.long().clamp(min=0, max=self.embedding.num_embeddings - 1)
        return self.embedding(indices)


class _SmoothBasisField(nn.Module):
    """Low-rank smooth 3D basis bank with shape ``[n_bases, 1, D, H, W]``."""

    def __init__(self, n_bases: int, spatial_shape: Sequence[int]):
        super().__init__()
        depth, height, width = [int(value) for value in spatial_shape]
        self.bases = nn.Parameter(
            0.01 * torch.randn(int(n_bases), 1, depth, height, width)
        )

    def compose(self, coefficients: torch.Tensor) -> torch.Tensor:
        # coefficients: [B, n_bases] -> field [B, 1, D, H, W]
        return torch.einsum("bk,kcdhw->bcdhw", coefficients, self.bases)


class _ChannelAffine(nn.Module):
    """Channel-wise affine transform initialized to the identity map."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, int(channels), 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, int(channels), 1, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.weight + self.bias


class LateStageSpatialModulation(nn.Module):
    """Build a smooth gamma field from age/sex/education and modulate features.

    Modulation follows the handoff formula ``F' = Norm(F + F * tanh(gamma))``.
    ``Norm`` is a channel-wise affine map initialized to identity, so a
    zero-initialized coefficient head yields ``F' = F`` at the start of training.
    """

    def __init__(
        self,
        channels: int,
        var_specs: Sequence[dict],
        spatial_shape: Sequence[int] = (4, 4, 4),
        embedding_dim: int = 16,
        sparse_regularization: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.var_specs = [dict(spec) for spec in var_specs]
        self.sparse_regularization = bool(sparse_regularization)
        self.last_gamma: Optional[torch.Tensor] = None

        encoders: List[nn.Module] = []
        coefficient_heads = nn.ModuleList()
        spatial_bases = nn.ModuleList()
        for spec in self.var_specs:
            n_bases = int(spec.get("n_bases", 4))
            if spec["type"] == "continuous":
                encoder = _RBFEncoder(
                    float(spec.get("min_val", -5.0)),
                    float(spec.get("max_val", 5.0)),
                    int(spec.get("n_centers", 8)),
                )
                in_dim = encoder.out_dim
            elif spec["type"] == "categorical":
                encoder = _CategoricalEncoder(
                    int(spec.get("n_cats", 2)), int(embedding_dim)
                )
                in_dim = encoder.out_dim
            else:
                raise ValueError(f"Unsupported variable type: {spec['type']}")
            head = nn.Sequential(
                nn.Linear(in_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, n_bases),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            encoders.append(encoder)
            coefficient_heads.append(head)
            spatial_bases.append(_SmoothBasisField(n_bases, self.spatial_shape))

        self.encoders = nn.ModuleList(encoders)
        self.coefficient_heads = coefficient_heads
        self.spatial_bases = spatial_bases
        self.norm = _ChannelAffine(self.channels)

    def _encode_variable(self, index: int, values: torch.Tensor) -> torch.Tensor:
        return self.encoders[index](values)

    def forward(self, features: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        if covariates.ndim != 2 or covariates.shape[1] < len(self.var_specs):
            raise ValueError(
                "covariates must have shape [B, >= n_vars] matching var_specs order"
            )
        batch, channels, depth, height, width = features.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, got {channels}"
            )

        gamma = features.new_zeros(batch, 1, *self.spatial_shape)
        for index, _spec in enumerate(self.var_specs):
            encoded = self._encode_variable(index, covariates[:, index])
            coefficients = self.coefficient_heads[index](encoded)
            gamma = gamma + self.spatial_bases[index].compose(coefficients)

        if gamma.shape[-3:] != (depth, height, width):
            gamma = F.interpolate(
                gamma,
                size=(depth, height, width),
                mode="trilinear",
                align_corners=False,
            )
        gamma = gamma.expand(-1, channels, -1, -1, -1)
        self.last_gamma = gamma
        # F' = Norm(F + F * tanh(gamma))
        return self.norm(features + features * torch.tanh(gamma))

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        tv_terms = []
        for bank in self.spatial_bases:
            bases = bank.bases
            for dim in (2, 3, 4):
                if bases.shape[dim] > 1:
                    tv_terms.append(bases.diff(dim=dim).abs().mean())
        device = self.spatial_bases[0].bases.device
        dtype = self.spatial_bases[0].bases.dtype
        journal_tv = (
            torch.stack(tv_terms).mean()
            if tv_terms
            else torch.zeros((), device=device, dtype=dtype)
        )
        losses = {"journal_tv": journal_tv}
        if self.sparse_regularization:
            sparse_terms = []
            for bank in self.spatial_bases:
                sparse_terms.append(bank.bases.abs().mean())
            if self.last_gamma is not None:
                sparse_terms.append(self.last_gamma.abs().mean())
            losses["journal_sparse"] = torch.stack(sparse_terms).mean()
        return losses


class JournalResNet3D(nn.Module):
    """Single-stream 3D ResNet with optional late-stage spatial modulation."""

    def __init__(
        self,
        layers: Sequence[int],
        num_classes: int = 2,
        base_channels: int = 32,
        spatial_shape: Sequence[int] = (4, 4, 4),
        embedding_dim: int = 16,
        sparse_regularization: bool = True,
        dropout: float = 0.1,
        var_specs: Optional[Sequence[dict]] = None,
    ):
        super().__init__()
        self.inplanes = int(base_channels)
        self.var_specs = list(var_specs or default_journal_var_specs())
        self.conv1 = nn.Conv3d(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, base_channels, layers[0])
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, layers[3], stride=2)
        channels = base_channels * 8 * BasicBlock.expansion
        self.modulation = LateStageSpatialModulation(
            channels=channels,
            var_specs=self.var_specs,
            spatial_shape=spatial_shape,
            embedding_dim=embedding_dim,
            sparse_regularization=sparse_regularization,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(channels, int(num_classes))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(
        self,
        image: torch.Tensor,
        covariates: torch.Tensor,
        modulate: bool = True,
    ) -> torch.Tensor:
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if modulate:
            x = self.modulation(x, covariates)
        return x

    def forward(self, image: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image, covariates, modulate=True)
        pooled = self.pool(features).flatten(1)
        return self.fc(self.dropout(pooled))

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        return self.modulation.get_regularization_losses()


def journal_resnet10(**kwargs) -> JournalResNet3D:
    return JournalResNet3D([1, 1, 1, 1], **kwargs)


def journal_resnet18(**kwargs) -> JournalResNet3D:
    return JournalResNet3D([2, 2, 2, 2], **kwargs)


__all__ = [
    "JournalResNet3D",
    "LateStageSpatialModulation",
    "default_journal_var_specs",
    "journal_resnet10",
    "journal_resnet18",
]
