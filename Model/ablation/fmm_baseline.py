"""Independent raw-volume Frequency Mixup Manipulation baseline.

The reference repository contains the outline of the FMM model, but its
training scripts depend on files that are not present in the public checkout.
This module keeps the published contracts explicit while making the encoder
usable with the variable MRI shapes used by this project.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Identity in the forward pass and sign reversal in the backward pass."""

    @staticmethod
    def forward(ctx, input_tensor: Tensor, coefficient: float):
        ctx.coefficient = float(coefficient)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.coefficient * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, coefficient: float = 1.0):
        super().__init__()
        self.coefficient = float(coefficient)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return GradientReversalFunction.apply(input_tensor, self.coefficient)


class BasicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class SpatialGate3d(nn.Module):
    """The max/mean channel-compression attention gate used by FMM."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.compress = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.attention: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        compressed = torch.cat((x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)), dim=1)
        attention = torch.sigmoid(self.compress(compressed))
        self.attention = attention
        return x * attention


def _validate_channels(channels: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(item) for item in channels)
    if len(values) != 10 or any(item <= 0 for item in values):
        raise ValueError("channels must contain ten positive convolution widths")
    return values


class FMMNet(nn.Module):
    """Ten-convolution 3D encoder with classifier and domain heads.

    ``pool_shape`` preserves the reference classifier contract (128*2*3*2
    for the default channels) while adaptive pooling makes small smoke inputs
    and the project's cropped volumes both runnable.
    """

    def __init__(
        self,
        *,
        channels: Iterable[int] = (8, 8, 16, 16, 32, 32, 64, 64, 128, 128),
        pool_shape: Sequence[int] = (2, 3, 2),
        num_classes: int = 2,
        classifier_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        widths = _validate_channels(tuple(channels))
        pool_shape = tuple(int(item) for item in pool_shape)
        if len(pool_shape) != 3 or any(item <= 0 for item in pool_shape):
            raise ValueError("pool_shape must contain three positive integers")
        self.channels = widths
        self.pool_shape = pool_shape
        self.feature_extractor = nn.ModuleList()
        in_channels = 1
        for layer_index, out_channels in enumerate(widths):
            self.feature_extractor.append(BasicConv3d(in_channels, out_channels))
            in_channels = out_channels
            if layer_index in {1, 3, 5, 7, 9}:
                self.feature_extractor.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.spatial_gate = SpatialGate3d()
        self.adaptive_pool = nn.AdaptiveAvgPool3d(pool_shape)
        feature_dim = widths[-1] * pool_shape[0] * pool_shape[1] * pool_shape[2]
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(feature_dim, int(classifier_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(classifier_hidden), int(num_classes)),
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, int(classifier_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(classifier_hidden), 1),
        )
        self.intensity_discriminator = nn.Sequential(
            nn.Linear(feature_dim, int(classifier_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(classifier_hidden), 1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"expected [B,1,D,H,W] input, got {tuple(x.shape)}")
        gate_seen = False
        for layer in self.feature_extractor:
            x = layer(x)
            # The paper places spatial attention after the first convolution.
            if isinstance(layer, BasicConv3d) and not gate_seen:
                x = self.spatial_gate(x)
                gate_seen = True
        pooled = self.adaptive_pool(x)
        return pooled.flatten(1), self.spatial_gate.attention

    def forward(self, x: Tensor, *, return_attention: bool = False) -> dict[str, Tensor]:
        features, attention = self.encode(x)
        result = {"logits": self.classifier(features), "features": features}
        if return_attention:
            if attention is None:
                raise RuntimeError("spatial attention was not computed")
            result["attention"] = attention
        return result

    def domain_logits(self, features: Tensor, *, coefficient: float = 1.0, head: str = "domain") -> Tensor:
        if head not in {"domain", "intensity"}:
            raise ValueError("head must be 'domain' or 'intensity'")
        reversed_features = GradientReversal(float(coefficient))(features)
        return (self.domain_discriminator if head == "domain" else self.intensity_discriminator)(reversed_features).squeeze(1)


__all__ = ["BasicConv3d", "FMMNet", "GradientReversal", "GradientReversalFunction", "SpatialGate3d"]
