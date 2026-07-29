"""ResNet3D backbone with APIS hooks after layer1/layer2."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from Model.backbone.film_backbone import BasicBlock


class DualShiftBackbone(nn.Module):
    def __init__(
        self,
        layers: Sequence[int] = (2, 2, 2, 2),
        base_channels: int = 32,
    ):
        super().__init__()
        self.inplanes = int(base_channels)
        self.base_channels = int(base_channels)
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, base_channels, layers[0])
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, layers[3], stride=2)
        self.out_channels = base_channels * 8 * BasicBlock.expansion
        self.layer1_channels = base_channels * BasicBlock.expansion
        self.layer2_channels = base_channels * 2 * BasicBlock.expansion

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
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_stem(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        return self.maxpool(x)

    def forward(
        self,
        image: torch.Tensor,
        *,
        shift_layer1: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shift_layer2: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> dict:
        x = self.forward_stem(image)
        f1 = self.layer1(x)
        if shift_layer1 is not None:
            f1 = shift_layer1(f1)
        f2 = self.layer2(f1)
        if shift_layer2 is not None:
            f2 = shift_layer2(f2)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}
