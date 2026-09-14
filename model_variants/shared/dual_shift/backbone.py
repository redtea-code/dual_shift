"""ResNet3D backbone with APIS hooks after layer1/layer2."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.layer3_channels = base_channels * 4 * BasicBlock.expansion

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
        update_bn_stats: bool = True,
        capture_bn_moments: bool = False,
        paired_bn_moments: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        """Run the backbone, optionally capturing or replaying BN batch moments.

        APIC v3_2 uses the moments measured on the paired clean pass for every
        shifted BN layer.  Replaying running moments is not equivalent and can
        manufacture a branch difference even when the intervention is zero.
        """
        if capture_bn_moments and paired_bn_moments is not None:
            raise ValueError("capture_bn_moments and paired_bn_moments are exclusive")
        captured: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        handles = []
        named_bn = {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        }
        if capture_bn_moments:
            for name, module in named_bn.items():
                def capture(mod, inputs, *, key=name):
                    value = inputs[0]
                    dims = (0,) + tuple(range(2, value.ndim))
                    captured[key] = (
                        value.mean(dim=dims).detach(),
                        value.var(dim=dims, unbiased=False).detach(),
                    )

                handles.append(module.register_forward_pre_hook(capture))
        elif paired_bn_moments is not None:
            missing = sorted(set(named_bn) - set(paired_bn_moments))
            if missing:
                raise ValueError(f"paired BN moments missing layers: {missing}")
            for name, module in named_bn.items():
                def replay(mod, inputs, output, *, key=name):
                    del output
                    mean, variance = paired_bn_moments[key]
                    value = inputs[0]
                    return F.batch_norm(
                        value,
                        mean.to(value.device, value.dtype),
                        variance.to(value.device, value.dtype),
                        mod.weight,
                        mod.bias,
                        training=False,
                        momentum=0.0,
                        eps=mod.eps,
                    )

                handles.append(module.register_forward_hook(replay))

        states = []
        if (not update_bn_stats or paired_bn_moments is not None) and self.training:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    states.append((module, module.training))
                    module.training = False
        try:
            x = self.forward_stem(image)
            f1 = self.layer1(x)
            if shift_layer1 is not None:
                f1 = shift_layer1(f1)
            f2 = self.layer2(f1)
            if shift_layer2 is not None:
                f2 = shift_layer2(f2)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            result = {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}
            if capture_bn_moments:
                result["bn_moments"] = captured
            return result
        finally:
            for handle in handles:
                handle.remove()
            for module, state in states:
                module.training = state
