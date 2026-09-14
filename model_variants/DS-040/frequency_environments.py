"""Pre-registered single-view frequency/resolution environments for GroupDRO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FREQUENCY_ENVIRONMENTS = ("original", "lowpass", "downsample_resample", "mild_blur")


class FrequencyEnvironmentAugment3D(nn.Module):
    """Assign each source training sample to one deterministic frequency environment.

    The returned tensor contains one transformed view per input sample.  This
    is deliberately not a paired-view consistency objective: GroupDRO receives
    only the supervised classification loss and the sampled environment ID.
    """

    def __init__(self, *, lowpass_kernel: int = 3, blur_sigma: float = 0.8) -> None:
        super().__init__()
        if lowpass_kernel < 3 or lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be odd and at least 3")
        if blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")
        self.lowpass_kernel = int(lowpass_kernel)
        self.blur_sigma = float(blur_sigma)
        radius = 1
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (axis / self.blur_sigma).square())
        kernel_1d /= kernel_1d.sum()
        kernel_3d = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
        self.register_buffer("blur_kernel", kernel_3d.reshape(1, 1, 3, 3, 3))

    @property
    def num_environments(self) -> int:
        return len(FREQUENCY_ENVIRONMENTS)

    def _lowpass(self, images: Tensor) -> Tensor:
        padding = self.lowpass_kernel // 2
        return F.avg_pool3d(images, self.lowpass_kernel, stride=1, padding=padding)

    @staticmethod
    def _downsample_resample(images: Tensor) -> Tensor:
        shape = tuple(int(value) for value in images.shape[-3:])
        reduced = tuple(max(2, (value + 1) // 2) for value in shape)
        down = F.interpolate(images, size=reduced, mode="trilinear", align_corners=False)
        return F.interpolate(down, size=shape, mode="trilinear", align_corners=False)

    def _mild_blur(self, images: Tensor) -> Tensor:
        channels = int(images.shape[1])
        kernel = self.blur_kernel.to(device=images.device, dtype=images.dtype).expand(
            channels, 1, -1, -1, -1
        )
        return F.conv3d(images, kernel, padding=1, groups=channels)

    def forward(
        self,
        images: Tensor,
        *,
        generator: torch.Generator | None = None,
        environment_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B,C,D,H,W]")
        batch_size = int(images.shape[0])
        if environment_ids is None:
            environment_ids = torch.randint(
                self.num_environments,
                (batch_size,),
                device=images.device,
                generator=generator,
            )
        else:
            environment_ids = environment_ids.to(device=images.device, dtype=torch.long).reshape(-1)
        if environment_ids.shape != (batch_size,):
            raise ValueError("environment_ids must have shape [B]")
        if environment_ids.numel() and (
            environment_ids.min() < 0 or environment_ids.max() >= self.num_environments
        ):
            raise ValueError("environment_ids contain an unknown frequency environment")

        output = images.clone()
        transforms = (None, self._lowpass, self._downsample_resample, self._mild_blur)
        for environment_id, transform in enumerate(transforms):
            if transform is None:
                continue
            mask = environment_ids == environment_id
            if mask.any():
                output[mask] = transform(images[mask])
        return output, environment_ids


__all__ = ["FREQUENCY_ENVIRONMENTS", "FrequencyEnvironmentAugment3D"]
