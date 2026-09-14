"""Differentiable raw-volume Fourier operations used by the FMM baseline."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def _check_volume_pair(source: Tensor, target: Tensor) -> None:
    if source.shape != target.shape:
        raise ValueError(f"source and target shapes must match, got {tuple(source.shape)} and {tuple(target.shape)}")
    if source.ndim != 5:
        raise ValueError("expected [B,C,D,H,W] volumes")


def fft_amplitude_phase(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return amplitude and phase of the spatial dimensions of a real volume."""
    if x.ndim != 5:
        raise ValueError("expected [B,C,D,H,W] volumes")
    spectrum = torch.fft.fftn(x, dim=(-3, -2, -1), norm="ortho")
    return torch.abs(spectrum), torch.angle(spectrum)


def ifft_amplitude_phase(amplitude: Tensor, phase: Tensor) -> Tensor:
    if amplitude.shape != phase.shape:
        raise ValueError("amplitude and phase shapes must match")
    complex_spectrum = torch.polar(amplitude, phase)
    return torch.fft.ifftn(complex_spectrum, dim=(-3, -2, -1), norm="ortho").real


def sample_lambda(
    batch_size: int,
    *,
    low: float = 0.0,
    high: float = 1.0,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("lambda interval must satisfy 0 <= low <= high <= 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if low == high:
        return torch.full((batch_size, 1, 1, 1, 1), low, device=device)
    values = torch.rand(batch_size, 1, 1, 1, 1, device=device, generator=generator)
    return low + (high - low) * values


def mix_amplitude(
    source: Tensor,
    target: Tensor,
    mixing: Tensor | float,
    *,
    phase_source: bool = False,
) -> Tensor:
    """Mix source/target amplitudes and keep target phase by default.

    ``mixing=0`` is the target volume and ``mixing=1`` combines source
    amplitude with target phase.  ``phase_source=True`` is useful for the
    Stage-I source intensity construction, where source phase is retained.
    """
    _check_volume_pair(source, target)
    source_amplitude, source_phase = fft_amplitude_phase(source)
    target_amplitude, target_phase = fft_amplitude_phase(target)
    if not torch.is_tensor(mixing):
        mixing = torch.as_tensor(mixing, dtype=source.dtype, device=source.device)
    mixing = mixing.to(dtype=source.dtype, device=source.device)
    while mixing.ndim < source.ndim:
        mixing = mixing.unsqueeze(-1)
    if mixing.ndim == 1:
        mixing = mixing.view(-1, 1, 1, 1, 1)
    if mixing.shape[0] not in {1, source.shape[0]}:
        raise ValueError("mixing must be scalar or have one value per batch item")
    amplitude = (1.0 - mixing) * target_amplitude + mixing * source_amplitude
    phase = source_phase if phase_source else target_phase
    return ifft_amplitude_phase(amplitude, phase)


def intensity_transform(
    x: Tensor,
    *,
    scale_range: Sequence[float] = (0.8, 1.2),
    noise_std: float = 0.05,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply the source-side intensity perturbation used before Stage I FFT."""
    if x.ndim != 5:
        raise ValueError("expected [B,C,D,H,W] volumes")
    if len(scale_range) != 2 or not 0 < float(scale_range[0]) <= float(scale_range[1]):
        raise ValueError("scale_range must contain two positive ascending values")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    low, high = (float(item) for item in scale_range)
    scale = low + (high - low) * torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device, generator=generator)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    spatial_std = x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1, 1)
    return scale * x + float(noise_std) * spatial_std * noise


def spectral_diagnostics(x: Tensor) -> dict[str, float]:
    amplitude, _phase = fft_amplitude_phase(x.detach())
    return {
        "amplitude_mean": float(amplitude.mean().cpu()),
        "amplitude_std": float(amplitude.std().cpu()),
        "amplitude_dc_mean": float(amplitude[..., 0, 0, 0].mean().cpu()),
    }


__all__ = [
    "fft_amplitude_phase",
    "ifft_amplitude_phase",
    "intensity_transform",
    "mix_amplitude",
    "sample_lambda",
    "spectral_diagnostics",
]
