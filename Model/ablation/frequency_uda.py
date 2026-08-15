"""Frequency-guided, target-adaptive ScaleTable models.

The target domain enters this method only through an offline, label-free
frequency prior.  At training and inference, the model consumes one image and
the existing CAPM table exactly as the ``layer5_pixel + original_capm``
baseline does.  It never receives a cohort identifier or a target diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .scale_table_transformer import ScaleTableInteractionAblation3D


BAND_NAMES = ("low", "mid", "high")
DEFAULT_BAND_EDGES = (0.0, 0.15, 0.35)
PRIOR_SCHEMA = "dualshift_feature_frequency_prior_v1"


def _as_band_tensor(values: Sequence[float], *, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.shape != (len(BAND_NAMES),):
        raise ValueError(f"{name} must contain exactly {len(BAND_NAMES)} values")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_band_edges(edges: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in edges)
    if len(values) != len(BAND_NAMES):
        raise ValueError("band_edges must contain the lower edge of low, mid, and high")
    if values[0] != 0.0 or not (0.0 < values[1] < values[2]):
        raise ValueError("band_edges must be [0.0, positive_mid, larger_high]")
    return values


def radial_band_masks_rfft(
    spatial_shape: Sequence[int],
    *,
    band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build an exhaustive low/mid/high radial partition for ``rfftn`` output.

    Frequencies above the high lower edge are included in the high mask so the
    masks cover every rFFT coefficient, including 3-D corner frequencies.
    """
    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("spatial_shape must contain three positive dimensions")
    _, mid, high = _validate_band_edges(band_edges)
    axes = (
        torch.fft.fftfreq(shape[0], device=device, dtype=dtype),
        torch.fft.fftfreq(shape[1], device=device, dtype=dtype),
        torch.fft.rfftfreq(shape[2], device=device, dtype=dtype),
    )
    grids = torch.meshgrid(*axes, indexing="ij")
    radius = torch.sqrt(sum(grid.square() for grid in grids))
    masks = torch.stack(
        (
            radius < mid,
            (radius >= mid) & (radius < high),
            radius >= high,
        ),
        dim=0,
    ).to(dtype=dtype)
    if not torch.allclose(masks.sum(dim=0), torch.ones_like(radius)):
        raise RuntimeError("frequency bands must form an exhaustive partition")
    return masks


def band_power_fractions(
    features: Tensor,
    *,
    band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-sample band fractions, masks, and the complex rFFT tensor."""
    if features.ndim != 5:
        raise ValueError("features must have shape [B,C,D,H,W]")
    spectrum = torch.fft.rfftn(features.float(), dim=(-3, -2, -1), norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    masks = radial_band_masks_rfft(
        features.shape[-3:], band_edges=band_edges, device=features.device, dtype=power.dtype
    )
    band_power = (power.unsqueeze(1) * masks[None, :, None]).sum(dim=(2, 3, 4, 5))
    total = power.sum(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1).clamp_min(eps)
    fractions = band_power / total
    return fractions, masks, spectrum


@dataclass(frozen=True)
class FrequencyPrior:
    """Serializable label-free source/target feature-spectrum summary."""

    band_edges: tuple[float, float, float]
    source_mean: tuple[float, float, float]
    source_std: tuple[float, float, float]
    target_mean: tuple[float, float, float]
    target_std: tuple[float, float, float]
    discrepancy: tuple[float, float, float]
    source_count: int
    target_count: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_band_edges(self.band_edges)
        for name in ("source_mean", "source_std", "target_mean", "target_std", "discrepancy"):
            _as_band_tensor(getattr(self, name), name=name)
        if any(value < 0 for value in self.source_std + self.target_std):
            raise ValueError("frequency standard deviations must be non-negative")
        discrepancy = _as_band_tensor(self.discrepancy, name="discrepancy")
        if (discrepancy < 0).any() or float(discrepancy.max()) > 1.0 + 1e-8:
            raise ValueError("discrepancy must lie in [0, 1]")
        if self.source_count < 1 or self.target_count < 1:
            raise ValueError("source_count and target_count must be positive")

    @classmethod
    def from_summaries(
        cls,
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        *,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
        metadata: Mapping[str, Any] | None = None,
        eps: float = 1e-6,
    ) -> "FrequencyPrior":
        source_mean = _as_band_tensor(source["mean"], name="source.mean")
        source_std = _as_band_tensor(source["std"], name="source.std").clamp_min(eps)
        target_mean = _as_band_tensor(target["mean"], name="target.mean")
        target_std = _as_band_tensor(target["std"], name="target.std").clamp_min(eps)
        pooled = torch.sqrt(0.5 * (source_std.square() + target_std.square())).clamp_min(eps)
        standardized_difference = (source_mean - target_mean).abs() / pooled
        maximum = standardized_difference.max().clamp_min(eps)
        discrepancy = standardized_difference / maximum
        return cls(
            band_edges=_validate_band_edges(band_edges),
            source_mean=tuple(float(value) for value in source_mean.tolist()),
            source_std=tuple(float(value) for value in source_std.tolist()),
            target_mean=tuple(float(value) for value in target_mean.tolist()),
            target_std=tuple(float(value) for value in target_std.tolist()),
            discrepancy=tuple(float(value) for value in discrepancy.tolist()),
            source_count=int(source["count"]),
            target_count=int(target["count"]),
            metadata=dict(metadata or {}),
        )

    def with_discrepancy(self, discrepancy: Sequence[float]) -> "FrequencyPrior":
        values = _as_band_tensor(discrepancy, name="discrepancy")
        if (values < 0).any() or float(values.max()) > 1.0 + 1e-8:
            raise ValueError("discrepancy must lie in [0, 1]")
        return FrequencyPrior(
            band_edges=self.band_edges,
            source_mean=self.source_mean,
            source_std=self.source_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            discrepancy=tuple(float(value) for value in values.tolist()),
            source_count=self.source_count,
            target_count=self.target_count,
            metadata=dict(self.metadata),
        )

    def uniform(self) -> "FrequencyPrior":
        return self.with_discrepancy((1.0, 1.0, 1.0))

    def permuted(self, seed: int) -> "FrequencyPrior":
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(BAND_NAMES), generator=generator).tolist()
        if order == list(range(len(BAND_NAMES))):
            order = order[1:] + order[:1]
        return self.with_discrepancy([self.discrepancy[index] for index in order])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRIOR_SCHEMA,
            "band_names": list(BAND_NAMES),
            "band_edges": list(self.band_edges),
            "source": {"mean": list(self.source_mean), "std": list(self.source_std), "count": self.source_count},
            "target_adapt": {"mean": list(self.target_mean), "std": list(self.target_std), "count": self.target_count},
            "discrepancy": list(self.discrepancy),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrequencyPrior":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != PRIOR_SCHEMA:
            raise ValueError(f"Unsupported frequency prior schema: {payload.get('schema')!r}")
        if tuple(payload.get("band_names", ())) != BAND_NAMES:
            raise ValueError("frequency prior band names do not match the model contract")
        return cls(
            band_edges=_validate_band_edges(payload["band_edges"]),
            source_mean=tuple(float(value) for value in payload["source"]["mean"]),
            source_std=tuple(float(value) for value in payload["source"]["std"]),
            target_mean=tuple(float(value) for value in payload["target_adapt"]["mean"]),
            target_std=tuple(float(value) for value in payload["target_adapt"]["std"]),
            discrepancy=tuple(float(value) for value in payload["discrepancy"]),
            source_count=int(payload["source"]["count"]),
            target_count=int(payload["target_adapt"]["count"]),
            metadata=dict(payload.get("metadata") or {}),
        )


class FeatureSpectrumAccumulator:
    """Streaming, label-free band-power summary for a feature population."""

    def __init__(self, *, band_edges: Sequence[float] = DEFAULT_BAND_EDGES) -> None:
        self.band_edges = _validate_band_edges(band_edges)
        self._sum = torch.zeros(len(BAND_NAMES), dtype=torch.float64)
        self._sum_sq = torch.zeros(len(BAND_NAMES), dtype=torch.float64)
        self._count = 0

    def update(self, features: Tensor) -> None:
        fractions, _, _ = band_power_fractions(features, band_edges=self.band_edges)
        values = fractions.detach().to(device="cpu", dtype=torch.float64)
        self._sum += values.sum(dim=0)
        self._sum_sq += values.square().sum(dim=0)
        self._count += int(values.shape[0])

    def summary(self) -> dict[str, Any]:
        if self._count < 1:
            raise RuntimeError("Cannot summarize an empty feature population")
        mean = self._sum / self._count
        variance = (self._sum_sq / self._count - mean.square()).clamp_min(0.0)
        return {
            "count": int(self._count),
            "mean": [float(value) for value in mean.tolist()],
            "std": [float(value) for value in variance.sqrt().tolist()],
        }


class DomainGuidedFrequencyGate3D(nn.Module):
    """Attenuate sample-specific energy in label-free cross-domain bands."""

    def __init__(
        self,
        prior: FrequencyPrior,
        *,
        max_strength: float = 0.5,
        init_strength: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (0.0 < max_strength <= 1.0):
            raise ValueError("max_strength must be in (0, 1]")
        if not (0.0 < init_strength < max_strength):
            raise ValueError("init_strength must be in (0, max_strength)")
        self.band_edges = prior.band_edges
        self.max_strength = float(max_strength)
        self.eps = float(eps)
        self.register_buffer("source_mean", torch.tensor(prior.source_mean, dtype=torch.float32))
        self.register_buffer("source_std", torch.tensor(prior.source_std, dtype=torch.float32).clamp_min(eps))
        self.register_buffer("discrepancy", torch.tensor(prior.discrepancy, dtype=torch.float32))
        ratio = torch.tensor(init_strength / max_strength, dtype=torch.float32)
        self.raw_strength = nn.Parameter(torch.logit(ratio))
        self.energy_scale = nn.Parameter(torch.zeros(len(BAND_NAMES)))
        self.energy_bias = nn.Parameter(torch.zeros(len(BAND_NAMES)))
        self.last_audit: dict[str, Tensor] | None = None

    @property
    def effective_strength(self) -> Tensor:
        return self.max_strength * torch.sigmoid(self.raw_strength)

    def forward(self, features: Tensor, *, return_audit: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        fractions, masks, spectrum = band_power_fractions(
            features, band_edges=self.band_edges, eps=self.eps
        )
        normalized_energy = (fractions - self.source_mean) / self.source_std
        reliability = torch.sigmoid(normalized_energy * self.energy_scale + self.energy_bias)
        attenuation = 1.0 - self.effective_strength * self.discrepancy * reliability
        scale = torch.einsum("bk,kdhw->bdhw", attenuation, masks)
        filtered = torch.fft.irfftn(
            spectrum * scale[:, None], s=tuple(features.shape[-3:]), dim=(-3, -2, -1), norm="ortho"
        ).to(dtype=features.dtype)
        residual = filtered - features
        identity_loss = residual.square().mean() / features.square().mean().detach().clamp_min(self.eps)
        audit = {
            "band_fractions": fractions,
            "reliability": reliability,
            "attenuation": attenuation,
            "effective_strength": self.effective_strength.reshape(1),
            "identity_loss": identity_loss,
        }
        self.last_audit = audit
        if return_audit:
            return filtered, audit
        return filtered

    def regularization_losses(self) -> dict[str, Tensor]:
        if self.last_audit is None:
            return {"frequency_identity": self.raw_strength.new_zeros(())}
        return {"frequency_identity": self.last_audit["identity_loss"]}


class FrequencyGuidedScaleTable3D(ScaleTableInteractionAblation3D):
    """``layer4 -> DGF-Gate -> layer5 -> original CAPM`` model instance."""

    def __init__(
        self,
        *,
        prior: FrequencyPrior,
        preset: str = "layer5_pixel",
        num_classes: int = 2,
        max_strength: float = 0.5,
        init_strength: float = 0.05,
        **kwargs: Any,
    ) -> None:
        if preset != "layer5_pixel":
            raise ValueError("FrequencyGuidedScaleTable3D requires preset='layer5_pixel'")
        super().__init__(
            preset=preset,
            interaction="original_capm",
            num_classes=num_classes,
            **kwargs,
        )
        self.frequency_gate = DomainGuidedFrequencyGate3D(
            prior, max_strength=max_strength, init_strength=init_strength
        )
        self.frequency_prior_metadata = dict(prior.metadata)

    def extract_layer4(self, image: Tensor) -> Tensor:
        """Extract the pre-CAPM ``layer4`` map used to construct an offline prior."""
        features = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(self, name)(features)
        return features

    def extract_features(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if force_capm:
            raise ValueError("original_capm has no force_capm path")
        features = self.extract_layer4(image)
        features, frequency_audit = self.frequency_gate(features, return_audit=True)
        features = self.layer5(features)
        features, capm_audit = self._apply_calibrator(features, table, False, return_audit)
        audit = {f"frequency_{key}": value for key, value in frequency_audit.items()}
        audit.update(capm_audit)
        return features, audit

    def forward(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, audit = self.extract_features(
            image, table, force_capm=force_capm, return_audit=return_audit
        )
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audit
        return logits

    def regularization_losses(self) -> dict[str, Tensor]:
        return self.frequency_gate.regularization_losses()

    def get_regularization_losses(self) -> dict[str, Tensor]:
        return self.regularization_losses()

    def load_source_baseline(self, checkpoint_path: str | Path, *, map_location: Any = "cpu") -> None:
        """Load a source-selected ``original_capm`` checkpoint without hiding mismatches."""
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state = payload.get("model_state", payload)
        incompatible = self.load_state_dict(state, strict=False)
        allowed_missing = {name for name in self.state_dict() if name.startswith("frequency_gate.")}
        missing = set(incompatible.missing_keys).difference(allowed_missing)
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Source baseline is not compatible with FrequencyGuidedScaleTable3D: "
                f"missing={sorted(missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
            )

    def experiment_signature(self) -> dict[str, Any]:
        signature = super().experiment_signature()
        signature.update(
            {
                "model_family": "frequency_uda",
                "frequency_insert_stage": "layer4_to_layer5",
                "frequency_band_names": BAND_NAMES,
                "frequency_band_edges": self.frequency_gate.band_edges,
                "frequency_prior_discrepancy": self.frequency_gate.discrepancy.detach().cpu().tolist(),
                "frequency_max_strength": self.frequency_gate.max_strength,
                "frequency_prior_metadata": dict(self.frequency_prior_metadata),
            }
        )
        return signature


def build_frequency_guided_model(
    *,
    prior_path: str | Path,
    prior_mode: str = "domain_guided",
    prior_seed: int = 0,
    **kwargs: Any,
) -> FrequencyGuidedScaleTable3D:
    """Instantiate full, uniform, or permuted-prior controls from one JSON prior."""
    prior = FrequencyPrior.load(prior_path)
    if prior_mode == "domain_guided":
        selected_prior = prior
    elif prior_mode == "uniform":
        selected_prior = prior.uniform()
    elif prior_mode == "permuted":
        selected_prior = prior.permuted(prior_seed)
    else:
        raise ValueError("prior_mode must be 'domain_guided', 'uniform', or 'permuted'")
    return FrequencyGuidedScaleTable3D(prior=selected_prior, **kwargs)


__all__ = [
    "BAND_NAMES",
    "DEFAULT_BAND_EDGES",
    "DomainGuidedFrequencyGate3D",
    "FeatureSpectrumAccumulator",
    "FrequencyGuidedScaleTable3D",
    "FrequencyPrior",
    "band_power_fractions",
    "build_frequency_guided_model",
    "radial_band_masks_rfft",
]
