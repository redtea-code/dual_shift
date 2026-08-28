"""CAPM-conditioned, label-free residual adaptation pilot.

This is deliberately a frozen-checkpoint experiment.  It estimates channel
moments from CAPM-adjusted source images and an image-only ``T_adapt``
population, then applies a bounded affine correction to that CAPM feature.
The statistics builder uses one fixed source-only demographic reference table
for both populations, so it does not materialize target covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .scale_table_transformer import ScaleTableInteractionAblation3D


RESIDUAL_STATS_SCHEMA = "dualshift_capm_residual_stats_v1"
TARGET_SPLIT_SCHEMA = "dualshift_capm_residual_target_split_v1"


def _subject_digest(subject_ids: Iterable[object]) -> str:
    payload = "\n".join(sorted(map(str, subject_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_capm_residual_target_split(
    path: str | Path,
    subject_ids: Iterable[object],
    adaptation_indices: Iterable[int],
    test_indices: Iterable[int],
    *,
    direction: str,
    adaptation_fraction: float,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the image-only target partition used by this pilot."""
    subjects = list(subject_ids)
    adaptation = [int(index) for index in adaptation_indices]
    test = [int(index) for index in test_indices]
    adaptation_subjects = sorted({str(subjects[index]) for index in adaptation})
    test_subjects = sorted({str(subjects[index]) for index in test})
    if not adaptation_subjects or not test_subjects:
        raise ValueError("target adaptation/test subjects must be non-empty")
    if set(adaptation_subjects).intersection(test_subjects):
        raise ValueError("target adaptation/test subjects must be disjoint")
    payload = {
        "schema": TARGET_SPLIT_SCHEMA,
        "direction": str(direction),
        "adaptation_fraction": float(adaptation_fraction),
        "seed": int(seed),
        "target_labels_read": False,
        "target_metrics_read": False,
        "target_adapt_subjects": adaptation_subjects,
        "target_test_subjects": test_subjects,
        "target_adapt_subject_digest": _subject_digest(adaptation_subjects),
        "target_test_subject_digest": _subject_digest(test_subjects),
        "metadata": dict(metadata or {}),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _channel_tensor(values: Any, *, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


@dataclass(frozen=True)
class CAPMResidualStats:
    """Serializable source/target channel-moment summary."""

    source_mean: tuple[float, ...]
    source_std: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    discrepancy: tuple[float, ...]
    source_count: int
    target_count: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        names = ("source_mean", "source_std", "target_mean", "target_std", "discrepancy")
        tensors = {name: _channel_tensor(getattr(self, name), name=name) for name in names}
        lengths = {int(tensor.numel()) for tensor in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("residual statistics must have the same channel count")
        if (tensors["source_std"] < 0).any() or (tensors["target_std"] < 0).any():
            raise ValueError("source_std and target_std must be non-negative")
        discrepancy = tensors["discrepancy"]
        if (discrepancy < 0).any() or float(discrepancy.max()) > 1.0 + 1e-8:
            raise ValueError("discrepancy must lie in [0, 1]")
        if self.source_count < 1 or self.target_count < 1:
            raise ValueError("source_count and target_count must be positive")

    @property
    def channels(self) -> int:
        return len(self.source_mean)

    @classmethod
    def from_summaries(
        cls,
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        eps: float = 1e-6,
    ) -> "CAPMResidualStats":
        source_mean = _channel_tensor(source["mean"], name="source.mean")
        target_mean = _channel_tensor(target["mean"], name="target.mean")
        source_std = _channel_tensor(source["std"], name="source.std").clamp_min(eps)
        target_std = _channel_tensor(target["std"], name="target.std").clamp_min(eps)
        if source_mean.shape != target_mean.shape or source_std.shape != target_std.shape:
            raise ValueError("source and target summaries must have equal channel counts")
        pooled = torch.sqrt(0.5 * (source_std.square() + target_std.square())).clamp_min(eps)
        mean_shift = (source_mean - target_mean).abs() / pooled
        scale_shift = torch.log(target_std / source_std).abs()
        raw = torch.sqrt(mean_shift.square() + scale_shift.square())
        maximum = raw.max()
        discrepancy = raw / maximum if float(maximum) > eps else torch.zeros_like(raw)
        return cls(
            source_mean=tuple(float(value) for value in source_mean.tolist()),
            source_std=tuple(float(value) for value in source_std.tolist()),
            target_mean=tuple(float(value) for value in target_mean.tolist()),
            target_std=tuple(float(value) for value in target_std.tolist()),
            discrepancy=tuple(float(value) for value in discrepancy.tolist()),
            source_count=int(source["count"]),
            target_count=int(target["count"]),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_STATS_SCHEMA,
            "channels": self.channels,
            "source": {
                "mean": list(self.source_mean),
                "std": list(self.source_std),
                "count": self.source_count,
            },
            "target_adapt": {
                "mean": list(self.target_mean),
                "std": list(self.target_std),
                "count": self.target_count,
            },
            "discrepancy": list(self.discrepancy),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CAPMResidualStats":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != RESIDUAL_STATS_SCHEMA:
            raise ValueError(f"Unsupported residual-statistics schema: {payload.get('schema')!r}")
        return cls(
            source_mean=tuple(float(value) for value in payload["source"]["mean"]),
            source_std=tuple(float(value) for value in payload["source"]["std"]),
            target_mean=tuple(float(value) for value in payload["target_adapt"]["mean"]),
            target_std=tuple(float(value) for value in payload["target_adapt"]["std"]),
            discrepancy=tuple(float(value) for value in payload["discrepancy"]),
            source_count=int(payload["source"]["count"]),
            target_count=int(payload["target_adapt"]["count"]),
            metadata=dict(payload.get("metadata") or {}),
        )


class FeatureMomentAccumulator:
    """Streaming channel moments for spatial feature maps."""

    def __init__(self) -> None:
        self._sum: Tensor | None = None
        self._sum_sq: Tensor | None = None
        self._elements = 0
        self._samples = 0

    @torch.no_grad()
    def update(self, features: Tensor) -> None:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, C, D, H, W]")
        values = features.detach().to(dtype=torch.float64)
        batch, channels = values.shape[:2]
        flattened = values.permute(1, 0, 2, 3, 4).reshape(channels, -1)
        channel_sum = flattened.sum(dim=1).cpu()
        channel_sum_sq = flattened.square().sum(dim=1).cpu()
        if self._sum is None:
            self._sum = torch.zeros(channels, dtype=torch.float64)
            self._sum_sq = torch.zeros(channels, dtype=torch.float64)
        if self._sum.numel() != channels:
            raise ValueError("all feature batches must have the same channel count")
        self._sum += channel_sum
        self._sum_sq += channel_sum_sq
        self._elements += int(flattened.shape[1])
        self._samples += int(batch)

    def summary(self) -> dict[str, Any]:
        if self._sum is None or self._elements < 1 or self._samples < 1:
            raise RuntimeError("Cannot summarize an empty feature population")
        mean = self._sum / self._elements
        variance = (self._sum_sq / self._elements - mean.square()).clamp_min(0.0)
        return {
            "count": int(self._samples),
            "elements": int(self._elements),
            "mean": [float(value) for value in mean.tolist()],
            "std": [float(value) for value in variance.sqrt().tolist()],
        }


class CAPMResidualAdapter3D(nn.Module):
    """Apply a fixed, bounded channel-wise correction to a spatial feature map."""

    def __init__(
        self,
        stats: CAPMResidualStats,
        *,
        max_strength: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (0.0 <= max_strength <= 1.0):
            raise ValueError("max_strength must lie in [0, 1]")
        self.max_strength = float(max_strength)
        self.eps = float(eps)
        self.register_buffer("source_mean", torch.tensor(stats.source_mean, dtype=torch.float32))
        self.register_buffer("source_std", torch.tensor(stats.source_std, dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(stats.target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(stats.target_std, dtype=torch.float32))
        self.register_buffer("discrepancy", torch.tensor(stats.discrepancy, dtype=torch.float32))
        self.last_audit: dict[str, Tensor] | None = None

    @property
    def effective_strength(self) -> Tensor:
        return self.discrepancy.new_tensor(self.max_strength)

    def forward(
        self,
        features: Tensor,
        *,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, C, D, H, W]")
        if features.shape[1] != self.source_mean.numel():
            raise ValueError(
                f"feature channels ({features.shape[1]}) do not match residual statistics "
                f"({self.source_mean.numel()})"
            )
        shape = (1, -1, 1, 1, 1)
        source_mean = self.source_mean.to(dtype=features.dtype).view(*shape)
        source_std = self.source_std.to(dtype=features.dtype).clamp_min(self.eps).view(*shape)
        target_mean = self.target_mean.to(dtype=features.dtype).view(*shape)
        target_std = self.target_std.to(dtype=features.dtype).clamp_min(self.eps).view(*shape)
        aligned = (features - target_mean) / target_std * source_std + source_mean
        residual = aligned - features
        gate = self.discrepancy.to(dtype=features.dtype).view(*shape)
        correction = self.effective_strength.to(dtype=features.dtype) * gate * residual
        output = features + correction
        denominator = features.square().mean().detach().clamp_min(self.eps)
        audit = {
            "residual_rms": residual.square().mean().sqrt().reshape(1),
            "correction_rms": correction.square().mean().sqrt().reshape(1),
            "effective_strength": self.effective_strength.reshape(1),
            "gate_activity": (gate > 0).to(features.dtype).mean().reshape(1),
            "identity_loss": correction.square().mean() / denominator,
        }
        self.last_audit = audit
        if return_audit:
            return output, audit
        return output


class CAPMResidualAdaptation3D(ScaleTableInteractionAblation3D):
    """``layer4 -> original CAPM -> residual correction -> classifier`` pilot."""

    def __init__(
        self,
        *,
        stats: CAPMResidualStats,
        preset: str = "layer4_pixel",
        num_classes: int = 2,
        max_strength: float = 0.25,
        **kwargs: Any,
    ) -> None:
        if preset != "layer4_pixel":
            raise ValueError("CAPMResidualAdaptation3D requires preset='layer4_pixel'")
        super().__init__(preset=preset, interaction="original_capm", num_classes=num_classes, **kwargs)
        if stats.channels != 512:
            raise ValueError("layer4 residual adaptation expects 512-channel statistics")
        self.residual_adapter = CAPMResidualAdapter3D(stats, max_strength=max_strength)
        self.residual_stats_metadata = dict(stats.metadata)

    def extract_layer4(self, image: Tensor) -> Tensor:
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
        apply_adaptation: bool = True,
        return_audit: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if force_capm:
            raise ValueError("original_capm has no force_capm path")
        features = self.extract_layer4(image)
        features, capm_audit = self._apply_calibrator(features, table, False, return_audit)
        audit: dict[str, Tensor] = dict(capm_audit)
        if apply_adaptation:
            features, residual_audit = self.residual_adapter(features, return_audit=True)
            audit.update({f"residual_{key}": value for key, value in residual_audit.items()})
        return features, audit

    def forward(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        force_capm: bool = False,
        apply_adaptation: bool = True,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, audit = self.extract_features(
            image,
            table,
            force_capm=force_capm,
            apply_adaptation=apply_adaptation,
            return_audit=return_audit,
        )
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audit
        return logits

    def experiment_signature(self) -> dict[str, Any]:
        signature = super().experiment_signature()
        signature.update(
            {
                "model_family": "capm_conditioned_residual_adaptation",
                "residual_stage": "post_capm_layer4_map",
                "adaptation_mode": "frozen_target_stats_no_retrain",
                "residual_stats_schema": RESIDUAL_STATS_SCHEMA,
                "residual_stats_metadata": dict(self.residual_stats_metadata),
                "max_strength": self.residual_adapter.max_strength,
            }
        )
        return signature


def _extract_layer4_map(model: Any, images: Tensor) -> Tensor:
    if hasattr(model, "extract_layer4"):
        return model.extract_layer4(images)
    required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("model must expose extract_layer4(image) or the standard ResNet layer modules")
    features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    for name in ("layer1", "layer2", "layer3", "layer4"):
        features = getattr(model, name)(features)
    return features


@torch.no_grad()
def build_capm_residual_stats_from_loaders(
    model: Any,
    source_loader: Iterable[Mapping[str, Any]],
    target_adapt_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    output_path: str | Path,
    reference_table: Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CAPMResidualStats:
    """Build statistics from image-only loaders and persist the JSON artifact.

    If ``reference_table`` is supplied, the frozen model's CAPM output is the
    statistic basis.  The same source-only reference is used for source and
    ``T_adapt`` so the target loader remains image-only.
    """
    if not hasattr(model, "extract_layer4"):
        required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
        if not all(hasattr(model, name) for name in required):
            raise TypeError(
                "model must expose extract_layer4(image) or the standard ResNet layer modules"
            )
    previous_mode = bool(model.training)
    model.eval()
    source_accumulator = FeatureMomentAccumulator()
    target_accumulator = FeatureMomentAccumulator()

    def accumulate(loader: Iterable[Mapping[str, Any]], accumulator: FeatureMomentAccumulator) -> None:
        for batch in loader:
            if set(batch) - {"image", "subject_id"}:
                raise ValueError("residual-statistics loaders may contain only image and subject_id")
            images = batch["image"].to(device)
            if reference_table is None:
                features = _extract_layer4_map(model, images)
            else:
                table = reference_table.to(device=device, dtype=images.dtype)
                if table.ndim == 1:
                    table = table.unsqueeze(0).expand(images.shape[0], -1)
                if table.ndim != 2 or table.shape[0] != images.shape[0]:
                    raise ValueError("reference_table must be [3] or [B, 3]")
                result = model.extract_features(images, table)
                features = result[0] if isinstance(result, tuple) else result
            accumulator.update(features)

    try:
        accumulate(source_loader, source_accumulator)
        accumulate(target_adapt_loader, target_accumulator)
    finally:
        model.train(previous_mode)

    supplied = dict(metadata or {})
    supplied.update(
        {
            "method": (
                "label_free_capm_adjusted_channel_moment_residual"
                if reference_table is not None
                else "label_free_layer4_channel_moment_residual"
            ),
            "target_labels_read": False,
            "target_metrics_read": False,
        }
    )
    stats = CAPMResidualStats.from_summaries(
        source_accumulator.summary(), target_accumulator.summary(), metadata=supplied
    )
    stats.save(output_path)
    return stats


def build_capm_residual_model(
    stats: CAPMResidualStats,
    *,
    preset: str = "layer4_pixel",
    **kwargs: Any,
) -> CAPMResidualAdaptation3D:
    return CAPMResidualAdaptation3D(stats=stats, preset=preset, **kwargs)


__all__ = [
    "CAPMResidualAdaptation3D",
    "CAPMResidualAdapter3D",
    "CAPMResidualStats",
    "FeatureMomentAccumulator",
    "RESIDUAL_STATS_SCHEMA",
    "TARGET_SPLIT_SCHEMA",
    "build_capm_residual_model",
    "build_capm_residual_stats_from_loaders",
    "save_capm_residual_target_split",
]
