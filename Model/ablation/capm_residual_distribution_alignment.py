"""DS-041 CAPM-conditioned source-free residual distribution alignment.

The module deliberately keeps the first DS-041 stage non-generative and
non-parametric at deployment time. A frozen ``original_capm`` backbone produces
``B4``. A source-only task-support projector optionally restricts the transport
to ``R4 = (I - P_task) B4``. Source statistics are serialized once; target
statistics are estimated from an image-only, subject-disjoint ``T_adapt`` view.

The K=1 path is a diagonal moment transport. The K>1 path is a compact,
per-channel diagonal GMM transport inspired by SFHarmony. GMM observations are
subject-level pooled vectors, never individual spatial activations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .capm_frequency_grl import PROJECTOR_SCHEMA, TaskSupportProjector
from .scale_table_transformer import ScaleTableInteractionAblation3D


RESIDUAL_DISTRIBUTION_STATS_SCHEMA = "dualshift_capm_residual_distribution_stats_v1"
RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA = (
    "dualshift_capm_residual_distribution_target_split_v1"
)


def load_task_support_projector_artifact(path: str | Path) -> TaskSupportProjector:
    """Load a standalone projector or the nested projector in a DS-040 report."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("projector artifact JSON must contain an object")
    candidate = payload
    if payload.get("schema") != PROJECTOR_SCHEMA:
        candidate = payload.get("projector")
    if not isinstance(candidate, Mapping) or candidate.get("schema") != PROJECTOR_SCHEMA:
        raise ValueError("projector artifact must be a TaskSupportProjector or DS-040 report")
    basis = torch.tensor(candidate.get("basis"), dtype=torch.float32)
    return TaskSupportProjector(
        basis,
        source_count=int(candidate.get("source_count", 0)),
        metadata=candidate.get("metadata") or {},
    )


def _finite_matrix(values: Any, *, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N, C] with N,C >= 1")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _finite_gmm(values: Any, *, name: str, shape: tuple[int, int]) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _subject_digest(subject_ids: Iterable[object]) -> str:
    payload = json.dumps(
        sorted({str(value) for value in subject_ids}), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_residual_distribution_target_split(
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
    """Persist a target split with an explicit label-blind adaptation contract."""
    subjects = list(subject_ids)
    adaptation = [int(index) for index in adaptation_indices]
    test = [int(index) for index in test_indices]
    if not adaptation or not test:
        raise ValueError("target adaptation/test indices must be non-empty")
    if any(index < 0 or index >= len(subjects) for index in adaptation + test):
        raise IndexError("target split index is outside subject_ids")
    if len(set(adaptation)) != len(adaptation) or len(set(test)) != len(test):
        raise ValueError("target split indices must not contain duplicates")
    if set(adaptation).intersection(test):
        raise ValueError("target adaptation/test indices must be disjoint")
    adaptation_subjects = sorted({str(subjects[index]) for index in adaptation})
    test_subjects = sorted({str(subjects[index]) for index in test})
    if set(adaptation_subjects).intersection(test_subjects):
        raise ValueError("target adaptation/test subjects must be disjoint")
    payload = {
        "schema": RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA,
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


def _normal_log_prob(values: Tensor, means: Tensor, stds: Tensor, weights: Tensor) -> Tensor:
    """Return per-observation log probability for a diagonal 1D GMM."""
    log_weights = weights.clamp_min(1e-12).log()
    log_scale = stds.clamp_min(1e-6).log()
    standardized = (values.unsqueeze(-1) - means) / stds.clamp_min(1e-6)
    return log_weights - log_scale - 0.5 * standardized.square() - 0.5 * torch.log(
        values.new_tensor(2.0 * torch.pi)
    )


@torch.no_grad()
def _fit_diagonal_gmm(
    values: Tensor,
    components: int,
    *,
    iterations: int = 30,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit one independent GMM per channel with deterministic quantile starts."""
    values = _finite_matrix(values, name="values")
    if int(components) not in {1, 2}:
        raise ValueError("DS-041 supports only components=1 or components=2")
    if values.shape[0] < components:
        raise ValueError("at least as many subject observations as components are required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    n, channels = values.shape
    sorted_values = values.sort(dim=0).values
    positions = torch.linspace(0, n - 1, components, dtype=torch.float64)
    indices = positions.round().long().clamp(0, n - 1)
    means = sorted_values.index_select(0, indices).T.contiguous()
    global_std = values.std(dim=0, unbiased=False).clamp_min(eps)
    stds = global_std.unsqueeze(1).expand(channels, components).clone()
    weights = values.new_full((channels, components), 1.0 / components)

    for _ in range(iterations):
        # [N, C, K], with one GMM per channel.
        log_prob = _normal_log_prob(
            values, means.unsqueeze(0), stds.unsqueeze(0), weights.unsqueeze(0)
        )
        responsibilities = (log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True)).exp()
        mass = responsibilities.sum(dim=0).clamp_min(eps)
        weights = mass / float(n)
        means = (responsibilities * values.unsqueeze(-1)).sum(dim=0) / mass
        variance = (
            responsibilities * (values.unsqueeze(-1) - means.unsqueeze(0)).square()
        ).sum(dim=0) / mass
        stds = variance.clamp_min(eps * eps).sqrt()

    # Component ordering is part of the serialized contract and removes label
    # switching between independently fitted source and target models.
    order = means.argsort(dim=1)
    means = means.gather(1, order)
    stds = stds.gather(1, order)
    weights = weights.gather(1, order)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return means, stds, weights


def _weighted_bhattacharyya(
    source_mean: Tensor,
    source_std: Tensor,
    source_weight: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    target_weight: Tensor,
) -> Tensor:
    pooled_variance = source_std.square() + target_std.square()
    distance = 0.25 * torch.log(
        0.25 * (source_std.square() / target_std.square().clamp_min(1e-12)
                + target_std.square() / source_std.square().clamp_min(1e-12)
                + 2.0)
    ) + (source_mean - target_mean).square() / (4.0 * pooled_variance.clamp_min(1e-12))
    pair_weight = 0.5 * (source_weight + target_weight)
    return (pair_weight * distance).sum(dim=1)


@dataclass(frozen=True)
class ResidualDistributionStats:
    """Serializable source/target diagonal GMM summary for pooled residuals."""

    source_mean: tuple[tuple[float, ...], ...]
    source_std: tuple[tuple[float, ...], ...]
    source_weight: tuple[tuple[float, ...], ...]
    target_mean: tuple[tuple[float, ...], ...]
    target_std: tuple[tuple[float, ...], ...]
    target_weight: tuple[tuple[float, ...], ...]
    discrepancy: tuple[float, ...]
    source_count: int
    target_count: int
    components: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.components not in {1, 2}:
            raise ValueError("DS-041 supports only components=1 or components=2")
        names = (
            "source_mean", "source_std", "source_weight",
            "target_mean", "target_std", "target_weight",
        )
        tensors = {
            name: _finite_gmm(
                getattr(self, name), name=name,
                shape=(len(getattr(self, "source_mean")), self.components),
            )
            for name in names
        }
        channels = tensors["source_mean"].shape[0]
        for name in names:
            if tensors[name].shape != (channels, self.components):
                raise ValueError("all GMM summaries must have equal channel/component shape")
        for name in ("source_std", "target_std"):
            if (tensors[name] <= 0).any():
                raise ValueError(f"{name} must be strictly positive")
        for name in ("source_weight", "target_weight"):
            weights = tensors[name]
            if (weights < 0).any() or not torch.allclose(
                weights.sum(dim=1), torch.ones(channels, dtype=weights.dtype), atol=1e-5
            ):
                raise ValueError(f"{name} must be non-negative and sum to one per channel")
        discrepancy = torch.as_tensor(self.discrepancy, dtype=torch.float64).flatten()
        if discrepancy.numel() != channels or not torch.isfinite(discrepancy).all():
            raise ValueError("discrepancy must match channel count and be finite")
        if (discrepancy < 0).any() or float(discrepancy.max()) > 1.0 + 1e-8:
            raise ValueError("discrepancy must lie in [0, 1]")
        if self.source_count < 1 or self.target_count < 1:
            raise ValueError("source_count and target_count must be positive")

    @property
    def channels(self) -> int:
        return len(self.source_mean)

    @classmethod
    def fit(
        cls,
        source_pooled: Any,
        target_pooled: Any,
        *,
        components: int = 1,
        iterations: int = 30,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResidualDistributionStats":
        source = _finite_matrix(source_pooled, name="source_pooled")
        target = _finite_matrix(target_pooled, name="target_pooled")
        if int(components) not in {1, 2}:
            raise ValueError("DS-041 supports only components=1 or components=2")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target pooled residuals must have equal channel counts")
        source_mean, source_std, source_weight = _fit_diagonal_gmm(
            source, int(components), iterations=iterations
        )
        target_mean, target_std, target_weight = _fit_diagonal_gmm(
            target, int(components), iterations=iterations
        )
        raw = _weighted_bhattacharyya(
            source_mean, source_std, source_weight,
            target_mean, target_std, target_weight,
        )
        maximum = raw.max()
        discrepancy = raw / maximum if float(maximum) > 1e-8 else torch.zeros_like(raw)
        supplied = dict(metadata or {})
        supplied.update({
            "schema": RESIDUAL_DISTRIBUTION_STATS_SCHEMA,
            "statistic_unit": "subject_level_gap_residual",
            "components": int(components),
            "gmm_iterations": int(iterations),
            "target_labels_read": False,
            "target_metrics_read": False,
        })
        return cls(
            source_mean=tuple(tuple(float(v) for v in row) for row in source_mean.tolist()),
            source_std=tuple(tuple(float(v) for v in row) for row in source_std.tolist()),
            source_weight=tuple(tuple(float(v) for v in row) for row in source_weight.tolist()),
            target_mean=tuple(tuple(float(v) for v in row) for row in target_mean.tolist()),
            target_std=tuple(tuple(float(v) for v in row) for row in target_std.tolist()),
            target_weight=tuple(tuple(float(v) for v in row) for row in target_weight.tolist()),
            discrepancy=tuple(float(v) for v in discrepancy.tolist()),
            source_count=int(source.shape[0]),
            target_count=int(target.shape[0]),
            components=int(components),
            metadata=supplied,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_DISTRIBUTION_STATS_SCHEMA,
            "channels": self.channels,
            "components": self.components,
            "source": {
                "mean": [list(row) for row in self.source_mean],
                "std": [list(row) for row in self.source_std],
                "weight": [list(row) for row in self.source_weight],
                "count": self.source_count,
            },
            "target_adapt": {
                "mean": [list(row) for row in self.target_mean],
                "std": [list(row) for row in self.target_std],
                "weight": [list(row) for row in self.target_weight],
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
    def load(cls, path: str | Path) -> "ResidualDistributionStats":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != RESIDUAL_DISTRIBUTION_STATS_SCHEMA:
            raise ValueError(f"Unsupported residual distribution schema: {payload.get('schema')!r}")
        components = int(payload["components"])
        source = payload["source"]
        target = payload["target_adapt"]
        return cls(
            source_mean=tuple(tuple(float(v) for v in row) for row in source["mean"]),
            source_std=tuple(tuple(float(v) for v in row) for row in source["std"]),
            source_weight=tuple(tuple(float(v) for v in row) for row in source["weight"]),
            target_mean=tuple(tuple(float(v) for v in row) for row in target["mean"]),
            target_std=tuple(tuple(float(v) for v in row) for row in target["std"]),
            target_weight=tuple(tuple(float(v) for v in row) for row in target["weight"]),
            discrepancy=tuple(float(v) for v in payload["discrepancy"]),
            source_count=int(source["count"]),
            target_count=int(target["count"]),
            components=components,
            metadata=dict(payload.get("metadata") or {}),
        )


class SubjectResidualAccumulator:
    """Collect one GAP residual vector per subject for summary fitting."""

    def __init__(self) -> None:
        self._rows: list[Tensor] = []
        self._subjects: set[str] = set()

    @torch.no_grad()
    def update(self, features: Tensor, subject_ids: Sequence[object] | None = None) -> None:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,C,D,H,W]")
        pooled = features.detach().mean(dim=(-3, -2, -1)).to(dtype=torch.float64).cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError("features must contain only finite values")
        if subject_ids is not None:
            ids = [str(value) for value in subject_ids]
            if len(ids) != pooled.shape[0]:
                raise ValueError("subject_ids must match feature batch size")
            if len(set(ids)) != len(ids):
                raise ValueError("each subject must contribute one observation; duplicate IDs in batch")
            duplicate = self._subjects.intersection(ids)
            if duplicate:
                raise ValueError(
                    "each subject must contribute one observation; duplicate IDs across batches="
                    f"{sorted(duplicate)[:3]}"
                )
            self._subjects.update(ids)
        self._rows.append(pooled)

    def pooled(self) -> Tensor:
        if not self._rows:
            raise RuntimeError("Cannot summarize an empty residual population")
        values = torch.cat(self._rows, dim=0)
        if not torch.isfinite(values).all():
            raise ValueError("pooled residual observations must be finite")
        return values

    @property
    def count(self) -> int:
        return int(sum(int(row.shape[0]) for row in self._rows))


class ResidualDistributionTransport3D(nn.Module):
    """Apply bounded diagonal or per-channel GMM residual transport."""

    def __init__(
        self,
        stats: ResidualDistributionStats,
        *,
        max_strength: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (0.0 <= max_strength <= 1.0):
            raise ValueError("max_strength must lie in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.max_strength = float(max_strength)
        self.eps = float(eps)
        self.components = int(stats.components)
        self.register_buffer("source_mean", torch.tensor(stats.source_mean, dtype=torch.float32))
        self.register_buffer("source_std", torch.tensor(stats.source_std, dtype=torch.float32))
        self.register_buffer("source_weight", torch.tensor(stats.source_weight, dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(stats.target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(stats.target_std, dtype=torch.float32))
        self.register_buffer("target_weight", torch.tensor(stats.target_weight, dtype=torch.float32))
        self.register_buffer("discrepancy", torch.tensor(stats.discrepancy, dtype=torch.float32))
        self.last_audit: dict[str, Tensor] | None = None

    def _select_components(self, features: Tensor) -> Tensor:
        if self.components == 1:
            return torch.zeros(
                (features.shape[0], features.shape[1]), dtype=torch.long, device=features.device
            )
        pooled = features.mean(dim=(-3, -2, -1))
        means = self.target_mean.to(device=features.device, dtype=features.dtype)
        stds = self.target_std.to(device=features.device, dtype=features.dtype).clamp_min(self.eps)
        # Per-channel MAP assignments avoid a learned target classifier and use
        # only the serialized target adaptation summary.
        score = ((pooled.unsqueeze(-1) - means.unsqueeze(0)) / stds.unsqueeze(0)).square()
        return score.argmin(dim=-1)

    def forward(
        self,
        features: Tensor,
        *,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,C,D,H,W]")
        if features.shape[1] != self.source_mean.shape[0]:
            raise ValueError(
                f"feature channels ({features.shape[1]}) do not match stats ({self.source_mean.shape[0]})"
            )
        shape = (1, features.shape[1], 1, 1, 1)
        assignments = self._select_components(features)
        source_mean = self.source_mean.to(device=features.device, dtype=features.dtype)
        source_std = self.source_std.to(device=features.device, dtype=features.dtype).clamp_min(self.eps)
        target_mean = self.target_mean.to(device=features.device, dtype=features.dtype)
        target_std = self.target_std.to(device=features.device, dtype=features.dtype).clamp_min(self.eps)
        if self.components > 1:
            batch_count = features.shape[0]
            source_table = source_mean.unsqueeze(0).expand(batch_count, -1, -1)
            source_scale_table = source_std.unsqueeze(0).expand(batch_count, -1, -1)
            target_table = target_mean.unsqueeze(0).expand(batch_count, -1, -1)
            target_scale_table = target_std.unsqueeze(0).expand(batch_count, -1, -1)
            component_index = assignments.unsqueeze(-1)
            source_mean_batch = torch.gather(source_table, 2, component_index).squeeze(-1)
            source_std_batch = torch.gather(source_scale_table, 2, component_index).squeeze(-1)
            target_mean_batch = torch.gather(target_table, 2, component_index).squeeze(-1)
            target_std_batch = torch.gather(target_scale_table, 2, component_index).squeeze(-1)
            source_mean_view = source_mean_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            source_std_view = source_std_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target_mean_view = target_mean_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target_std_view = target_std_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            source_mean_view = source_mean[:, :1].T.reshape(*shape)
            source_std_view = source_std[:, :1].T.reshape(*shape)
            target_mean_view = target_mean[:, :1].T.reshape(*shape)
            target_std_view = target_std[:, :1].T.reshape(*shape)
        aligned = (features - target_mean_view) / target_std_view * source_std_view + source_mean_view
        residual = aligned - features
        gate = self.discrepancy.to(device=features.device, dtype=features.dtype).reshape(*shape)
        correction = float(self.max_strength) * gate * residual
        output = features + correction
        denominator = features.square().mean().detach().clamp_min(self.eps)
        audit = {
            "correction_rms": correction.square().mean().sqrt().reshape(1),
            "residual_rms": residual.square().mean().sqrt().reshape(1),
            "effective_strength": features.new_tensor(self.max_strength).reshape(1),
            "gate_activity": (gate > 0).to(features.dtype).mean().reshape(1),
            "identity_loss": correction.square().mean().reshape(1) / denominator,
            "component_entropy": self._component_entropy().to(features.device).reshape(1),
            "finite": torch.isfinite(output).all().to(features.dtype).reshape(1),
        }
        if self.components > 1:
            audit["component_fraction"] = torch.bincount(
                assignments.reshape(-1), minlength=self.components
            ).to(features.dtype) / float(assignments.numel())
        self.last_audit = audit
        if return_audit:
            return output, audit
        return output

    def _component_entropy(self) -> Tensor:
        weights = self.target_weight.mean(dim=0).clamp_min(1e-8)
        return -(weights * weights.log()).sum()


class CAPMResidualDistributionAlignment3D(ScaleTableInteractionAblation3D):
    """Frozen ``layer4 -> original CAPM -> residual transport -> classifier``."""

    def __init__(
        self,
        *,
        stats: ResidualDistributionStats,
        projector: TaskSupportProjector | None = None,
        transport_scope: str = "residual",
        preset: str = "layer4_pixel",
        num_classes: int = 2,
        max_strength: float = 0.25,
        **kwargs: Any,
    ) -> None:
        if preset != "layer4_pixel":
            raise ValueError("DS-041 requires preset='layer4_pixel'")
        if transport_scope not in {"full", "residual"}:
            raise ValueError("transport_scope must be 'full' or 'residual'")
        if transport_scope == "residual" and projector is None:
            raise ValueError("residual transport requires a source task-support projector")
        if stats.channels != 512:
            raise ValueError("layer4 transport expects 512-channel statistics")
        if projector is not None and projector.channels != stats.channels:
            raise ValueError("projector and transport statistics must have equal channels")
        super().__init__(preset=preset, interaction="original_capm", num_classes=num_classes, **kwargs)
        self.projector = projector
        self.transport_scope = transport_scope
        self.transport = ResidualDistributionTransport3D(stats, max_strength=max_strength)
        self.stats_metadata = dict(stats.metadata)

    def extract_layer4(self, image: Tensor) -> Tensor:
        features = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(self, name)(features)
        return features

    def _apply_transport(
        self, capm_features: Tensor, *, apply_transport: bool, return_audit: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.transport_scope == "residual":
            if self.projector is None:
                raise RuntimeError("residual transport has no projector")
            task_component = self.projector.project(capm_features)
            transport_input = capm_features - task_component
        else:
            task_component = capm_features.new_zeros(capm_features.shape)
            transport_input = capm_features
        if not apply_transport:
            zero = capm_features.new_zeros(1)
            return capm_features, {
                "transport_applied": zero,
                "task_component_rms": task_component.square().mean().sqrt().reshape(1),
                "transport_input_rms": transport_input.square().mean().sqrt().reshape(1),
                "projected_correction_rms": zero,
                "output_rms": capm_features.square().mean().sqrt().reshape(1),
                "anchor_drift_rms": zero,
                "finite": torch.isfinite(capm_features).all().to(capm_features.dtype).reshape(1),
            }
        transported, audit = self.transport(transport_input, return_audit=True)
        raw_correction = transported - transport_input
        if self.transport_scope == "residual":
            # Channel-wise statistics do not preserve an arbitrary oblique
            # task subspace. Project the realized correction back to the
            # residual complement before recombining with the CAPM anchor.
            correction = self.projector.residual(raw_correction)  # type: ignore[union-attr]
            output = capm_features + correction
        else:
            correction = raw_correction
            output = transported
        audit = dict(audit)
        audit["transport_applied"] = capm_features.new_ones(1)
        audit["task_component_rms"] = task_component.square().mean().sqrt().reshape(1)
        audit["transport_input_rms"] = transport_input.square().mean().sqrt().reshape(1)
        audit["projected_correction_rms"] = correction.square().mean().sqrt().reshape(1)
        audit["output_rms"] = output.square().mean().sqrt().reshape(1)
        if self.projector is not None:
            anchor_delta = self.projector.project(output - capm_features)
            audit["anchor_drift_rms"] = anchor_delta.square().mean().sqrt().reshape(1)
        else:
            audit["anchor_drift_rms"] = (output - capm_features).square().mean().sqrt().reshape(1)
        return output, audit

    def extract_features(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        apply_transport: bool = True,
        return_audit: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        features = self.extract_layer4(image)
        result = self._apply_calibrator(features, table, False, return_audit)
        capm_features, capm_audit = result if isinstance(result, tuple) else (result, {})
        output, transport_audit = self._apply_transport(
            capm_features, apply_transport=apply_transport, return_audit=return_audit
        )
        audit = dict(capm_audit)
        audit.update({f"distribution_{key}": value for key, value in transport_audit.items()})
        return output, audit

    def forward(
        self,
        image: Tensor,
        table: Tensor | None = None,
        *,
        apply_transport: bool = True,
        return_audit: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, audit = self.extract_features(
            image, table, apply_transport=apply_transport, return_audit=return_audit
        )
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audit
        return logits

    def experiment_signature(self) -> dict[str, Any]:
        signature = super().experiment_signature()
        signature.update({
            "model_family": "capm_conditioned_source_free_residual_distribution_alignment",
            "transport_scope": self.transport_scope,
            "transport_components": self.transport.components,
            "transport_stats_schema": RESIDUAL_DISTRIBUTION_STATS_SCHEMA,
            "statistic_unit": "subject_level_gap_residual",
            "adaptation_mode": "frozen_target_summary_no_retrain",
            "max_strength": self.transport.max_strength,
            "projector_schema": None if self.projector is None else self.projector.metadata.get("schema"),
        })
        return signature


@torch.no_grad()
def build_residual_distribution_stats_from_loaders(
    model: Any,
    source_loader: Iterable[Mapping[str, Any]],
    target_adapt_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    output_path: str | Path,
    reference_table: Tensor | None = None,
    projector: TaskSupportProjector | None = None,
    components: int = 1,
    gmm_iterations: int = 30,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualDistributionStats:
    """Build stats from image-only loaders and persist the source-free artifact."""
    required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    if not hasattr(model, "extract_layer4") and not all(hasattr(model, name) for name in required):
        raise TypeError("model must expose extract_layer4(image) or standard ResNet layer modules")
    if int(components) not in {1, 2}:
        raise ValueError("DS-041 supports only components=1 or components=2")
    previous_mode = bool(model.training)
    model.eval()
    source_accumulator = SubjectResidualAccumulator()
    target_accumulator = SubjectResidualAccumulator()

    def extract_capm(images: Tensor) -> Tensor:
        if reference_table is None:
            if hasattr(model, "extract_layer4"):
                return model.extract_layer4(images)
            features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
            for name in ("layer1", "layer2", "layer3", "layer4"):
                features = getattr(model, name)(features)
            return features
        table = reference_table.to(device=images.device, dtype=images.dtype)
        if table.ndim == 1:
            table = table.unsqueeze(0).expand(images.shape[0], -1)
        if table.ndim != 2 or table.shape[0] != images.shape[0] or table.shape[1] != 3:
            raise ValueError("reference_table must be [3] or [B, 3]")
        if not hasattr(model, "extract_layer4"):
            features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
            for name in ("layer1", "layer2", "layer3", "layer4"):
                features = getattr(model, name)(features)
        else:
            features = model.extract_layer4(images)
        result = model._apply_calibrator(features, table, False, False)
        return result[0] if isinstance(result, tuple) else result

    def accumulate(loader: Iterable[Mapping[str, Any]], accumulator: SubjectResidualAccumulator) -> None:
        for batch in loader:
            if set(batch) - {"image", "subject_id"}:
                raise ValueError("DS-041 statistics loaders may contain only image and subject_id")
            images = batch["image"].to(device)
            capm_features = extract_capm(images)
            selected = projector.residual(capm_features) if projector is not None else capm_features
            accumulator.update(selected, batch["subject_id"])

    try:
        accumulate(source_loader, source_accumulator)
        accumulate(target_adapt_loader, target_accumulator)
    finally:
        model.train(previous_mode)

    supplied = dict(metadata or {})
    supplied.update({
        "method": "source_free_capm_conditioned_residual_distribution_alignment",
        "transport_scope": "residual" if projector is not None else "full",
        "target_labels_read": False,
        "target_metrics_read": False,
        "source_observation_count": source_accumulator.count,
        "target_observation_count": target_accumulator.count,
    })
    stats = ResidualDistributionStats.fit(
        source_accumulator.pooled(),
        target_accumulator.pooled(),
        components=components,
        iterations=gmm_iterations,
        metadata=supplied,
    )
    stats.save(output_path)
    return stats


@torch.no_grad()
def audit_synthetic_perturbation(
    model: CAPMResidualDistributionAlignment3D,
    clean_images: Tensor,
    perturbed_images: Tensor,
    table: Tensor,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Measure residual-distance recovery without using target labels."""
    if clean_images.shape != perturbed_images.shape:
        raise ValueError("clean_images and perturbed_images must have identical shapes")
    device_obj = torch.device(device)
    was_training = bool(model.training)
    model.eval()
    try:
        clean_control, _ = model(clean_images.to(device_obj), table.to(device_obj), apply_transport=False, return_audit=True)
        perturbed_control, _ = model(perturbed_images.to(device_obj), table.to(device_obj), apply_transport=False, return_audit=True)
        clean_adapt, _ = model(clean_images.to(device_obj), table.to(device_obj), apply_transport=True, return_audit=True)
        perturbed_adapt, _ = model(perturbed_images.to(device_obj), table.to(device_obj), apply_transport=True, return_audit=True)
        before = (perturbed_control - clean_control).float().square().mean().sqrt()
        after = (perturbed_adapt - clean_adapt).float().square().mean().sqrt()
        recovery = 1.0 - after / before.clamp_min(1e-6)
        return {
            "logit_distance_before": float(before.cpu()),
            "logit_distance_after": float(after.cpu()),
            "logit_recovery_fraction": float(recovery.cpu()),
            "clean_prediction_agreement": float(
                (clean_control.argmax(dim=1) == clean_adapt.argmax(dim=1)).float().mean().cpu()
            ),
            "perturbed_prediction_agreement": float(
                (perturbed_control.argmax(dim=1) == perturbed_adapt.argmax(dim=1)).float().mean().cpu()
            ),
            "finite": bool(torch.isfinite(clean_adapt).all() and torch.isfinite(perturbed_adapt).all()),
        }
    finally:
        model.train(was_training)


def apply_bounded_intensity_perturbation(
    images: Tensor, *, scale: float = 1.1, bias: float = 0.0
) -> Tensor:
    """Create a deterministic ESPA-style additive/multiplicative test perturbation."""
    if not torch.isfinite(images).all():
        raise ValueError("images must be finite")
    if scale <= 0:
        raise ValueError("scale must be positive")
    return images * float(scale) + float(bias)


def apply_fourier_amplitude_perturbation(images: Tensor, *, amplitude_scale: float = 1.1) -> Tensor:
    """Apply a phase-preserving, high-frequency raw-volume perturbation."""
    if images.ndim != 5:
        raise ValueError("images must have shape [B,C,D,H,W]")
    if amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be positive")
    dims = (-3, -2, -1)
    spectrum = torch.fft.rfftn(images, dim=dims, norm="ortho")
    spatial = tuple(int(size) for size in images.shape[-3:])
    axes = (
        torch.fft.fftfreq(spatial[0], device=images.device),
        torch.fft.fftfreq(spatial[1], device=images.device),
        torch.fft.rfftfreq(spatial[2], device=images.device),
    )
    grid = torch.meshgrid(*axes, indexing="ij")
    radius = torch.sqrt(sum(axis.square() for axis in grid))
    radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)
    scale = 1.0 + (float(amplitude_scale) - 1.0) * radius
    perturbed = spectrum * scale.reshape(1, 1, *scale.shape)
    return torch.fft.irfftn(perturbed, s=spatial, dim=dims, norm="ortho")


__all__ = [
    "CAPMResidualDistributionAlignment3D",
    "RESIDUAL_DISTRIBUTION_STATS_SCHEMA",
    "RESIDUAL_DISTRIBUTION_TARGET_SPLIT_SCHEMA",
    "ResidualDistributionStats",
    "ResidualDistributionTransport3D",
    "SubjectResidualAccumulator",
    "apply_bounded_intensity_perturbation",
    "apply_fourier_amplitude_perturbation",
    "audit_synthetic_perturbation",
    "build_residual_distribution_stats_from_loaders",
    "load_task_support_projector_artifact",
    "save_residual_distribution_target_split",
]
