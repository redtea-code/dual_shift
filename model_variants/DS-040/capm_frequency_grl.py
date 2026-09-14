"""CAPM-conditioned raw-Fourier adaptation for DS-040.

The classifier always sees the complete CAPM feature map.  A domain and an
intensity discriminator can instead receive either that map (``full``) or
the complement of a frozen, source-task support subspace (``residual``).
This separation is the central DS-040 intervention; it is intentionally
small enough to make the full/residual comparison auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from Model.ablation.fmm_baseline import GradientReversal, SpatialGate3d
from Model.ablation.scale_table_transformer import ScaleTableInteractionAblation3D
from training.fmm_frequency import intensity_transform, mix_amplitude, sample_lambda


PROJECTOR_SCHEMA = "dualshift_capm_task_projector_v1"


class TaskSupportProjector(nn.Module):
    """Frozen channel projector estimated from source diagnosis gradients.

    ``basis`` contains orthonormal columns spanning directions supported by
    the source classifier.  Residual adaptation uses ``(I - P)B``; no target
    image or target metadata is consumed while fitting this projector.
    """

    def __init__(self, basis: Tensor, *, source_count: int = 0, metadata: Mapping[str, Any] | None = None):
        super().__init__()
        basis = torch.as_tensor(basis, dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 1:
            raise ValueError("basis must have shape [channels, rank]")
        if basis.shape[1] > basis.shape[0]:
            raise ValueError("projector rank cannot exceed channel count")
        gram = basis.T @ basis
        if not torch.allclose(gram, torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-4, rtol=1e-4):
            raise ValueError("projector basis must be orthonormal")
        self.register_buffer("basis", basis.contiguous())
        self.source_count = int(source_count)
        self.metadata = dict(metadata or {})

    @property
    def channels(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def project(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError(f"features must have shape [B,{self.channels},D,H,W]")
        basis = self.basis.to(device=features.device, dtype=features.dtype)
        coefficients = torch.einsum("cq,bcdhw->bqdhw", basis, features)
        return torch.einsum("cq,bqdhw->bcdhw", basis, coefficients)

    def residual(self, features: Tensor) -> Tensor:
        return features - self.project(features)

    @classmethod
    @torch.no_grad()
    def fit_from_pooled_features(
        cls,
        pooled_features: Tensor,
        labels: Tensor,
        classifier: nn.Linear,
        *,
        rank: int = 32,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TaskSupportProjector":
        """Fit the top eigenspace of per-subject source CE gradients."""
        if pooled_features.ndim != 2:
            raise ValueError("pooled_features must have shape [N,C]")
        labels = labels.to(device=pooled_features.device, dtype=torch.long).flatten()
        if len(labels) != len(pooled_features) or len(labels) < 2:
            raise ValueError("features and labels must contain at least two matching samples")
        if classifier.in_features != pooled_features.shape[1]:
            raise ValueError("classifier width does not match pooled features")
        weights = classifier.weight.detach().to(pooled_features)
        logits = F.linear(pooled_features, weights, classifier.bias.detach().to(pooled_features) if classifier.bias is not None else None)
        probabilities = logits.softmax(dim=1)
        gradients = probabilities @ weights - weights.index_select(0, labels)
        centered = gradients - gradients.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        if int(rank) < 1:
            raise ValueError("projector rank must be positive")
        effective_rank = min(int(rank), pooled_features.shape[1])
        order = torch.argsort(eigenvalues, descending=True)[:effective_rank]
        basis = eigenvectors[:, order]
        supplied = dict(metadata or {})
        supplied.update({
            "schema": PROJECTOR_SCHEMA,
            "fit_method": "source_ce_gradient_covariance",
            "rank": int(effective_rank),
            "channels": int(pooled_features.shape[1]),
            "top_eigenvalue_mass": float(eigenvalues[order].sum() / eigenvalues.clamp_min(0).sum().clamp_min(1e-12)),
            "target_labels_read": False,
            "target_metrics_read": False,
        })
        return cls(basis, source_count=len(labels), metadata=supplied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROJECTOR_SCHEMA,
            "channels": self.channels,
            "rank": self.rank,
            "source_count": self.source_count,
            "basis": self.basis.detach().cpu().tolist(),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str) -> None:
        import json
        from pathlib import Path
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TaskSupportProjector":
        import json
        from pathlib import Path
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != PROJECTOR_SCHEMA:
            raise ValueError(f"Unsupported projector schema: {payload.get('schema')!r}")
        return cls(torch.tensor(payload["basis"], dtype=torch.float32), source_count=int(payload.get("source_count", 0)), metadata=payload.get("metadata") or {})


def _pooled(features: Tensor) -> Tensor:
    if features.ndim != 5:
        raise ValueError("CAPM features must have shape [B,C,D,H,W]")
    return features.mean(dim=(-3, -2, -1))


def _binary_auc(logits: Tensor, labels: Tensor) -> Tensor:
    """Exact pairwise AUC for the synthetic domain/intensity labels."""
    positives = logits[labels > 0.5]
    negatives = logits[labels <= 0.5]
    if not len(positives) or not len(negatives):
        return logits.new_zeros(())
    pairwise = positives[:, None] - negatives[None, :]
    return (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean().detach()


class CAPMFrequencyGRL3D(nn.Module):
    """ResNet10 CAPM feature path with selectable full/residual GRL inputs."""

    def __init__(
        self,
        *,
        projector: TaskSupportProjector | None = None,
        grl_mode: str = "full",
        domain_grl: bool = False,
        intensity_grl: bool = False,
        num_classes: int = 2,
        layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        input_shape: tuple[int, int, int] = (160, 196, 160),
        classifier_dropout: float = 0.3,
        grl_coefficient: float = 1.0,
        use_table_concat: bool = False,
    ) -> None:
        super().__init__()
        if grl_mode not in {"full", "residual"}:
            raise ValueError("grl_mode must be 'full' or 'residual'")
        if grl_mode == "residual" and projector is None:
            raise ValueError("residual GRL mode requires a projector")
        self.grl_mode = grl_mode
        self.domain_grl = bool(domain_grl)
        self.intensity_grl = bool(intensity_grl)
        self.grl_coefficient = float(grl_coefficient)
        self.use_table_concat = bool(use_table_concat)
        self.backbone = ScaleTableInteractionAblation3D(
            preset="layer4_pixel",
            interaction="original_capm",
            num_classes=num_classes,
            layers=layers,
            input_shape=input_shape,
            classifier_dropout=classifier_dropout,
        )
        self.projector = projector
        self.spatial_gate = SpatialGate3d()
        self.last_attention: Tensor | None = None
        self.domain_discriminator = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.table_concat_classifier = nn.Linear(518, num_classes) if self.use_table_concat else None
        if self.table_concat_classifier is not None:
            nn.init.zeros_(self.table_concat_classifier.weight)
            nn.init.zeros_(self.table_concat_classifier.bias)
        self.intensity_discriminator = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    @property
    def classifier(self) -> nn.Linear:
        return self.backbone.fc

    def capm_features(self, image: Tensor, table: Tensor) -> Tensor:
        features = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(image))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(self.backbone, name)(features)
        # Keep the fixed attention branch table-free and before CAPM; the
        # classifier and both GRL placements therefore share the same B4.
        features = self.spatial_gate(features)
        self.last_attention = self.spatial_gate.attention
        features, _audit = self.backbone._apply_calibrator(features, table, False, False)
        return features

    def adversarial_features(self, capm_features: Tensor) -> Tensor:
        if self.grl_mode == "full":
            return capm_features
        if self.projector is None:
            raise RuntimeError("residual mode has no projector")
        return self.projector.residual(capm_features)

    def logits_from_features(self, features: Tensor, table_concat: Tensor | None = None) -> Tensor:
        pooled = self.backbone.dropout(self.backbone.pool(features).flatten(1))
        if self.use_table_concat:
            if table_concat is None or table_concat.ndim != 2 or table_concat.shape[1] != 6:
                raise ValueError("P0-M requires table_concat with shape [B,6]")
            return self.table_concat_classifier(torch.cat((pooled, table_concat), dim=1))
        return self.backbone.fc(pooled)

    def forward(self, image: Tensor, table: Tensor, *, table_concat: Tensor | None = None, return_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        features = self.capm_features(image, table)
        logits = self.logits_from_features(features, table_concat)
        return (logits, features) if return_features else logits

    def domain_logits(self, features: Tensor) -> Tensor:
        if not self.domain_grl:
            raise RuntimeError("domain GRL is disabled")
        return self.domain_discriminator(GradientReversal(self.grl_coefficient)(_pooled(self.adversarial_features(features)))).squeeze(1)

    def intensity_logits(self, features: Tensor) -> Tensor:
        if not self.intensity_grl:
            raise RuntimeError("intensity GRL is disabled")
        return self.intensity_discriminator(GradientReversal(self.grl_coefficient)(_pooled(self.adversarial_features(features)))).squeeze(1)

    def experiment_signature(self) -> dict[str, Any]:
        return {
            "model_family": "capm_conditioned_frequency_grl",
            "preset": "layer4_pixel",
            "capm_variables": ("age", "sex", "education"),
            "grl_mode": self.grl_mode,
            "domain_grl": self.domain_grl,
            "intensity_grl": self.intensity_grl,
            "grl_coefficient": self.grl_coefficient,
            "projector_rank": None if self.projector is None else self.projector.rank,
            "use_table_concat": self.use_table_concat,
        }


@dataclass(frozen=True)
class FrequencyBatch:
    source_image: Tensor
    source_intensity: Tensor
    target_style: Tensor


def make_frequency_batch(
    source_image: Tensor,
    target_image: Tensor,
    *,
    generator: torch.Generator | None = None,
    scale_range: tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.05,
    lambda_range: tuple[float, float] = (0.0, 1.0),
) -> FrequencyBatch:
    """Create the locked FMM intensity and target-style images."""
    source_intensity_raw = intensity_transform(source_image, scale_range=scale_range, noise_std=noise_std, generator=generator)
    source_intensity = mix_amplitude(source_intensity_raw, source_image, 1.0, phase_source=True)
    mixing = sample_lambda(source_image.shape[0], low=lambda_range[0], high=lambda_range[1], device=source_image.device, generator=generator)
    target_style = mix_amplitude(source_image, target_image, mixing)
    return FrequencyBatch(source_image=source_image, source_intensity=source_intensity, target_style=target_style)


def compute_capm_frequency_losses(
    model: CAPMFrequencyGRL3D,
    batch: FrequencyBatch,
    source_table: Tensor,
    labels: Tensor,
    *,
    lambda_domain: float = 1.0,
    lambda_intensity: float = 1.0,
    lambda_attention: float = 1.0,
    lambda_anchor: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute DS-040 losses without reading any target label or covariate."""
    source_features = model.capm_features(batch.source_image, source_table)
    source_attention = model.last_attention
    intensity_features = model.capm_features(batch.source_intensity, source_table)
    intensity_attention = model.last_attention
    target_features = model.capm_features(batch.target_style, source_table)
    labels = labels.to(device=source_features.device, dtype=torch.long).flatten()
    cls = 0.5 * (F.cross_entropy(model.logits_from_features(source_features), labels) + F.cross_entropy(model.logits_from_features(intensity_features), labels))
    zero = cls.new_zeros(())
    domain = zero
    intensity = zero
    domain_accuracy = zero
    intensity_accuracy = zero
    domain_auc = zero
    intensity_auc = zero
    if model.domain_grl:
        features = torch.cat((source_features, target_features), dim=0)
        domain_labels = torch.cat((torch.zeros(len(source_features), device=labels.device), torch.ones(len(target_features), device=labels.device)))
        domain_logits = model.domain_logits(features)
        domain = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
        domain_accuracy = ((domain_logits.detach() >= 0).to(domain_labels.dtype) == domain_labels).float().mean()
        domain_auc = _binary_auc(domain_logits.detach(), domain_labels)
    if model.intensity_grl:
        features = torch.cat((source_features, intensity_features), dim=0)
        intensity_labels = torch.cat((torch.zeros(len(source_features), device=labels.device), torch.ones(len(intensity_features), device=labels.device)))
        intensity_logits = model.intensity_logits(features)
        intensity = F.binary_cross_entropy_with_logits(intensity_logits, intensity_labels)
        intensity_accuracy = ((intensity_logits.detach() >= 0).to(intensity_labels.dtype) == intensity_labels).float().mean()
        intensity_auc = _binary_auc(intensity_logits.detach(), intensity_labels)
    # The attention term is a shared, table-free spatial-energy consistency
    # constraint; it is present in every frequency cell, including F0.
    if source_attention is None or intensity_attention is None:
        raise RuntimeError("spatial attention was not computed")
    attention = F.mse_loss(source_attention / source_attention.mean(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6), intensity_attention / intensity_attention.mean(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6))
    if model.projector is None:
        anchor_source = source_features.mean(dim=(-3, -2, -1))
        anchor_intensity = intensity_features.mean(dim=(-3, -2, -1))
    else:
        anchor_source = model.projector.project(source_features).mean(dim=(-3, -2, -1))
        anchor_intensity = model.projector.project(intensity_features).mean(dim=(-3, -2, -1))
    anchor = F.mse_loss(anchor_source, anchor_intensity)
    adversarial = model.adversarial_features(source_features)
    residual_norm = adversarial.square().mean().sqrt()
    full_norm = source_features.square().mean().sqrt()
    target_adversarial = model.adversarial_features(target_features)
    full_mean_shift = (source_features.mean(dim=(0, -3, -2, -1)) - target_features.mean(dim=(0, -3, -2, -1))).square().mean().sqrt().detach()
    adversarial_mean_shift = (adversarial.mean(dim=(0, -3, -2, -1)) - target_adversarial.mean(dim=(0, -3, -2, -1))).square().mean().sqrt().detach()
    full_mmd_proxy = (source_features.mean(dim=(0, -3, -2, -1)) - target_features.mean(dim=(0, -3, -2, -1))).square().mean().detach()
    adversarial_mmd_proxy = (adversarial.mean(dim=(0, -3, -2, -1)) - target_adversarial.mean(dim=(0, -3, -2, -1))).square().mean().detach()
    source_amplitude = torch.fft.fftn(batch.source_image.detach(), dim=(-3, -2, -1), norm="ortho").abs().mean()
    target_style_amplitude = torch.fft.fftn(batch.target_style.detach(), dim=(-3, -2, -1), norm="ortho").abs().mean()
    total = cls + float(lambda_domain) * domain + float(lambda_intensity) * intensity + float(lambda_attention) * attention + float(lambda_anchor) * anchor
    return total, {"classification": cls, "domain": domain, "intensity": intensity, "attention": attention, "anchor": anchor, "domain_accuracy": domain_accuracy, "intensity_accuracy": intensity_accuracy, "domain_auc": domain_auc, "intensity_auc": intensity_auc, "adversarial_feature_rms": residual_norm, "full_feature_rms": full_norm, "full_feature_mean_shift": full_mean_shift, "adversarial_feature_mean_shift": adversarial_mean_shift, "full_feature_mmd_proxy": full_mmd_proxy, "adversarial_feature_mmd_proxy": adversarial_mmd_proxy, "source_amplitude_mean": source_amplitude.detach(), "target_style_amplitude_mean": target_style_amplitude.detach()}


__all__ = [
    "CAPMFrequencyGRL3D",
    "FrequencyBatch",
    "PROJECTOR_SCHEMA",
    "TaskSupportProjector",
    "compute_capm_frequency_losses",
    "make_frequency_batch",
]
