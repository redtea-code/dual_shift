"""Cross-modal relation-preserving UDA model for DS-042.

The implementation keeps MRI and tabular inputs explicit.  It exposes the
intermediate representations required by the experiment plan and leaves
domain/batch interpretation to the training diagnostics rather than baking a
scanner label into the model.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_table_transformer import build_scale_table_ablation, demographic_var_specs


def _as_missing(
    value: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor:
    if value is None:
        return torch.zeros(
            reference.shape[0], 1, device=reference.device, dtype=reference.dtype
        )
    return value.reshape(reference.shape[0], 1).to(device=reference.device, dtype=reference.dtype)


def coral_loss(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Linear CORAL loss with a finite zero for singleton batches."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("CORAL inputs must have shape [batch, features]")
    if source.shape[1] != target.shape[1]:
        raise ValueError("CORAL feature dimensions must match")
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_zeros(())
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.transpose(0, 1).matmul(source_centered) / (source.shape[0] - 1)
    target_cov = target_centered.transpose(0, 1).matmul(target_centered) / (target.shape[0] - 1)
    scale = max(int(source.shape[1]), 1)
    return (source_cov - target_cov).square().mean() / float(scale * scale) + eps * 0.0


def paired_relation_loss(
    image_shared: torch.Tensor, table_shared: torch.Tensor
) -> torch.Tensor:
    """Cosine relation loss for subject-level MRI/table pairs."""
    if image_shared.shape != table_shared.shape:
        raise ValueError("paired relation inputs must have identical shapes")
    if image_shared.ndim != 2:
        raise ValueError("paired relation inputs must have shape [batch, features]")
    return (1.0 - F.cosine_similarity(image_shared, table_shared, dim=1)).mean()


class _ResidualAdapter(nn.Module):
    """Zero-initialized bounded residual adapter."""

    def __init__(self, dimension: int, max_strength: float) -> None:
        super().__init__()
        if max_strength < 0:
            raise ValueError("max_strength must be non-negative")
        self.max_strength = float(max_strength)
        self.linear = nn.Linear(int(dimension), int(dimension))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, value: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled or self.max_strength == 0.0:
            return value
        return value + self.max_strength * torch.tanh(self.linear(value))


class CMRPUDA3D(nn.Module):
    """MRI+table model with explicit relation-preserving UDA representations."""

    def __init__(
        self,
        *,
        preset: str = "layer4_pixel",
        layers: Sequence[int] = (1, 1, 1, 1),
        input_shape: Sequence[int] = (160, 196, 160),
        representation_dim: int = 128,
        table_hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_adapter_strength: float = 0.1,
        use_image: bool = True,
        use_table: bool = True,
    ) -> None:
        super().__init__()
        if int(representation_dim) < 2:
            raise ValueError("representation_dim must be at least 2")
        if int(table_hidden_dim) < 2:
            raise ValueError("table_hidden_dim must be at least 2")
        self.preset = str(preset)
        self.input_shape = tuple(int(item) for item in input_shape)
        self.representation_dim = int(representation_dim)
        self.use_image = bool(use_image)
        self.use_table = bool(use_table)
        self.table_variables = tuple(spec["name"] for spec in demographic_var_specs())

        # Reuse the tested 3-D backbone but keep its table calibrator disabled.
        self.image_backbone = build_scale_table_ablation(
            preset=self.preset,
            interaction="image_only",
            num_classes=num_classes,
            layers=tuple(int(item) for item in layers),
            input_shape=self.input_shape,
        )
        image_dim = int(self.image_backbone.fc.in_features)
        table_dim = len(self.table_variables) * 2
        self.mri_encoder = nn.Sequential(
            nn.Linear(image_dim, self.representation_dim),
            nn.LayerNorm(self.representation_dim),
            nn.GELU(),
        )
        self.table_encoder = nn.Sequential(
            nn.Linear(table_dim, int(table_hidden_dim)),
            nn.LayerNorm(int(table_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(table_hidden_dim), self.representation_dim),
            nn.LayerNorm(self.representation_dim),
            nn.GELU(),
        )
        self.mri_shared = nn.Linear(self.representation_dim, self.representation_dim)
        self.table_shared = nn.Linear(self.representation_dim, self.representation_dim)
        self.mri_specific = nn.Linear(self.representation_dim, self.representation_dim)
        self.table_specific = nn.Linear(self.representation_dim, self.representation_dim)
        self.mri_adapter = _ResidualAdapter(self.representation_dim, max_adapter_strength)
        self.table_adapter = _ResidualAdapter(self.representation_dim, max_adapter_strength)
        fusion_dim = self.representation_dim * 5
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.representation_dim * 2),
            nn.LayerNorm(self.representation_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.representation_dim * 2, self.representation_dim),
            nn.LayerNorm(self.representation_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(self.representation_dim, int(num_classes))

    def _image_embedding(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_image:
            zeros = image.new_zeros(image.shape[0], self.representation_dim)
            return zeros, image.new_zeros(image.shape[0], 1, 1, 1, 1)
        feature_map, _ = self.image_backbone.extract_features(image, table=None)
        pooled = self.image_backbone.pool(feature_map).flatten(1)
        return self.mri_encoder(pooled), feature_map

    def forward_with_repr(
        self,
        image: torch.Tensor,
        covariates: torch.Tensor,
        *,
        age_missing: torch.Tensor | None = None,
        sex_missing: torch.Tensor | None = None,
        education_missing: torch.Tensor | None = None,
        apply_adapter: bool = True,
    ) -> dict[str, torch.Tensor]:
        if covariates.ndim != 2 or covariates.shape[1] < 3:
            raise ValueError("covariates must have shape [B, 3]")
        if image.shape[0] != covariates.shape[0]:
            raise ValueError("image and covariates batch sizes must match")
        mri_raw, feature_map = self._image_embedding(image)
        if self.use_table:
            missing = torch.cat(
                (
                    _as_missing(age_missing, covariates),
                    _as_missing(sex_missing, covariates),
                    _as_missing(education_missing, covariates),
                ),
                dim=1,
            )
            table_input = torch.cat((covariates[:, :3].float(), missing), dim=1)
            table_raw = self.table_encoder(table_input)
        else:
            table_raw = covariates.new_zeros(covariates.shape[0], self.representation_dim)
        mri_shared = self.mri_shared(mri_raw)
        table_shared = self.table_shared(table_raw)
        mri_specific = self.mri_specific(mri_raw)
        table_specific = self.table_specific(table_raw)
        mri_specific = self.mri_adapter(mri_specific, enabled=apply_adapter)
        table_specific = self.table_adapter(table_specific, enabled=apply_adapter)
        joint_input = torch.cat(
            (
                mri_shared,
                table_shared,
                mri_specific,
                table_specific,
                mri_shared * table_shared,
            ),
            dim=1,
        )
        joint = self.fusion(joint_input)
        return {
            "logits": self.classifier(joint),
            "mri_embedding": mri_raw,
            "table_embedding": table_raw,
            "mri_shared": mri_shared,
            "table_shared": table_shared,
            "mri_specific": mri_specific,
            "table_specific": table_specific,
            "joint": joint,
            "feature_map": feature_map,
        }

    def forward(
        self,
        image: torch.Tensor,
        covariates: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward_with_repr(image, covariates, **kwargs)["logits"]

    def adapter_penalty(self) -> torch.Tensor:
        penalty = self.classifier.weight.new_zeros(())
        for module in (self.mri_adapter, self.table_adapter):
            penalty = penalty + sum((parameter.square().mean() for parameter in module.parameters()))
        return penalty

    def experiment_signature(self) -> dict[str, Any]:
        return {
            "model_family": "cmrp_uda",
            "preset": self.preset,
            "input_shape": self.input_shape,
            "representation_dim": self.representation_dim,
            "table_variables": self.table_variables,
            "use_image": self.use_image,
            "use_table": self.use_table,
            "max_adapter_strength": self.mri_adapter.max_strength,
        }


def build_cmrp_uda_model(**kwargs: Any) -> CMRPUDA3D:
    return CMRPUDA3D(**kwargs)


__all__ = [
    "CMRPUDA3D",
    "build_cmrp_uda_model",
    "coral_loss",
    "paired_relation_loss",
]
