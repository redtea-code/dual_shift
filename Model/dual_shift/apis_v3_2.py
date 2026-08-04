"""APIC v3_2-A: source-only fixed style memory with bounded statistic transport.

The module deliberately owns the mechanism state (teacher, PCA, prototypes and
audits) so that the training loop can keep the same clean/shifted interface as
APIC v3.  It never consumes acquisition metadata or target-domain statistics.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.dual_shift.apis import channel_stats


APIC_V3_2_VARIANTS = ("v3_2_balanced_style_memory",)


def _stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = channel_stats(features)
    return mean.flatten(1), std.clamp_min(1e-4).flatten(1)


class StyleTeacher(nn.Module):
    """Frozen stem/layer1/layer2 copied after clean warm-up."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.conv1 = copy.deepcopy(backbone.conv1)
        self.bn1 = copy.deepcopy(backbone.bn1)
        self.relu = copy.deepcopy(backbone.relu)
        self.maxpool = copy.deepcopy(backbone.maxpool)
        self.layer1 = copy.deepcopy(backbone.layer1)
        self.layer2 = copy.deepcopy(backbone.layer2)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Parent DualShiftResNet3D.train() must not switch teacher BN layers
        # back to batch-statistics mode.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        return self.layer1(x), self.layer2(self.layer1(x))


class FixedStyleBankAPICV32(nn.Module):
    """APIC v3_2-A implementation.

    Source style descriptors are collected during clean warm-up, projected with
    a frozen PCA basis, and clustered once.  The intervention transports the
    *relative* teacher-statistic difference from a factual prototype to another
    supported source prototype, applied to the current student feature stats.
    """

    def __init__(
        self,
        *,
        layer1_channels: int,
        layer2_channels: int,
        alpha_max: float = 0.25,
        style_dim: int = 16,
        memory_size: int = 4,
        temperature: float = 0.5,
        rms_min: float = 0.001,
        rms_max: float = 0.05,
        min_cluster_count: int = 2,
        max_observed: int = 20000,
    ) -> None:
        super().__init__()
        if memory_size < 2:
            raise ValueError("memory_size must be >= 2")
        self.variant = "v3_2_balanced_style_memory"
        self.layer1_channels = int(layer1_channels)
        self.layer2_channels = int(layer2_channels)
        self.raw_dim = 4 * (self.layer1_channels + self.layer2_channels)
        self.style_dim = int(min(style_dim, self.raw_dim))
        self.memory_size = int(memory_size)
        self.temperature = float(max(temperature, 1e-3))
        self.alpha_max = float(alpha_max)
        self.rms_min = float(rms_min)
        self.rms_max = float(rms_max)
        self.min_cluster_count = int(min_cluster_count)
        self.max_observed = int(max_observed)
        self.enabled = True
        self.current_alpha = 0.0
        self.teacher: Optional[StyleTeacher] = None
        self._observed: list[torch.Tensor] = []
        self._observed_count = 0
        self._finalized = False
        self._state: Dict[str, torch.Tensor] = {}
        self._last_audit: Dict[str, torch.Tensor] = {}

        self.register_buffer("descriptor_center", torch.zeros(self.raw_dim))
        self.register_buffer("descriptor_scale", torch.ones(self.raw_dim))
        self.register_buffer("pca_mean", torch.zeros(self.raw_dim))
        self.register_buffer("pca_components", torch.zeros(self.style_dim, self.raw_dim))
        self.register_buffer("style_prototypes", torch.zeros(self.memory_size, self.style_dim))
        self.register_buffer("style_counts", torch.zeros(self.memory_size))
        self.register_buffer("prototype_radii", torch.ones(self.memory_size))
        self.register_buffer("style_valid", torch.zeros(self.memory_size, dtype=torch.bool))
        self.register_buffer("prototype_mu1", torch.zeros(self.memory_size, self.layer1_channels))
        self.register_buffer("prototype_logstd1", torch.zeros(self.memory_size, self.layer1_channels))
        self.register_buffer("prototype_mu2", torch.zeros(self.memory_size, self.layer2_channels))
        self.register_buffer("prototype_logstd2", torch.zeros(self.memory_size, self.layer2_channels))

    @property
    def mode_name(self) -> str:
        return self.variant

    @property
    def finalized(self) -> bool:
        return bool(self._finalized)

    def set_alpha(self, alpha: float) -> None:
        self.current_alpha = float(max(0.0, min(alpha, self.alpha_max)))

    @staticmethod
    def _descriptor(layer1: torch.Tensor, layer2: torch.Tensor) -> torch.Tensor:
        def one(features: torch.Tensor) -> torch.Tensor:
            mean, std = _stats(features)
            padded = F.pad(features, (1, 1, 1, 1, 1, 1), mode="replicate")
            low = F.avg_pool3d(padded, kernel_size=3, stride=1)
            high = features - low
            low_e = low.float().square().mean(dim=(2, 3, 4)).sqrt().to(mean.dtype)
            high_e = high.float().square().mean(dim=(2, 3, 4)).sqrt().to(mean.dtype)
            return torch.cat([mean, std, low_e, high_e], dim=1)

        return torch.cat([one(layer1), one(layer2)], dim=1)

    def observe_source(self, layer1: torch.Tensor, layer2: torch.Tensor) -> None:
        """Collect clean source descriptors before the bank is frozen."""
        if self._finalized or self._observed_count >= self.max_observed:
            return
        value = self._descriptor(layer1.detach(), layer2.detach()).float().cpu()
        remaining = self.max_observed - self._observed_count
        self._observed.append(value[:remaining])
        self._observed_count += int(min(value.shape[0], remaining))

    @torch.no_grad()
    def freeze_teacher(self, backbone: nn.Module) -> None:
        if self.teacher is None:
            self.teacher = StyleTeacher(backbone).to(next(backbone.parameters()).device)
            self.teacher.eval()
            # Warm-up student descriptors use train-mode BN statistics and are
            # intentionally discarded.  B0 rebuilds the bank with this frozen
            # teacher in eval mode.
            self._observed.clear()
            self._observed_count = 0

    @staticmethod
    def _kmeans(values: torch.Tensor, k: int, iterations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        n = values.shape[0]
        if n == 0:
            raise ValueError("cannot fit style bank without source descriptors")
        if n < k:
            indices = torch.arange(n, device=values.device)
            indices = indices.repeat((k + n - 1) // n)[:k]
        else:
            # Farthest-point deterministic initialization is stable across GPUs.
            indices = [0]
            for _ in range(1, k):
                chosen = values[torch.tensor(indices, device=values.device)]
                distance = (values[:, None] - chosen[None]).square().sum(dim=2).min(dim=1).values
                indices.append(int(distance.argmax().item()))
            indices = torch.tensor(indices, device=values.device)
        centers = values[indices].clone()
        assignment = torch.zeros(n, dtype=torch.long, device=values.device)
        for _ in range(iterations):
            assignment = (values[:, None] - centers[None]).square().sum(dim=2).argmin(dim=1)
            for slot in range(k):
                mask = assignment == slot
                if bool(mask.any()):
                    centers[slot] = values[mask].mean(dim=0)
        return centers, assignment

    @torch.no_grad()
    def finalize_style_bank(self) -> None:
        if self._finalized:
            return
        if not self._observed:
            self._finalized = False
            return
        raw = torch.cat(self._observed, dim=0).float()
        center = raw.median(dim=0).values
        scale = (raw - center).abs().median(dim=0).values.clamp_min(1e-4)
        normalized = (raw - center) / scale
        self.descriptor_center.copy_(center.to(self.descriptor_center.device))
        self.descriptor_scale.copy_(scale.to(self.descriptor_scale.device))
        self.pca_mean.copy_(normalized.mean(dim=0).to(self.pca_mean.device))
        centered = normalized - normalized.mean(dim=0, keepdim=True)
        q = min(self.style_dim, max(1, centered.shape[0] - 1), centered.shape[1])
        _, _, vector = torch.pca_lowrank(centered, q=q, center=False)
        components = vector[:, : self.style_dim].transpose(0, 1).contiguous()
        if components.shape[0] < self.style_dim:
            components = F.pad(components, (0, 0, 0, self.style_dim - components.shape[0]))
        self.pca_components.copy_(components.to(self.pca_components.device))
        projected = (centered @ components.t()).contiguous()
        prototypes, assignments = self._kmeans(projected, self.memory_size)
        # Repair empty/under-occupied slots before exposing the bank.  This is
        # a deterministic occupancy constraint, not a target-domain heuristic.
        if projected.shape[0] >= self.memory_size * self.min_cluster_count:
            for slot in range(self.memory_size):
                while int((assignments == slot).sum().item()) < self.min_cluster_count:
                    counts = torch.stack([(assignments == item).sum() for item in range(self.memory_size)])
                    donor = int(counts.argmax().item())
                    candidates = torch.nonzero(assignments == donor, as_tuple=False).flatten()
                    if candidates.numel() <= self.min_cluster_count:
                        break
                    donor_distance = (projected[candidates] - prototypes[donor]).square().sum(dim=1)
                    moved = candidates[int(donor_distance.argmax().item())]
                    assignments[moved] = slot
                    prototypes[slot] = projected[assignments == slot].mean(dim=0)
                    prototypes[donor] = projected[assignments == donor].mean(dim=0)
        self.style_prototypes.copy_(prototypes.to(self.style_prototypes.device))
        for slot in range(self.memory_size):
            mask = assignments == slot
            count = int(mask.sum().item())
            if count < self.min_cluster_count and raw.shape[0] >= self.memory_size:
                continue
            self.style_valid[slot] = True
            self.style_counts[slot] = float(count)
            distances = (projected[mask] - prototypes[slot]).square().sum(dim=1).sqrt()
            self.prototype_radii[slot] = float(distances.quantile(0.95).clamp_min(1e-3).item())
        # Store raw-stat targets by recovering the descriptor fields from each cluster.
        raw_dim1 = 4 * self.layer1_channels
        for slot in range(self.memory_size):
            mask = assignments == slot
            if not bool(mask.any()):
                continue
            mean = raw[mask].mean(dim=0)
            m1, s1, low1, high1 = torch.split(mean[:raw_dim1], self.layer1_channels, dim=0)
            offset = raw_dim1
            m2, s2, low2, high2 = torch.split(mean[offset:], self.layer2_channels, dim=0)
            del low1, high1, low2, high2
            self.prototype_mu1[slot].copy_(m1.to(self.prototype_mu1.device))
            self.prototype_logstd1[slot].copy_(s1.clamp_min(1e-4).log().to(self.prototype_logstd1.device))
            self.prototype_mu2[slot].copy_(m2.to(self.prototype_mu2.device))
            self.prototype_logstd2[slot].copy_(s2.clamp_min(1e-4).log().to(self.prototype_logstd2.device))
        self._finalized = bool(self.style_valid.sum().item() >= 2)
        self._observed.clear()

    @torch.no_grad()
    def _teacher_stats(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.teacher is None:
            raise RuntimeError("APIC v3_2 style teacher has not been frozen")
        return self.teacher(image)

    def _project(self, raw: torch.Tensor) -> torch.Tensor:
        normalized = (raw - self.descriptor_center.to(raw.device)) / self.descriptor_scale.to(raw.device)
        centered = normalized - self.pca_mean.to(raw.device)
        return centered @ self.pca_components.to(raw.device).t()

    def prepare_style_condition(
        self,
        image: torch.Tensor,
        student_layer1: torch.Tensor,
        student_layer2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._finalized:
            self.observe_source(student_layer1, student_layer2)
            self.finalize_style_bank()
        if not self._finalized:
            valid = image.new_zeros((image.shape[0],), dtype=torch.bool)
            self._state = {"gate": image.new_zeros((image.shape[0],))}
            return image.new_zeros((image.shape[0], 1)), valid
        with torch.no_grad():
            if self.teacher is None:
                # Small synthetic/unit tests may call the module directly
                # before a backbone is attached; the model path always freezes
                # the teacher at the clean -> APIC phase boundary.
                teacher_l1, teacher_l2 = student_layer1.detach(), student_layer2.detach()
            else:
                teacher_l1, teacher_l2 = self._teacher_stats(image)
            raw = self._descriptor(teacher_l1, teacher_l2)
            style = self._project(raw)
        valid_slots = torch.nonzero(self.style_valid, as_tuple=False).flatten()
        distances = (style[:, None] - self.style_prototypes.to(style.device)[valid_slots][None]).square().sum(dim=2)
        nearest = distances.argmin(dim=1)
        src = valid_slots[nearest]
        # Pick the nearest different supported prototype.  With K=4 this is a
        # deterministic source-only alternative and never uses target metadata.
        sorted_slots = distances.argsort(dim=1)
        target = valid_slots[sorted_slots[:, 1]]
        src_dist = distances.gather(1, nearest[:, None]).sqrt().squeeze(1)
        support_radius = self.prototype_radii.to(style.device)[src]
        valid = src_dist <= support_radius
        separation = (self.style_prototypes[src] - self.style_prototypes[target]).norm(dim=1)
        valid = valid & (separation > 1e-4)
        confidence = torch.softmax(-distances / self.temperature, dim=1).max(dim=1).values
        gate = valid.to(style.dtype) * (0.20 + 0.60 * confidence)
        self._state = {
            "src": src,
            "target": target,
            "gate": gate,
            "style": style,
            "confidence": confidence,
            "entropy": (-(torch.softmax(-distances / self.temperature, dim=1).clamp_min(1e-8).log() * torch.softmax(-distances / self.temperature, dim=1)).sum(dim=1)),
            "delta_mu1": self.prototype_mu1[target] - self.prototype_mu1[src],
            "delta_logstd1": self.prototype_logstd1[target] - self.prototype_logstd1[src],
            "delta_mu2": self.prototype_mu2[target] - self.prototype_mu2[src],
            "delta_logstd2": self.prototype_logstd2[target] - self.prototype_logstd2[src],
        }
        self._state["gate"] = self._state["gate"].to(image.device)
        return style, valid

    def _apply(self, features: torch.Tensor, layer: int) -> torch.Tensor:
        gate = self._state.get("gate")
        if gate is None or self.current_alpha <= 0:
            return features
        mean, std = _stats(features)
        if layer == 1:
            delta_mu = self._state["delta_mu1"].to(features.device)
            delta_log = self._state["delta_logstd1"].to(features.device)
        else:
            delta_mu = self._state["delta_mu2"].to(features.device)
            delta_log = self._state["delta_logstd2"].to(features.device)
        target_mu = mean + delta_mu
        target_std = std * delta_log.exp().clamp(0.5, 2.0)
        shape = (features.shape[0], features.shape[1]) + (1,) * (features.ndim - 2)
        transformed = target_std.reshape(shape) * (features - mean.reshape(shape)) / std.reshape(shape) + target_mu.reshape(shape)
        delta = transformed - features
        base_rms = features.flatten(1).square().mean(dim=1).sqrt().clamp_min(1e-6)
        rms = delta.flatten(1).square().mean(dim=1).sqrt() / base_rms
        batch_shape = (features.shape[0],) + (1,) * (features.ndim - 1)
        clip = (self.rms_max / rms.clamp_min(1e-6)).clamp(max=1.0)
        delta = delta * clip.reshape(batch_shape)
        keep = (gate * self.current_alpha).reshape(batch_shape).to(features.dtype)
        shifted = features + keep * delta
        realized = (shifted - features).flatten(1).square().mean(dim=1).sqrt() / base_rms
        target_error = (delta.flatten(1).square().mean(dim=1).sqrt() / base_rms).detach()
        self._last_audit[f"layer{layer}"] = {"realized": realized, "target_error": target_error}
        return shifted

    def make_shift_fns(self, condition=None, *, valid_mask=None):
        del condition, valid_mask
        self._last_audit = {}
        return (lambda features: self._apply(features, 1), lambda features: self._apply(features, 2))

    def audit_tensors(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        n = reference.shape[0]
        realized = [item["realized"] for item in self._last_audit.values()]
        target = [item["target_error"] for item in self._last_audit.values()]
        if realized:
            realized_ps = torch.stack(realized).mean(dim=0)
            target_ps = torch.stack(target).mean(dim=0)
        else:
            realized_ps = reference.new_zeros((n,))
            target_ps = reference.new_zeros((n,))
        gate = self._state.get("gate", reference.new_zeros((n,))).to(reference.device)
        return {
            "strength": realized_ps.mean(),
            "feature_strength": realized_ps.mean(),
            "coefficient_l2": realized_ps.mean(),
            "coefficient_l2_per_sample": realized_ps,
            "realized_per_sample": realized_ps,
            "style_target_error_per_sample": target_ps,
            "style_confidence": self._state.get("confidence", reference.new_zeros((n,))).mean(),
            "style_entropy": self._state.get("entropy", reference.new_zeros((n,))).mean(),
            "condition_gate": gate.mean(),
            "style_delta": target_ps.mean(),
            "effective_slots": reference.new_tensor(float(self.style_valid.sum().item())),
            "max_slot_share": reference.new_tensor(float(self.style_counts.max().item() / self.style_counts.sum().clamp_min(1.0).item())),
        }


def build_apis_v3_2(variant: str, *, layer1_channels: int, layer2_channels: int, alpha_max: float, style_dim: int = 16, memory_size: int = 4, temperature: float = 0.5, rms_min: float = 0.001, rms_max: float = 0.05, **_) -> nn.Module:
    if variant not in APIC_V3_2_VARIANTS:
        raise ValueError(f"Unknown APIC v3_2 variant {variant!r}")
    return FixedStyleBankAPICV32(
        layer1_channels=layer1_channels,
        layer2_channels=layer2_channels,
        alpha_max=alpha_max,
        style_dim=style_dim,
        memory_size=memory_size,
        temperature=temperature,
        rms_min=rms_min,
        rms_max=rms_max,
    )
