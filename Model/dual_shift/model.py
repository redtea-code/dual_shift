"""DualShiftResNet3D: CDT + APIS joint model."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from Model.dual_shift.acquisition_encoder import AcquisitionDescriptorEncoder
from Model.dual_shift.apis import APISModule
from Model.dual_shift.apis_v3 import APIS_V3_VARIANTS, build_apis_v3
from Model.dual_shift.apis_v3_2 import APIC_V3_2_VARIANTS, build_apis_v3_2
from Model.dual_shift.backbone import DualShiftBackbone
from Model.dual_shift.demographic_encoder import (
    DemographicEncoder,
    LowRankDemographicFusion,
)
from Model.dual_shift.demographic_transport import ContinuousDemographicTransport
from Model.dual_shift.mixstyle import MixStyle
from Model.dual_shift.outputs import DualShiftOutput
from Model.dual_shift.protocol_prototypes import ProtocolPrototypeBank


class BoundedAcquisitionFiLM(nn.Module):
    """Source-fitted acquisition conditioning with bounded feature modulation."""

    def __init__(self, acquisition_dim: int, feature_dim: int, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(max(0.0, alpha))
        self.controller = nn.Linear(int(acquisition_dim), int(feature_dim) * 2)
        # Start exactly at the identity so the conditioning path cannot create a
        # chance performance jump before it receives gradient evidence.
        nn.init.zeros_(self.controller.weight)
        nn.init.zeros_(self.controller.bias)

    def forward(
        self, features: torch.Tensor, acquisition_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_raw, beta_raw = self.controller(acquisition_embedding).chunk(2, dim=1)
        gamma = 1.0 + self.alpha * torch.tanh(gamma_raw)
        # Keep additive modulation on the same per-sample scale as the feature map.
        rms = features.flatten(1).square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        beta = self.alpha * torch.tanh(beta_raw) * rms
        shape = (features.shape[0], features.shape[1]) + (1,) * (features.ndim - 2)
        shifted = features * gamma.reshape(shape) + beta.reshape(shape)
        relative_rms = (
            (shifted - features).flatten(1).square().mean(dim=1).sqrt()
            / rms.flatten().clamp_min(1e-6)
        )
        return shifted, relative_rms


class DualShiftResNet3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 32,
        layers: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.1,
        fusion_rank: int = 16,
        acquisition_out_dim: int = 32,
        alpha_max: float = 0.25,
        apis_basis_count: int = 4,
        apis_rank: int = 8,
        use_apis: bool = True,
        use_cdt: bool = True,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        prototype_min_subjects: int = 8,
        use_scan_film: bool = False,
        scan_film_alpha: float = 0.1,
        use_demographics: bool = True,
        apis_variant: str = "v2_residual",
        apis_style_dim: int = 16,
        apis_memory_size: int = 8,
        apis_memory_beta: float = 0.95,
        apis_style_temperature: float = 0.5,
        apis_rms_min: float = 0.001,
        apis_rms_max: float = 0.05,
        apis_delta_min: float = 0.02,
        apis_delta_max: float = 0.50,
        apis_g_min: float = 0.20,
        apis_g_max: float = 0.80,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_apis = bool(use_apis) and not bool(use_mixstyle)
        self.use_cdt = bool(use_cdt)
        self.use_mixstyle = bool(use_mixstyle)
        self.use_scan_film = bool(use_scan_film)
        self.use_demographics = bool(use_demographics)
        self.apis_variant = str(apis_variant)
        self.backbone = DualShiftBackbone(layers=layers, base_channels=base_channels)
        self.demographic_encoder = DemographicEncoder()
        self.fusion = LowRankDemographicFusion(
            feature_dim=self.backbone.out_channels,
            demographic_dim=self.demographic_encoder.out_dim,
            rank=fusion_rank,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(self.backbone.out_channels, self.num_classes)
        self.acquisition_encoder = AcquisitionDescriptorEncoder(out_dim=acquisition_out_dim)
        self.scan_film = BoundedAcquisitionFiLM(
            acquisition_out_dim,
            self.backbone.out_channels,
            alpha=scan_film_alpha,
        )
        apis_kwargs = {
            "layer1_channels": self.backbone.layer1_channels,
            "layer2_channels": self.backbone.layer2_channels,
            "acquisition_dim": acquisition_out_dim,
            "alpha_max": alpha_max,
            "basis_count": apis_basis_count,
            "rank": apis_rank,
        }
        if self.apis_variant == "v2_residual":
            self.apis = APISModule(**apis_kwargs)
        elif self.apis_variant in APIC_V3_2_VARIANTS:
            self.apis = build_apis_v3_2(
                self.apis_variant,
                style_dim=apis_style_dim,
                memory_size=apis_memory_size,
                temperature=apis_style_temperature,
                rms_min=apis_rms_min,
                rms_max=apis_rms_max,
                delta_min=apis_delta_min,
                delta_max=apis_delta_max,
                g_min=apis_g_min,
                g_max=apis_g_max,
                **apis_kwargs,
            )
        elif self.apis_variant in APIS_V3_VARIANTS:
            self.apis = build_apis_v3(
                self.apis_variant,
                layer3_channels=self.backbone.layer3_channels,
                style_dim=apis_style_dim,
                memory_size=apis_memory_size,
                memory_beta=apis_memory_beta,
                temperature=apis_style_temperature,
                **apis_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown apis_variant {self.apis_variant!r}; expected "
                f"'v2_residual', one of {APIS_V3_VARIANTS}, or {APIC_V3_2_VARIANTS}"
            )
        self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.prototype_bank = ProtocolPrototypeBank(min_subjects=prototype_min_subjects)
        self.cdt = ContinuousDemographicTransport()
        self._apis_active = False
        self._building_style_bank = False
        # Negative-control flag: shuffle acquisition embeddings within the batch.
        self.shuffle_acquisition = False

    def set_phase(
        self,
        *,
        apis_active: bool,
        cdt_enabled: bool,
        alpha: float,
        prepare_style_bank: bool = False,
    ) -> None:
        self._apis_active = bool(apis_active) and self.use_apis
        self._building_style_bank = bool(prepare_style_bank)
        self.apis.enabled = self._apis_active
        self.apis.set_alpha(alpha if self._apis_active else 0.0)
        if self.apis_variant in APIC_V3_2_VARIANTS and (
            self._apis_active or self._building_style_bank
        ):
            # The teacher is copied exactly once, at the clean -> APIC boundary.
            self.apis.freeze_teacher(self.backbone)
        if self.apis_variant in APIC_V3_2_VARIANTS and self._apis_active:
            self.apis.finalize_style_bank(strict=True)
            if not self.apis.finalized:
                raise RuntimeError(
                    "APIC v3_2 mechanism_calibration did not support two style slots"
                )
        self.cdt.enabled = bool(cdt_enabled) and self.use_cdt
        # MixStyle only during training forward; keep modules activated flag aligned.
        self.mixstyle1.set_activated(self.use_mixstyle)
        self.mixstyle2.set_activated(self.use_mixstyle)

    def _heads(
        self,
        layer4: torch.Tensor,
        covariates: torch.Tensor,
        *,
        age_missing: Optional[torch.Tensor] = None,
        sex_missing: Optional[torch.Tensor] = None,
        education_missing: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool(layer4).flatten(1)
        if not self.use_demographics:
            # Keep the diagnostic path strictly image-only. Covariates remain in
            # the batch schema for compatible loaders, but cannot affect logits.
            logits = self.classifier(self.dropout(pooled))
            empty_demo = pooled.new_zeros((pooled.shape[0], 0))
            return logits, pooled, empty_demo
        demo = self.demographic_encoder(
            covariates,
            age_missing=age_missing,
            sex_missing=sex_missing,
            education_missing=education_missing,
        )
        fused = self.fusion(pooled, demo)
        logits = self.classifier(self.dropout(fused))
        return logits, fused, demo

    def begin_epoch(self) -> None:
        self.prototype_bank.begin_epoch()

    def end_epoch_update(self) -> None:
        self.prototype_bank.end_epoch_update()

    def forward(
        self,
        image: torch.Tensor,
        covariates: torch.Tensor,
        *,
        acquisitions: Optional[Sequence[Mapping[str, Any]]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        subject_ids: Optional[Sequence[str]] = None,
        update_prototypes: bool = False,
        force_clean_only: bool = False,
        age_missing: Optional[torch.Tensor] = None,
        sex_missing: Optional[torch.Tensor] = None,
        education_missing: Optional[torch.Tensor] = None,
    ) -> DualShiftOutput:
        head_kwargs = {
            "age_missing": age_missing,
            "sex_missing": sex_missing,
            "education_missing": education_missing,
        }
        if self.training and self.use_mixstyle and not force_clean_only:
            mix_feats = self.backbone(
                image,
                shift_layer1=self.mixstyle1,
                shift_layer2=self.mixstyle2,
            )
            logits, embedding, demo = self._heads(
                mix_feats["layer4"], covariates, **head_kwargs
            )
            return DualShiftOutput(
                clean_logits=logits,
                clean_embedding=embedding,
                demographic_embedding=demo,
                shift_strength=image.new_zeros(()),
                extras={"mixstyle": True},
            )

        paired_v3_2 = bool(
            self.apis_variant in APIC_V3_2_VARIANTS
            and self.training
            and self._apis_active
            and not force_clean_only
        )
        clean_feats = self.backbone(image, capture_bn_moments=paired_v3_2)
        acq_emb = None
        scan_film_strength = image.new_zeros(())
        if (
            self.use_scan_film
            and acquisitions is not None
            and self.acquisition_encoder.is_fitted_
        ):
            acq_emb = self.acquisition_encoder(acquisitions)
            clean_layer4, film_per_sample = self.scan_film(clean_feats["layer4"], acq_emb)
            scan_film_strength = film_per_sample.mean()
        else:
            clean_layer4 = clean_feats["layer4"]
        head_rng = None
        if paired_v3_2 and self.training and self.dropout.p > 0:
            head_rng = {
                "cpu_before": torch.random.get_rng_state(),
                "cuda_before": torch.cuda.get_rng_state(image.device) if image.is_cuda else None,
            }
        clean_logits, clean_embedding, demo = self._heads(
            clean_layer4, covariates, **head_kwargs
        )
        if head_rng is not None:
            head_rng["cpu_after"] = torch.random.get_rng_state()
            head_rng["cuda_after"] = (
                torch.cuda.get_rng_state(image.device) if image.is_cuda else None
            )

        # Collect source-only descriptors during clean warm-up.  The collection
        # is detached and finalized when the APIC phase begins.
        if (
            self.apis_variant in APIC_V3_2_VARIANTS
            and (self.training or self._building_style_bank)
            and not self.apis.finalized
        ):
            if self.apis.teacher is not None:
                with torch.no_grad():
                    teacher_l1, teacher_l2 = self.apis.teacher(image)
                self.apis.observe_source(teacher_l1, teacher_l2, subject_ids=subject_ids)
            else:
                self.apis.observe_source(
                    clean_feats["layer1"], clean_feats["layer2"], subject_ids=subject_ids
                )

        run_style_memory = (
            self.apis_variant in {"v3_style_memory", *APIC_V3_2_VARIANTS}
            and self.training
            and self._apis_active
            and not force_clean_only
        )
        if self.apis_variant in {"v3_style_memory", *APIC_V3_2_VARIANTS}:
            if not run_style_memory:
                return DualShiftOutput(
                    clean_logits=clean_logits,
                    clean_embedding=clean_embedding,
                    demographic_embedding=demo,
                    shift_strength=image.new_zeros(()),
                    extras={
                        "apis_mode": "v3_style_memory",
                        "scan_film_strength": scan_film_strength,
                    },
                )
            if self.apis_variant == "v3_style_memory":
                condition, valid_mask = self.apis.prepare_style_condition(
                    clean_feats["layer1"],
                    clean_feats["layer2"],
                    clean_feats["layer3"],
                    update_memory=True,
                )
            else:
                condition, valid_mask = self.apis.prepare_style_condition(
                    image,
                    clean_feats["layer1"],
                    clean_feats["layer2"],
                    sample_ids=sample_ids,
                )
            shift1, shift2 = self.apis.make_shift_fns(
                condition, valid_mask=valid_mask
            )
            shifted_feats = self.backbone(
                image,
                shift_layer1=shift1,
                shift_layer2=shift2,
                update_bn_stats=self.apis_variant not in APIC_V3_2_VARIANTS,
                paired_bn_moments=clean_feats.get("bn_moments"),
            )
            shifted_layer4 = shifted_feats["layer4"]
            if self.use_scan_film and acq_emb is not None:
                shifted_layer4, _ = self.scan_film(shifted_layer4, acq_emb)
            if head_rng is not None:
                torch.random.set_rng_state(head_rng["cpu_before"])
                if image.is_cuda:
                    torch.cuda.set_rng_state(head_rng["cuda_before"], image.device)
            try:
                shifted_logits, shifted_embedding, _ = self._heads(
                    shifted_layer4, covariates, **head_kwargs
                )
            finally:
                if head_rng is not None:
                    torch.random.set_rng_state(head_rng["cpu_after"])
                    if image.is_cuda:
                        torch.cuda.set_rng_state(head_rng["cuda_after"], image.device)
            if self.apis_variant in APIC_V3_2_VARIANTS:
                # Unsupported rows are a strict final-output fallback.  This is
                # stronger than relying on a zero residual inside a mixed batch.
                mask = valid_mask.reshape(-1, 1)
                shifted_logits = torch.where(mask, shifted_logits, clean_logits)
                shifted_embedding = torch.where(mask, shifted_embedding, clean_embedding)
            audit = self.apis.audit_tensors(image)
            valid_frac = valid_mask.float().mean()
            return DualShiftOutput(
                clean_logits=clean_logits,
                shifted_logits=shifted_logits,
                clean_embedding=clean_embedding,
                shifted_embedding=shifted_embedding,
                demographic_embedding=demo,
                shift_strength=audit["strength"],
                extras={
                    "apis_mode": self.apis_variant,
                    "apis_coefficient_l2": audit["coefficient_l2"],
                    "apis_coefficient_l2_per_sample": audit[
                        "coefficient_l2_per_sample"
                    ],
                    "valid_intervention_mask": valid_mask,
                    "valid_intervention_frac": valid_frac,
                    "scan_film_strength": scan_film_strength,
                    "apis_feature_strength": audit["feature_strength"],
                    "style_confidence": audit["style_confidence"],
                    "style_entropy": audit["style_entropy"],
                    "style_delta": audit["style_delta"],
                    "prototype_relative_separation": audit[
                        "prototype_relative_separation"
                    ],
                    "condition_gate": audit["condition_gate"],
                    "apis_style_target_error_per_sample": audit.get(
                        "style_target_error_per_sample"
                    ),
                    "apis_rms_per_sample": audit.get("realized_per_sample"),
                    "apis_rms_layer1_per_sample": audit.get("rms_layer1_per_sample"),
                    "apis_rms_layer2_per_sample": audit.get("rms_layer2_per_sample"),
                    "apis_effective_slots": audit.get("effective_slots"),
                    "apis_max_slot_share": audit.get("max_slot_share"),
                },
            )

        run_apis = (
            self.training
            and self._apis_active
            and not force_clean_only
            and acquisitions is not None
            and self.acquisition_encoder.is_fitted_
        )
        if not run_apis:
            return DualShiftOutput(
                clean_logits=clean_logits,
                clean_embedding=clean_embedding,
                demographic_embedding=demo,
                shift_strength=image.new_zeros(()),
                extras={"scan_film_strength": scan_film_strength},
            )

        if acq_emb is None:
            acq_emb = self.acquisition_encoder(acquisitions)
        domain_keys = self.acquisition_encoder.domain_keys(acquisitions)
        # Collect into pending only; never mutate the frozen sampling bank mid-epoch.
        # Prototype updates always use factual (unshuffled) descriptors.
        if update_prototypes and subject_ids is not None:
            self.prototype_bank.collect_epoch_statistics(
                domain_keys,
                subject_ids,
                acq_emb.detach(),
                clean_feats["layer1"].detach(),
                clean_feats["layer2"].detach(),
            )

        target_acq_emb, selected = self.prototype_bank.sample_target_embeddings(
            domain_keys,
            device=image.device,
            embedding_dim=acq_emb.shape[-1],
        )
        valid_mask = selected >= 0
        valid_frac = valid_mask.float().mean()
        if target_acq_emb is None or not bool(valid_mask.any().item()):
            return DualShiftOutput(
                clean_logits=clean_logits,
                clean_embedding=clean_embedding,
                demographic_embedding=demo,
                shift_strength=image.new_zeros(()),
                selected_protocol_index=selected,
                extras={
                    "apis_skipped": "no_valid_observed_protocol",
                    "valid_intervention_mask": valid_mask,
                    "valid_intervention_frac": valid_frac,
                },
            )
        # Invalid rows keep a placeholder target; residual mask zeros the
        # intervention at layer1/layer2 so BN/features stay on the clean path.
        safe_target = target_acq_emb.detach().clone()
        if not bool(valid_mask.all().item()):
            safe_target[~valid_mask] = acq_emb.detach()[~valid_mask]
        # Negative control: shuffle only the factual descriptor used to build
        # the APIS condition; prototype bank and domain keys stay factual.
        factual_for_condition = acq_emb
        if bool(getattr(self, "shuffle_acquisition", False)) and acq_emb.shape[0] > 1:
            perm = torch.randperm(acq_emb.shape[0], device=acq_emb.device)
            factual_for_condition = acq_emb[perm]
        if self.apis_variant == "v2_residual":
            condition = self.apis.protocol_condition(
                factual_for_condition, safe_target
            )
        else:
            condition = self.apis.protocol_condition(
                factual_for_condition,
                safe_target,
                layer1_features=clean_feats["layer1"],
            )
        shifted_image = image
        if hasattr(self.apis, "make_shifted_image"):
            shifted_image = self.apis.make_shifted_image(
                image,
                condition,
                valid_mask=valid_mask,
            )
        shift1, shift2 = self.apis.make_shift_fns(condition, valid_mask=valid_mask)
        shifted_feats = self.backbone(
            shifted_image,
            shift_layer1=shift1,
            shift_layer2=shift2,
        )
        shifted_layer4 = shifted_feats["layer4"]
        if self.use_scan_film:
            shifted_layer4, _ = self.scan_film(shifted_layer4, acq_emb)
        shifted_logits, shifted_embedding, _ = self._heads(
            shifted_layer4, covariates, **head_kwargs
        )
        apis_audit = self.apis.audit_tensors(image)
        return DualShiftOutput(
            clean_logits=clean_logits,
            shifted_logits=shifted_logits,
            clean_embedding=clean_embedding,
            shifted_embedding=shifted_embedding,
            demographic_embedding=demo,
            shift_strength=apis_audit["strength"],
            selected_protocol_index=selected,
            extras={
                "apis_mode": getattr(
                    self.apis, "mode_name", "observed_protocol_residual"
                ),
                "apis_coefficient_l2": apis_audit["coefficient_l2"],
                "apis_coefficient_l2_per_sample": apis_audit.get(
                    "coefficient_l2_per_sample"
                ),
                "protocol_condition_norm": condition.norm(dim=1).mean(),
                "valid_intervention_mask": valid_mask,
                "valid_intervention_frac": valid_frac,
                "scan_film_strength": scan_film_strength,
                "apis_feature_strength": apis_audit.get("feature_strength"),
                "apis_intensity_strength": apis_audit.get("intensity_strength"),
                "protocol_distance": apis_audit.get("protocol_distance"),
                "condition_gate": apis_audit.get("condition_gate"),
            },
        )

    def fit_acquisition_encoder(self, acquisitions: Sequence[Mapping[str, Any]]) -> None:
        self.acquisition_encoder.fit(acquisitions)
        # Move newly created embeddings to model device.
        device = next(self.backbone.parameters()).device
        self.acquisition_encoder.to(device)
