"""DualShiftResNet3D: CDT + APIS joint model."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from Model.dual_shift.acquisition_encoder import AcquisitionDescriptorEncoder
from Model.dual_shift.apis import APISModule
from Model.dual_shift.backbone import DualShiftBackbone
from Model.dual_shift.demographic_encoder import (
    DemographicEncoder,
    LowRankDemographicFusion,
)
from Model.dual_shift.demographic_transport import ContinuousDemographicTransport
from Model.dual_shift.mixstyle import MixStyle
from Model.dual_shift.outputs import DualShiftOutput
from Model.dual_shift.protocol_prototypes import ProtocolPrototypeBank


class DualShiftResNet3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 32,
        layers: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.1,
        fusion_rank: int = 16,
        acquisition_out_dim: int = 32,
        alpha_max: float = 0.5,
        use_apis: bool = True,
        use_cdt: bool = True,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        prototype_min_subjects: int = 8,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_apis = bool(use_apis) and not bool(use_mixstyle)
        self.use_cdt = bool(use_cdt)
        self.use_mixstyle = bool(use_mixstyle)
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
        self.apis = APISModule(alpha_max=alpha_max)
        self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.prototype_bank = ProtocolPrototypeBank(min_subjects=prototype_min_subjects)
        self.cdt = ContinuousDemographicTransport()
        self._apis_active = False

    def set_phase(self, *, apis_active: bool, cdt_enabled: bool, alpha: float) -> None:
        self._apis_active = bool(apis_active) and self.use_apis
        self.apis.enabled = self._apis_active
        self.apis.set_alpha(alpha if self._apis_active else 0.0)
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
        del sample_ids  # reserved for trainer-side CDT bookkeeping
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

        clean_feats = self.backbone(image)
        clean_logits, clean_embedding, demo = self._heads(
            clean_feats["layer4"], covariates, **head_kwargs
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
            )

        acq_emb = self.acquisition_encoder(acquisitions)
        domain_keys = self.acquisition_encoder.domain_keys(acquisitions)
        # Collect into pending only; never mutate the frozen sampling bank mid-epoch.
        if update_prototypes and subject_ids is not None:
            self.prototype_bank.collect_epoch_statistics(
                domain_keys,
                subject_ids,
                acq_emb.detach(),
                clean_feats["layer1"].detach(),
                clean_feats["layer2"].detach(),
            )

        mean1, std1, mean2, std2, selected = self.prototype_bank.sample_targets(
            domain_keys, device=image.device
        )
        if mean1 is None:
            return DualShiftOutput(
                clean_logits=clean_logits,
                clean_embedding=clean_embedding,
                demographic_embedding=demo,
                shift_strength=image.new_zeros(()),
                selected_protocol_index=selected,
                extras={"apis_skipped": "no_valid_prototype"},
            )
        shift1, shift2 = self.apis.make_shift_fns(
            target_mean1=mean1,
            target_std1=std1,
            target_mean2=mean2,
            target_std2=std2,
        )
        shifted_feats = self.backbone(image, shift_layer1=shift1, shift_layer2=shift2)
        shifted_logits, shifted_embedding, _ = self._heads(
            shifted_feats["layer4"], covariates, **head_kwargs
        )
        return DualShiftOutput(
            clean_logits=clean_logits,
            shifted_logits=shifted_logits,
            clean_embedding=clean_embedding,
            shifted_embedding=shifted_embedding,
            demographic_embedding=demo,
            shift_strength=image.new_tensor(self.apis.current_alpha),
            selected_protocol_index=selected,
        )

    def fit_acquisition_encoder(self, acquisitions: Sequence[Mapping[str, Any]]) -> None:
        self.acquisition_encoder.fit(acquisitions)
        # Move newly created embeddings to model device.
        device = next(self.backbone.parameters()).device
        self.acquisition_encoder.to(device)
