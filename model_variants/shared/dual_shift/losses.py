"""Dual-shift joint losses."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _weighted_mean(
    per_sample: torch.Tensor, sample_weights: Optional[torch.Tensor]
) -> torch.Tensor:
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.to(per_sample.device).reshape(-1)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor],
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    return _weighted_mean(per_sample, sample_weights)


def js_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample JS, aggregated with optional CDT weights."""
    log_p = F.log_softmax(logits_a, dim=1)
    log_q = F.log_softmax(logits_b, dim=1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-8).log()
    kl_pm = (p * (log_p - log_m)).sum(dim=1)
    kl_qm = (q * (log_q - log_m)).sum(dim=1)
    per_sample = 0.5 * (kl_pm + kl_qm)
    return _weighted_mean(per_sample, sample_weights)


def feature_cosine_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    per_sample = 1.0 - (a * b).sum(dim=1)
    return _weighted_mean(per_sample, sample_weights)


def compute_dual_shift_loss(
    *,
    clean_logits: torch.Tensor,
    labels: torch.Tensor,
    shifted_logits: Optional[torch.Tensor] = None,
    clean_embedding: Optional[torch.Tensor] = None,
    shifted_embedding: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    kl_value: float = 0.0,
    shift_strength: float = 0.0,
    alpha_max: float = 0.25,
    lambda_shift: float = 1.0,
    lambda_js: float = 0.25,
    lambda_feat: float = 0.1,
    lambda_kl: float = 0.05,
    enable_apis: bool = True,
    penalize_alpha_overflow: bool = False,
    intervention_penalty: Optional[torch.Tensor] = None,
    lambda_intervention: float = 0.001,
    intervention_mask: Optional[torch.Tensor] = None,
    v3_2_mode: bool = False,
    rms_per_sample: Optional[torch.Tensor] = None,
    style_target_error_per_sample: Optional[torch.Tensor] = None,
    rms_min: float = 0.001,
    rms_max: float = 0.05,
    lambda_style: float = 0.1,
    lambda_rms: float = 0.1,
) -> Dict[str, torch.Tensor]:
    clean_ce = _weighted_ce(clean_logits, labels, sample_weights, class_weights)
    losses = {
        "clean_ce": clean_ce,
        "shift_ce": clean_logits.new_zeros(()),
        "js": clean_logits.new_zeros(()),
        "feat": clean_logits.new_zeros(()),
        "kl": clean_logits.new_tensor(float(kl_value)),
        "strength": clean_logits.new_zeros(()),
        "intervention": clean_logits.new_zeros(()),
        "style": clean_logits.new_zeros(()),
        "rms_band": clean_logits.new_zeros(()),
    }
    total = clean_ce + lambda_kl * losses["kl"]
    shift_weights = sample_weights
    apis_active = bool(enable_apis and shifted_logits is not None)
    if apis_active and intervention_mask is not None:
        mask = intervention_mask.to(device=clean_logits.device, dtype=torch.float32).reshape(
            -1
        )
        if sample_weights is None:
            shift_weights = mask
        else:
            shift_weights = sample_weights.to(clean_logits.device).reshape(-1) * mask
        apis_active = bool(float(mask.sum().detach()) > 0.0)
    if apis_active and shifted_logits is not None:
        clean_per = F.cross_entropy(clean_logits, labels, weight=class_weights, reduction="none")
        shift_per = F.cross_entropy(shifted_logits, labels, weight=class_weights, reduction="none")
        mask = intervention_mask.to(clean_logits.device, dtype=clean_per.dtype) if intervention_mask is not None else torch.ones_like(clean_per)
        base_weights = torch.ones_like(clean_per) if sample_weights is None else sample_weights.to(clean_logits.device).reshape(-1)
        denom = base_weights.sum().clamp_min(1e-8)
        if v3_2_mode:
            # Every sample contributes once; supported rows use the mean of the
            # factual and shifted CE, while unsupported rows are exact clean CE.
            shift_ce = (((1.0 - mask) * clean_per + mask * 0.5 * (clean_per + shift_per)) * base_weights).sum() / denom
            js_per = 0.5 * (F.kl_div(F.log_softmax(clean_logits, 1), F.softmax(shifted_logits, 1), reduction="none").sum(1) + F.kl_div(F.log_softmax(shifted_logits, 1), F.softmax(clean_logits, 1), reduction="none").sum(1))
            js = (js_per * mask * base_weights).sum() / denom
        else:
            shift_ce = _weighted_ce(shifted_logits, labels, shift_weights, class_weights)
            js = js_divergence(clean_logits, shifted_logits, shift_weights)
        losses["shift_ce"] = shift_ce
        losses["js"] = js
        if v3_2_mode:
            # shift_ce already implements the revision-4 classification loss:
            # unsupported rows are clean CE and supported rows receive 50/50
            # clean/shifted CE. Do not average clean CE a second time.
            total = shift_ce + lambda_js * js
        else:
            total = total + lambda_shift * shift_ce + lambda_js * js
        if clean_embedding is not None and shifted_embedding is not None:
            if v3_2_mode:
                feature_per = 1.0 - (
                    F.normalize(clean_embedding, dim=1)
                    * F.normalize(shifted_embedding, dim=1)
                ).sum(dim=1)
                feat = (feature_per * mask * base_weights).sum() / denom
            else:
                feat = feature_cosine_loss(
                    clean_embedding, shifted_embedding, shift_weights
                )
            losses["feat"] = feat
            total = total + lambda_feat * feat
        if penalize_alpha_overflow:
            strength = clean_logits.new_tensor(
                max(0.0, float(shift_strength) - float(alpha_max)) ** 2
            )
            losses["strength"] = strength
            total = total + strength
        if intervention_penalty is not None:
            pen = intervention_penalty.to(clean_logits.device)
            if intervention_mask is not None and pen.ndim > 0:
                mask = intervention_mask.to(device=pen.device, dtype=pen.dtype).reshape(
                    -1
                )
                pen = pen.reshape(-1)
                pen_denom = mask.sum().clamp_min(1.0)
                pen = (pen * mask).sum() / pen_denom
            elif pen.ndim > 0:
                pen = pen.mean()
            losses["intervention"] = pen
            if not v3_2_mode:
                total = total + lambda_intervention * losses["intervention"]
        if v3_2_mode:
            if style_target_error_per_sample is not None:
                target_error = style_target_error_per_sample.to(clean_logits.device).reshape(-1)
                losses["style"] = (target_error * base_weights * mask).sum() / denom
                total = total + float(lambda_style) * losses["style"]
            if rms_per_sample is not None:
                rms = rms_per_sample.to(clean_logits.device).reshape(-1)
                band = F.relu(float(rms_min) - rms).square() + F.relu(rms - float(rms_max)).square()
                # RMS band is a mechanism audit, not a trainable objective.
                losses["rms_band"] = (band * base_weights * mask).sum() / denom
    with torch.no_grad():
        losses["per_sample_clean"] = F.cross_entropy(
            clean_logits, labels, reduction="none"
        )
        if shifted_logits is not None:
            losses["per_sample_shift"] = F.cross_entropy(
                shifted_logits, labels, reduction="none"
            )
        else:
            losses["per_sample_shift"] = losses["per_sample_clean"]
    losses["total"] = total
    return losses
