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
        shift_ce = _weighted_ce(shifted_logits, labels, shift_weights, class_weights)
        js = js_divergence(clean_logits, shifted_logits, shift_weights)
        losses["shift_ce"] = shift_ce
        losses["js"] = js
        total = total + lambda_shift * shift_ce + lambda_js * js
        if clean_embedding is not None and shifted_embedding is not None:
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
                denom = mask.sum().clamp_min(1.0)
                pen = (pen * mask).sum() / denom
            elif pen.ndim > 0:
                pen = pen.mean()
            losses["intervention"] = pen
            total = total + lambda_intervention * losses["intervention"]
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
