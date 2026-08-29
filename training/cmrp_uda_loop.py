"""Training/evaluation helpers for the DS-042 CMRP-UDA variants."""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F

from Model.ablation.cmrp_uda import CMRPUDA3D, coral_loss, paired_relation_loss
from training.journal_metrics import compute_journal_metrics


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _table_inputs(batch: dict[str, Any], variant: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    table = batch["covariates"]
    masks = {
        "age_missing": batch.get("age_missing"),
        "sex_missing": batch.get("sex_missing"),
        "education_missing": batch.get("education_missing"),
    }
    if variant == "cmrp_missingness_only":
        table = torch.zeros_like(table)
    if variant == "cmrp_shuffled_table":
        permutation = torch.randperm(table.shape[0], device=table.device)
        table = table.index_select(0, permutation)
        masks = {
            key: None if value is None else value.index_select(0, permutation)
            for key, value in masks.items()
        }
    return table, masks


def _forward(model: CMRPUDA3D, batch: dict[str, Any], variant: str, *, apply_adapter: bool):
    table, masks = _table_inputs(batch, variant)
    return model.forward_with_repr(
        batch["image"],
        table,
        **masks,
        apply_adapter=apply_adapter,
    )


def _mean_or_zero(values: Iterable[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    values = list(values)
    return torch.stack(values).mean() if values else reference.new_zeros(())


def _cmrp_loss(
    model: CMRPUDA3D,
    source_out: dict[str, torch.Tensor],
    target_out: dict[str, torch.Tensor] | None,
    source_clean: dict[str, torch.Tensor],
    source_labels: torch.Tensor,
    *,
    target_aug_out: dict[str, torch.Tensor] | None,
    variant: str,
    class_weights: torch.Tensor | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config.get("cmrp_uda") or {}
    task = F.cross_entropy(source_out["logits"], source_labels, weight=class_weights)
    source_relation = paired_relation_loss(
        source_out["mri_shared"], source_out["table_shared"]
    )
    target_relation = source_out["logits"].new_zeros(())
    alignment = source_out["logits"].new_zeros(())
    if target_out is not None:
        target_relation = paired_relation_loss(
            target_out["mri_shared"], target_out["table_shared"]
        )
        if target_aug_out is not None:
            target_relation = target_relation + paired_relation_loss(
                target_out["mri_shared"], target_aug_out["mri_shared"]
            )
        # Keep the named alignment ablations executable from the variant list;
        # the YAML default remains the primary MRI-specific alignment.
        align_scope = str(cfg.get("alignment_scope", "mri_specific"))
        align_scope = {
            "cmrp_joint_uda": "joint",
            "cmrp_mri_specific": "mri_specific",
            "cmrp_table_specific": "table_specific",
            "cmrp_both_specific": "both_specific",
        }.get(variant, align_scope)
        if align_scope == "joint":
            alignment = coral_loss(source_out["joint"], target_out["joint"])
        elif align_scope == "table_specific":
            alignment = coral_loss(source_out["table_specific"], target_out["table_specific"])
        elif align_scope == "both_specific":
            alignment = coral_loss(source_out["mri_specific"], target_out["mri_specific"]) + coral_loss(
                source_out["table_specific"], target_out["table_specific"]
            )
        else:
            alignment = coral_loss(source_out["mri_specific"], target_out["mri_specific"])

    if variant in {"cmrp_no_source_relation", "cmrp_shuffled_table", "cmrp_missingness_only", "cmrp_table_only"}:
        source_relation = source_out["logits"].new_zeros(())
    if variant in {"cmrp_no_target_relation", "cmrp_shuffled_table", "cmrp_missingness_only", "cmrp_table_only"}:
        target_relation = source_out["logits"].new_zeros(())
    if variant == "cmrp_no_alignment":
        alignment = source_out["logits"].new_zeros(())

    identity = (source_out["joint"] - source_clean["joint"].detach()).square().mean()
    identity = identity / source_clean["joint"].detach().square().mean().clamp_min(1e-6)
    proximal = model.adapter_penalty()
    if variant == "cmrp_no_identity":
        identity = source_out["logits"].new_zeros(())
        proximal = source_out["logits"].new_zeros(())

    weights = {
        "task": float(cfg.get("lambda_task", 1.0)),
        "source_relation": float(cfg.get("lambda_source_relation", 0.1)),
        "target_relation": float(cfg.get("lambda_target_relation", 0.1)),
        "alignment": float(cfg.get("lambda_alignment", 0.1)),
        "proximal": float(cfg.get("lambda_prox", 1e-4)),
        "identity": float(cfg.get("lambda_identity", 0.1)),
    }
    total = (
        weights["task"] * task
        + weights["source_relation"] * source_relation
        + weights["target_relation"] * target_relation
        + weights["alignment"] * alignment
        + weights["proximal"] * proximal
        + weights["identity"] * identity
    )
    return total, {
        "task": task.detach(),
        "source_relation": source_relation.detach(),
        "target_relation": target_relation.detach(),
        "alignment": alignment.detach(),
        "proximal": proximal.detach(),
        "identity": identity.detach(),
    }


def run_cmrp_train_epoch(
    model: CMRPUDA3D,
    source_loader,
    target_adapt_loader,
    device: torch.device,
    *,
    optimizer,
    variant: str,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> dict[str, float | None]:
    model.train(True)
    if variant != "cmrp_source_only" and target_adapt_loader is None:
        raise ValueError("CMRP-UDA variants require an unlabeled target adaptation loader")
    weight_tensor = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    target_iter = None if target_adapt_loader is None else iter(target_adapt_loader)
    totals: dict[str, float] = {key: 0.0 for key in ("loss", "task", "source_relation", "target_relation", "alignment", "proximal", "identity")}
    steps = 0
    for raw_source in source_loader:
        source = _move(raw_source, device)
        target = None
        target_aug = None
        if target_iter is not None:
            try:
                raw_target = next(target_iter)
            except StopIteration:
                target_iter = iter(target_adapt_loader)
                raw_target = next(target_iter)
            target = _move(raw_target, device)
        optimizer.zero_grad(set_to_none=True)
        source_out = _forward(model, source, variant, apply_adapter=True)
        # Evaluate the clean anchor without dropout noise.  The model returns
        # to train mode immediately so the supervised/target paths keep their
        # normal stochastic regularization and BatchNorm updates.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            source_clean = _forward(model, source, variant, apply_adapter=False)
        if was_training:
            model.train(True)
        target_out = None
        if target is not None:
            target_out = _forward(model, target, variant, apply_adapter=True)
            aug = dict(target)
            noise_scale = float((config.get("cmrp_uda") or {}).get("target_aug_noise", 0.01))
            aug["image"] = target["image"] + noise_scale * torch.randn_like(target["image"])
            target_aug = _forward(model, aug, variant, apply_adapter=True)
        loss, parts = _cmrp_loss(
            model,
            source_out,
            target_out,
            source_clean,
            source["label"],
            target_aug_out=target_aug,
            variant=variant,
            class_weights=weight_tensor,
            config=config,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CMRP loss for variant {variant}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float((config.get("cmrp_uda") or {}).get("max_grad_norm", 5.0)))
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals[key] += float(value.cpu())
        steps += 1
    if not steps:
        raise RuntimeError("CMRP source loader produced no batches")
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def run_cmrp_eval_epoch(
    model: CMRPUDA3D,
    loader,
    device: torch.device,
    *,
    variant: str,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.train(False)
    weight_tensor = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    environments: list[int] = []
    subjects: list[str] = []
    folders: list[str] = []
    losses = 0.0
    count = 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = _forward(model, batch, variant, apply_adapter=True)
        logits = output["logits"]
        loss = F.cross_entropy(logits, batch["label"], weight=weight_tensor)
        logits_all.append(logits.detach().cpu())
        labels_all.append(batch["label"].detach().cpu())
        env = batch.get("environment_id")
        environments.extend(
            [int(value) for value in (env.detach().cpu().tolist() if torch.is_tensor(env) else env or [0] * len(logits))]
        )
        subjects.extend([str(value) for value in batch["subject_id"]])
        folders.extend([str(value) for value in batch["folder"]])
        losses += float(loss.detach().cpu()) * len(logits)
        count += len(logits)
    if not logits_all:
        raise RuntimeError("CMRP evaluation loader produced no batches")
    logits_np = torch.cat(logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    env_np = torch.as_tensor(environments, dtype=torch.long).numpy()
    return {
        "loss": losses / max(count, 1),
        "logits": logits_np,
        "labels": labels_np,
        "environments": env_np,
        "subjects": subjects,
        "folders": folders,
        "field_strengths": [float("nan")] * len(labels_np),
        "metrics": compute_journal_metrics(
            logits_np,
            labels_np,
            env_np,
            input_type="logits",
            subject_ids=subjects,
            aggregate="none",
        ),
    }


__all__ = ["run_cmrp_eval_epoch", "run_cmrp_train_epoch"]
