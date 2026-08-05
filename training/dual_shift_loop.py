"""Training helpers for DualShift (CDT + APIS) journal variants."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from Model.dual_shift.losses import compute_dual_shift_loss
from Model.dual_shift.model import DualShiftResNet3D


def unbatch_acquisitions(batch_acquisition) -> Optional[List[Dict[str, Any]]]:
    if batch_acquisition is None:
        return None
    if isinstance(batch_acquisition, list):
        return batch_acquisition
    if not isinstance(batch_acquisition, Mapping):
        return None
    if not batch_acquisition:
        return []
    first = next(iter(batch_acquisition.values()))
    n = len(first)
    rows = []
    for index in range(n):
        row = {}
        for key, value in batch_acquisition.items():
            if torch.is_tensor(value):
                item = value[index]
                if item.numel() == 1:
                    number = item.item()
                    row[key] = None if isinstance(number, float) and not np.isfinite(number) else number
                else:
                    row[key] = item.detach().cpu()
            else:
                item = value[index]
                if item is None or item == "":
                    row[key] = None
                else:
                    row[key] = item
        rows.append(row)
    return rows


def phase_schedule(epoch: int, config: Mapping, *, variant: str | None = None) -> Dict[str, Any]:
    ds = config.get("dual_shift") or {}
    warm_clean = int(ds.get("warm_clean_epochs", 5))
    warm_apis = int(ds.get("warm_apis_epochs", 5))
    alpha_max = float(ds.get("alpha_max", 0.5))
    if variant in {"ce_x", "ce_xd"}:
        return {
            "apis_active": False,
            "cdt_enabled": False,
            "alpha": 0.0,
            "update_prototypes": False,
            "phase": "clean",
        }
    if variant in {"mixstyle", "mixstyle_x", "mixstyle_xd"}:
        return {
            "apis_active": False,
            "cdt_enabled": False,
            "alpha": 0.0,
            "update_prototypes": False,
            "phase": "mixstyle",
        }
    if variant == "apic_v3_2_x":
        if epoch < warm_clean:
            return {
                "apis_active": False,
                "cdt_enabled": False,
                "alpha": 0.0,
                "update_prototypes": False,
                "prepare_style_bank": False,
                "phase": "clean_warmup",
            }
        if epoch == warm_clean:
            return {
                "apis_active": False,
                "cdt_enabled": False,
                "alpha": 0.0,
                "update_prototypes": False,
                "prepare_style_bank": True,
                "phase": "style_bank_build",
            }
        if epoch < warm_clean + warm_apis + 1:
            progress = (epoch - warm_clean) / max(warm_apis, 1)
            return {
                "apis_active": True,
                "cdt_enabled": False,
                "alpha": alpha_max * progress,
                "update_prototypes": False,
                "prepare_style_bank": False,
                "phase": "apis_warmup",
            }
        return {
            "apis_active": True,
            "cdt_enabled": False,
            "alpha": alpha_max,
            "update_prototypes": False,
            "prepare_style_bank": False,
            "phase": "joint",
        }
    if variant == "film_scan":
        return {
            "apis_active": False,
            "cdt_enabled": False,
            "alpha": 0.0,
            "update_prototypes": False,
            "phase": "scan_film",
        }
    if variant == "cdt_only":
        # Align budget with DualShift: uniform through clean+APIS warm-up epochs,
        # enable CDT only when DualShift would enter the joint phase.
        if epoch < warm_clean + warm_apis:
            return {
                "apis_active": False,
                "cdt_enabled": False,
                "alpha": 0.0,
                "update_prototypes": False,
                "phase": "clean_warmup",
            }
        return {
            "apis_active": False,
            "cdt_enabled": True,
            "alpha": 0.0,
            "update_prototypes": False,
            "phase": "cdt",
        }
    # epoch is 0-based
    if epoch < warm_clean:
        return {
            "apis_active": False,
            "cdt_enabled": False,
            "alpha": 0.0,
            "update_prototypes": False,
            "phase": "clean_warmup",
        }
    if epoch < warm_clean + warm_apis:
        progress = (epoch - warm_clean + 1) / max(warm_apis, 1)
        return {
            "apis_active": True,
            "cdt_enabled": False,
            "alpha": alpha_max * progress,
            "update_prototypes": True,
            "phase": "apis_warmup",
        }
    return {
        "apis_active": True,
        "cdt_enabled": True,
        "alpha": alpha_max,
        "update_prototypes": True,
        "phase": "joint",
    }


def initialize_dual_shift_controllers(
    model: DualShiftResNet3D,
    train_subset,
    config: Mapping,
) -> None:
    dataset = train_subset.dataset
    indices = train_subset.indices
    ds = config.get("dual_shift") or {}
    needs_acquisition = model.use_scan_film or (
        model.use_apis and model.apis_variant != "v3_style_memory"
    )
    if needs_acquisition:
        acquisitions = []
        for index in indices:
            record = dataset.records[int(index)]
            acquisitions.append(record.get("acquisition") or {})
        if any(acquisitions):
            model.fit_acquisition_encoder(acquisitions)
    if model.use_cdt:
        sample_ids = [str(dataset.records[int(index)]["folder"]) for index in indices]
        covariates = train_subset.covariates
        model.cdt.initialize(
            sample_ids=sample_ids,
            subject_ids=[str(dataset.subject_ids[int(i)]) for i in indices],
            labels=[int(dataset.labels[int(i)]) for i in indices],
            age=covariates[:, 0].tolist(),
            sex=covariates[:, 1].tolist(),
            education=covariates[:, 2].tolist(),
        )
        model.cdt.age_bandwidth = float(ds.get("age_bandwidth", 1.0))
        model.cdt.education_bandwidth = float(ds.get("education_bandwidth", 1.0))
        model.cdt.cross_sex_rho = float(ds.get("cross_sex_rho", 0.5))
        model.cdt.ema_beta = float(ds.get("ema_beta", 0.9))
        model.cdt.ess_ratio_min = float(ds.get("ess_ratio_min", 0.2))
        model.cdt.kl_lambda = float(ds.get("lambda_kl", 0.05))
    model.prototype_bank.min_subjects = int(ds.get("prototype_min_subjects", 8))


def run_dual_shift_epoch(
    model: DualShiftResNet3D,
    loader,
    device,
    *,
    optimizer=None,
    config=None,
    class_weights=None,
    epoch: int = 0,
    collect_prototype_updates: bool = False,
    variant: str | None = None,
):
    from training.journal_metrics import compute_journal_metrics

    training = optimizer is not None
    schedule = phase_schedule(epoch, config or {}, variant=variant)
    bank_build = bool(training and schedule.get("phase") == "style_bank_build")
    model.train(training and not bank_build)
    if training:
        model.begin_epoch()
        model.set_phase(
            apis_active=schedule["apis_active"],
            cdt_enabled=schedule["cdt_enabled"],
            alpha=schedule["alpha"],
            prepare_style_bank=bool(schedule.get("prepare_style_bank", False)),
        )
        if model.use_cdt:
            model.cdt.recompute_weights(
                force_uniform=not bool(schedule["cdt_enabled"])
            )
    else:
        model.set_phase(apis_active=False, cdt_enabled=False, alpha=0.0)

    ds = (config or {}).get("dual_shift") or {}
    weight_tensor = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    total_loss = 0.0
    count = 0
    intervention_penalty_sum = 0.0
    valid_intervention_sum = 0.0
    apic_audit_keys = (
        "apis_feature_strength",
        "style_confidence",
        "style_entropy",
        "style_delta",
        "prototype_relative_separation",
        "condition_gate",
        "apis_effective_slots",
        "apis_max_slot_share",
    )
    apic_audit_sums = {key: 0.0 for key in apic_audit_keys}
    loss_component_keys = ("clean_ce", "shift_ce", "js", "feat", "intervention", "style", "rms_band")
    loss_component_sums = {key: 0.0 for key in loss_component_keys}
    logits_all, labels_all, env_all, subjects, folders = [], [], [], [], []
    field_strengths: List[float] = []
    for raw_batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
            if key != "acquisition"
        }
        acquisitions = unbatch_acquisitions(raw_batch.get("acquisition"))
        sample_ids = [str(folder) for folder in raw_batch["folder"]]
        subject_ids = [str(subject) for subject in raw_batch["subject_id"]]
        with torch.set_grad_enabled(training and not bank_build):
            outputs = model(
                batch["image"],
                batch["covariates"],
                acquisitions=acquisitions,
                sample_ids=sample_ids,
                subject_ids=subject_ids,
                update_prototypes=bool(
                    training and schedule["update_prototypes"] and collect_prototype_updates
                ),
                force_clean_only=not training,
                age_missing=batch.get("age_missing"),
                sex_missing=batch.get("sex_missing"),
                education_missing=batch.get("education_missing"),
            )
            sample_weights = None
            if training and model.use_cdt:
                sample_weights = model.cdt.batch_weights(sample_ids).to(device)
            extras = outputs.extras or {}
            intervention_penalty = extras.get("apis_coefficient_l2_per_sample")
            if intervention_penalty is None:
                intervention_penalty = extras.get("apis_coefficient_l2")
            intervention_mask = extras.get("valid_intervention_mask")
            v3_2_mode = getattr(model, "apis_variant", None) == "v3_2_balanced_style_memory"
            loss_dict = compute_dual_shift_loss(
                clean_logits=outputs.clean_logits,
                labels=batch["label"],
                shifted_logits=outputs.shifted_logits if training else None,
                clean_embedding=outputs.clean_embedding if training else None,
                shifted_embedding=outputs.shifted_embedding if training else None,
                sample_weights=sample_weights,
                class_weights=weight_tensor,
                kl_value=model.cdt.kl_penalty() if (training and model.cdt.enabled) else 0.0,
                shift_strength=float(schedule["alpha"]),
                alpha_max=float(ds.get("alpha_max", 0.5)),
                lambda_shift=float(ds.get("lambda_shift", 1.0)),
                lambda_js=float(ds.get("lambda_js", 0.25)),
                lambda_feat=float(ds.get("lambda_feat", 0.1)),
                lambda_kl=float(ds.get("lambda_kl", 0.05)),
                enable_apis=bool(
                    training
                    and schedule["apis_active"]
                    and model.use_apis
                ),
                intervention_penalty=intervention_penalty if training else None,
                lambda_intervention=float(ds.get("lambda_intervention", 0.001)),
                intervention_mask=intervention_mask if training else None,
                v3_2_mode=v3_2_mode,
                rms_per_sample=extras.get("apis_rms_per_sample") if training else None,
                style_target_error_per_sample=(
                    extras.get("apis_style_target_error_per_sample") if training else None
                ),
                rms_min=float(ds.get("rms_min", 0.001)),
                rms_max=float(ds.get("rms_max", 0.05)),
                lambda_style=float(ds.get("lambda_style", 0.1)),
                lambda_rms=float(ds.get("lambda_rms", 0.1)),
            )
            train_loss = loss_dict["total"]
            batch_size = len(batch["label"])
            for key in loss_component_keys:
                value = loss_dict.get(key)
                if value is not None:
                    loss_component_sums[key] += float(value.detach()) * batch_size
            if training and not bank_build:
                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()
                if model.use_cdt:
                    model.cdt.update_losses(
                        sample_ids,
                        loss_dict["per_sample_clean"].detach().cpu().tolist(),
                        loss_dict["per_sample_shift"].detach().cpu().tolist(),
                    )
                tracked = train_loss
                intervention_penalty_sum += float(loss_dict["intervention"].detach()) * len(
                    batch["label"]
                )
                frac = extras.get("valid_intervention_frac")
                if frac is not None:
                    valid_intervention_sum += float(frac.detach()) * len(batch["label"])
                for key in apic_audit_keys:
                    value = extras.get(key)
                    if value is not None:
                        apic_audit_sums[key] += float(value.detach().mean()) * len(
                            batch["label"]
                        )
            else:
                tracked = F.cross_entropy(outputs.clean_logits, batch["label"])
        total_loss += float(tracked.detach()) * batch_size
        count += batch_size
        logits_all.append(outputs.clean_logits.detach().cpu())
        labels_all.append(batch["label"].detach().cpu())
        env_all.append(batch["environment_id"].detach().cpu())
        subjects.extend(raw_batch["subject_id"])
        folders.extend(raw_batch["folder"])
        if acquisitions:
            for record in acquisitions:
                try:
                    field_strengths.append(float((record or {}).get("field_strength")))
                except (TypeError, ValueError):
                    field_strengths.append(float("nan"))
        else:
            field_strengths.extend([float("nan")] * batch_size)

    logits_np = torch.cat(logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    env_np = torch.cat(env_all).numpy()
    metrics = compute_journal_metrics(
        logits_np,
        labels_np,
        env_np,
        input_type="logits",
        subject_ids=subjects,
        aggregate="none",
    )
    if training and schedule.get("update_prototypes"):
        model.end_epoch_update()
    result = {
        "loss": total_loss / max(count, 1),
        "logits": logits_np,
        "labels": labels_np,
        "environments": env_np,
        "subjects": subjects,
        "folders": folders,
        "field_strengths": field_strengths,
        "metrics": metrics,
        "phase": schedule["phase"],
        "alpha": schedule["alpha"],
        "apis_coefficient_l2": intervention_penalty_sum / max(count, 1),
        "valid_intervention_frac": valid_intervention_sum / max(count, 1),
    }
    result.update(
        {key: value / max(count, 1) for key, value in apic_audit_sums.items()}
    )
    result.update(
        {
            "clean_ce": loss_component_sums["clean_ce"] / max(count, 1),
            "shift_ce": loss_component_sums["shift_ce"] / max(count, 1),
            "js": loss_component_sums["js"] / max(count, 1),
            "feature_consistency": loss_component_sums["feat"] / max(count, 1),
            "intervention_penalty": loss_component_sums["intervention"]
            / max(count, 1),
            "style_target_loss": loss_component_sums["style"] / max(count, 1),
            "rms_band_loss": loss_component_sums["rms_band"] / max(count, 1),
        }
    )
    if getattr(model, "apis_variant", None) in {"v3_style_memory", "v3_2_balanced_style_memory"}:
        style_valid = getattr(model.apis, "style_valid", None)
        style_counts = getattr(model.apis, "style_counts", None)
        if style_valid is not None:
            result["style_memory_valid_slots"] = int(style_valid.sum().item())
        if style_counts is not None:
            result["style_memory_total_assignments"] = float(style_counts.sum().item())
    return result
