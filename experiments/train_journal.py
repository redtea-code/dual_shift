"""Complete leakage-safe source-to-target journal experiment entry point."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate


def journal_collate(batch):
    """Default collate, but keep acquisition rows as a Python list of dicts."""
    if not batch:
        return default_collate(batch)
    if "acquisition" not in batch[0]:
        return default_collate(batch)
    acquisitions = [item.get("acquisition") for item in batch]
    stripped = [{key: value for key, value in item.items() if key != "acquisition"} for item in batch]
    collated = default_collate(stripped)
    collated["acquisition"] = acquisitions
    return collated

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.journal_dataset import (
    CovariatePreprocessor,
    JournalManifestDataset,
    JournalNiftiDataset,
    JournalSubset,
    build_journal_dataset,
)
from Model.backbone.backdoor_backbone import resnet10_backdoor
from Model.backbone.daft_backbone import resnet10_daft
from Model.backbone.film_backbone import resnet10_film
from Model.backbone.journal_resnet import journal_resnet10, journal_resnet18
from Model.dual_shift import DualShiftResNet3D
from Model.dual_shift.metadata_baseline import MetadataConcatBaseline
from training.dual_shift_loop import (
    initialize_dual_shift_controllers,
    run_dual_shift_epoch,
)
from training.group_dro import GroupDRO
from training.journal_metrics import (
    bootstrap_confidence_intervals,
    compute_journal_metrics,
    compute_metrics_by_field_strength,
    paired_bootstrap_difference,
    save_json_summary,
)
from utils.claim_holdout import (
    assert_no_holdout_leak,
    filter_indices_excluding_subjects,
    load_holdout_subjects,
)
from utils.journal_protocol import (
    DemographicEnvironmentBuilder,
    assert_disjoint_subjects,
)


VARIANT_STAGES = {
    "ce_only": "A",
    "groupdro": "A",
    "spatial": "B",
    "spatial_groupdro": "C",
    # External fusion baselines under the same journal protocol (cov3).
    "film": "baselines",
    "daft": "baselines",
    "hyperfusion": "baselines",
    # Claim baseline: image + acquisition concat (no demographics).
    "metadata": "baselines",
    # Original Wave0 methods re-run under journal protocol (same split/cov3).
    "gamma": "gamma_concat",
    "concat": "gamma_concat",
    # Dual-dictionary on frozen Stage A CE features (cross-cohort check).
    "dual_dict_linear": "dictionary",
    "dual_dict_core": "dictionary",
    # Dual-shift journal mainline (CDT + APIS).
    "dual_shift": "dual_shift",
    "cdt_only": "dual_shift",
    "apis_only": "dual_shift",  # legacy alias → residual APIS v2 behavior
    "apis_v2": "dual_shift",
    "apis_v2_shuffle": "dual_shift",
    "mixstyle": "dual_shift",
}

# Display names: ce_only with class_weighted_ce is weighted CE, not plain CE.
VARIANT_DISPLAY = {
    "ce_only": "weighted_ce",
    "groupdro": "groupdro",
    "spatial": "spatial",
    "spatial_groupdro": "spatial_groupdro",
    "film": "film",
    "daft": "daft",
    "hyperfusion": "hyperfusion",
    "metadata": "metadata_acq_concat",
    "gamma": "gamma",
    "concat": "concat",
    "dual_dict_linear": "dual_dict_linear",
    "dual_dict_core": "dual_dict_core",
    "dual_shift": "dual_shift",
    "cdt_only": "cdt_only",
    "apis_only": "apis_v2",
    "apis_v2": "apis_v2",
    "apis_v2_shuffle": "apis_v2_shuffle",
    "mixstyle": "mixstyle",
}

EXTERNAL_FUSION_VARIANTS = frozenset({"film", "daft", "hyperfusion", "concat"})
METADATA_VARIANTS = frozenset({"metadata"})
PATCH_GAMMA_VARIANTS = frozenset({"gamma"})
DICTIONARY_VARIANTS = frozenset({"dual_dict_linear", "dual_dict_core"})
DUAL_SHIFT_VARIANTS = frozenset(
    {"dual_shift", "cdt_only", "apis_only", "apis_v2", "apis_v2_shuffle", "mixstyle"}
)
JOURNAL_SPATIAL_VARIANTS = frozenset({"spatial", "spatial_groupdro"})
JOURNAL_BACKBONE_VARIANTS = frozenset(
    {"ce_only", "groupdro", "spatial", "spatial_groupdro"}
)

VARIANT_ALIASES = {
    # Historical name retained; residual APIS v2 is the current implementation.
    "apis_only": "apis_v2",
}


def _default_gamma_backdoor_kwargs() -> dict:
    """Match Wave0 B_gamma defaults (table_feature→cov3; patch-γ + TV/sparsity)."""
    return {
        "return_gamma": True,
        "gamma_range": True,
        "gamma_mode": "patch",
        "shuffle_tabular": False,
        "shuffle_seed": 0,
        "gamma_dropout": 0.2,
        "gamma_dropout_rescale": True,
        "group_sharing": 200,
        "spatial_smooth_mode": "both",
        "spatial_smooth_lambda": 0.01,
        "spatial_filter_alpha": 0.4,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dataset(config: dict, cohort: str) -> JournalNiftiDataset | JournalManifestDataset:
    return build_journal_dataset(config, cohort)


def _subject_labels(dataset: JournalNiftiDataset, subjects: np.ndarray) -> list[int]:
    labels = []
    for subject in subjects:
        subject_labels = dataset.labels[dataset.subject_ids == subject]
        labels.append(int(np.bincount(subject_labels).argmax()))
    return labels


def _can_stratify(labels: Sequence[int], test_size: float) -> bool:
    counts = np.bincount(labels)
    n_subjects = len(labels)
    holdout = max(1, round(n_subjects * test_size))
    return bool(np.min(counts[counts > 0]) >= 2 and holdout >= len(counts[counts > 0]))


def _split_source(
    dataset: JournalNiftiDataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """Subject-level 6/2/2 (or configured) source split with optional stratification."""
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError(
            "train_ratio, val_ratio, and test_ratio must be positive and sum to 1"
        )
    subjects = np.unique(dataset.subject_ids)
    if len(subjects) < 3:
        raise ValueError("Source dataset needs at least three subjects for 6/2/2 split")
    subject_labels = _subject_labels(dataset, subjects)
    holdout_ratio = float(val_ratio + test_ratio)
    stratify_holdout = _can_stratify(subject_labels, holdout_ratio)
    train_subjects, holdout_subjects = train_test_split(
        subjects,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=subject_labels if stratify_holdout else None,
    )
    holdout_labels = _subject_labels(dataset, holdout_subjects)
    relative_test = float(test_ratio / holdout_ratio)
    stratify_test = _can_stratify(holdout_labels, relative_test)
    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout_labels if stratify_test else None,
    )
    train_indices = np.flatnonzero(np.isin(dataset.subject_ids, train_subjects))
    val_indices = np.flatnonzero(np.isin(dataset.subject_ids, val_subjects))
    test_indices = np.flatnonzero(np.isin(dataset.subject_ids, test_subjects))
    assert_disjoint_subjects(train_subjects, val_subjects, test_subjects)
    return train_indices, val_indices, test_indices


def _assert_close_mapping(left: dict, right: dict, *, label: str, rtol=1e-5, atol=1e-6):
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        raise ValueError(
            f"{label} key mismatch: only_left={sorted(left_keys - right_keys)} "
            f"only_right={sorted(right_keys - left_keys)}"
        )
    for key in sorted(left_keys):
        a, b = left[key], right[key]
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                raise ValueError(f"{label}.{key} length mismatch")
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                if isinstance(left_item, (int, float)) and isinstance(
                    right_item, (int, float)
                ):
                    if not np.isclose(left_item, right_item, rtol=rtol, atol=atol):
                        raise ValueError(
                            f"{label}.{key}[{index}] mismatch: {left_item} vs {right_item}"
                        )
                elif left_item != right_item:
                    raise ValueError(
                        f"{label}.{key}[{index}] mismatch: {left_item!r} vs {right_item!r}"
                    )
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not np.isclose(float(a), float(b), rtol=rtol, atol=atol):
                raise ValueError(f"{label}.{key} mismatch: {a} vs {b}")
        elif a != b:
            raise ValueError(f"{label}.{key} mismatch: {a!r} vs {b!r}")


def _fit_protocol(
    source,
    train_indices,
    val_indices,
    test_indices,
    target,
    environment_config,
    *,
    frozen_manifest=None,
):
    fitted_preprocessor = CovariatePreprocessor(
        scale_continuous=bool(environment_config.get("scale_continuous", True))
    ).fit(
        source.raw_age[train_indices],
        source.raw_sex[train_indices],
        source.raw_education[train_indices],
        source.subject_ids[train_indices],
    )
    if frozen_manifest and frozen_manifest.get("covariate_preprocessor"):
        frozen_preprocessor = CovariatePreprocessor.from_dict(
            frozen_manifest["covariate_preprocessor"]
        )
        _assert_close_mapping(
            fitted_preprocessor.to_dict(),
            frozen_preprocessor.to_dict(),
            label="covariate_preprocessor",
        )
        preprocessor = frozen_preprocessor
    else:
        preprocessor = fitted_preprocessor

    # Environment uses the same imputed sex names as the model covariates.
    def filled(dataset, indices):
        age = np.where(
            np.isfinite(dataset.raw_age[indices]),
            dataset.raw_age[indices],
            preprocessor.age_median_,
        )
        education = np.where(
            np.isfinite(dataset.raw_education[indices]),
            dataset.raw_education[indices],
            preprocessor.education_median_,
        )
        sex = preprocessor.transform_sex_names(dataset.raw_sex[indices])
        return age, sex, education

    builder = DemographicEnvironmentBuilder(
        n_age_bins=int(environment_config["n_age_bins"]),
        n_education_bins=int(environment_config["n_education_bins"]),
        min_group_size=int(environment_config["min_group_size"]),
        min_per_class=int(environment_config.get("min_per_class", 1)),
        mode=str(environment_config.get("mode", "age_sex")),
    )
    train_raw = filled(source, train_indices)
    train_env = builder.fit_transform(
        *train_raw,
        subject_ids=source.subject_ids[train_indices],
        labels=source.labels[train_indices],
    )
    if frozen_manifest and frozen_manifest.get("environment"):
        frozen_env = frozen_manifest["environment"]
        # Always validate edges against the frozen Stage A protocol.
        if "age_edges" in frozen_env:
            if not np.allclose(
                builder.age_edges_, np.asarray(frozen_env["age_edges"], dtype=float)
            ):
                raise ValueError("Frozen age_edges do not match refit protocol")
        if "education_edges" in frozen_env:
            if not np.allclose(
                builder.education_edges_,
                np.asarray(frozen_env["education_edges"], dtype=float),
            ):
                raise ValueError("Frozen education_edges do not match refit protocol")
        if frozen_env.get("environment_to_id") or (
            frozen_env.get("detailed_groups") is not None
            and frozen_env.get("age_sex_groups") is not None
        ):
            restored = DemographicEnvironmentBuilder.from_dict(
                {
                    **frozen_env,
                    "mode": environment_config.get("mode", frozen_env.get("mode", "age_sex")),
                    "n_age_bins": environment_config.get(
                        "n_age_bins", frozen_env.get("n_age_bins", 3)
                    ),
                    "n_education_bins": environment_config.get(
                        "n_education_bins", frozen_env.get("n_education_bins", 3)
                    ),
                    "min_group_size": environment_config.get(
                        "min_group_size", frozen_env.get("min_group_size", 5)
                    ),
                    "min_per_class": environment_config.get(
                        "min_per_class", frozen_env.get("min_per_class", 1)
                    ),
                }
            )
            if restored.environment_to_id_ != builder.environment_to_id_:
                raise ValueError(
                    "Frozen environment_to_id mapping does not match refit protocol"
                )
            restored.train_subject_ids_ = frozenset(
                source.subject_ids[train_indices].tolist()
            )
            builder = restored
            train_env = builder.transform(
                *train_raw,
                subject_ids=source.subject_ids[train_indices],
                validate_no_train_overlap=False,
            )
        elif frozen_env.get("names"):
            fitted_names = {str(k): v for k, v in builder.environment_names_.items()}
            frozen_names = {str(k): v for k, v in frozen_env["names"].items()}
            if fitted_names != frozen_names:
                raise ValueError(
                    "Frozen environment names do not match refit protocol"
                )

    val_raw = filled(source, val_indices)
    val_env = builder.transform(
        *val_raw,
        subject_ids=source.subject_ids[val_indices],
        validate_no_train_overlap=True,
    )
    test_raw = filled(source, test_indices)
    test_env = builder.transform(
        *test_raw,
        subject_ids=source.subject_ids[test_indices],
        validate_no_train_overlap=True,
    )
    target_indices = np.arange(len(target))
    target_raw = filled(target, target_indices)
    target_env = builder.transform(
        *target_raw,
        subject_ids=target.subject_ids,
        validate_no_train_overlap=False,
    )
    return preprocessor, builder, train_env, val_env, test_env, target_env


def _make_journal_backbone(config, num_classes):
    """Build the journal ResNet used by ce_only / GroupDRO / dictionary."""
    model_config = config["model"]
    factory = {
        "journal_resnet10": journal_resnet10,
        "journal_resnet18": journal_resnet18,
    }[model_config["name"]]
    var_specs = [
        {
            "name": "age",
            "type": "continuous",
            "min_val": -5.0,
            "max_val": 5.0,
            "n_centers": int(model_config.get("continuous_centers", 8)),
            "n_bases": int(model_config.get("continuous_bases", 4)),
        },
        {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 2},
        {
            "name": "education",
            "type": "continuous",
            "min_val": -5.0,
            "max_val": 5.0,
            "n_centers": int(model_config.get("continuous_centers", 8)),
            "n_bases": int(model_config.get("continuous_bases", 4)),
        },
    ]
    return factory(
        num_classes=num_classes,
        base_channels=int(model_config["base_channels"]),
        spatial_shape=tuple(model_config["spatial_shape"]),
        embedding_dim=int(model_config["embedding_dim"]),
        sparse_regularization=float(config["loss"]["lambda_sparse"]) > 0,
        dropout=float(model_config.get("dropout", 0.0)),
        var_specs=var_specs,
    )


def _backbone_feature_dim(backbone) -> int:
    return int(backbone.fc.in_features)


def _make_model(config, num_classes, variant):
    """Build journal ResNet or an external fusion baseline.

    External baselines (FiLM / DAFT / HyperFusion / Concat) and patch-Gamma
    consume the same cov3 vector ``[age, sex, education]`` (txt_dim=3).
    Dictionary variants wrap a journal backbone + DualDictionaryModel head.
    """
    model_config = config["model"]
    baseline_cfg = config.get("baselines", {})
    covariate_dim = 3

    if variant in DICTIONARY_VARIANTS:
        from Model.dictionary.journal_dual_dict import make_journal_dual_dict

        backbone = _make_journal_backbone(config, num_classes)
        feature_dim = _backbone_feature_dim(backbone)
        return make_journal_dual_dict(
            backbone,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dict_cfg=config.get("dictionary") or {},
            preset=variant,
        )

    if variant in DUAL_SHIFT_VARIANTS:
        ds = config.get("dual_shift") or {}
        model_config = config["model"]
        use_apis = variant in {"dual_shift", "apis_only", "apis_v2", "apis_v2_shuffle"}
        model = DualShiftResNet3D(
            num_classes=num_classes,
            base_channels=int(model_config.get("base_channels", 32)),
            dropout=float(model_config.get("dropout", 0.1)),
            fusion_rank=int(ds.get("fusion_rank", 16)),
            acquisition_out_dim=int(ds.get("acquisition_out_dim", 32)),
            alpha_max=float(ds.get("alpha_max", 0.25)),
            apis_basis_count=int(ds.get("apis_basis_count", 4)),
            apis_rank=int(ds.get("apis_rank", 8)),
            use_apis=use_apis,
            use_cdt=variant in {"dual_shift", "cdt_only"},
            use_mixstyle=variant == "mixstyle",
            mixstyle_p=float(ds.get("mixstyle_p", 0.5)),
            mixstyle_alpha=float(ds.get("mixstyle_alpha", 0.1)),
            prototype_min_subjects=int(ds.get("prototype_min_subjects", 8)),
        )
        model.shuffle_acquisition = bool(variant == "apis_v2_shuffle")
        return model

    if variant == "film":
        return resnet10_film(
            txt_dim=covariate_dim,
            num_classes=num_classes,
            film_stages=str(baseline_cfg.get("film_stages", "last")),
            feature=False,
        )
    if variant == "daft":
        return resnet10_daft(
            txt_dim=covariate_dim,
            num_classes=num_classes,
            feature=False,
        )
    if variant == "hyperfusion":
        from Model.comparison.factories import make_hyperfusion

        return make_hyperfusion(
            in_channels=1,
            num_tabular=covariate_dim,
            num_classes=num_classes,
            img_feat_dim=int(baseline_cfg.get("hyper_img_feat_dim", 128)),
            hidden_dim=int(baseline_cfg.get("hyper_hidden_dim", 128)),
            dropout=float(baseline_cfg.get("hyper_dropout", model_config.get("dropout", 0.1))),
            base_channels=int(baseline_cfg.get("hyper_base_channels", 8)),
        )
    if variant == "metadata":
        ds = config.get("dual_shift") or {}
        return MetadataConcatBaseline(
            num_classes=num_classes,
            base_channels=int(model_config.get("base_channels", 32)),
            acquisition_out_dim=int(ds.get("acquisition_out_dim", 32)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if variant == "concat":
        from Model.comparison.factories import make_concat_fusion

        return make_concat_fusion(
            in_channels=1,
            num_tabular=covariate_dim,
            num_classes=num_classes,
            img_feat_dim=int(baseline_cfg.get("concat_img_feat_dim", 128)),
            tab_feat_dim=int(baseline_cfg.get("concat_tab_feat_dim", 128)),
            hidden_dim=int(baseline_cfg.get("concat_hidden_dim", 128)),
            dropout=float(
                baseline_cfg.get("concat_dropout", model_config.get("dropout", 0.1))
            ),
            base_channels=int(baseline_cfg.get("concat_base_channels", 8)),
        )
    if variant == "gamma":
        backdoor_kwargs = dict(baseline_cfg.get("gamma_backdoor_kwargs") or {})
        if not backdoor_kwargs:
            backdoor_kwargs = _default_gamma_backdoor_kwargs()
        input_shape = tuple(
            int(v) for v in baseline_cfg.get("gamma_input_shape", (160, 196, 160))
        )
        return resnet10_backdoor(
            txt_dim=covariate_dim,
            num_classes=num_classes,
            z_dim=int(baseline_cfg.get("gamma_z_dim", 128)),
            feature=False,
            use_class_head=bool(baseline_cfg.get("gamma_use_class_head", True)),
            backdoor_kwargs=backdoor_kwargs,
            input_shape=input_shape,
        )

    return _make_journal_backbone(config, num_classes)


def _logits(model, batch, spatial, variant=None):
    image = batch["image"]
    covariates = batch["covariates"]
    if variant in DICTIONARY_VARIANTS:
        return model(image, covariates)
    if variant in METADATA_VARIANTS:
        raise RuntimeError("metadata variant must pass acquisitions via _run_epoch")
    if variant in EXTERNAL_FUSION_VARIANTS or variant in PATCH_GAMMA_VARIANTS:
        # FiLM/DAFT/Gamma: (x, txt); HyperFusion/Concat: (image, table).
        return model(image, covariates)
    if spatial:
        return model(image, covariates)
    features = model.forward_features(image, covariates, modulate=False)
    return model.fc(model.dropout(model.pool(features).flatten(1)))


def _run_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    dro=None,
    config=None,
    spatial=False,
    class_weights=None,
    variant=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    logits_all, labels_all, env_all, subjects, folders = [], [], [], [], []
    field_strengths = []
    weight_tensor = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    for raw_batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
            if key != "acquisition"
        }
        acquisitions = raw_batch.get("acquisition")
        with torch.set_grad_enabled(training):
            if variant in DUAL_SHIFT_VARIANTS:
                raise RuntimeError(
                    "Dual-shift variants must use run_dual_shift_epoch, not _run_epoch"
                )
            if variant in DICTIONARY_VARIANTS:
                dict_out = model.forward_dict(batch["image"], batch["covariates"])
                logits = dict_out["logits"]
            elif variant in METADATA_VARIANTS:
                if not isinstance(acquisitions, list):
                    raise RuntimeError("metadata variant requires acquisition rows")
                logits = model(batch["image"], acquisitions)
            else:
                logits = _logits(model, batch, spatial, variant=variant)
            if dro is None:
                objective = F.cross_entropy(
                    logits, batch["label"], weight=weight_tensor
                )
            else:
                # GroupDRO uses per-sample CE; class weights applied per sample.
                sample = F.cross_entropy(
                    logits, batch["label"], reduction="none", weight=weight_tensor
                )
                group_losses, present = dro.compute_group_losses(
                    sample, batch["environment_id"]
                )
                if training:
                    dro.update(group_losses, present)
                weights = dro.group_weights.to(device)
                present_weights = weights * present.to(weights.dtype)
                denominator = present_weights.sum().clamp_min(
                    torch.finfo(present_weights.dtype).tiny
                )
                objective = (present_weights * group_losses).sum() / denominator
            train_loss = objective
            if variant in DICTIONARY_VARIANTS and training:
                # Supervised dual-dict core loss on frozen CE GAP features.
                from Model.dictionary.journal_dual_dict import resolve_dictionary_preset
                from Model.dictionary.losses import compute_core_losses

                preset_cfg = resolve_dictionary_preset(
                    config.get("dictionary") or {}, variant
                )
                ages = batch["covariates"][:, 0]
                train_loss, _ = compute_core_losses(
                    dict_out["H"],
                    dict_out["H_hat"],
                    dict_out["Q_d"],
                    dict_out["Q_c"],
                    model.dict_model.D,
                    model.dict_model.C,
                    logits,
                    batch["label"],
                    dict_out["age_pred"],
                    ages,
                    loss_weights=preset_cfg["loss_weights"],
                )
            if spatial and training:
                regularizers = model.get_regularization_losses()
                train_loss = (
                    train_loss
                    + float(config["loss"]["lambda_tv"]) * regularizers["journal_tv"]
                    + float(config["loss"]["lambda_sparse"])
                    * regularizers.get("journal_sparse", train_loss.new_zeros(()))
                )
            if variant in PATCH_GAMMA_VARIANTS and training:
                # Wave0 B_gamma regs: backdoor_sparsity / backdoor_smoothness.
                regularizers = model.get_regularization_losses()
                loss_cfg = config.get("loss", {})
                baseline_cfg = config.get("baselines", {})
                sparsity_w = float(
                    baseline_cfg.get(
                        "gamma_backdoor_sparsity",
                        loss_cfg.get("backdoor_sparsity", 0.1),
                    )
                )
                smooth_w = float(
                    baseline_cfg.get(
                        "gamma_backdoor_smoothness",
                        loss_cfg.get("backdoor_smoothness", 0.2),
                    )
                )
                train_loss = (
                    train_loss
                    + sparsity_w
                    * regularizers.get("backdoor_sparsity", train_loss.new_zeros(()))
                    + smooth_w
                    * regularizers.get("backdoor_smoothness", train_loss.new_zeros(()))
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                optimizer.step()
                if variant in DICTIONARY_VARIANTS:
                    model.project_dictionary_norms()
                tracked = train_loss
            else:
                # Selection / reporting use pure classification loss for all variants.
                tracked = objective
        batch_size = len(batch["label"])
        total_loss += float(tracked.detach()) * batch_size
        count += batch_size
        logits_all.append(logits.detach().cpu())
        labels_all.append(batch["label"].detach().cpu())
        env_all.append(batch["environment_id"].detach().cpu())
        subjects.extend(raw_batch["subject_id"])
        folders.extend(raw_batch["folder"])
        if isinstance(acquisitions, list):
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
    return {
        "loss": total_loss / max(count, 1),
        "logits": logits_np,
        "labels": labels_np,
        "environments": env_np,
        "subjects": subjects,
        "folders": folders,
        "field_strengths": field_strengths,
        "metrics": metrics,
    }


def _save_predictions(path, result):
    probabilities = torch.softmax(torch.from_numpy(result["logits"]), dim=1).numpy()
    fields = [
        "subject_id",
        "folder",
        "label",
        "predicted_label",
        "environment_id",
        "field_strength",
    ]
    fields += [f"probability_{i}" for i in range(probabilities.shape[1])]
    strengths = result.get("field_strengths") or [float("nan")] * len(probabilities)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            row = {
                "subject_id": result["subjects"][index],
                "folder": result["folders"][index],
                "label": int(result["labels"][index]),
                "predicted_label": int(probability.argmax()),
                "environment_id": int(result["environments"][index]),
                "field_strength": strengths[index],
            }
            row.update(
                {f"probability_{i}": float(value) for i, value in enumerate(probability)}
            )
            writer.writerow(row)
    return probabilities


def _gamma_summary(model):
    bases = [item.bases.detach().cpu().numpy() for item in model.modulation.spatial_bases]
    names = [spec["name"] for spec in model.var_specs]
    per_variable = {}
    for name, value in zip(names, bases):
        diffs = []
        for axis in (2, 3, 4):
            if value.shape[axis] > 1:
                diffs.append(np.abs(np.diff(value, axis=axis)).mean())
        per_variable[name] = {
            "basis_abs_mean": float(np.abs(value).mean()),
            "basis_min": float(value.min()),
            "basis_max": float(value.max()),
            "basis_tv": float(np.mean(diffs)) if diffs else 0.0,
        }
    return {"per_variable": per_variable}


def _patch_gamma_summary(model):
    """Summarize Wave0 patch-gamma from ResNetBackdoorBackbone."""
    gamma, spatial = model.get_gamma()
    if gamma is None:
        return {"kind": "patch_gamma", "available": False}
    gamma_np = gamma.detach().cpu().numpy()
    summary = {
        "kind": "patch_gamma",
        "available": True,
        "abs_mean": float(np.abs(gamma_np).mean()),
        "mean": float(gamma_np.mean()),
        "min": float(gamma_np.min()),
        "max": float(gamma_np.max()),
        "n_patches": int(gamma_np.shape[-1]) if gamma_np.ndim >= 2 else int(gamma_np.size),
    }
    if spatial is not None:
        summary["spatial_shape"] = [int(v) for v in spatial]
    return summary


def _selection_key(result, config, *, variant: str | None = None):
    """Select checkpoints on subject-aggregated validation metrics.

    Dual-shift variants use the designed composite source-validation score when
    eligible under the collapse guard:

        S = AUC + 0.25 * macroF1 - 0.10 * Brier - 0.10 * groupGap
    """
    from sklearn.metrics import recall_score

    eval_cfg = config.get("evaluation", {})
    aggregate = str(eval_cfg.get("aggregate", "subject_mean"))
    subject_metrics = compute_journal_metrics(
        result["logits"],
        result["labels"],
        result["environments"],
        input_type="logits",
        subject_ids=result["subjects"],
        aggregate=aggregate if aggregate != "none" else "subject_mean",
    )
    auc = float(subject_metrics["auc"])
    macro_f1 = float(subject_metrics.get("macro_f1") or 0.0)
    brier = float(subject_metrics.get("brier") or 0.0)
    group_gap = 0.0
    if subject_metrics.get("worst_group_auc") is not None and np.isfinite(
        subject_metrics.get("worst_group_auc")
    ):
        group_gap = max(0.0, auc - float(subject_metrics["worst_group_auc"]))
    composite = auc + 0.25 * macro_f1 - 0.10 * brier - 0.10 * group_gap
    if not np.isfinite(auc):
        composite = -float(result["loss"])
    score_tuple = (1, composite) if np.isfinite(auc) else (0, composite)

    ds = config.get("dual_shift") or {}
    guard = ds.get("collapse_guard") or {}
    apply_guard = bool(guard.get("enabled", True)) and (
        variant in DUAL_SHIFT_VARIANTS if variant is not None else False
    )
    use_composite = bool(ds.get("use_composite_selection", True)) and (
        variant in DUAL_SHIFT_VARIANTS if variant is not None else False
    )
    if not apply_guard:
        if use_composite:
            return (1, composite)
        return (1, auc) if np.isfinite(auc) else (0, -float(result["loss"]))

    min_sen = guard.get("min_sensitivity", 0.15)
    min_spe = guard.get("min_specificity", 0.15)
    min_recall = guard.get("min_class_recall", 0.15)
    eligible = True
    reasons = []
    sen = subject_metrics.get("sensitivity")
    spe = subject_metrics.get("specificity")
    recalls = None
    logits = np.asarray(result["logits"])
    labels = np.asarray(result["labels"])
    if logits.ndim == 2 and logits.shape[1] > 2:
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        if "subject_ids" in result:
            from training.journal_metrics import aggregate_subject_predictions

            probs, labels, _, _ = aggregate_subject_predictions(
                probs,
                labels,
                result["subjects"],
                result.get("environments"),
                input_type="probabilities",
            )
        predicted = probs.argmax(axis=1)
        recalls = recall_score(
            labels,
            predicted,
            labels=list(range(probs.shape[1])),
            average=None,
            zero_division=0,
        ).tolist()
        for index, value in enumerate(recalls):
            if float(value) < float(min_recall):
                eligible = False
                reasons.append(f"class{index}_recall {value:.4f} < {min_recall}")
    else:
        if sen is not None and np.isfinite(sen) and float(sen) < float(min_sen):
            eligible = False
            reasons.append(f"sensitivity {sen:.4f} < {min_sen}")
        if spe is not None and np.isfinite(spe) and float(spe) < float(min_spe):
            eligible = False
            reasons.append(f"specificity {spe:.4f} < {min_spe}")

    result["collapse_guard"] = {
        "eligible": eligible,
        "reasons": reasons,
        "metrics": {
            "sensitivity": sen,
            "specificity": spe,
            "recalls": recalls,
            "auc": auc,
            "macro_f1": macro_f1,
            "brier": brier,
            "group_gap": group_gap,
            "composite": composite,
        },
    }
    primary = composite if use_composite else auc
    return (1 if eligible else 0, primary if np.isfinite(primary) else -float(result["loss"]))


def _class_weights_from_labels(labels: np.ndarray, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def _train_variant(
    variant,
    config,
    train_loader,
    val_loader,
    test_loader,
    target_loader,
    output_dir,
    num_classes,
    num_groups,
    device,
    class_weights=None,
):
    # Fair comparison: every variant starts from the same RNG state.
    seed_everything(int(config["seed"]))
    spatial = variant in JOURNAL_SPATIAL_VARIANTS
    use_dro = variant in {"groupdro", "spatial_groupdro"}
    external = variant in EXTERNAL_FUSION_VARIANTS
    patch_gamma = variant in PATCH_GAMMA_VARIANTS
    dictionary = variant in DICTIONARY_VARIANTS
    dual_shift = variant in DUAL_SHIFT_VARIANTS
    metadata = variant in METADATA_VARIANTS
    model = _make_model(config, num_classes, variant).to(device)
    if dictionary:
        from Model.dictionary.journal_dual_dict import resolve_dictionary_preset

        dict_cfg = resolve_dictionary_preset(config.get("dictionary") or {}, variant)
    else:
        dict_cfg = {}
    if dual_shift:
        initialize_dual_shift_controllers(model, train_loader.dataset, config)
        parameters = model.parameters()
        lr = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"]["weight_decay"])
    elif metadata:
        # Fit acquisition vocab/stats on source train only (same rule as APIS).
        train_subset = train_loader.dataset
        acquisitions = []
        for index in train_subset.indices:
            record = train_subset.dataset.records[int(index)]
            acquisitions.append(record.get("acquisition") or {})
        if not any(acquisitions):
            raise RuntimeError("metadata baseline requires acquisition rows on train set")
        model.fit_acquisition_encoder(acquisitions)
        parameters = model.parameters()
        lr = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"]["weight_decay"])
    elif dictionary:
        ce_ckpt = Path(output_dir) / "ce_only" / "best_checkpoint.pt"
        if not ce_ckpt.exists():
            raise FileNotFoundError(
                f"Dictionary variant {variant} requires frozen Stage A ce_only "
                f"checkpoint at {ce_ckpt}"
            )
        checkpoint = torch.load(ce_ckpt, map_location=device, weights_only=False)
        model.load_backbone_state(checkpoint["model_state"])
        model.freeze_backbone()
        parameters = list(model.dictionary_parameters())
        lr = float(dict_cfg.get("lr", 1e-3))
        weight_decay = float(
            (config.get("dictionary") or {}).get(
                "weight_decay", config["training"]["weight_decay"]
            )
        )
    elif spatial or external or patch_gamma:
        parameters = model.parameters()
        lr = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"]["weight_decay"])
    else:
        for parameter in model.modulation.parameters():
            parameter.requires_grad_(False)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        lr = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
    dro = (
        GroupDRO(num_groups, step_size=float(config["groupdro"]["step_size"])).to(device)
        if use_dro
        else None
    )
    variant_dir = Path(output_dir) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "best_checkpoint.pt"
    best_key = None
    history = []
    for epoch in range(int(config["training"]["epochs"])):
        if dual_shift:
            train_result = run_dual_shift_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                config=config,
                class_weights=class_weights,
                epoch=epoch,
                collect_prototype_updates=True,
                variant=variant,
            )
            val_result = run_dual_shift_epoch(
                model,
                val_loader,
                device,
                config=config,
                class_weights=None,
                epoch=epoch,
                variant=variant,
            )
        else:
            train_result = _run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                dro=dro,
                config=config,
                spatial=spatial,
                class_weights=class_weights,
                variant=variant,
            )
            val_result = _run_epoch(
                model,
                val_loader,
                device,
                dro=None,
                config=config,
                spatial=spatial,
                class_weights=None,
                variant=variant,
            )
        key = _selection_key(val_result, config, variant=variant)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "val_auc": key[-1] if isinstance(key[-1], (int, float)) else None,
                "phase": train_result.get("phase"),
                "apis_coefficient_l2": train_result.get("apis_coefficient_l2"),
                "valid_intervention_frac": train_result.get("valid_intervention_frac"),
            }
        )
        print(
            f"[journal] {variant} epoch {epoch + 1}/{int(config['training']['epochs'])} "
            f"train_loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} "
            f"val_auc={history[-1]['val_auc']} "
            f"phase={train_result.get('phase')} "
            f"apis_l2={train_result.get('apis_coefficient_l2')} "
            f"valid_frac={train_result.get('valid_intervention_frac')}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            payload = {
                "epoch": epoch + 1,
                "variant": variant,
                "stage": VARIANT_STAGES[variant],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "groupdro_state": None if dro is None else dro.state_dict(),
                "selection": {
                    "val_score": key[-1] if key[0] else None,
                    "val_loss": val_result["loss"],
                    "seed": int(config["seed"]),
                    "collapse_guard": val_result.get("collapse_guard"),
                },
            }
            if dual_shift:
                payload["acquisition_encoder_extra"] = (
                    model.acquisition_encoder.to_state_dict_extra()
                )
                payload["prototype_bank"] = model.prototype_bank.state_dict()
                payload["cdt"] = model.cdt.state_dict()
            if metadata:
                payload["acquisition_encoder_extra"] = (
                    model.acquisition_encoder.to_state_dict_extra()
                )
            torch.save(payload, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if dual_shift or metadata:
        if checkpoint.get("acquisition_encoder_extra"):
            model.acquisition_encoder.load_state_dict_extra(
                checkpoint["acquisition_encoder_extra"]
            )
            model.acquisition_encoder.to(device)
    if dual_shift:
        if checkpoint.get("prototype_bank"):
            model.prototype_bank.load_state_dict(checkpoint["prototype_bank"])
        if checkpoint.get("cdt"):
            model.cdt.load_state_dict(checkpoint["cdt"])
    model.load_state_dict(checkpoint["model_state"])
    if dro is not None and checkpoint["groupdro_state"] is not None:
        dro.load_state_dict(checkpoint["groupdro_state"])
    if dual_shift:
        val_result = run_dual_shift_epoch(
            model, val_loader, device, config=config, epoch=0, variant=variant
        )
        test_result = run_dual_shift_epoch(
            model, test_loader, device, config=config, epoch=0, variant=variant
        )
        target_result = run_dual_shift_epoch(
            model, target_loader, device, config=config, epoch=0, variant=variant
        )
    else:
        val_result = _run_epoch(
            model,
            val_loader,
            device,
            dro=None,
            config=config,
            spatial=spatial,
            variant=variant,
        )
        test_result = _run_epoch(
            model,
            test_loader,
            device,
            dro=None,
            config=config,
            spatial=spatial,
            variant=variant,
        )
        target_result = _run_epoch(
            model,
            target_loader,
            device,
            dro=None,
            config=config,
            spatial=spatial,
            variant=variant,
        )
    eval_cfg = config.get("evaluation", {})
    aggregate = str(eval_cfg.get("aggregate", "subject_mean"))
    cluster = bool(eval_cfg.get("cluster_by_subject", True))
    reports = {}
    for split, result in (
        ("source_val", val_result),
        ("source_test", test_result),
        ("target", target_result),
    ):
        probabilities = _save_predictions(variant_dir / f"{split}_predictions.csv", result)
        scan_metrics = compute_journal_metrics(
            probabilities,
            result["labels"],
            result["environments"],
            input_type="probabilities",
            subject_ids=result["subjects"],
            aggregate="none",
        )
        subject_metrics = compute_journal_metrics(
            probabilities,
            result["labels"],
            result["environments"],
            input_type="probabilities",
            subject_ids=result["subjects"],
            aggregate=aggregate,
        )
        ci = bootstrap_confidence_intervals(
            probabilities,
            result["labels"],
            result["environments"],
            n_bootstrap=int(eval_cfg.get("bootstrap_samples", 1000)),
            random_state=int(config["seed"]),
            input_type="probabilities",
            subject_ids=result["subjects"],
            cluster_by_subject=cluster,
            aggregate=aggregate,
        )
        reports[split] = {
            "loss": result["loss"],
            "metrics_scan": scan_metrics,
            "metrics": subject_metrics,
            "bootstrap_ci": ci,
        }
        strengths = result.get("field_strengths")
        if strengths is not None and len(strengths) == len(result["labels"]):
            reports[split]["metrics_by_field_strength"] = (
                compute_metrics_by_field_strength(
                    probabilities,
                    result["labels"],
                    strengths,
                    result["environments"],
                    input_type="probabilities",
                    subject_ids=result["subjects"],
                    aggregate=aggregate,
                )
            )
    baseline_name = VARIANT_DISPLAY.get(variant, variant)
    if variant == "ce_only" and not bool(
        config.get("training", {}).get("class_weighted_ce", False)
    ):
        baseline_name = "unweighted_ce"
    reports.update(
        {
            "variant": variant,
            "variant_display_name": baseline_name,
            "class_weighted_ce": bool(
                config.get("training", {}).get("class_weighted_ce", False)
            ),
            "stage": VARIANT_STAGES[variant],
            "history": history,
            "gamma": (
                _gamma_summary(model)
                if spatial
                else (_patch_gamma_summary(model) if patch_gamma else None)
            ),
            "group_weights": None if dro is None else dro.group_weights.cpu().tolist(),
            "initialization_seed": int(config["seed"]),
        }
    )
    save_json_summary(variant_dir / "journal_metrics.json", reports)
    save_json_summary(variant_dir / "gamma_summary.json", reports["gamma"] or {})
    return reports


def _config_fingerprint(config: dict) -> str:
    def normalise(value):
        if isinstance(value, dict):
            return {str(key): normalise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        return value

    payload = json.dumps(normalise(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest(
    source,
    target,
    train_indices,
    val_indices,
    test_indices,
    direction,
    preprocessor,
    builder,
    config,
):
    def rows(dataset, indices, split, cohort):
        return [
            {
                "cohort": cohort,
                "split": split,
                "index": int(index),
                "subject_id": dataset.records[index]["subject_id"],
                "folder": dataset.records[index]["folder"],
                "label": int(dataset.records[index]["label"]),
            }
            for index in indices
        ]

    def folders(dataset, indices):
        return sorted(dataset.records[index]["folder"] for index in indices)

    source_name, target_name = direction.split("_to_")
    return {
        "direction": direction,
        "protocol": {
            "covariates": "cov3_age_sex_edu",
            "environment": f"env_{config.get('environments', {}).get('mode', 'age_sex')}",
            "metrics": "subject_mean",
            "baseline_ce_name": (
                "weighted_ce"
                if config.get("training", {}).get("class_weighted_ce", False)
                else "unweighted_ce"
            ),
        },
        "initialization_seed": int(config["seed"]),
        "label_mapping": dict(source.label_mapping),
        "preprocessing_version": str(
            config.get("data", {}).get("preprocessing_version", "journal_v1")
        ),
        "config_hash": _config_fingerprint(config),
        "source_train_subjects": sorted(set(source.subject_ids[train_indices].tolist())),
        "source_val_subjects": sorted(set(source.subject_ids[val_indices].tolist())),
        "source_test_subjects": sorted(set(source.subject_ids[test_indices].tolist())),
        "source_train_folders": folders(source, train_indices),
        "source_val_folders": folders(source, val_indices),
        "source_test_folders": folders(source, test_indices),
        "target_subjects": sorted(set(target.subject_ids.tolist())),
        "target_folders": folders(target, range(len(target))),
        "samples": rows(source, train_indices, "source_train", source_name)
        + rows(source, val_indices, "source_val", source_name)
        + rows(source, test_indices, "source_test", source_name)
        + rows(target, range(len(target)), "target", target_name),
        "covariate_preprocessor": preprocessor.to_dict(),
        "environment": builder.to_dict(),
        "dataset_fingerprint": {
            "source_n": int(len(source)),
            "target_n": int(len(target)),
            "source_subjects": int(len(set(source.subject_ids.tolist()))),
            "target_subjects": int(len(set(target.subject_ids.tolist()))),
        },
    }


def _read_prediction_table(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    probability_columns = sorted(
        [name for name in rows[0] if name.startswith("probability_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    keys = [(row["subject_id"], row["folder"]) for row in rows]
    probabilities = np.asarray(
        [[float(row[name]) for name in probability_columns] for row in rows],
        dtype=float,
    )
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    environments = np.asarray(
        [int(row["environment_id"]) for row in rows], dtype=np.int64
    )
    return keys, probabilities, labels, environments


def _paired_variant_comparisons(output_dir, variants, config):
    """Compare every trained variant with CE-only on identical target samples."""
    if "ce_only" not in variants:
        return {}
    baseline = _read_prediction_table(
        Path(output_dir) / "ce_only" / "target_predictions.csv"
    )
    eval_cfg = config.get("evaluation", {})
    comparisons = {}
    for variant in variants:
        if variant == "ce_only":
            continue
        candidate = _read_prediction_table(
            Path(output_dir) / variant / "target_predictions.csv"
        )
        if candidate[0] != baseline[0] or not np.array_equal(candidate[2], baseline[2]):
            raise ValueError(
                f"Cannot pair {variant} with ce_only: target sample order differs"
            )
        subjects = [key[0] for key in baseline[0]]
        metric_results = {}
        for metric in (
            "accuracy",
            "macro_f1",
            "auc",
            "brier",
            "ece",
            "worst_group_auc",
            "group_gap",
            "sensitivity",
            "specificity",
        ):
            metric_results[metric] = paired_bootstrap_difference(
                candidate[1],
                baseline[1],
                baseline[2],
                baseline[3],
                metric=metric,
                n_bootstrap=int(eval_cfg.get("bootstrap_samples", 1000)),
                random_state=int(config["seed"]),
                input_type="probabilities",
                subject_ids=subjects,
                cluster_by_subject=bool(eval_cfg.get("cluster_by_subject", True)),
                aggregate=str(eval_cfg.get("aggregate", "subject_mean")),
            )
        comparisons[f"{variant}_minus_ce_only"] = metric_results
    return comparisons


def _indices_from_manifest(dataset, subject_ids):
    subjects = set(map(str, subject_ids))
    indices = [
        index
        for index, subject in enumerate(dataset.subject_ids.tolist())
        if str(subject) in subjects
    ]
    if not indices:
        raise ValueError("Split manifest subjects do not match dataset")
    return np.asarray(indices, dtype=np.int64)


def _validate_frozen_folders(dataset, indices, expected_folders, *, split_name: str):
    if expected_folders is None:
        return
    actual = sorted(dataset.records[index]["folder"] for index in indices)
    expected = sorted(map(str, expected_folders))
    if actual != expected:
        raise ValueError(
            f"Frozen {split_name} folder list mismatch "
            f"(actual={len(actual)}, expected={len(expected)})"
        )


def _load_frozen_split(source, manifest_path: str):
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    train_indices = _indices_from_manifest(source, manifest["source_train_subjects"])
    val_indices = _indices_from_manifest(source, manifest["source_val_subjects"])
    test_indices = _indices_from_manifest(source, manifest["source_test_subjects"])
    assert_disjoint_subjects(
        source.subject_ids[train_indices],
        source.subject_ids[val_indices],
        source.subject_ids[test_indices],
    )
    _validate_frozen_folders(
        source, train_indices, manifest.get("source_train_folders"), split_name="train"
    )
    _validate_frozen_folders(
        source, val_indices, manifest.get("source_val_folders"), split_name="val"
    )
    _validate_frozen_folders(
        source, test_indices, manifest.get("source_test_folders"), split_name="test"
    )
    if "label_mapping" in manifest and dict(source.label_mapping) != {
        str(k) if not isinstance(k, (int, str)) else k: v
        for k, v in manifest["label_mapping"].items()
    }:
        # Compare with stringified keys for JSON round-trip safety.
        current = {str(k): int(v) for k, v in source.label_mapping.items()}
        frozen = {str(k): int(v) for k, v in manifest["label_mapping"].items()}
        if current != frozen:
            raise ValueError("Frozen label_mapping does not match dataset")
    return train_indices, val_indices, test_indices, manifest


def evaluate_stage_a_gate(ce_metrics: dict, groupdro_metrics: dict, gate_cfg: dict) -> dict:
    """GroupDRO vs weighted/unweighted CE gate on subject-level target metrics."""
    ce = ce_metrics["target"]["metrics"]
    gd = groupdro_metrics["target"]["metrics"]
    checks = {
        "auc_ok": (ce["auc"] - gd["auc"])
        <= float(gate_cfg.get("max_auc_drop", 0.01)),
        "worst_group_ok": gd["worst_group_auc"]
        >= ce["worst_group_auc"] - float(gate_cfg.get("worst_group_tolerance", 0.0)),
        "macro_f1_ok": (ce["macro_f1"] - gd["macro_f1"])
        <= float(gate_cfg.get("max_macro_f1_drop", 0.03)),
        "calibration_ok": (
            (ce["ece"] - gd["ece"])
            >= float(gate_cfg.get("min_ece_improvement", 0.005))
            or (ce["brier"] - gd["brier"])
            >= float(gate_cfg.get("min_brier_improvement", 0.005))
        ),
    }
    return {
        "stage": "A",
        "baseline_display_name": ce_metrics.get("variant_display_name", "weighted_ce"),
        "candidate": "groupdro",
        "checks": checks,
        "passed": bool(all(checks.values())),
        "delta": {
            key: float(gd[key] - ce[key])
            for key in ("auc", "macro_f1", "ece", "brier", "worst_group_auc")
        },
    }


def evaluate_stage_b_gate(ce_metrics: dict, spatial_metrics: dict, gate_cfg: dict) -> dict:
    """Spatial non-inferiority vs CE on subject-level target metrics."""
    ce = ce_metrics["target"]["metrics"]
    sp = spatial_metrics["target"]["metrics"]
    checks = {
        "auc_ok": (ce["auc"] - sp["auc"])
        <= float(gate_cfg.get("max_auc_drop", 0.01)),
        "worst_group_ok": sp["worst_group_auc"]
        >= ce["worst_group_auc"] - float(gate_cfg.get("worst_group_tolerance", 0.0)),
        "macro_f1_ok": (ce["macro_f1"] - sp["macro_f1"])
        <= float(gate_cfg.get("max_macro_f1_drop", 0.03)),
        "calibration_ok": (
            sp["ece"] <= ce["ece"] + float(gate_cfg.get("max_ece_worsen", 0.01))
            or sp["brier"] <= ce["brier"] + float(gate_cfg.get("max_brier_worsen", 0.01))
        ),
    }
    return {
        "stage": "B",
        "baseline_display_name": ce_metrics.get("variant_display_name", "weighted_ce"),
        "candidate": "spatial",
        "checks": checks,
        "passed": bool(all(checks.values())),
        "delta": {
            key: float(sp[key] - ce[key])
            for key in ("auc", "macro_f1", "ece", "brier", "worst_group_auc")
        },
    }


def run(
    config: dict,
    direction: str,
    output_dir: str,
    variants=None,
    device="cuda",
    *,
    skip_existing: bool = True,
    split_manifest: str | None = None,
):
    seed_everything(int(config["seed"]))
    source_name, target_name = direction.split("_to_")
    source, target = _dataset(config, source_name), _dataset(config, target_name)
    train_cfg = config["training"]
    train_ratio = float(train_cfg.get("train_ratio", 0.6))
    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    test_ratio = float(train_cfg.get("test_ratio", 0.2))
    claim_cfg = config.get("claim") or {}
    holdout_subjects: set[str] = set()
    holdout_path = claim_cfg.get("exclude_subjects_json")
    holdout_key = str(claim_cfg.get("exclude_subjects_key", "subjects_le_30d"))
    if holdout_path:
        holdout_subjects = load_holdout_subjects(holdout_path, key=holdout_key)
    frozen_manifest = None
    if split_manifest:
        train_indices, val_indices, test_indices, frozen_manifest = _load_frozen_split(
            source, split_manifest
        )
    else:
        train_indices, val_indices, test_indices = _split_source(
            source, train_ratio, val_ratio, test_ratio, int(config["seed"])
        )
    if holdout_subjects:
        # Mechanism subjects must not enter train/val/selection; drop from all
        # source splits. Target cohort may still contain them for E3 eval only.
        train_indices = np.asarray(
            filter_indices_excluding_subjects(
                source.subject_ids, train_indices, holdout_subjects
            ),
            dtype=int,
        )
        val_indices = np.asarray(
            filter_indices_excluding_subjects(
                source.subject_ids, val_indices, holdout_subjects
            ),
            dtype=int,
        )
        test_indices = np.asarray(
            filter_indices_excluding_subjects(
                source.subject_ids, test_indices, holdout_subjects
            ),
            dtype=int,
        )
        for name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ):
            assert_no_holdout_leak(
                source.subject_ids, indices, holdout_subjects, split_name=name
            )
            if len(indices) == 0:
                raise RuntimeError(f"hold-out exclusion emptied source {name} split")
    assert_disjoint_subjects(
        source.subject_ids[train_indices],
        source.subject_ids[val_indices],
        source.subject_ids[test_indices],
    )
    preprocessor, builder, train_env, val_env, test_env, target_env = _fit_protocol(
        source,
        train_indices,
        val_indices,
        test_indices,
        target,
        config["environments"],
        frozen_manifest=frozen_manifest,
    )
    train_set = JournalSubset(source, train_indices, preprocessor, train_env)
    val_set = JournalSubset(source, val_indices, preprocessor, val_env)
    test_set = JournalSubset(source, test_indices, preprocessor, test_env)
    target_set = JournalSubset(target, np.arange(len(target)), preprocessor, target_env)
    batch_size = int(config["training"]["batch_size"])
    if batch_size < 2:
        raise ValueError("training.batch_size must be >= 2 for BatchNorm stability")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {"num_workers": int(config["training"]["num_workers"])}
    requested = [
        VARIANT_ALIASES.get(str(item), str(item))
        for item in list(variants or config.get("variants") or [])
    ]
    use_subject_balance = bool(
        (config.get("dual_shift") or {}).get("subject_balanced_sampler", True)
    ) and any(v in DUAL_SHIFT_VARIANTS or v == "ce_only" for v in requested)
    # Apply subject-balanced sampling for DualShift screening runs (incl. fair CE).
    train_sampler = None
    if use_subject_balance:
        subject_ids = [
            str(source.subject_ids[int(index)]) for index in train_set.indices
        ]
        counts = {}
        for subject_id in subject_ids:
            counts[subject_id] = counts.get(subject_id, 0) + 1
        weights = [1.0 / counts[subject_id] for subject_id in subject_ids]
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator if train_sampler is None else None,
        drop_last=len(train_set) >= batch_size,
        collate_fn=journal_collate,
        **common,
    )
    eval_batch = int(config["training"]["eval_batch_size"])
    val_loader = DataLoader(
        val_set, batch_size=eval_batch, collate_fn=journal_collate, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch, collate_fn=journal_collate, **common
    )
    target_loader = DataLoader(
        target_set, batch_size=eval_batch, collate_fn=journal_collate, **common
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest = _manifest(
        source,
        target,
        train_indices,
        val_indices,
        test_indices,
        direction,
        preprocessor,
        builder,
        config,
    )
    manifest["split_ratios"] = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    manifest["frozen_split_manifest"] = split_manifest
    manifest["claim_holdout"] = {
        "enabled": bool(holdout_subjects),
        "path": holdout_path,
        "key": holdout_key if holdout_path else None,
        "n_excluded_subjects": int(len(holdout_subjects)),
        "excluded_subjects": sorted(holdout_subjects),
    }
    manifest["match_audit"] = {
        "source": getattr(source, "match_audit", {}),
        "target": getattr(target, "match_audit", {}),
    }
    save_json_summary(Path(output_dir) / "split_manifest.json", manifest)
    save_json_summary(
        Path(output_dir) / "metadata_match_audit.json",
        manifest["match_audit"],
    )
    selected = requested
    unknown = set(selected).difference(VARIANT_STAGES)
    if unknown:
        raise ValueError(f"Unknown journal variants: {sorted(unknown)}")
    resolved_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    num_classes = len(set(source.label_mapping.values()))
    class_weights = None
    if bool(train_cfg.get("class_weighted_ce", False)):
        class_weights = _class_weights_from_labels(
            source.labels[train_indices], num_classes
        )
    results = {}
    for variant in selected:
        metrics_path = Path(output_dir) / variant / "journal_metrics.json"
        if skip_existing and metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as handle:
                results[variant] = json.load(handle)
            print(f"[journal] skip existing variant: {variant}", flush=True)
            continue
        results[variant] = _train_variant(
            variant,
            config,
            train_loader,
            val_loader,
            test_loader,
            target_loader,
            output_dir,
            num_classes,
            len(builder.environment_to_id_),
            resolved_device,
            class_weights=class_weights,
        )
    comparisons = _paired_variant_comparisons(output_dir, selected, config)
    save_json_summary(Path(output_dir) / "paired_comparisons.json", comparisons)
    save_json_summary(Path(output_dir) / "summary.json", results)
    gate_cfg = config.get("study", {}).get("gate", {})
    if "ce_only" in results and "groupdro" in results:
        stage_a_gate = evaluate_stage_a_gate(
            results["ce_only"], results["groupdro"], gate_cfg
        )
        save_json_summary(Path(output_dir) / "stage_a_gate.json", stage_a_gate)
    if "ce_only" in results and "spatial" in results:
        stage_b_gate = evaluate_stage_b_gate(
            results["ce_only"], results["spatial"], gate_cfg
        )
        save_json_summary(Path(output_dir) / "stage_b_gate.json", stage_b_gate)
    return results


def _finite_mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "mean": float(values.mean()) if len(values) else float("nan"),
        "std": float(values.std()) if len(values) else float("nan"),
        "n": int(len(values)),
    }


def run_study(config, directions, seeds, output_dir, variants=None, device="cuda"):
    """Run the pre-registered multi-seed, bidirectional external study."""
    selected = variants or config["variants"]
    records = []
    for direction in directions:
        for seed in seeds:
            seeded_config = copy.deepcopy(config)
            seeded_config["seed"] = int(seed)
            run_dir = Path(output_dir) / direction / f"seed_{seed}"
            result = run(
                seeded_config,
                direction,
                str(run_dir),
                variants=selected,
                device=device,
            )
            for variant, report in result.items():
                metrics = report["target"]["metrics"]
                records.append(
                    {
                        "direction": direction,
                        "seed": int(seed),
                        "variant": variant,
                        **{
                            name: metrics[name]
                            for name in (
                                "accuracy",
                                "macro_f1",
                                "auc",
                                "brier",
                                "ece",
                                "worst_group_auc",
                                "group_gap",
                            )
                        },
                    }
                )

    aggregates = {}
    for direction in directions:
        aggregates[direction] = {}
        for variant in selected:
            subset = [
                record
                for record in records
                if record["direction"] == direction and record["variant"] == variant
            ]
            aggregates[direction][variant] = {
                metric: _finite_mean_std([record[metric] for record in subset])
                for metric in (
                    "accuracy",
                    "macro_f1",
                    "auc",
                    "brier",
                    "ece",
                    "worst_group_auc",
                    "group_gap",
                )
            }

    gate = {"evaluated": False, "pass": False, "directions": {}}
    candidate = "spatial_groupdro"
    if "ce_only" in selected and candidate in selected:
        gate["evaluated"] = True
        thresholds = config.get("study", {}).get("gate", {})
        max_auc_drop = float(thresholds.get("max_auc_drop", 0.01))
        ece_threshold = float(thresholds.get("min_ece_improvement", 0.0))
        brier_threshold = float(thresholds.get("min_brier_improvement", 0.0))
        worst_tolerance = float(thresholds.get("worst_group_tolerance", 0.0))
        max_f1_drop = float(thresholds.get("max_macro_f1_drop", 0.03))
        direction_passes = []
        for direction in directions:
            baseline = aggregates[direction]["ce_only"]
            proposed = aggregates[direction][candidate]
            auc_delta = proposed["auc"]["mean"] - baseline["auc"]["mean"]
            ece_improvement = baseline["ece"]["mean"] - proposed["ece"]["mean"]
            brier_improvement = baseline["brier"]["mean"] - proposed["brier"]["mean"]
            worst_delta = (
                proposed["worst_group_auc"]["mean"]
                - baseline["worst_group_auc"]["mean"]
            )
            f1_delta = proposed["macro_f1"]["mean"] - baseline["macro_f1"]["mean"]
            passed = (
                auc_delta >= -max_auc_drop
                and worst_delta >= -worst_tolerance
                and f1_delta >= -max_f1_drop
                and (
                    ece_improvement >= ece_threshold
                    or brier_improvement >= brier_threshold
                )
            )
            gate["directions"][direction] = {
                "auc_delta": auc_delta,
                "ece_improvement": ece_improvement,
                "brier_improvement": brier_improvement,
                "worst_group_auc_delta": worst_delta,
                "macro_f1_delta": f1_delta,
                "pass": bool(passed),
            }
            direction_passes.append(bool(passed))
        gate["pass"] = any(direction_passes)

    summary = {
        "directions": list(directions),
        "seeds": [int(seed) for seed in seeds],
        "variants": list(selected),
        "records": records,
        "aggregates": aggregates,
        "stage_c_gate": gate,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json_summary(Path(output_dir) / "study_summary.json", summary)
    return summary


def _make_smoke_data(root: str, config: dict):
    rng = np.random.default_rng(12)
    manifest_root = Path(root) / "scan_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for cohort, count in (("ADNI", 12), ("NACC", 8)):
        cohort_root = Path(root) / cohort
        cohort_root.mkdir(parents=True)
        rows = []
        manifest_rows = []
        spec = config["cohorts"][cohort]
        for index in range(count):
            subject = f"{cohort}{index:04d}"
            date = f"2020_01_{index + 1:02d}"
            # Match local folder convention: 1=CN, 3=AD.
            label = 1 if index % 2 == 0 else 3
            folder_name = f"{subject}-{date}-{label}"
            folder = cohort_root / folder_name
            folder.mkdir()
            image = rng.normal(
                (0 if label == 1 else 1) * 0.5, 1, size=(16, 16, 16)
            ).astype(np.float32)
            image_path = folder / spec["image_filename"]
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            rows.append(
                {
                    spec["columns"]["id"][0]: subject,
                    spec["columns"]["date"][0]: date.replace("_", "-"),
                    spec["columns"]["sex"][0]: "F" if index % 2 == 0 else "M",
                    spec["columns"]["age"][0]: 60 + index,
                    # Diagnosis matches folder label for audit path.
                    "DIAGNOSIS": label,
                    "label": label,
                }
            )
            manufacturer = "Siemens" if index % 2 == 0 else "GE"
            field_strength = 3.0 if cohort == "NACC" or index % 3 else 1.5
            tr_raw = 2300.0 if field_strength >= 2.5 else 8.9
            manifest_rows.append(
                {
                    "dataset": cohort,
                    "subject_id": subject,
                    "scan_date": date.replace("_", "-"),
                    "pre_folder": folder_name,
                    "pre_label": label,
                    "csv_visdate": date.replace("_", "-"),
                    "csv_label": label,
                    "csv_match_days": 0,
                    "csv_matched": True,
                    "age": 60 + index,
                    "sex": "female" if index % 2 == 0 else "male",
                    "education": 12 + (index % 5),
                    "age_missing": 0,
                    "sex_missing": 0,
                    "education_missing": 0,
                    "manufacturer": manufacturer,
                    "scanner_model": "Skyra" if manufacturer == "Siemens" else "SIGNA",
                    "field_strength": field_strength,
                    "sequence_family": "MPRAGE",
                    "tr_raw": tr_raw,
                    "tr_mode": "inversion_cycle" if tr_raw >= 100 else "short_cycle",
                    "te_ms": 2.98,
                    "ti_ms": 900.0,
                    "flip_angle": 9.0,
                    "slice_thickness": 1.0,
                    "pixel_spacing_x": 1.0,
                    "pixel_spacing_y": 1.0,
                    "acceleration": "Accelerated",
                    "site_id": 1 + (index % 3),
                    "metadata_match_quality": "exact_image",
                    "source_image_id": f"{cohort}_{index}",
                    "image_aligned": True,
                    "image_path": str(image_path),
                }
            )
        csv_path = Path(root) / f"{cohort}.csv"
        import pandas as pd

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        manifest_csv = manifest_root / f"{cohort}_scan_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
        spec["image_root"] = str(cohort_root)
        spec["metadata_csv"] = str(csv_path)
    config["scan_manifest"] = {
        "enabled": True,
        "root": str(manifest_root),
        "files": {
            "ADNI": "ADNI_scan_manifest.csv",
            "NACC": "NACC_scan_manifest.csv",
        },
    }
    config["training"].update(
        {
            "epochs": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "image_shape": [32, 32, 32],
        }
    )
    config["model"].update(
        {"name": "journal_resnet10", "base_channels": 2, "spatial_shape": [1, 1, 1]}
    )
    # Dual-shift warmups must fit into the single smoke epoch.
    config.setdefault("dual_shift", {})
    config["dual_shift"].update(
        {
            "warm_clean_epochs": 0,
            "warm_apis_epochs": 0,
            "alpha_max": 0.25,
            "prototype_min_subjects": 1,
            "collapse_guard": {"enabled": False},
        }
    )
    config["environments"].update(
        {
            "n_age_bins": 1,
            "n_education_bins": 1,
            "min_group_size": 1,
            "min_per_class": 1,
            "mode": "age_sex",
        }
    )
    config["metadata_match"] = {
        "max_days": 90,
        "require_diagnosis_match": True,
        "on_diagnosis_mismatch": "exclude",
    }
    config["evaluation"]["bootstrap_samples"] = 10
    config["evaluation"]["aggregate"] = "subject_mean"
    config["evaluation"]["cluster_by_subject"] = True
    config["training"]["class_weighted_ce"] = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="config/journal.yaml")
    parser.add_argument(
        "--direction",
        choices=["ADNI_to_NACC", "NACC_to_ADNI"],
        default="ADNI_to_NACC",
    )
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANT_STAGES))
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--study", action="store_true")
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["ADNI_to_NACC", "NACC_to_ADNI"],
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--force-variants",
        action="store_true",
        help="Retrain variants even when journal_metrics.json already exists",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Freeze source split/subjects from an existing Stage A manifest",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.seed is not None:
        config["seed"] = args.seed
    output_dir = args.output_dir or os.path.join(
        config["output_root"],
        f"{args.direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    skip_existing = not args.force_variants
    if args.smoke_test:
        with tempfile.TemporaryDirectory(prefix="journal_smoke_") as temporary:
            _make_smoke_data(temporary, config)
            run(
                config,
                args.direction,
                output_dir,
                variants=args.variants,
                device="cpu",
                skip_existing=False,
            )
    elif args.study:
        study_config = config.get("study", {})
        run_study(
            config,
            args.directions
            or study_config.get(
                "directions", ["ADNI_to_NACC", "NACC_to_ADNI"]
            ),
            args.seeds or study_config.get("seeds", [42, 43, 44, 45, 46]),
            output_dir,
            variants=args.variants,
            device=args.device,
        )
    else:
        run(
            config,
            args.direction,
            output_dir,
            variants=args.variants,
            device=args.device,
            skip_existing=skip_existing,
            split_manifest=args.split_manifest,
        )
    print(f"[journal] completed: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
