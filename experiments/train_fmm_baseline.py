"""Run the independent FMM raw-volume UDA baseline and its controls.

The runner has two data paths:

* real data: the existing scan-filtered journal dataset factory;
* ``--smoke-test``: deterministic synthetic volumes for a full one-epoch
  source/target training and audit check.

Target adaptation batches are represented by a label-blind dataset view. The
target holdout is evaluated only after checkpoint selection on source
validation, and never participates in optimization or selection.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.ablation.fmm_baseline import FMMNet, GradientReversal
from data.journal_dataset import build_journal_dataset
from training.fmm_frequency import (
    intensity_transform,
    mix_amplitude,
    sample_lambda,
    spectral_diagnostics,
)
from training.fmm_protocol import (
    FMMDatasetView,
    SyntheticFMMDataset,
    assert_disjoint_subjects,
    split_source_indices,
    split_target_indices,
    subject_digest,
)


UPSTREAM_REPOSITORY = "https://github.com/ku-milab/FMM"
UPSTREAM_COMMIT = "580625cee5bfc1474fe8700e530ade07ac5e9776"
VARIANTS = ("b0_ref", "b1_fmm", "b1a_no_source_fft", "b1b_no_attention", "b1c_no_grl", "g0_no_grl", "g1_domain_only", "g2_intensity_only", "g3_both_grl")
DS038_VARIANTS = ("g0_no_grl", "g1_domain_only", "g2_intensity_only", "g3_both_grl")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_hash(payload: object) -> str:
    def normalise(value):
        if isinstance(value, Mapping):
            return {str(key): normalise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        return value

    encoded = json.dumps(normalise(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_file_hash(dataset) -> str | None:
    """Hash the manifest/metadata file when a real journal dataset exposes one."""
    candidates = [getattr(dataset, "manifest_csv", None), getattr(dataset, "metadata_csv", None)]
    for candidate in candidates:
        if candidate and os.path.isfile(str(candidate)):
            digest = hashlib.sha256()
            with open(candidate, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return None


def _variant_flags(variant: str) -> dict[str, bool]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown FMM variant {variant!r}; choose from {VARIANTS}")
    if variant in DS038_VARIANTS:
        return {
            "source_stage": True,
            "inter_stage": True,
            "source_fft": True,
            "attention": True,
            "domain_grl": variant in {"g1_domain_only", "g3_both_grl"},
            "intensity_grl": variant in {"g2_intensity_only", "g3_both_grl"},
        }
    return {
        "source_stage": variant != "b0_ref",
        "inter_stage": variant != "b0_ref",
        "source_fft": variant not in {"b0_ref", "b1a_no_source_fft"},
        "attention": variant not in {"b1b_no_attention", "b0_ref"},
        "grl": variant not in {"b1c_no_grl", "b0_ref"},
    }



def _head_diagnostics(logits: torch.Tensor, labels: torch.Tensor, *, gradient_norm: float, coefficient: float) -> dict[str, float | None]:
    probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
    target = labels.detach().cpu().numpy().astype(np.int64)
    predicted = (probabilities >= 0.5).astype(np.int64)
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits.detach(), labels.detach()).cpu()),
        "accuracy": float(np.mean(predicted == target)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "auc": float(roc_auc_score(target, probabilities)) if len(np.unique(target)) == 2 else None,
        "gradient_norm": float(gradient_norm),
        "grl_coefficient": float(coefficient),
    }


def _grl_gradient_probe(*, coefficient: float = 1.0) -> dict[str, float]:
    feature_with = torch.tensor([[1.0, -2.0]], requires_grad=True)
    feature_without = feature_with.detach().clone().requires_grad_(True)
    discriminator_with = nn.Linear(2, 1, bias=False)
    discriminator_without = nn.Linear(2, 1, bias=False)
    discriminator_without.load_state_dict(discriminator_with.state_dict())
    labels = torch.ones(1, 1)
    loss_with = F.binary_cross_entropy_with_logits(
        GradientReversal(float(coefficient))(feature_with) @ discriminator_with.weight.t(), labels
    )
    loss_without = F.binary_cross_entropy_with_logits(
        feature_without @ discriminator_without.weight.t(), labels
    )
    loss_with.backward(); loss_without.backward()
    return {
        "encoder_gradient_with_grl": float(feature_with.grad[0, 0]),
        "encoder_gradient_without_grl": float(feature_without.grad[0, 0]),
        "discriminator_gradient_with_grl": float(discriminator_with.weight.grad[0, 0]),
        "discriminator_gradient_without_grl": float(discriminator_without.weight.grad[0, 0]),
    }


def _mechanism_record(head: str, *, epoch: int, step: int, is_best_checkpoint: bool, metrics: dict) -> dict:
    return {"head": str(head), "epoch": int(epoch), "step": int(step), "is_best_checkpoint": bool(is_best_checkpoint), "metrics": metrics}


def _module_grad_norm(module: nn.Module) -> float:
    values = [parameter.grad.detach().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm().cpu()) if values else 0.0


def _frozen_feature_domain_probe(source: torch.Tensor, target: torch.Tensor, *, seed: int = 42, epochs: int = 25) -> dict[str, float]:
    source = source.detach().float().cpu()
    target = target.detach().float().cpu()
    features = torch.cat((source, target), dim=0)
    labels = torch.cat((torch.zeros(len(source)), torch.ones(len(target))))
    order = torch.randperm(len(features), generator=torch.Generator().manual_seed(int(seed)))
    probe = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    for _ in range(int(epochs)):
        logits = probe(features[order]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, labels[order])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(probe(features)).squeeze(1).numpy()
    target_labels = labels.numpy().astype(np.int64)
    predicted = (probabilities >= 0.5).astype(np.int64)
    source_mean, target_mean = source.mean(0), target.mean(0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target_labels, predicted)),
        "auc": float(roc_auc_score(target_labels, probabilities)),
        "mmd": float((source_mean - target_mean).pow(2).mean()),
        "source_norm": float(source.norm(dim=1).mean()),
        "target_norm": float(target.norm(dim=1).mean()),
    }


def _required_artifacts_complete(output: Path) -> bool:
    return all((output / name).is_file() for name in ("best.pt", "summary.json", "audit.json", "config.yaml", "predictions.json"))


def _make_loader(dataset, batch_size: int, *, shuffle: bool, workers: int, drop_last: bool = False):
    if len(dataset) == 0:
        raise ValueError("cannot create a loader for an empty split")
    if drop_last and len(dataset) < batch_size:
        drop_last = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        drop_last=bool(drop_last),
        pin_memory=False,
    )


def _classification_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor | None:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    if np.any(counts == 0):
        return None
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _subject_mean_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    if not rows:
        return {"n_subjects": 0, "balanced_accuracy": None, "accuracy": None, "auc": None}
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject_id"]), []).append(row)
    labels, probabilities = [], []
    for subject_id in sorted(by_subject):
        samples = by_subject[subject_id]
        labels.append(int(round(np.mean([int(item["label"]) for item in samples]))))
        probabilities.append(float(np.mean([float(item["probability"]) for item in samples])))
    predictions = (np.asarray(probabilities) >= 0.5).astype(np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    result: dict[str, float | int | None] = {
        "n_subjects": int(len(labels_array)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "accuracy": float(np.mean(predictions == labels_array)),
        "auc": None,
    }
    if len(np.unique(labels_array)) == 2:
        result["auc"] = float(roc_auc_score(labels_array, probabilities))
    return result


@torch.no_grad()
def evaluate(model: FMMNet, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    for batch in loader:
        if "label" not in batch:
            raise AssertionError("evaluation loader must explicitly expose labels")
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)["logits"]
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch["label"].cpu().numpy()
        subjects = batch["subject_id"]
        for subject, label, probability in zip(subjects, labels, probabilities):
            rows.append({"subject_id": str(subject), "label": int(label), "probability": float(probability)})
    return _subject_mean_metrics(rows), rows


def _next_target(target_iterator, target_loader):
    try:
        return next(target_iterator), target_iterator
    except StopIteration:
        target_iterator = iter(target_loader)
        return next(target_iterator), target_iterator


def _batch_images(batch: Mapping, device: torch.device) -> torch.Tensor:
    images = batch["image"].to(device, non_blocking=True).float()
    if images.ndim != 5:
        raise ValueError(f"expected image batch [B,1,D,H,W], got {tuple(images.shape)}")
    return images


def _train_epoch(
    model: FMMNet,
    source_loader: DataLoader,
    target_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    flags: Mapping[str, bool],
    config: Mapping,
    criterion: nn.Module,
    rng: torch.Generator,
    sampled_target_subjects: set[str],
    diagnostics: dict,
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    losses = {"total": [], "classification": [], "domain": [], "attention": [], "intensity": []}
    fmm_cfg = config.get("fmm", {}) or {}
    domain_enabled = flags.get("domain_grl", flags.get("grl", False))
    intensity_enabled = flags.get("intensity_grl", flags.get("grl", False))
    lambda_grl = float(fmm_cfg.get("grl_coefficient", 1.0)) if (domain_enabled or intensity_enabled) else 0.0
    lambda_domain = float(fmm_cfg.get("lambda_domain", 1.0)) if domain_enabled else 0.0
    lambda_attention = float(fmm_cfg.get("lambda_attention", 1.0)) if flags["attention"] else 0.0
    lambda_intensity = float(fmm_cfg.get("lambda_intensity", 1.0)) if intensity_enabled else 0.0
    diagnostics.setdefault("domain", [])
    diagnostics.setdefault("intensity", [])
    intensity_cfg = fmm_cfg.get("intensity_transform", {}) or {}
    target_iterator = iter(target_loader) if target_loader is not None else None
    for step, source_batch in enumerate(source_loader):
        source_images = _batch_images(source_batch, device)
        labels = source_batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        total = source_images.new_zeros(())
        cls_loss = source_images.new_zeros(())
        domain_loss = source_images.new_zeros(())
        attention_loss = source_images.new_zeros(())
        intensity_loss = source_images.new_zeros(())

        if flags["source_stage"]:
            transformed = intensity_transform(
                source_images,
                scale_range=intensity_cfg.get("scale_range", (0.8, 1.2)),
                noise_std=float(intensity_cfg.get("noise_std", 0.05)),
                generator=rng,
            )
            source_invariant = (
                mix_amplitude(transformed, source_images, 1.0, phase_source=True)
                if flags["source_fft"]
                else transformed
            )
            out_source = model(source_images, return_attention=flags["attention"])
            out_invariant = model(source_invariant, return_attention=flags["attention"])
            cls_loss = 0.5 * (criterion(out_source["logits"], labels) + criterion(out_invariant["logits"], labels))
            if intensity_enabled:
                intensity_features = torch.cat((out_source["features"], out_invariant["features"]), dim=0)
                intensity_features.retain_grad()
                intensity_labels = torch.cat((torch.zeros(len(labels), device=device), torch.ones(len(labels), device=device)))
                intensity_logits = model.domain_logits(intensity_features, coefficient=lambda_grl, head="intensity")
                intensity_loss = F.binary_cross_entropy_with_logits(intensity_logits, intensity_labels)
                diagnostics["intensity"].append(_mechanism_record("intensity", epoch=epoch, step=step, is_best_checkpoint=False, metrics=_head_diagnostics(intensity_logits, intensity_labels, gradient_norm=0.0, coefficient=lambda_grl)) | {"features": intensity_features})
            total = cls_loss + lambda_intensity * intensity_loss
            diagnostics.setdefault("source_stage", spectral_diagnostics(source_invariant[:1].detach()))
        else:
            out_source = model(source_images, return_attention=False)
            cls_loss = criterion(out_source["logits"], labels)
            total = cls_loss

        if flags["inter_stage"]:
            if target_iterator is None or target_loader is None:
                raise RuntimeError("FMM inter-domain stage requires a target adaptation loader")
            target_batch, target_iterator = _next_target(target_iterator, target_loader)
            if "label" in target_batch:
                raise AssertionError("target adaptation batch exposed diagnosis labels")
            sampled_target_subjects.update(map(str, target_batch["subject_id"]))
            target_images = _batch_images(target_batch, device)
            batch_size = min(source_images.shape[0], target_images.shape[0])
            if batch_size <= 0:
                continue
            source_for_inter = source_images[:batch_size]
            source_out = model(source_for_inter, return_attention=flags["attention"])
            target_images = target_images[:batch_size]
            mixing = sample_lambda(
                batch_size,
                low=float(fmm_cfg.get("lambda_low", 0.0)),
                high=float(fmm_cfg.get("lambda_high", 1.0)),
                device=device,
                generator=rng,
            )
            target_style = mix_amplitude(source_for_inter, target_images, mixing)
            target_out = model(target_style, return_attention=flags["attention"])
            if domain_enabled:
                domain_features = torch.cat((source_out["features"], target_out["features"]), dim=0)
                domain_features.retain_grad()
                domain_labels = torch.cat((torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)))
                domain_logits = model.domain_logits(domain_features, coefficient=lambda_grl, head="domain")
                domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
                diagnostics["domain"].append(_mechanism_record("domain", epoch=epoch, step=step, is_best_checkpoint=False, metrics=_head_diagnostics(domain_logits, domain_labels, gradient_norm=0.0, coefficient=lambda_grl)) | {"features": domain_features})
            if flags["attention"]:
                attention_loss = F.mse_loss(source_out["attention"], target_out["attention"])
            total = total + lambda_domain * domain_loss + lambda_attention * attention_loss
            diagnostics.setdefault("inter_stage", spectral_diagnostics(target_style[:1].detach()))
        total.backward()
        for head_name in ("domain", "intensity"):
            if diagnostics.get(head_name):
                entry = diagnostics[head_name][-1]
                features = entry.pop("features", None)
                if features is not None and features.grad is not None:
                    entry["metrics"]["shared_encoder_feature_gradient_norm"] = float(features.grad.detach().norm().cpu())
                head_module = model.domain_discriminator if head_name == "domain" else model.intensity_discriminator
                entry["metrics"]["discriminator_parameter_gradient_norm"] = _module_grad_norm(head_module)
        optimizer.step()
        losses["total"].append(float(total.detach().cpu()))
        losses["classification"].append(float(cls_loss.detach().cpu()))
        losses["domain"].append(float(domain_loss.detach().cpu()))
        losses["attention"].append(float(attention_loss.detach().cpu()))
        losses["intensity"].append(float(intensity_loss.detach().cpu()))
    if not losses["total"]:
        raise RuntimeError("training produced no optimizer steps")
    return {key: float(np.mean(values)) for key, values in losses.items()}


def _synthetic_data(config: dict):
    shape = tuple(config.get("training", {}).get("image_shape", (32, 32, 32)))
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    source_images = torch.randn((12, 1, *shape), generator=generator)
    source_labels = [index % 2 for index in range(12)]
    target_images = torch.randn((8, 1, *shape), generator=generator) + 0.15
    target_labels = [index % 2 for index in range(8)]
    return SyntheticFMMDataset(source_images, source_labels, "S"), SyntheticFMMDataset(target_images, target_labels, "T")


def run(config: dict, direction: str, output_dir: str, *, variant: str, device: str, smoke_test: bool = False) -> dict:
    seed_everything(int(config.get("seed", 42)))
    flags = _variant_flags(variant)
    source_name, target_name = direction.split("_to_")
    if smoke_test:
        source, target = _synthetic_data(config)
    else:
        source = build_journal_dataset(config, source_name)
        target = build_journal_dataset(config, target_name)
    assert_disjoint_subjects(source.subject_ids, target.subject_ids)
    train_cfg = config.get("training", {}) or {}
    train_indices, val_indices, test_indices = split_source_indices(
        source,
        float(train_cfg.get("train_ratio", 0.6)),
        float(train_cfg.get("val_ratio", 0.2)),
        float(train_cfg.get("test_ratio", 0.2)),
        int(config.get("split_seed", config.get("seed", 42))),
    )
    target_cfg = config.get("target_split", {}) or {}
    target_adapt_indices, target_test_indices = split_target_indices(
        target,
        float(target_cfg.get("adapt_ratio", 0.5)),
        int(target_cfg.get("seed", config.get("split_seed", config.get("seed", 42)))),
    )
    source_train = FMMDatasetView(source, train_indices, include_label=True)
    source_val = FMMDatasetView(source, val_indices, include_label=True)
    source_test = FMMDatasetView(source, test_indices, include_label=True)
    target_adapt = FMMDatasetView(target, target_adapt_indices, include_label=False)
    target_test = FMMDatasetView(target, target_test_indices, include_label=True)
    assert_disjoint_subjects(source_train.subject_ids, source_val.subject_ids, source_test.subject_ids)
    assert_disjoint_subjects(target_adapt.subject_ids, target_test.subject_ids)

    device_obj = torch.device(device)
    model_cfg = config.get("model", {}) or {}
    model = FMMNet(
        channels=model_cfg.get("channels", (8, 8, 16, 16, 32, 32, 64, 64, 128, 128)),
        pool_shape=model_cfg.get("pool_shape", (2, 3, 2)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        classifier_hidden=int(model_cfg.get("classifier_hidden", 64)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    ).to(device_obj)
    labels = np.asarray(source.labels)[train_indices]
    class_weights = _classification_weights(labels, device_obj) if bool(train_cfg.get("class_weighted_ce", True)) else None
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    batch_size = int(train_cfg.get("batch_size", 4))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    workers = int(train_cfg.get("num_workers", 0))
    source_loader = _make_loader(source_train, batch_size, shuffle=True, workers=workers, drop_last=True)
    target_loader = None if variant == "b0_ref" else _make_loader(target_adapt, batch_size, shuffle=True, workers=workers, drop_last=True)
    val_loader = _make_loader(source_val, eval_batch_size, shuffle=False, workers=workers)
    source_test_loader = _make_loader(source_test, eval_batch_size, shuffle=False, workers=workers)
    target_test_loader = _make_loader(target_test, eval_batch_size, shuffle=False, workers=workers)
    rng = torch.Generator(device=device_obj).manual_seed(int(config.get("seed", 42)) + 101)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sampled_target_subjects: set[str] = set()
    diagnostics: dict = {}
    history = []
    best_score = -float("inf")
    best_epoch = -1
    best_path = output / "best.pt"
    epochs = int(train_cfg.get("epochs", 1))
    for epoch in range(epochs):
        losses = _train_epoch(
            model,
            source_loader,
            target_loader,
            optimizer,
            device_obj,
            flags=flags,
            config=config,
            criterion=criterion,
            rng=rng,
            sampled_target_subjects=sampled_target_subjects,
            diagnostics=diagnostics,
            epoch=epoch,
        )
        val_metrics, _ = evaluate(model, val_loader, device_obj)
        record = {"epoch": epoch, "losses": losses, "source_val": val_metrics}
        history.append(record)
        score = val_metrics["balanced_accuracy"]
        if score is not None and float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": variant}, best_path)
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    for head_name in ("domain", "intensity"):
        for record in diagnostics.get(head_name, []):
            record["is_best_checkpoint"] = record["epoch"] == best_epoch
    @torch.no_grad()
    def _loader_features(loader):
        if loader is None:
            return torch.empty((0, model.feature_dim))
        model.eval(); values = []
        for batch in loader:
            values.append(model.encode(_batch_images(batch, device_obj))[0].cpu())
        return torch.cat(values) if values else torch.empty((0, model.feature_dim))
    source_features = _loader_features(source_loader)
    target_features = _loader_features(target_loader)
    frozen_probe = _frozen_feature_domain_probe(source_features, target_features, seed=int(config.get("seed", 42))) if len(target_features) else None
    source_val_metrics, source_val_rows = evaluate(model, val_loader, device_obj)
    source_test_metrics, source_test_rows = evaluate(model, source_test_loader, device_obj)
    target_test_metrics, target_test_rows = evaluate(model, target_test_loader, device_obj)
    summary = {
        "variant": variant,
        "direction": direction,
        "best_epoch": int(best_epoch),
        "source_val": source_val_metrics,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "history": history,
        "mechanism_diagnostics": {key: diagnostics.get(key, []) for key in ("domain", "intensity")},
        "frozen_feature_domain_probe": frozen_probe,
    }
    audit = {
        "git_commit": _git_commit(),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "config_hash": _json_hash(config),
        "seed": int(config.get("seed", 42)),
        "split_seed": int(config.get("split_seed", config.get("seed", 42))),
        "dataset_file_hashes": {"source": _dataset_file_hash(source), "target": _dataset_file_hash(target)},
        "variant_flags": flags,
        "target_label_access_during_training": False,
        "target_labels_used_for_selection": False,
        "target_test_evaluation_after_fit": True,
        "source_subject_digests": {
            "train": subject_digest(source_train.subject_ids),
            "val": subject_digest(source_val.subject_ids),
            "test": subject_digest(source_test.subject_ids),
        },
        "target_subject_digests": {
            "adapt": subject_digest(target_adapt.subject_ids),
            "test": subject_digest(target_test.subject_ids),
        },
        "sampled_target_subject_ids": sorted(sampled_target_subjects),
        "spectral_diagnostics": {key: value for key, value in diagnostics.items() if key not in {"domain", "intensity"}},
        "mechanism_diagnostics": {key: diagnostics.get(key, []) for key in ("domain", "intensity")},
        "frozen_feature_domain_probe": frozen_probe,
        "split_sizes": {
            "source_train": int(len(source_train)),
            "source_val": int(len(source_val)),
            "source_test": int(len(source_test)),
            "target_adapt": int(len(target_adapt)),
            "target_test": int(len(target_test)),
        },
    }
    (output / "status.json").write_text(json.dumps({"status": "writing", "variant": variant, "seed": int(config.get("seed", 42))}, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "predictions.json").write_text(
        json.dumps({"source_val": source_val_rows, "source_test": source_test_rows, "target_test": target_test_rows}, indent=2),
        encoding="utf-8",
    )
    if not _required_artifacts_complete(output):
        raise RuntimeError("run completed without all required DS-038 artifacts")
    (output / "status.json").write_text(json.dumps({"status": "complete", "variant": variant, "seed": int(config.get("seed", 42)), "best_epoch": int(best_epoch)}, indent=2), encoding="utf-8")
    return {"summary": summary, "audit": audit}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="fmm_baseline_scan_filtered_1p5t_mci_ad.yaml")
    parser.add_argument("--direction", choices=["ADNI_to_NACC", "NACC_to_ADNI"], default="ADNI_to_NACC")
    parser.add_argument("--variant", choices=VARIANTS, default="b1_fmm")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.smoke_test:
        config.setdefault("training", {}).update({"epochs": 1, "batch_size": 2, "eval_batch_size": 2, "image_shape": [32, 32, 32]})
        config.setdefault("model", {}).update({"channels": [2, 2, 4, 4, 8, 8, 16, 16, 32, 32], "pool_shape": [1, 1, 1], "dropout": 0.0})
    output_dir = args.output_dir or os.path.join(
        config.get("output_root", "outputs/fmm_baseline"),
        f"{args.direction}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    result = run(config, args.direction, output_dir, variant=args.variant, device="cpu" if args.smoke_test else args.device, smoke_test=args.smoke_test)
    print(json.dumps({"output_dir": str(output_dir), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
