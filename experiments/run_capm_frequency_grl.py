"""Run the DS-040 CAPM-conditioned frequency/GRL audit.

The default ``--smoke-test`` path executes the complete P0 -> projector ->
variant workflow on deterministic synthetic volumes.  Real-data execution
uses the existing scan-filtered journal dataset and source-validation
selection.  Target adaptation batches are image/subject-only views; labels
are touched only by the final exploratory target report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Model.ablation.capm_frequency_grl import (
    CAPMFrequencyGRL3D,
    TaskSupportProjector,
    compute_capm_frequency_losses,
    make_frequency_batch,
)
from training.fmm_protocol import FMMDatasetView, split_target_indices, subject_digest


VARIANTS = ("P0", "F0", "F1", "F2", "F3", "R1", "R2", "R3")
VARIANT_FLAGS = {
    "P0": {"frequency": False, "domain_grl": False, "intensity_grl": False, "grl_mode": "full"},
    "F0": {"frequency": True, "domain_grl": False, "intensity_grl": False, "grl_mode": "full"},
    "F1": {"frequency": True, "domain_grl": True, "intensity_grl": False, "grl_mode": "full"},
    "F2": {"frequency": True, "domain_grl": False, "intensity_grl": True, "grl_mode": "full"},
    "F3": {"frequency": True, "domain_grl": True, "intensity_grl": True, "grl_mode": "full"},
    "R1": {"frequency": True, "domain_grl": True, "intensity_grl": False, "grl_mode": "residual"},
    "R2": {"frequency": True, "domain_grl": False, "intensity_grl": True, "grl_mode": "residual"},
    "R3": {"frequency": True, "domain_grl": True, "intensity_grl": True, "grl_mode": "residual"},
}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ImageOnlyView(Dataset):
    """Expose only image and subject ID from a journal dataset."""

    def __init__(self, dataset: Any, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset.records[int(self.indices[index])]
        return {"image": self.dataset._load_image(record["path"]), "subject_id": record["subject_id"]}


class SyntheticDataset(Dataset):
    def __init__(self, images: Tensor, labels: Tensor | None, prefix: str):
        self.images = images.float().contiguous()
        self.labels = None if labels is None else labels.long().contiguous()
        self.subject_ids = np.asarray([f"{prefix}{i:03d}" for i in range(len(images))], dtype=object)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {"image": self.images[index], "subject_id": str(self.subject_ids[index])}
        if self.labels is not None:
            item["label"] = self.labels[index]
            item["covariates"] = torch.tensor([65.0 + float(index % 4), float(index % 2), 12.0 + float(index % 5)])
        return item


def _loader(dataset: Dataset, batch_size: int, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
    if len(dataset) < 1:
        raise ValueError("cannot create a loader for an empty split")
    # Avoid a singleton training batch at the final ResNet stage, where
    # BatchNorm3d has no within-batch variance.  Never drop an entire split.
    effective_drop_last = bool(drop_last and len(dataset) > int(batch_size))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=0, drop_last=effective_drop_last)


def _clone_model(flags: Mapping[str, Any], projector: TaskSupportProjector | None, *, input_shape: tuple[int, int, int], layers: tuple[int, int, int, int], grl_coefficient: float) -> CAPMFrequencyGRL3D:
    return CAPMFrequencyGRL3D(
        projector=projector,
        grl_mode=str(flags["grl_mode"]),
        domain_grl=bool(flags["domain_grl"]),
        intensity_grl=bool(flags["intensity_grl"]),
        input_shape=input_shape,
        layers=layers,
        grl_coefficient=grl_coefficient,
        classifier_dropout=0.0,
    )


def _module_grad_norm(module: nn.Module) -> float:
    gradients = [parameter.grad.detach().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(gradients).norm().cpu()) if gradients else 0.0


@torch.no_grad()
def _evaluate(model: CAPMFrequencyGRL3D, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    was_training = model.training
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    subjects: list[str] = []
    try:
        for batch in loader:
            if "label" not in batch or "covariates" not in batch:
                raise AssertionError("evaluation loader must explicitly expose labels and covariates")
            logits = model(batch["image"].to(device), batch["covariates"].to(device))
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(batch["label"].long().cpu().tolist())
            subjects.extend(map(str, batch["subject_id"]))
    finally:
        model.train(was_training)
    predictions = (np.asarray(probabilities) >= 0.5).astype(np.int64)
    target = np.asarray(labels, dtype=np.int64)
    result: dict[str, Any] = {
        "n_scans": int(len(target)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "accuracy": float(np.mean(predictions == target)),
        "auc": float(roc_auc_score(target, probabilities)) if len(np.unique(target)) == 2 else None,
    }
    result["predictions"] = [{"subject_id": subject, "label": int(label), "probability": float(probability)} for subject, label, probability in zip(subjects, labels, probabilities)]
    return result


@torch.no_grad()
def _fit_projector(model: CAPMFrequencyGRL3D, loader: DataLoader, device: torch.device, rank: int) -> TaskSupportProjector:
    model.eval()
    pooled: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in loader:
        if "label" not in batch or "covariates" not in batch:
            raise AssertionError("projector fitting requires source labels only")
        _, features = model(batch["image"].to(device), batch["covariates"].to(device), return_features=True)
        pooled.append(features.mean(dim=(-3, -2, -1)).cpu())
        labels.append(batch["label"].long().cpu())
    return TaskSupportProjector.fit_from_pooled_features(torch.cat(pooled), torch.cat(labels), model.classifier, rank=rank, metadata={"source_subject_count": len({str(s) for batch in loader for s in batch["subject_id"]})})


def _train_source_epoch(model: CAPMFrequencyGRL3D, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    values: list[float] = []
    for batch in loader:
        image, table, labels = batch["image"].to(device), batch["covariates"].to(device), batch["label"].long().to(device)
        logits = model(image, table)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values))


def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_p0_with_source_selection(model: CAPMFrequencyGRL3D, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int) -> list[dict[str, float]]:
    """Train P0 and select only by source validation balanced accuracy."""
    best_state: dict[str, Tensor] | None = None
    best_score = -float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        train_loss = _train_source_epoch(model, train_loader, optimizer, device)
        validation = _evaluate(model, val_loader, device)
        score = float(validation["balanced_accuracy"])
        history.append({"epoch": float(epoch), "train_loss": train_loss, "source_val_balanced_accuracy": score})
        if score > best_score:
            best_score = score
            best_state = _snapshot(model)
    if best_state is None:
        raise RuntimeError("source-only P0 training produced no checkpoint")
    model.load_state_dict(best_state)
    return history


def _train_variant_with_source_selection(model: CAPMFrequencyGRL3D, train_loader: DataLoader, val_loader: DataLoader, target_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, *, epochs: int, seed: int, weights: Mapping[str, float]) -> list[dict[str, float]]:
    """Train a frequency variant while selecting checkpoints on source val only."""
    best_state: dict[str, Tensor] | None = None
    best_score = -float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        metrics = _train_frequency_epoch(model, train_loader, target_loader, optimizer, device, generator=torch.Generator(device=device).manual_seed(int(seed) + epoch), weights=weights)
        validation = _evaluate(model, val_loader, device)
        score = float(validation["balanced_accuracy"])
        history.append({"epoch": float(epoch), **metrics, "source_val_balanced_accuracy": score})
        if score > best_score:
            best_score = score
            best_state = _snapshot(model)
    if best_state is None:
        raise RuntimeError("frequency variant training produced no checkpoint")
    model.load_state_dict(best_state)
    return history


def _train_frequency_epoch(model: CAPMFrequencyGRL3D, source_loader: DataLoader, target_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, *, generator: torch.Generator, weights: Mapping[str, float]) -> dict[str, float]:
    model.train()
    target_iterator = iter(target_loader)
    totals: dict[str, list[float]] = {name: [] for name in ("total", "classification", "domain", "intensity", "attention", "anchor", "domain_accuracy", "intensity_accuracy", "domain_auc", "intensity_auc", "adversarial_feature_rms", "full_feature_rms", "full_feature_mean_shift", "adversarial_feature_mean_shift", "full_feature_mmd_proxy", "adversarial_feature_mmd_proxy", "source_amplitude_mean", "target_style_amplitude_mean", "encoder_grad_norm", "discriminator_grad_norm")}
    for source_batch in source_loader:
        try:
            target_batch = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_batch = next(target_iterator)
        n = min(len(source_batch["image"]), len(target_batch["image"]))
        if n < 1:
            continue
        source_image = source_batch["image"][:n].to(device)
        target_image = target_batch["image"][:n].to(device)
        table = source_batch["covariates"][:n].to(device)
        labels = source_batch["label"][:n].long().to(device)
        generated = make_frequency_batch(source_image, target_image, generator=generator)
        loss, parts = compute_capm_frequency_losses(model, generated, table, labels, lambda_domain=weights["domain"], lambda_intensity=weights["intensity"], lambda_attention=weights["attention"], lambda_anchor=weights["anchor"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        totals["encoder_grad_norm"].append(_module_grad_norm(model.backbone))
        discriminator_gradients = []
        if model.domain_grl:
            discriminator_gradients.extend(model.domain_discriminator.parameters())
        if model.intensity_grl:
            discriminator_gradients.extend(model.intensity_discriminator.parameters())
        discriminator_gradients = [parameter.grad.detach().norm() for parameter in discriminator_gradients if parameter.grad is not None]
        totals["discriminator_grad_norm"].append(float(torch.stack(discriminator_gradients).norm().cpu()) if discriminator_gradients else 0.0)
        optimizer.step()
        totals["total"].append(float(loss.detach().cpu()))
        for name, value in parts.items():
            if name in totals:
                totals[name].append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) if values else 0.0 for name, values in totals.items()}


def _synthetic_loaders(seed: int, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(int(seed))
    source_images = torch.randn(12, 1, 32, 32, 32, generator=generator)
    source_labels = torch.tensor([0, 1] * 6)
    target_images = torch.randn(8, 1, 32, 32, 32, generator=generator) + 0.15
    source_train = SyntheticDataset(source_images[:8], source_labels[:8], "s-train-")
    source_val = SyntheticDataset(source_images[8:10], source_labels[8:10], "s-val-")
    source_test = SyntheticDataset(source_images[10:], source_labels[10:], "s-test-")
    target_adapt = SyntheticDataset(target_images[:4], None, "t-adapt-")
    target_test = SyntheticDataset(target_images[4:], torch.tensor([0, 1, 0, 1]), "t-test-")
    return (_loader(source_train, batch_size, shuffle=True, drop_last=True), _loader(source_val, batch_size, shuffle=False), _loader(source_test, batch_size, shuffle=False), _loader(target_adapt, batch_size, shuffle=True, drop_last=True)), _loader(target_test, batch_size, shuffle=False)


def run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], CAPMFrequencyGRL3D]:
    seed_everything(args.seed)
    (source_train, source_val, source_test, target_adapt), target_test = _synthetic_loaders(args.seed, args.batch_size)
    device = torch.device("cpu")
    input_shape = (32, 32, 32)
    layers = (1, 1, 1, 1)
    p0 = _clone_model(VARIANT_FLAGS["P0"], None, input_shape=input_shape, layers=layers, grl_coefficient=args.grl_coefficient).to(device)
    optimizer = torch.optim.Adam(p0.parameters(), lr=args.learning_rate)
    history = _train_p0_with_source_selection(p0, source_train, source_val, optimizer, device, args.epochs)
    projector = _fit_projector(p0, source_train, device, args.projector_rank)
    flags = VARIANT_FLAGS[args.variant]
    model = _clone_model(flags, projector, input_shape=input_shape, layers=layers, grl_coefficient=args.grl_coefficient).to(device)
    model.load_state_dict(p0.state_dict(), strict=False)
    variant_history = []
    if args.variant == "P0":
        selected = p0
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        variant_history.extend(_train_variant_with_source_selection(model, source_train, source_val, target_adapt, optimizer, device, epochs=args.epochs, seed=args.seed, weights={"domain": 1.0 if flags["domain_grl"] else 0.0, "intensity": 1.0 if flags["intensity_grl"] else 0.0, "attention": 1.0, "anchor": 0.1}))
        selected = model
    report = {
        "schema": "dualshift_ds040_smoke_report_v1",
        "status": "smoke_only",
        "variant": args.variant,
        "seed": args.seed,
        "model_signature": selected.experiment_signature(),
        "p0_history": history,
        "variant_history": variant_history,
        "source_val": _evaluate(selected, source_val, device),
        "source_test": _evaluate(selected, source_test, device),
        "target_test": _evaluate(selected, target_test, device),
        "projector": projector.to_dict(),
        "target_labels_used_for_adaptation": False,
        "target_labels_used_for_selection": False,
        "target_metrics_read_for_final_report": True,
    }
    return report, selected


def _build_real_loaders(config: dict[str, Any], direction: str, source_split: str | None, target_adapt_ratio: float, seed: int, batch_size: int):
    from experiments.train_journal import _dataset, _fit_protocol, _load_frozen_split, journal_collate
    source_name, target_name = direction.split("_to_", maxsplit=1)
    source, target = _dataset(config, source_name), _dataset(config, target_name)
    if source_split:
        train_indices, val_indices, test_indices, manifest = _load_frozen_split(source, source_split)
    else:
        from training.fmm_protocol import split_source_indices
        ratios = config.get("training", {})
        train_indices, val_indices, test_indices = split_source_indices(source, float(ratios.get("train_ratio", .6)), float(ratios.get("val_ratio", .2)), float(ratios.get("test_ratio", .2)), seed)
        manifest = None
    adapt_indices, target_test_indices = split_target_indices(target, target_adapt_ratio, seed)
    env_cfg = config.get("environments") or {"mode": "age_sex", "n_age_bins": 3, "n_education_bins": 3, "min_group_size": 8, "min_per_class": 2, "scale_continuous": True}
    # Only the disjoint final holdout is allowed to expose target covariates;
    # T_adapt remains an image/subject-only view below.
    preprocessor, _builder, train_env, val_env, test_env, _target_env = _fit_protocol(source, train_indices, val_indices, test_indices, target, env_cfg, frozen_manifest=manifest, target_indices=target_test_indices)
    # This call intentionally constructs only the source table for adaptation;
    # target covariates are not requested by the image-only view.
    from data.journal_dataset import JournalSubset
    source_train = JournalSubset(source, train_indices, preprocessor, train_env)
    source_val = JournalSubset(source, val_indices, preprocessor, val_env)
    source_test = JournalSubset(source, test_indices, preprocessor, test_env)
    target_adapt = ImageOnlyView(target, adapt_indices)
    target_test = JournalSubset(target, target_test_indices, preprocessor, np.zeros(len(target_test_indices), dtype=np.int64))
    collate = journal_collate
    kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": 0, "collate_fn": collate}
    return _loader(source_train, batch_size, shuffle=True, drop_last=True), _loader(source_val, batch_size, shuffle=False), _loader(source_test, batch_size, shuffle=False), _loader(target_adapt, batch_size, shuffle=True, drop_last=True), _loader(target_test, batch_size, shuffle=False), {"source": source, "target": target, "train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices, "target_adapt_indices": adapt_indices, "target_test_indices": target_test_indices}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="R3")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--direction", choices=("ADNI_to_NACC", "NACC_to_ADNI"), default="ADNI_to_NACC")
    parser.add_argument("--source-split", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/ds040_capm_frequency_grl/report.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--projector-rank", type=int, default=32)
    parser.add_argument("--grl-coefficient", type=float, default=1.0)
    parser.add_argument("--target-adapt-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch-size and epochs must be positive")
    if args.smoke_test:
        report, selected = run_smoke(args)
    else:
        if args.config is None:
            parser.error("--config is required without --smoke-test")
        with args.config.open(encoding="utf-8") as handle:
            config = __import__("yaml").safe_load(handle)
        seed_everything(args.seed)
        train_loader, val_loader, test_loader, target_adapt_loader, target_test_loader, inventory = _build_real_loaders(config, args.direction, str(args.source_split) if args.source_split else None, args.target_adapt_ratio, args.seed, args.batch_size)
        input_shape = tuple(int(v) for v in (config.get("training") or {}).get("image_shape", (160, 196, 160)))
        layers = tuple(int(v) for v in (config.get("scale_table_ablation") or {}).get("layers", (1, 1, 1, 1)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Real execution currently follows the same bounded pilot as smoke;
        # the per-variant loop is intentionally explicit for auditability.
        p0 = _clone_model(VARIANT_FLAGS["P0"], None, input_shape=input_shape, layers=layers, grl_coefficient=args.grl_coefficient).to(device)
        opt = torch.optim.Adam(p0.parameters(), lr=args.learning_rate)
        p0_history = _train_p0_with_source_selection(p0, train_loader, val_loader, opt, device, args.epochs)
        projector = _fit_projector(p0, train_loader, device, args.projector_rank)
        flags = VARIANT_FLAGS[args.variant]
        selected = p0
        variant_history: list[dict[str, float]] = []
        if args.variant != "P0":
            selected = _clone_model(flags, projector, input_shape=input_shape, layers=layers, grl_coefficient=args.grl_coefficient).to(device)
            selected.load_state_dict(p0.state_dict(), strict=False)
            opt = torch.optim.Adam(selected.parameters(), lr=args.learning_rate)
            variant_history = _train_variant_with_source_selection(selected, train_loader, val_loader, target_adapt_loader, opt, device, epochs=args.epochs, seed=args.seed, weights={"domain": float(flags["domain_grl"]), "intensity": float(flags["intensity_grl"]), "attention": 1.0, "anchor": 0.1})
        report = {"schema": "dualshift_ds040_report_v1", "status": "completed_code_path", "variant": args.variant, "direction": args.direction, "seed": args.seed, "p0_history": p0_history, "variant_history": variant_history, "source_val": _evaluate(selected, val_loader, device), "source_test": _evaluate(selected, test_loader, device), "target_test": _evaluate(selected, target_test_loader, device), "target_labels_used_for_adaptation": False, "target_labels_used_for_selection": False, "target_metrics_read_for_final_report": True, "source_subject_digest": subject_digest(inventory["source"].subject_ids[inventory["train_indices"]]), "target_adapt_subject_digest": subject_digest(inventory["target"].subject_ids[inventory["target_adapt_indices"]]), "target_test_subject_digest": subject_digest(inventory["target"].subject_ids[inventory["target_test_indices"]]), "model_signature": selected.experiment_signature(), "projector": projector.to_dict()}
    report["variant_flags"] = dict(VARIANT_FLAGS[args.variant])
    report["selection_source_validation_only"] = True
    report["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    torch.save({"model_state": selected.state_dict(), "report_schema": report["schema"]}, args.output.with_name("best.pt"))
    audit = {
        "schema": "dualshift_ds040_audit_v1",
        "variant": args.variant,
        "seed": int(args.seed),
        "variant_flags": dict(VARIANT_FLAGS[args.variant]),
        "target_labels_used_for_adaptation": False,
        "target_labels_used_for_selection": False,
        "target_metrics_read_for_final_report": True,
        "selection_source_validation_only": True,
        "code_sha256": report["code_sha256"],
        "model_signature": selected.experiment_signature(),
    }
    args.output.with_name("audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.output.with_name("predictions.json").write_text(json.dumps({"source_val": report["source_val"].get("predictions", []), "source_test": report["source_test"].get("predictions", []), "target_test": report["target_test"].get("predictions", [])}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variant": report["variant"], "source_val_balanced_accuracy": report["source_val"]["balanced_accuracy"], "target_test_balanced_accuracy": report["target_test"]["balanced_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
