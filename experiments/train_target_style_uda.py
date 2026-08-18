"""Run the target-style feature transport + CAPM UDA probe.

The script is intentionally separate from ``train_journal.py``.  It creates a
subject-disjoint ``T_adapt``/``T_test`` split, consumes only unlabeled target
images during adaptation, and selects checkpoints using source validation
balanced accuracy.  It is an experimental baseline for the multi-seed audit,
not a new method claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.journal_dataset import JournalSubset
from experiments.train_journal import (
    _class_weights_from_labels,
    _dataset,
    _fit_protocol,
    _split_source,
    journal_collate,
    seed_everything,
)
from Model.ablation.scale_table_transformer import build_scale_table_ablation
from Model.ablation.target_style_transport import TargetStyleCAPM
from training.journal_metrics import compute_journal_metrics, save_json_summary
from utils.journal_protocol import assert_disjoint_subjects


VARIANTS = ("capm", "target_style_capm")


class _UnlabeledTargetView(Dataset):
    """Expose target images only, preventing accidental label/table use."""

    def __init__(self, subset: JournalSubset):
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        return {"image": self.subset[index]["image"]}


def _target_subject_split(dataset, adapt_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < adapt_ratio < 1.0:
        raise ValueError("target_split.adapt_ratio must lie strictly between zero and one")
    subjects = np.unique(dataset.subject_ids.astype(str))
    if len(subjects) < 2:
        raise ValueError("target dataset needs at least two subjects for T_adapt/T_test")
    rng = np.random.default_rng(int(seed))
    ordered = subjects[rng.permutation(len(subjects))]
    n_adapt = int(round(len(ordered) * float(adapt_ratio)))
    n_adapt = min(max(n_adapt, 1), len(ordered) - 1)
    adapt_subjects = set(ordered[:n_adapt].tolist())
    adapt = np.flatnonzero(np.isin(dataset.subject_ids.astype(str), list(adapt_subjects)))
    test = np.flatnonzero(~np.isin(dataset.subject_ids.astype(str), list(adapt_subjects)))
    return adapt.astype(np.int64), test.astype(np.int64)


def _make_model(config: dict, num_classes: int, device: torch.device) -> TargetStyleCAPM:
    model_cfg = config.get("model") or {}
    ablation_cfg = config.get("scale_table_ablation") or {}
    input_shape = tuple(
        int(value)
        for value in ablation_cfg.get(
            "input_shape", (config.get("training") or {}).get("image_shape", (128, 128, 128))
        )
    )
    backbone = build_scale_table_ablation(
        preset=str(ablation_cfg.get("preset", "layer4_pixel")),
        interaction="capm",
        num_classes=num_classes,
        layers=tuple(int(value) for value in ablation_cfg.get("layers", (1, 1, 1, 1))),
        spatial_shape=tuple(int(value) for value in ablation_cfg.get("spatial_shape", (4, 4, 4))),
        transformer_dim=int(ablation_cfg.get("transformer_dim", 128)),
        num_heads=int(ablation_cfg.get("num_heads", 4)),
        transformer_dropout=float(ablation_cfg.get("transformer_dropout", 0.1)),
        classifier_dropout=float(ablation_cfg.get("classifier_dropout", model_cfg.get("dropout", 0.3))),
        gate_init=float(ablation_cfg.get("gate_init", 0.95)),
        input_shape=input_shape,
    )
    return TargetStyleCAPM(
        backbone,
        transport_strength=float((config.get("target_style_transport") or {}).get("strength", 0.5)),
        force_capm=True,
    ).to(device)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _evaluate(model: TargetStyleCAPM, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    logits, labels, environments, subjects, folders = [], [], [], [], []
    for raw in loader:
        batch = _batch_to_device(raw, device)
        output = model.predict(batch["image"], batch["covariates"])
        logits.append(output.cpu())
        labels.append(batch["label"].cpu())
        environments.append(batch["environment_id"].cpu())
        subjects.extend(raw["subject_id"])
        folders.extend(raw["folder"])
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy()
    env_np = torch.cat(environments).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    metrics = compute_journal_metrics(
        probabilities,
        labels_np,
        env_np,
        input_type="probabilities",
        subject_ids=subjects,
        aggregate="subject_mean",
        folders=folders,
    )
    return {"metrics": metrics, "logits": logits_np, "labels": labels_np}


def _loss(output: dict, labels: torch.Tensor, class_weights: torch.Tensor | None, cfg: dict, variant: str) -> torch.Tensor:
    clean = F.cross_entropy(output["clean_logits"], labels, weight=class_weights)
    if variant == "capm":
        return clean
    mixed = F.cross_entropy(output["mixed_logits"], labels, weight=class_weights)
    clean_prob = torch.softmax(output["clean_logits"].detach(), dim=1)
    consistency = F.kl_div(
        F.log_softmax(output["mixed_logits"], dim=1),
        clean_prob,
        reduction="batchmean",
    )
    transport_cfg = cfg.get("target_style_transport") or {}
    return (
        clean
        + float(transport_cfg.get("lambda_mixed_ce", 1.0)) * mixed
        + float(transport_cfg.get("lambda_consistency", 0.1)) * consistency
    )


def _run_one(config: dict, direction: str, output_dir: Path, variant: str, device_name: str) -> dict:
    seed_everything(int(config["seed"]))
    source_name, target_name = direction.split("_to_")
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    training_cfg = config["training"]
    train_idx, val_idx, source_test_idx = _split_source(
        source,
        float(training_cfg.get("train_ratio", 0.6)),
        float(training_cfg.get("val_ratio", 0.2)),
        float(training_cfg.get("test_ratio", 0.2)),
        int(config.get("split_seed", config["seed"])),
    )
    target_adapt_idx, target_test_idx = _target_subject_split(
        target,
        float((config.get("target_split") or {}).get("adapt_ratio", 0.5)),
        int((config.get("target_split") or {}).get("seed", config["seed"])),
    )
    assert_disjoint_subjects(
        source.subject_ids[train_idx], source.subject_ids[val_idx], source.subject_ids[source_test_idx]
    )
    target_ordered = np.concatenate([target_adapt_idx, target_test_idx])
    preprocessor, builder, train_env, val_env, source_test_env, target_env = _fit_protocol(
        source,
        train_idx,
        val_idx,
        source_test_idx,
        target,
        config["environments"],
        target_indices=target_ordered,
    )
    train_set = JournalSubset(source, train_idx, preprocessor, train_env)
    val_set = JournalSubset(source, val_idx, preprocessor, val_env)
    source_test_set = JournalSubset(source, source_test_idx, preprocessor, source_test_env)
    target_adapt_set = JournalSubset(
        target, target_adapt_idx, preprocessor, target_env[: len(target_adapt_idx)]
    )
    target_test_set = JournalSubset(
        target, target_test_idx, preprocessor, target_env[len(target_adapt_idx) :]
    )
    batch_size = int(training_cfg.get("batch_size", 2))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    common = {"num_workers": int(training_cfg.get("num_workers", 0)), "collate_fn": journal_collate}
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_set) >= batch_size,
        **common,
    )
    target_loader = DataLoader(
        _UnlabeledTargetView(target_adapt_set),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(target_adapt_set) >= batch_size,
        **common,
    )
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, **common)
    source_test_loader = DataLoader(source_test_set, batch_size=eval_batch_size, shuffle=False, **common)
    target_test_loader = DataLoader(target_test_set, batch_size=eval_batch_size, shuffle=False, **common)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = _make_model(config, len(source.label_mapping), device)
    class_weights = None
    if bool(training_cfg.get("class_weighted_ce", True)):
        class_weights = torch.as_tensor(
            _class_weights_from_labels(source.labels[train_idx], len(source.label_mapping)),
            dtype=torch.float32,
            device=device,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_epoch = 0
    history = []
    target_iter = None if variant == "capm" else iter(target_loader)
    epochs = int(training_cfg.get("epochs", 50))
    for epoch in range(epochs):
        model.train()
        losses = []
        amplitudes = []
        for raw_source in train_loader:
            source_batch = _batch_to_device(raw_source, device)
            target_image = None
            if target_iter is not None:
                try:
                    raw_target = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    raw_target = next(target_iter)
                target_image = raw_target["image"].to(device)
                if target_image.shape[0] != source_batch["image"].shape[0]:
                    repeats = int(np.ceil(source_batch["image"].shape[0] / target_image.shape[0]))
                    target_image = target_image.repeat((repeats, 1, 1, 1, 1))[: source_batch["image"].shape[0]]
            output = model(
                source_batch["image"],
                source_batch["covariates"],
                target_image=target_image,
                return_audit=True,
            )
            objective = _loss(output, source_batch["label"], class_weights, config, variant)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach().cpu()))
            if "transport_audit" in output:
                amplitudes.append(float(output["transport_audit"]["amplitude_l1"].cpu()))
        val = _evaluate(model, val_loader, device)
        val_ba = float(val["metrics"].get("balanced_accuracy", float("nan")))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_balanced_accuracy": val_ba,
                "target_amplitude_l1": float(np.mean(amplitudes)) if amplitudes else None,
            }
        )
        if np.isfinite(val_ba) and val_ba > best_score:
            best_score = val_ba
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": best_epoch,
                    "variant": variant,
                    "seed": int(config["seed"]),
                },
                output_dir / "best_checkpoint.pt",
            )
    if not (output_dir / "best_checkpoint.pt").exists():
        raise RuntimeError("no checkpoint was selected; source validation BA was non-finite")
    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    source_val = _evaluate(model, val_loader, device)
    source_test = _evaluate(model, source_test_loader, device)
    target_test = _evaluate(model, target_test_loader, device)
    report = {
        "variant": variant,
        "direction": direction,
        "seed": int(config["seed"]),
        "best_epoch": int(best_epoch),
        "source_val": source_val["metrics"],
        "source_test": source_test["metrics"],
        "target_test": target_test["metrics"],
        "history": history,
        "protocol": {
            "source_train_n": int(len(train_set)),
            "source_val_n": int(len(val_set)),
            "source_test_n": int(len(source_test_set)),
            "target_adapt_n": int(len(target_adapt_set)),
            "target_test_n": int(len(target_test_set)),
            "target_labels_used_in_training": False,
            "target_covariates_used_in_transport": False,
            "target_image_information": "layer4_spatial_fft_amplitude_only",
            "source_phase_preserved": True,
            "target_test_used_for_selection": False,
            "split_seed": int(config.get("split_seed", config["seed"])),
            "target_split_seed": int((config.get("target_split") or {}).get("seed", config["seed"])),
            "target_adapt_subjects": sorted(set(target.subject_ids[target_adapt_idx].astype(str).tolist())),
            "target_test_subjects": sorted(set(target.subject_ids[target_test_idx].astype(str).tolist())),
        },
        "config": config,
    }
    save_json_summary(output_dir / "metrics.json", report)
    return report


def run_study(config: dict, direction: str, seeds: Sequence[int], output_root: str, variants: Iterable[str], device: str) -> None:
    root = Path(output_root)
    for seed in seeds:
        for variant in variants:
            seeded = copy.deepcopy(config)
            seeded["seed"] = int(seed)
            run_dir = root / direction / f"seed_{seed}" / variant
            print(f"[target-style] {direction} seed={seed} variant={variant}", flush=True)
            _run_one(seeded, direction, run_dir, variant, device)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction", default="ADNI_to_NACC")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_root = args.output_root or config.get("output_root", "outputs/target_style_capm_uda")
    seeds = args.seeds or [int(config.get("seed", 42))]
    run_study(config, args.direction, seeds, output_root, args.variants, args.device)


if __name__ == "__main__":
    main()
