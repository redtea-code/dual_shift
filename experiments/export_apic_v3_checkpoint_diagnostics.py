"""Export sample-level APIC v3 memory and counterfactual shift diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.journal_dataset import CovariatePreprocessor, JournalSubset
from experiments.apic_v3_protocol import config_fingerprint
from experiments.train_journal import _dataset, _make_model, journal_collate
from Model.dual_shift.acquisition_encoder import (
    canonicalize_field_strength,
    canonicalize_manufacturer,
    canonicalize_sequence_family,
)
from training.journal_metrics import compute_journal_metrics


SPLIT_NAMES = ("source_train", "source_val", "source_test", "target")


def relative_rms_delta(shifted: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, clean.ndim))
    numerator = (shifted.float() - clean.float()).square().mean(dim=dims).sqrt()
    denominator = clean.float().square().mean(dim=dims).sqrt().clamp_min(1e-8)
    return numerator / denominator


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _folder_indices(dataset, expected_folders: list[str], split: str) -> np.ndarray:
    by_folder = {
        str(record["folder"]): index
        for index, record in enumerate(dataset.records)
    }
    missing = [folder for folder in expected_folders if str(folder) not in by_folder]
    if missing:
        raise ValueError(
            f"{split}: {len(missing)} manifest folders are absent from the dataset"
        )
    return np.asarray(
        [by_folder[str(folder)] for folder in expected_folders], dtype=np.int64
    )


def _build_loaders(config: dict, manifest: dict, batch_size: int, workers: int):
    source_name, target_name = str(manifest["direction"]).split("_to_")
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    preprocessor = CovariatePreprocessor.from_dict(manifest["covariate_preprocessor"])
    definitions = {
        "source_train": (source, manifest["source_train_folders"]),
        "source_val": (source, manifest["source_val_folders"]),
        "source_test": (source, manifest["source_test_folders"]),
        "target": (target, manifest["target_folders"]),
    }
    loaders = {}
    for split, (dataset, folders) in definitions.items():
        indices = _folder_indices(dataset, list(folders), split)
        subset = JournalSubset(
            dataset, indices, preprocessor, np.zeros(len(indices), dtype=int)
        )
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=journal_collate,
        )
    return loaders


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=1)


def _js_per_sample(
    clean_logits: torch.Tensor, shifted_logits: torch.Tensor
) -> torch.Tensor:
    p = _probabilities(clean_logits)
    q = _probabilities(shifted_logits)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
        + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
    )


def _quantiles(values: list[float]) -> dict:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(array) == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    summary = {}
    numeric = (
        "style_confidence",
        "style_entropy",
        "style_delta",
        "condition_gate",
        "layer1_relative_rms",
        "layer2_relative_rms",
        "embedding_cosine_distance",
        "js_divergence",
    )
    for split in SPLIT_NAMES:
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        labels = np.asarray([row["label"] for row in selected], dtype=int)
        clean = np.asarray(
            [
                [row["clean_probability_0"], row["clean_probability_1"]]
                for row in selected
            ]
        )
        shifted = np.asarray(
            [
                [row["shifted_probability_0"], row["shifted_probability_1"]]
                for row in selected
            ]
        )
        summary[split] = {
            "n": len(selected),
            "clean_metrics": compute_journal_metrics(
                clean, labels, input_type="probabilities"
            ),
            "shifted_metrics": compute_journal_metrics(
                shifted, labels, input_type="probabilities"
            ),
            "prediction_flip_rate": float(
                np.mean([row["prediction_flip"] for row in selected])
            ),
            **{
                key: _quantiles([float(row[key]) for row in selected])
                for key in numeric
            },
        }
    return summary


def _association(rows: list[dict], field: str) -> float | None:
    pairs = [
        (str(row["prototype_slot"]), str(row.get(field, "Missing")))
        for row in rows
    ]
    if len({left for left, _ in pairs}) < 2 or len({right for _, right in pairs}) < 2:
        return None
    return float(
        normalized_mutual_info_score(
            [left for left, _ in pairs], [right for _, right in pairs]
        )
    )


def _reference_prediction_checks(rows: list[dict], variant_dir: Path) -> dict:
    checks = {}
    for split in ("source_val", "source_test", "target"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        path = variant_dir / f"{split}_predictions.csv"
        if not path.exists():
            checks[split] = {"available": False, "path": str(path)}
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reference_rows = list(csv.DictReader(handle))
        by_folder = {str(row["folder"]): row for row in reference_rows}
        missing = []
        label_mismatches = 0
        probability_differences = []
        for row in selected:
            reference = by_folder.get(str(row["folder"]))
            if reference is None:
                missing.append(str(row["folder"]))
                continue
            label_mismatches += int(int(reference["label"]) != int(row["label"]))
            for class_index in (0, 1):
                probability_differences.append(
                    abs(
                        float(reference[f"probability_{class_index}"])
                        - float(row[f"clean_probability_{class_index}"])
                    )
                )
        maximum = max(probability_differences) if probability_differences else None
        checks[split] = {
            "available": True,
            "path": str(path),
            "n_diagnostic": len(selected),
            "n_matched": len(selected) - len(missing),
            "n_missing": len(missing),
            "missing_folders": missing[:20],
            "label_mismatches": label_mismatches,
            "max_probability_abs_difference": maximum,
            "matches_within_1e_5": bool(
                not missing
                and label_mismatches == 0
                and maximum is not None
                and maximum <= 1e-5
            ),
        }
    return checks


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No diagnostic rows were generated")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _diagnose_loader(
    model, loader, split: str, device: torch.device, max_samples: int | None,
    variant: str = "apic_v3_x",
):
    rows = []
    apic = model.apis
    valid_slots = torch.nonzero(apic.style_valid, as_tuple=False).flatten()
    prototypes = apic.style_prototypes[valid_slots].detach()
    for batch in loader:
        if max_samples is not None and len(rows) >= max_samples:
            break
        image = batch["image"].to(device)
        covariates = batch["covariates"].to(device)
        labels = batch["label"].to(device)
        head_kwargs = {
            key: batch[key].to(device)
            for key in ("age_missing", "sex_missing", "education_missing")
        }
        with torch.no_grad():
            clean_feats = model.backbone(image)
            clean_logits, clean_embedding, _ = model._heads(
                clean_feats["layer4"], covariates, **head_kwargs
            )
            if variant == "apic_v3_2_x":
                teacher_l1, teacher_l2 = apic._teacher_stats(image)
                style = apic._project(apic._descriptor(teacher_l1, teacher_l2))
            else:
                style = torch.tanh(apic.style_encoder(apic._style_stats(clean_feats["layer1"], clean_feats["layer2"])))
            if len(valid_slots):
                distances = torch.cdist(style.float(), prototypes.float(), p=2)
                assignment = torch.softmax(-distances / apic.temperature, dim=1)
                local_slot = assignment.argmax(dim=1)
                slots = valid_slots[local_slot]
            else:
                assignment = style.new_zeros((len(style), 0))
                slots = torch.full((len(style),), -1, device=device, dtype=torch.long)
            if variant == "apic_v3_2_x":
                condition, valid_mask = apic.prepare_style_condition(
                    image,
                    clean_feats["layer1"],
                    clean_feats["layer2"],
                    sample_ids=list(batch.get("subject_id") or []),
                )
                state = apic._state
                target_style = apic.style_prototypes[state["target"]]
                confidence = state["confidence"]
                entropy = state["entropy"]
                gate = state["gate"].detach()
            else:
                target_style, confidence, entropy = apic._target_style(style)
                condition, valid_mask = apic.prepare_style_condition(
                    clean_feats["layer1"], clean_feats["layer2"], clean_feats["layer3"], update_memory=False
                )
                gate = apic._gate.detach()
            shift1, shift2 = apic.make_shift_fns(condition, valid_mask=valid_mask)
            shifted_feats = model.backbone(
                image, shift_layer1=shift1, shift_layer2=shift2,
                update_bn_stats=variant != "apic_v3_2_x",
            )
            shifted_logits, shifted_embedding, _ = model._heads(
                shifted_feats["layer4"], covariates, **head_kwargs
            )
            clean_probs = _probabilities(clean_logits)
            shifted_probs = _probabilities(shifted_logits)
            layer1_delta = relative_rms_delta(
                shifted_feats["layer1"], clean_feats["layer1"]
            )
            layer2_delta = relative_rms_delta(
                shifted_feats["layer2"], clean_feats["layer2"]
            )
            cosine = 1.0 - F.cosine_similarity(
                clean_embedding.float(), shifted_embedding.float(), dim=1
            )
            js = _js_per_sample(clean_logits, shifted_logits)
            clean_ce = F.cross_entropy(clean_logits, labels, reduction="none")
            shift_ce = F.cross_entropy(shifted_logits, labels, reduction="none")
        acquisitions = batch.get("acquisition") or [{} for _ in range(len(labels))]
        for index in range(len(labels)):
            if max_samples is not None and len(rows) >= max_samples:
                break
            acquisition = acquisitions[index] or {}
            row = {
                "split": split,
                "subject_id": str(batch["subject_id"][index]),
                "folder": str(batch["folder"][index]),
                "label": int(labels[index]),
                "field_strength": canonicalize_field_strength(
                    acquisition.get("field_strength")
                ),
                "manufacturer": canonicalize_manufacturer(
                    acquisition.get("manufacturer")
                ),
                "sequence_family": canonicalize_sequence_family(
                    acquisition.get("sequence_family")
                ),
                "acquisition_json": json.dumps(
                    {
                        key: _json_value(value)
                        for key, value in acquisition.items()
                    },
                    sort_keys=True,
                ),
                "prototype_slot": int(slots[index]),
                "style_confidence": float(confidence[index]),
                "style_entropy": float(entropy[index]),
                "style_delta": float(
                    (target_style[index] - style[index])
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                ),
                "condition_gate": float(gate[index]),
                "valid_intervention": bool(valid_mask[index]),
                "layer1_relative_rms": float(layer1_delta[index]),
                "layer2_relative_rms": float(layer2_delta[index]),
                "embedding_cosine_distance": float(cosine[index]),
                "js_divergence": float(js[index]),
                "clean_ce": float(clean_ce[index]),
                "shifted_ce": float(shift_ce[index]),
                "clean_probability_0": float(clean_probs[index, 0]),
                "clean_probability_1": float(clean_probs[index, 1]),
                "shifted_probability_0": float(shifted_probs[index, 0]),
                "shifted_probability_1": float(shifted_probs[index, 1]),
                "clean_prediction": int(clean_probs[index].argmax()),
                "shifted_prediction": int(shifted_probs[index].argmax()),
                "prediction_flip": bool(
                    clean_probs[index].argmax() != shifted_probs[index].argmax()
                ),
            }
            for dim, value in enumerate(style[index].detach().cpu().tolist()):
                row[f"style_{dim}"] = value
            for dim, value in enumerate(target_style[index].detach().cpu().tolist()):
                row[f"target_style_{dim}"] = value
            for position, slot in enumerate(valid_slots.detach().cpu().tolist()):
                row[f"prototype_probability_slot{slot}"] = float(
                    assignment[index, position]
                )
            rows.append(row)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Exact path-remapped YAML used for training."
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        help="seed*/direction directory containing split_manifest.json.",
    )
    parser.add_argument("--checkpoint", help="Defaults to JOB_DIR/apic_v3_x/best_checkpoint.pt")
    parser.add_argument("--output-dir", help="Defaults to JOB_DIR/apic_v3_x/diagnostics")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--alpha",
        type=float,
        help="Diagnostic shift strength; defaults to config alpha_max.",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Optional maximum number per selected split."
    )
    parser.add_argument("--allow-config-hash-mismatch", action="store_true")
    parser.add_argument("--variant", choices=("apic_v3_x", "apic_v3_2_x"), default="apic_v3_x")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    job_dir = Path(args.job_dir).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_match = re.search(r"seed(\d+)", str(job_dir).replace("\\", "/"))
    if seed_match:
        config["seed"] = int(seed_match.group(1))
    manifest_path = job_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("config_hash")
    actual_hash = config_fingerprint(config)
    if expected_hash != actual_hash and not args.allow_config_hash_mismatch:
        raise SystemExit(
            "Config hash mismatch. Use the exact path-remapped training YAML, or pass "
            "--allow-config-hash-mismatch only after manually auditing the difference."
        )
    resolved_device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    variant_dir = job_dir / args.variant
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else variant_dir / "best_checkpoint.pt"
    )
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)
    if checkpoint.get("variant") != args.variant:
        raise SystemExit(
            f"Checkpoint variant must be {args.variant}, got {checkpoint.get('variant')!r}"
        )
    num_classes = len(
        set(int(value) for value in config["task"]["label_mapping"].values())
    )
    model = _make_model(config, num_classes, args.variant).to(resolved_device)
    if args.variant == "apic_v3_2_x":
        # The frozen teacher is created at the phase boundary during training.
        # Recreate its module structure before loading checkpoint teacher keys.
        model.apis.freeze_teacher(model.backbone)
    if checkpoint.get("acquisition_encoder_extra"):
        model.acquisition_encoder.load_state_dict_extra(
            checkpoint["acquisition_encoder_extra"]
        )
        model.acquisition_encoder.to(resolved_device)
    if checkpoint.get("prototype_bank") is not None:
        model.prototype_bank.load_state_dict(checkpoint["prototype_bank"])
    if checkpoint.get("cdt") is not None:
        model.cdt.load_state_dict(checkpoint["cdt"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if args.variant == "apic_v3_2_x":
        # `_finalized` is a Python flag, not a buffer; restore it from the
        # checkpointed style bank so diagnostics do not rebuild prototypes.
        model.apis._finalized = bool(int(model.apis.style_valid.sum().item()) >= 2)
        if not model.apis._finalized:
            raise SystemExit(
                "APIC v3_2 checkpoint style bank is not finalized "
                f"(valid_slots={int(model.apis.style_valid.sum().item())})"
            )
    alpha = float(
        args.alpha
        if args.alpha is not None
        else (config.get("dual_shift") or {}).get("alpha_max", 0.25)
    )
    if alpha < 0.0 or alpha > float(model.apis.alpha_max):
        raise SystemExit(
            f"--alpha must be in [0, {model.apis.alpha_max}], got {alpha}"
        )
    model.apis.enabled = True
    model.apis.set_alpha(alpha)
    if int(model.apis.style_valid.sum().item()) < 2:
        raise SystemExit("APIC checkpoint has fewer than two valid style-memory slots")
    loaders = _build_loaders(config, manifest, args.batch_size, args.num_workers)
    rows = []
    for split in args.splits:
        rows.extend(
            _diagnose_loader(
                model, loaders[split], split, resolved_device, args.max_samples, args.variant
            )
        )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else variant_dir / "diagnostics"
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "sample_diagnostics.csv", rows)
    reference_checks = _reference_prediction_checks(
        rows, checkpoint_path.parent
    )
    summary = {
        "config": str(config_path),
        "config_hash": actual_hash,
        "config_sha256": _sha256(config_path),
        "manifest_config_hash": expected_hash,
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection": checkpoint.get("selection"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(resolved_device),
        "diagnostic_alpha": alpha,
        "counterfactual_only": True,
        "reference_prediction_checks": reference_checks,
        "memory_valid_slots": int(model.apis.style_valid.sum().item()),
        "memory_counts": model.apis.style_counts.detach().cpu().tolist(),
        "splits": _summarize_rows(rows),
        "prototype_assignment_nmi": {
            key: _association(rows, key)
            for key in (
                "split",
                "label",
                "field_strength",
                "manufacturer",
                "sequence_family",
            )
        },
    }
    (output / "diagnostic_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    mismatched = [
        split
        for split, check in reference_checks.items()
        if check.get("available") and not check.get("matches_within_1e_5")
    ]
    if mismatched:
        print(
            "WARNING: reconstructed clean predictions differ from official artifacts "
            f"for {mismatched}",
            file=sys.stderr,
        )
    print(f"Wrote {len(rows)} sample diagnostics to {output}")


if __name__ == "__main__":
    main()
