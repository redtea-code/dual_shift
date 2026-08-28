"""Build a leakage-safe global raw/feature spectral atlas for Plan 34.

The command uses only S_train and unlabeled T_adapt.  It deliberately writes
descriptive, subject-level frequency evidence instead of a target-driven gate
coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.ablation import build_scale_table_ablation
from Model.ablation.frequency_uda import (
    BAND_NAMES,
    DEFAULT_BAND_EDGES,
    FrequencyGuidedScaleTable3D,
    FrequencyPrior,
    band_power_fractions,
)
from experiments.frequency_uda import (
    ImageSubjectSubset,
    build_frequency_prior_from_loaders,
    earliest_subject_indices,
)
from experiments.train_journal import _dataset, _load_frozen_split
from training.frequency_environments import FREQUENCY_ENVIRONMENTS, FrequencyEnvironmentAugment3D


ATLAS_SCHEMA = "dualshift_global_spectral_atlas_v1"
VALID_STAGES = ("raw", "layer1", "layer2", "layer3", "layer4", "layer5")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_digest(subject_ids: Iterable[object]) -> str:
    payload = "\n".join(sorted({str(value) for value in subject_ids})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_stages(stages: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(stage) for stage in stages)
    if not selected:
        raise ValueError("at least one atlas stage is required")
    invalid = set(selected).difference(VALID_STAGES)
    if invalid:
        raise ValueError(f"unknown atlas stages: {sorted(invalid)}")
    if len(set(selected)) != len(selected):
        raise ValueError("atlas stages must be unique")
    return selected


def extract_spatial_stage(model: torch.nn.Module, images: Tensor, stage: str) -> Tensor:
    """Extract a raw or spatial backbone map without CAPM/table modulation."""
    if stage == "raw":
        return images
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown atlas stage {stage!r}")
    features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    for name in ("layer1", "layer2", "layer3", "layer4"):
        features = getattr(model, name)(features)
        if stage == name:
            return features
    if stage == "layer5":
        return model.layer5(features)
    raise RuntimeError(f"could not extract stage {stage!r}")


def collect_subject_band_fractions(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    stages: Sequence[str],
    population: str,
    device: str | torch.device = "cpu",
    environment: str = "original",
) -> pd.DataFrame:
    """Collect one low/mid/high fraction row per subject/stage/band.

    The loader contract is image plus subject ID only.  It intentionally does
    not request a label-bearing dataset item for target subjects.
    """
    selected = _validate_stages(stages)
    device = torch.device(device)
    rows: list[dict[str, Any]] = []
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                images = torch.as_tensor(batch["image"], dtype=torch.float32, device=device)
                subject_ids = [str(value) for value in batch["subject_id"]]
                if len(subject_ids) != images.shape[0]:
                    raise ValueError("image and subject_id batch dimensions differ")
                for stage in selected:
                    features = extract_spatial_stage(model, images, stage)
                    fractions, _, _ = band_power_fractions(features)
                    for sample_index, subject_id in enumerate(subject_ids):
                        for band_index, band in enumerate(BAND_NAMES):
                            rows.append(
                                {
                                    "population": population,
                                    "environment": environment,
                                    "stage": stage,
                                    "subject_id": subject_id,
                                    "band": band,
                                    "fraction": float(fractions[sample_index, band_index].detach().cpu()),
                                }
                            )
    finally:
        model.train(previous_training)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No rows collected for population {population!r}")
    return frame


def summarize_band_fractions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"population", "environment", "stage", "subject_id", "band", "fraction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"band-fraction frame is missing {sorted(missing)}")
    return (
        frame.groupby(["population", "environment", "stage", "band"], as_index=False)
        .agg(subject_count=("subject_id", "nunique"), mean=("fraction", "mean"), std=("fraction", "std"))
        .fillna({"std": 0.0})
    )


def bootstrap_band_stability(source: np.ndarray, target: np.ndarray, *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    """Measure target-adapt discrepancy without selecting a model or target metric."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != len(BAND_NAMES) or target.shape[1] != len(BAND_NAMES):
        raise ValueError("source and target must have shape [N, 3]")
    if len(source) < 2 or len(target) < 2:
        raise ValueError("bootstrap requires at least two source and target subjects")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    def discrepancy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        pooled = np.sqrt((left.std(axis=0) ** 2 + right.std(axis=0) ** 2) / 2.0)
        return np.divide(np.abs(left.mean(axis=0) - right.mean(axis=0)), pooled, out=np.zeros_like(pooled), where=pooled > 0)

    raw_d = discrepancy(source, target)
    normalized = raw_d / raw_d.max() if raw_d.max() > 0 else np.zeros_like(raw_d)
    rng = np.random.default_rng(int(seed))
    samples = np.stack(
        [
            discrepancy(
                source[rng.integers(len(source), size=len(source))],
                target[rng.integers(len(target), size=len(target))],
            )
            for _ in range(n_bootstrap)
        ]
    )
    highest = samples.argmax(axis=1)
    return {
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "raw_d": raw_d.tolist(),
        "normalized_discrepancy": normalized.tolist(),
        "ci_low": np.quantile(samples, 0.025, axis=0).tolist(),
        "ci_high": np.quantile(samples, 0.975, axis=0).tolist(),
        "highest_band_probability": {
            band: float(np.mean(highest == index)) for index, band in enumerate(BAND_NAMES)
        },
    }


def _fraction_matrix(frame: pd.DataFrame, population: str, stage: str) -> np.ndarray:
    subset = frame.loc[(frame.population == population) & (frame.stage == stage) & (frame.environment == "original")]
    pivot = subset.pivot(index="subject_id", columns="band", values="fraction").reindex(columns=list(BAND_NAMES))
    if pivot.isna().any().any():
        raise RuntimeError(f"incomplete subject/band rows for {population}/{stage}")
    return pivot.to_numpy(dtype=float)


def atlas_bootstrap_report(frame: pd.DataFrame, *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    stages = sorted(set(frame.stage))
    return {
        stage: bootstrap_band_stability(
            _fraction_matrix(frame, "S_train", stage),
            _fraction_matrix(frame, "T_adapt", stage),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for stage in stages
    }


def _source_model(config: dict[str, Any], checkpoint: str | Path) -> torch.nn.Module:
    ablation = config.get("scale_table_ablation") or {}
    model = build_scale_table_ablation(
        preset="layer5_pixel",
        interaction="original_capm",
        num_classes=len(set(config["task"]["label_mapping"].values())),
        layers=tuple(int(value) for value in ablation.get("layers", (2, 2, 2, 2))),
        spatial_shape=tuple(int(value) for value in ablation.get("spatial_shape", (4, 4, 4))),
        transformer_dim=int(ablation.get("transformer_dim", 128)),
        num_heads=int(ablation.get("num_heads", 4)),
        transformer_dropout=float(ablation.get("transformer_dropout", 0.1)),
        classifier_dropout=float(ablation.get("classifier_dropout", 0.3)),
        gate_init=float(ablation.get("gate_init", 0.95)),
        input_shape=tuple(int(value) for value in ablation.get("input_shape", config.get("training", {}).get("image_shape", (160, 196, 160)))),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state", payload), strict=True)
    return model


def _placeholder_prior() -> FrequencyPrior:
    summary = {"count": 1, "mean": [1 / 3, 1 / 3, 1 / 3], "std": [1.0, 1.0, 1.0]}
    return FrequencyPrior.from_summaries(summary, summary, metadata={"placeholder": True})


def _prior_reproduction_model(
    config: dict[str, Any], checkpoint: str | Path
) -> FrequencyGuidedScaleTable3D:
    """Load the exact pre-gate layer4 path used by the original prior builder."""
    ablation = config.get("scale_table_ablation") or {}
    model = FrequencyGuidedScaleTable3D(
        prior=_placeholder_prior(),
        preset="layer5_pixel",
        num_classes=len(set(config["task"]["label_mapping"].values())),
        layers=tuple(int(value) for value in ablation.get("layers", (2, 2, 2, 2))),
        spatial_shape=tuple(int(value) for value in ablation.get("spatial_shape", (4, 4, 4))),
        transformer_dim=int(ablation.get("transformer_dim", 128)),
        num_heads=int(ablation.get("num_heads", 4)),
        transformer_dropout=float(ablation.get("transformer_dropout", 0.1)),
        classifier_dropout=float(ablation.get("classifier_dropout", 0.3)),
        gate_init=float(ablation.get("gate_init", 0.95)),
        input_shape=tuple(
            int(value)
            for value in ablation.get(
                "input_shape", config.get("training", {}).get("image_shape", (160, 196, 160))
            )
        ),
    )
    model.load_source_baseline(checkpoint)
    return model


def compare_frequency_priors(
    reference: FrequencyPrior,
    recomputed: FrequencyPrior,
    *,
    expected_metadata: dict[str, object],
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare P0 values and provenance without looking at target labels/metrics."""
    if atol < 0 or rtol < 0:
        raise ValueError("P0 tolerances must be non-negative")
    numeric_fields = {
        "band_edges": (reference.band_edges, recomputed.band_edges),
        "source_mean": (reference.source_mean, recomputed.source_mean),
        "source_std": (reference.source_std, recomputed.source_std),
        "target_mean": (reference.target_mean, recomputed.target_mean),
        "target_std": (reference.target_std, recomputed.target_std),
        "discrepancy": (reference.discrepancy, recomputed.discrepancy),
    }
    numeric: dict[str, dict[str, Any]] = {}
    for name, (expected, actual) in numeric_fields.items():
        expected_array = np.asarray(expected, dtype=float)
        actual_array = np.asarray(actual, dtype=float)
        numeric[name] = {
            "reference": expected_array.tolist(),
            "recomputed": actual_array.tolist(),
            "max_abs_error": float(np.abs(expected_array - actual_array).max()),
            "matches": bool(np.allclose(expected_array, actual_array, atol=atol, rtol=rtol)),
        }
    counts = {
        "source_count": {
            "reference": reference.source_count,
            "recomputed": recomputed.source_count,
            "matches": reference.source_count == recomputed.source_count,
        },
        "target_count": {
            "reference": reference.target_count,
            "recomputed": recomputed.target_count,
            "matches": reference.target_count == recomputed.target_count,
        },
    }
    metadata = {
        key: {
            "reference": reference.metadata.get(key),
            "recomputed": recomputed.metadata.get(key),
            "expected": expected,
            "matches": (
                reference.metadata.get(key) == expected
                and recomputed.metadata.get(key) == expected
            ),
        }
        for key, expected in expected_metadata.items()
    }
    blinding = {
        key: {
            "reference": reference.metadata.get(key),
            "recomputed": recomputed.metadata.get(key),
            "matches": (
                reference.metadata.get(key) is False
                and recomputed.metadata.get(key) is False
            ),
        }
        for key in ("target_labels_read", "target_metrics_read")
    }
    passed = (
        all(item["matches"] for item in numeric.values())
        and all(item["matches"] for item in counts.values())
        and all(item["matches"] for item in metadata.values())
        and all(item["matches"] for item in blinding.values())
    )
    return {
        "passed": bool(passed),
        "tolerance": {"atol": float(atol), "rtol": float(rtol)},
        "numeric": numeric,
        "counts": counts,
        "metadata": metadata,
        "blinding": blinding,
    }


def reproduce_frequency_prior(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    source_loader: Iterable[dict[str, Any]],
    target_loader: Iterable[dict[str, Any]],
    source_split: str | Path,
    target_split: str | Path,
    source_checkpoint: str | Path,
    reference_prior: str | Path,
    output_dir: str | Path,
    direction: str,
    device: torch.device,
) -> dict[str, Any]:
    """Recreate the C4 layer4 prior and write a P0 pass/fail report."""
    output = Path(output_dir)
    reference = FrequencyPrior.load(reference_prior)
    target_payload = json.loads(Path(target_split).read_text(encoding="utf-8"))
    expected_metadata = {
        "direction": direction,
        "source_split_manifest_sha256": _sha256(source_split),
        "target_split_sha256": _sha256(target_split),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "config_sha256": _sha256(config_path),
        "target_test_subject_digest": target_payload["target_test_subject_digest"],
    }
    model = _prior_reproduction_model(config, source_checkpoint).to(device)
    recomputed_path = output / "frequency_prior_recomputed.json"
    recomputed = build_frequency_prior_from_loaders(
        model,
        source_loader,
        target_loader,
        device=device,
        output_path=recomputed_path,
        metadata=expected_metadata,
    )
    report = compare_frequency_priors(
        reference,
        recomputed,
        expected_metadata=expected_metadata,
    )
    report.update(
        {
            "reference_prior": str(reference_prior),
            "reference_prior_sha256": _sha256(reference_prior),
            "recomputed_prior": str(recomputed_path),
            "target_labels_read": False,
            "target_metrics_read": False,
        }
    )
    (output / "frequency_prior_reproduction.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _target_adapt_indices(target: Any, split_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if payload.get("target_labels_read") is not False or payload.get("target_metrics_read") is not False:
        raise ValueError("target split does not certify label/metric blinding")
    adapt_ids = {str(value) for value in payload.get("target_adapt_subjects", [])}
    test_ids = {str(value) for value in payload.get("target_test_subjects", [])}
    if not adapt_ids or not test_ids or adapt_ids.intersection(test_ids):
        raise ValueError("target split has no valid subject-disjoint T_adapt population")
    if not payload.get("target_test_subject_digest"):
        raise ValueError("target split is missing the T_test subject digest")
    indices = [index for index, subject_id in enumerate(target.subject_ids.tolist()) if str(subject_id) in adapt_ids]
    return earliest_subject_indices(target.subject_ids, indices, target.records), payload


def run_atlas(*, config: dict[str, Any], config_path: str | Path, direction: str, source_split: str | Path, target_split: str | Path, source_checkpoint: str | Path, reference_prior: str | Path, output_dir: str | Path, stages: Sequence[str], n_bootstrap: int = 1000, batch_size: int = 2, device: str = "cpu") -> dict[str, Any]:
    """Build all primary atlas artifacts from frozen source and unlabeled T_adapt."""
    if direction != "ADNI_to_NACC":
        raise ValueError("global atlas is restricted to the primary ADNI_to_NACC direction")
    selected = _validate_stages(stages)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_name, target_name = direction.split("_to_")
    source = _dataset(config, source_name)
    target = _dataset(config, target_name)
    source_train, _, _, _ = _load_frozen_split(source, str(source_split))
    source_indices = earliest_subject_indices(source.subject_ids, source_train, source.records)
    adapt_indices, target_split_payload = _target_adapt_indices(target, target_split)
    source_set = ImageSubjectSubset(source, source_indices)
    target_set = ImageSubjectSubset(target, adapt_indices)
    source_loader = DataLoader(source_set, batch_size=int(batch_size), shuffle=False)
    target_loader = DataLoader(target_set, batch_size=int(batch_size), shuffle=False)
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    prior_reproduction = reproduce_frequency_prior(
        config=config,
        config_path=config_path,
        source_loader=source_loader,
        target_loader=target_loader,
        source_split=source_split,
        target_split=target_split,
        source_checkpoint=source_checkpoint,
        reference_prior=reference_prior,
        output_dir=output,
        direction=direction,
        device=resolved_device,
    )
    if not prior_reproduction["passed"]:
        raise RuntimeError(
            "P0 frequency_prior reproduction failed; inspect frequency_prior_reproduction.json "
            "before interpreting or training any new variant"
        )
    model = _source_model(config, source_checkpoint).to(resolved_device)
    source_rows = collect_subject_band_fractions(model, source_loader, stages=selected, population="S_train", device=resolved_device)
    target_rows = collect_subject_band_fractions(model, target_loader, stages=selected, population="T_adapt", device=resolved_device)
    source_environment_rows: list[pd.DataFrame] = []
    augmenter_cfg = config.get("frequency_mixstyle") or config.get("frequency_uda") or {}
    augmenter = FrequencyEnvironmentAugment3D(
        lowpass_kernel=int(augmenter_cfg.get("lowpass_kernel", 3)),
        blur_sigma=float(augmenter_cfg.get("blur_sigma", 0.8)),
    ).to(resolved_device)
    for environment_id, environment in enumerate(FREQUENCY_ENVIRONMENTS):
        rows: list[pd.DataFrame] = []
        for batch in source_loader:
            images = torch.as_tensor(batch["image"], dtype=torch.float32, device=resolved_device)
            transformed, _ = augmenter(images, environment_ids=torch.full((images.shape[0],), environment_id, device=resolved_device))
            rows.append(_collect_single_batch(model, transformed, batch["subject_id"], selected, "S_train_environment", environment))
        source_environment_rows.append(pd.concat(rows, ignore_index=True))
    all_rows = pd.concat([source_rows, target_rows], ignore_index=True)
    environment_rows = pd.concat(source_environment_rows, ignore_index=True)
    all_rows.to_csv(output / "per_subject_band_fractions.csv", index=False)
    summarize_band_fractions(all_rows).to_csv(output / "band_summary.csv", index=False)
    summarize_band_fractions(environment_rows).to_csv(output / "source_environment_band_summary.csv", index=False)
    bootstrap = atlas_bootstrap_report(all_rows, n_bootstrap=n_bootstrap, seed=int(config.get("seed", 0)))
    (output / "bootstrap_stability.json").write_text(json.dumps(bootstrap, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "schema": ATLAS_SCHEMA,
        "direction": direction,
        "stages": list(selected),
        "n_bootstrap": int(n_bootstrap),
        "target_labels_read": False,
        "target_metrics_read": False,
        "target_test_accessed": False,
        "source_split": str(source_split),
        "target_split": str(target_split),
        "source_checkpoint": str(source_checkpoint),
        "reference_prior": str(reference_prior),
        "config": str(config_path),
        "hashes": {
            "config": _sha256(config_path),
            "source_split": _sha256(source_split),
            "target_split": _sha256(target_split),
            "source_checkpoint": _sha256(source_checkpoint),
            "reference_prior": _sha256(reference_prior),
        },
        "frequency_prior_reproduction": prior_reproduction,
        "target_test_subject_digest_from_split": target_split_payload["target_test_subject_digest"],
        "populations": {
            "S_train": {"count": len(source_set), "subject_digest": _subject_digest(source_set.dataset.subject_ids[index] for index in source_set.indices)},
            "T_adapt": {"count": len(target_set), "subject_digest": _subject_digest(target_set.dataset.subject_ids[index] for index in target_set.indices)},
        },
    }
    (output / "atlas_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return provenance


def _collect_single_batch(model: torch.nn.Module, images: Tensor, subject_ids: Sequence[object], stages: Sequence[str], population: str, environment: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for stage in stages:
                fractions, _, _ = band_power_fractions(extract_spatial_stage(model, images, stage))
                for sample_index, subject_id in enumerate(subject_ids):
                    for band_index, band in enumerate(BAND_NAMES):
                        rows.append({"population": population, "environment": environment, "stage": stage, "subject_id": str(subject_id), "band": band, "fraction": float(fractions[sample_index, band_index].detach().cpu())})
    finally:
        model.train(previous_training)
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC",))
    parser.add_argument("--source-split", required=True, type=Path)
    parser.add_argument("--target-split", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--reference-prior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stages", nargs="+", default=["raw", "layer3", "layer4"], choices=VALID_STAGES)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = run_atlas(
        config=config,
        config_path=args.config,
        direction=args.direction,
        source_split=args.source_split,
        target_split=args.target_split,
        source_checkpoint=args.source_checkpoint,
        reference_prior=args.reference_prior,
        output_dir=args.output_dir,
        stages=args.stages,
        n_bootstrap=args.n_bootstrap,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
