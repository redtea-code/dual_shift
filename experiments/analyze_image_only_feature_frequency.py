"""Frequency-domain audit of frozen image-only backbone feature maps."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_frequency_domain import DEFAULT_BANDS, radial_frequency_grid, select_earliest_subject_rows
from experiments.train_journal import _make_model

PRESET_FEATURE_LAYERS = {
    "layer3_patch2": "layer3",
    "layer4_pixel": "layer4",
    "layer5_pixel": "layer5",
}


def selected_feature_layer(preset: str) -> str:
    try:
        return PRESET_FEATURE_LAYERS[preset]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen image-only preset: {preset!r}") from exc


def feature_power_summary(feature_map: torch.Tensor, bands=DEFAULT_BANDS) -> dict[str, float]:
    """Summarize mean channel-wise 3D FFT power for one [C,D,H,W] feature map."""
    if feature_map.ndim != 4:
        raise ValueError(f"expected feature map [C, D, H, W], got {tuple(feature_map.shape)}")
    values = feature_map.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("feature map contains non-finite values")
    power = np.abs(np.fft.fftshift(np.fft.fftn(values, axes=(1, 2, 3)), axes=(1, 2, 3))) ** 2
    power = power.mean(axis=0, dtype=np.float64) / float(np.prod(values.shape[1:]))
    radius = radial_frequency_grid(values.shape[1:])
    total = float(power.sum())
    result = {"total_log_power": float(np.log1p(total))}
    for name, (lower, upper) in bands.items():
        mask = (radius >= lower) & (radius <= upper if upper >= 0.5 else radius < upper)
        energy = float(power[mask].sum())
        result[f"{name}_energy"] = energy
        result[f"{name}_fraction"] = energy / total if total else 0.0
    result["high_fraction"] = result["high_fraction"]
    result["spectral_centroid"] = float((power * radius).sum() / total) if total else 0.0
    positive = (power > 0) & (radius > 0)
    result["spectral_slope"] = float(np.polyfit(np.log(radius[positive]), np.log(power[positive]), 1)[0]) if positive.sum() >= 2 else 0.0
    return result


def capture_selected_features(model, layer_name: str, image: torch.Tensor, covariates=None) -> torch.Tensor:
    """Run one inference and return the named backbone layer's detached output."""
    try:
        layer = getattr(model, layer_name)
    except AttributeError as exc:
        raise ValueError(f"model has no feature layer {layer_name!r}") from exc
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            if covariates is None:
                model(image, None)
            else:
                model(image, covariates)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one capture from {layer_name!r}, got {len(captured)}")
    return captured[0]



FREQUENCY_FEATURE_COLUMNS = [
    "total_log_power", "low_energy", "mid_energy", "high_energy",
    "low_fraction", "mid_fraction", "high_fraction", "spectral_centroid", "spectral_slope",
]


def classifier_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return only approved fixed frequency summary fields present in a table."""
    return [column for column in FREQUENCY_FEATURE_COLUMNS if column in frame.columns]


def compare_checkpoint_cohorts(frame: pd.DataFrame, random_state: int = 42) -> dict:
    """Run fixed source-target feature statistics and frequency-only domain CV."""
    columns = classifier_feature_columns(frame)
    source = frame.loc[frame["cohort_role"] == "source"]
    target = frame.loc[frame["cohort_role"] == "target"]
    rng = np.random.default_rng(random_state)
    comparisons = []
    for column in columns:
        left, right = source[column].to_numpy(), target[column].to_numpy()
        pooled = np.sqrt(((len(left)-1)*left.var(ddof=1)+(len(right)-1)*right.var(ddof=1))/(len(left)+len(right)-2))
        d = float((left.mean()-right.mean())/pooled) if pooled else 0.0
        bootstrap = [rng.choice(left, len(left), replace=True).mean() - rng.choice(right, len(right), replace=True).mean() for _ in range(1000)]
        comparisons.append({"feature":column,"source_mean":float(left.mean()),"target_mean":float(right.mean()),"cohens_d_source_minus_target":d,"mann_whitney_p":float(mannwhitneyu(left,right,alternative="two-sided").pvalue),"mean_difference_ci95": [float(value) for value in np.quantile(bootstrap,[0.025,0.975])]})
    X=frame[columns].to_numpy(); y=(frame["cohort_role"] == "target").astype(int).to_numpy()
    cv=StratifiedKFold(5,shuffle=True,random_state=random_state)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=random_state))
    probabilities=cross_val_predict(model,X,y,cv=cv,method="predict_proba")[:,1]
    predictions=(probabilities>=0.5).astype(int)
    return {"features":columns,"comparison":comparisons,"domain_classifier":{"balanced_accuracy":float(balanced_accuracy_score(y,predictions)),"auroc":float(roc_auc_score(y,probabilities)),"n_subjects":int(len(frame)),"cv":"5-fold stratified, seed 42, standardized logistic regression"},"domain_predictions": {"probability_target": probabilities.tolist(), "predicted_target": predictions.tolist()}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_frozen_image_only_model(config_path: Path, checkpoint_path: Path, device: str):
    config = _load_config(config_path)
    model = _make_model(config, num_classes=2, variant="image_only").to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def extract_checkpoint_cohorts(
    *, config_path: Path, checkpoint_path: Path, preset: str, direction: str,
    adni_manifest: Path, nacc_manifest: Path, output_dir: Path, device: str,
    max_subjects: int | None = None,
) -> dict:
    """Extract earliest-visit frequency summaries from source and target cohorts."""
    source_name, target_name = direction.split("_to_")
    selected_layer = selected_feature_layer(preset)
    if checkpoint_path.parent.name != "image_only":
        raise ValueError("feature frequency audit only permits image_only checkpoints")
    config = _load_config(config_path)
    cfg_preset = str((config.get("scale_table_ablation") or {}).get("preset"))
    if cfg_preset != preset:
        raise ValueError(f"config preset {cfg_preset!r} does not match requested {preset!r}")
    model, checkpoint = _load_frozen_image_only_model(config_path, checkpoint_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures, audits = [], [], []
    manifests = {"ADNI": adni_manifest, "NACC": nacc_manifest}
    for role, cohort in (("source", source_name), ("target", target_name)):
        frame = pd.read_csv(manifests[cohort])
        selected, audit = select_earliest_subject_rows(frame)
        if max_subjects is not None:
            selected = selected.iloc[:max_subjects].copy()
        audit.update({"cohort": cohort, "cohort_role": role, "requested_rows": int(len(selected))})
        for record in selected.to_dict("records"):
            try:
                volume = nib.load(str(record["image_path"])).get_fdata(dtype=np.float32)
                image = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)
                features = capture_selected_features(model, selected_layer, image, None)[0]
                summary = feature_power_summary(features)
                rows.append({
                    "checkpoint_id": checkpoint_path.parent.parent.name, "preset": preset,
                    "direction": direction, "selected_layer": selected_layer, "cohort_role": role,
                    "cohort": cohort, "subject_id": str(record["subject_id"]),
                    "scan_date": str(record.get("scan_date", "")),
                    "field_strength": str(record.get("field_strength", "")),
                    "feature_channels": int(features.shape[0]),
                    "feature_depth": int(features.shape[1]), "feature_height": int(features.shape[2]),
                    "feature_width": int(features.shape[3]), **summary,
                })
            except Exception as exc:
                failures.append({"cohort": cohort, "cohort_role": role, "subject_id": str(record.get("subject_id", "")), "image_path": str(record.get("image_path", "")), "error": repr(exc)})
        audit["processed_rows"] = sum(row["cohort_role"] == role for row in rows)
        audit["failed_rows"] = sum(failure["cohort_role"] == role for failure in failures)
        audits.append(audit)
    pd.DataFrame(rows).to_csv(output_dir / "subject_frequency_features.csv", index=False)
    (output_dir / "selection_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    (output_dir / "failed_images.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    provenance = {
        "analysis": "frozen_image_only_feature_frequency", "preset": preset, "selected_layer": selected_layer,
        "direction": direction, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)), "config_path": str(config_path),
        "config_sha256": _sha256(config_path), "source": source_name, "target": target_name,
        "bands": DEFAULT_BANDS, "deduplication": "earliest_visit",
        "manifest_sha256": {name: _sha256(path) for name, path in manifests.items()},
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"rows": len(rows), "failures": len(failures), "output_dir": str(output_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--preset", required=True, choices=sorted(PRESET_FEATURE_LAYERS))
    parser.add_argument("--direction", required=True, choices=["ADNI_to_NACC", "NACC_to_ADNI"])
    parser.add_argument("--adni-manifest", required=True)
    parser.add_argument("--nacc-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args(argv)
    result = extract_checkpoint_cohorts(
        config_path=Path(args.config_path), checkpoint_path=Path(args.checkpoint_path), preset=args.preset,
        direction=args.direction, adni_manifest=Path(args.adni_manifest), nacc_manifest=Path(args.nacc_manifest),
        output_dir=Path(args.output_dir), device=args.device, max_subjects=args.max_subjects,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
