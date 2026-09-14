from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

def bootstrap_discrepancy_stability(source_fractions, target_fractions, n_bootstrap: int, seed: int) -> dict[str, Any]:
    source, target = np.asarray(source_fractions, float), np.asarray(target_fractions, float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or n_bootstrap <= 0:
        raise ValueError("fraction arrays must be 2-D with matching columns and positive bootstrap count")
    signed_mean_delta = source.mean(0) - target.mean(0)
    def discrepancy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        pooled_std = np.sqrt((left.std(0) ** 2 + right.std(0) ** 2) / 2)
        return np.divide(np.abs(left.mean(0) - right.mean(0)), pooled_std, out=np.zeros_like(pooled_std), where=pooled_std > 0)
    raw_d = discrepancy(source, target)
    normalized = raw_d / raw_d.max() if raw_d.max() > 0 else np.zeros_like(raw_d)
    rng = np.random.default_rng(seed)
    samples = np.stack([discrepancy(source[rng.integers(len(source), size=len(source))], target[rng.integers(len(target), size=len(target))]) for _ in range(n_bootstrap)])
    ranks = np.argsort(-samples, axis=1)
    probs = np.zeros((source.shape[1], source.shape[1]))
    for rank in range(source.shape[1]):
        for band in range(source.shape[1]): probs[band, rank] = np.mean(ranks[:, rank] == band)
    return {"n_bootstrap": int(n_bootstrap), "seed": int(seed), "discrepancy": signed_mean_delta.tolist(), "signed_mean_delta": signed_mean_delta.tolist(), "raw_d": raw_d.tolist(), "normalized_discrepancy": normalized.tolist(), "ci_low": np.quantile(samples, .025, 0).tolist(), "ci_high": np.quantile(samples, .975, 0).tolist(), "rank_probabilities": probs.tolist()}

def summarize_gate_audit(audits: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(audits); fields = {}
    for key in sorted({k for row in rows for k in row}):
        values = np.asarray([row[key] for row in rows if np.isscalar(row.get(key))], float)
        if len(values): fields[key] = {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}
    return {"count": len(rows), "fields": fields}

def _scalar_record_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return value[0] if len(value) else None
    return value


def summarize_covariate_support(source_records: Iterable[dict[str, Any]], target_records: Iterable[dict[str, Any]] | None = None, *, sex_impute: int = 0) -> pd.DataFrame:
    """Summarize raw covariates and frozen numeric sex support.

    Sex categories are normalized as female/male/unknown, while the encoded
    statistics use the source-fitted binary encoding and its unknown impute.
    """
    populations = {"source": list(source_records)} if target_records is None else {"source": list(source_records), "T_adapt": list(target_records)}
    covariates = sorted({key for records in populations.values() for record in records for key in record})
    rows: list[dict[str, Any]] = []
    values_by_population: dict[tuple[str, str], pd.Series] = {}
    for population, records in populations.items():
        for covariate in covariates:
            raw = [_scalar_record_value(record.get(covariate)) for record in records]
            if covariate == "sex":
                def category(value):
                    text = str(value).strip().lower()
                    if text in {"0", "2", "f", "female", "woman"}: return "female"
                    if text in {"1", "m", "male", "man"}: return "male"
                    return "unknown"
                categories = [category(value) for value in raw]
                encoded = pd.Series([0.0 if value == "female" else 1.0 if value == "male" else float(sex_impute) for value in categories])
                values = encoded
                category_counts = json.dumps({key: categories.count(key) for key in ("female", "male", "unknown") if key in categories}, sort_keys=True)
                category_fractions = json.dumps({key: categories.count(key) / len(categories) for key in ("female", "male", "unknown") if key in categories}, sort_keys=True)
                extra = {"category_counts": category_counts, "category_fractions": category_fractions, "encoded_mean": float(encoded.mean()) if len(encoded) else np.nan, "encoded_min": float(encoded.min()) if len(encoded) else np.nan, "encoded_max": float(encoded.max()) if len(encoded) else np.nan}
                missing_count = 0
            else:
                values = pd.to_numeric(pd.Series(raw), errors="coerce")
                extra = {"category_counts": np.nan, "category_fractions": np.nan, "encoded_mean": np.nan, "encoded_min": np.nan, "encoded_max": np.nan}
                missing_count = int(values.isna().sum())
            values_by_population[(population, covariate)] = values
            valid = values.dropna()
            rows.append({"population": population, "covariate": covariate, "count": len(values), "missing_count": missing_count, "missing_fraction": float(missing_count / len(values)) if len(values) else np.nan, "min": float(valid.min()) if len(valid) else np.nan, "max": float(valid.max()) if len(valid) else np.nan, "standardized_mean_difference": np.nan, **extra})
    if target_records is not None:
        for row in rows:
            source = values_by_population[("source", row["covariate"])].dropna().to_numpy(float)
            target = values_by_population[("T_adapt", row["covariate"])].dropna().to_numpy(float)
            if len(source) and len(target):
                pooled = np.sqrt((source.std() ** 2 + target.std() ** 2) / 2)
                if pooled > 0: row["standardized_mean_difference"] = float((source.mean() - target.mean()) / pooled)
    return pd.DataFrame(rows, columns=["population", "covariate", "count", "missing_count", "missing_fraction", "min", "max", "standardized_mean_difference", "category_counts", "category_fractions", "encoded_mean", "encoded_min", "encoded_max"])


# Task 2 runner contract.  This boundary intentionally accepts loader factories
# so tests and callers can inject source/T_adapt-only data access.
_REQUIRED_AUDIT_ARTIFACTS = (
    "prior_provenance.json", "source_target_band_summary.csv",
    "prior_bootstrap_stability.json", "gate_activity.csv",
    "pre_post_spectrum_summary.csv", "c4_identity_forward_comparison.json",
    "source_environment_audit.csv", "covariate_support_audit.csv",
    "pre_post_discrepancy.json",
)


def validate_target_split_contract(split: dict[str, Any]) -> None:
    """Reject target split metadata that permits labels or target metrics."""
    if split.get("target_labels_read") is not False:
        raise ValueError("target_labels_read must be false")
    if split.get("target_metrics_read") is not False:
        raise ValueError("target_metrics_read must be false")


def validate_target_subject_disjointness(split: dict[str, Any]) -> set[str]:
    """Validate T_adapt/T_test IDs using metadata only; never inspect records."""
    adapt = {str(value) for value in split.get("target_adapt_subjects", ())}
    test = {str(value) for value in split.get("target_test_subjects", ())}
    if not adapt or not test or adapt.intersection(test):
        raise ValueError("target adaptation/test subjects must be non-empty and disjoint")
    return adapt


def compare_gate_forwards(forward: Any, model_inputs: Any) -> dict[str, Any]:
    """Compare enabled/disabled forwards on the identical input object."""
    enabled = forward(model_inputs, True)
    disabled = forward(model_inputs, False)
    feature_diff = np.asarray(enabled["features"]) - np.asarray(disabled["features"])
    logit_diff = np.asarray(enabled["logits"]) - np.asarray(disabled["logits"])
    finite = bool(np.isfinite(feature_diff).all() and np.isfinite(logit_diff).all())
    return {
        "paired_input_equal": True,
        "feature_abs_diff": float(np.abs(feature_diff).mean()),
        "logit_abs_diff": float(np.abs(logit_diff).mean()),
        "finite": finite,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        import json
        return json.load(handle)


def _call_factory(factory: Any, *, split: str) -> list[dict[str, Any]]:
    if factory is None:
        return []
    if not callable(factory):
        return list(factory)
    try:
        value = factory(split=split)
    except TypeError:
        value = factory()
    return list(value() if callable(value) else value)


def _earliest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        subjects = row.get("subject_id", [])
        if isinstance(subjects, (str, int)):
            subjects = [subjects]
        for subject in subjects:
            key = str(subject)
            candidate = dict(row)
            candidate["subject_id"] = [subject]
            order = (str(row.get("scan_date", "")), str(row.get("folder", "")))
            if key not in selected or order < selected[key].get("_early_key", ("~", "~")):
                candidate["_early_key"] = order
                selected[key] = candidate
    for row in selected.values():
        row.pop("_early_key", None)
    return list(selected.values())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def _sha256(path: str | Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    if not Path(path).exists(): return "missing"
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _with_frozen_covariates(rows: list[dict[str, Any]], source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach train-fitted CAPM covariates without accessing target labels."""
    payload = source_manifest.get("covariate_preprocessor")
    if payload is None:
        return rows
    from data.journal_dataset import CovariatePreprocessor
    preprocessor = CovariatePreprocessor.from_dict(payload)
    transformed = []
    for row in rows:
        values, _ = preprocessor.transform(
            [_scalar_record_value(row.get("age"))],
            [_scalar_record_value(row.get("sex"))],
            [_scalar_record_value(row.get("education"))],
        )
        transformed.append({**row, "covariates": values})
    return transformed


def _subject_digest(rows: list[dict[str, Any]]) -> str:
    import hashlib
    subjects = sorted(str(_scalar_record_value(row.get("subject_id"))) for row in rows)
    return hashlib.sha256("\n".join(subjects).encode()).hexdigest()


def _spectrum_rows(rows: list[dict[str, Any]], model_forward: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from Model.ablation.frequency_uda import band_power_fractions
    records, audits = [], []
    for population, row in rows:
        result = model_forward(row, True)
        pre, post = np.asarray(result.get("pre_gate_features", result.get("pre_features", row["image"]))), np.asarray(result.get("post_gate_features", result["features"]))
        for stage, values in (("pre", pre), ("post", post)):
            while values.ndim < 5: values = values[None]
            fractions, _, _ = band_power_fractions(__import__("torch").as_tensor(values, dtype=__import__("torch").float32))
            fractions = fractions.detach().cpu().numpy()
            for band, name in enumerate(("low", "mid", "high")):
                records.append({"population": population, "stage": stage, "band": name, "mean": float(fractions[:, band].mean()), "std": float(fractions[:, band].std()), "count": int(len(fractions))})
        audit = result.get("audit", {})
        audits.append({"population": population, "subject_id": str(row.get("subject_id", "")), **{k: float(np.asarray(v).mean()) for k, v in audit.items() if np.asarray(v).size and np.issubdtype(np.asarray(v).dtype, np.number)}})
    return records, audits


def run_c4_mechanism_audit(
    *, resolved_config: dict[str, Any], source_split: str | Path, target_split: str | Path, prior: str | Path,
    source_original_capm_checkpoint: str | Path, final_c4_checkpoint: str | Path, output_dir: str | Path, seed: int,
    direction: str, device: str = "cpu", n_bootstrap: int = 1000, bootstrap_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a frozen C4 audit using source rows and unlabeled T_adapt only."""
    import torch
    target_metadata = _read_json(target_split)
    validate_target_split_contract(target_metadata)
    adapt_subjects = validate_target_subject_disjointness(target_metadata)
    source_manifest = _read_json(source_split) if Path(source_split).exists() else {}
    source_rows = _earliest_rows(_call_factory(resolved_config.get("source_loader_factory"), split="S_train"))
    source_val_rows = _earliest_rows(_call_factory(resolved_config.get("source_val_loader_factory"), split="S_val"))
    adapt_rows = _earliest_rows(_call_factory(resolved_config.get("target_adapt_loader_factory"), split="target_adapt"))
    if any(set(map(str, row.get("subject_id", []))) - adapt_subjects for row in adapt_rows): raise ValueError("target adaptation loader returned a subject outside T_adapt")
    source_rows = _with_frozen_covariates(source_rows, source_manifest)
    source_val_rows = _with_frozen_covariates(source_val_rows, source_manifest)
    adapt_rows = _with_frozen_covariates(adapt_rows, source_manifest)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    prior_payload = _read_json(prior) if Path(prior).exists() else {}
    model_forward = resolved_config.get("model_forward")
    if model_forward is None:
        raise ValueError("resolved_config must provide model_forward bound to frozen checkpoints")
    all_rows = [("S_train", row) for row in source_rows] + [("S_val", row) for row in source_val_rows] + [("T_adapt", row) for row in adapt_rows]
    spectrum_rows, gate_rows = _spectrum_rows(all_rows, model_forward)
    source_fractions, target_fractions, post_source_fractions, post_target_fractions = [], [], [], []
    for population, row in all_rows:
        result = model_forward(row, True)
        values = result.get("pre_gate_features", result.get("pre_features", row["image"]))
        array = np.asarray(values)
        while array.ndim < 5: array = array[None]
        tensor = torch.as_tensor(array, dtype=torch.float32)
        fractions, _, _ = __import__("Model.ablation.frequency_uda", fromlist=["band_power_fractions"]).band_power_fractions(tensor)
        if population == "S_train": source_fractions.append(fractions[0].detach().cpu().numpy())
        elif population == "T_adapt": target_fractions.append(fractions[0].detach().cpu().numpy())
        post = np.asarray(result.get("post_gate_features", result["features"])); post = np.asarray(post)
        while post.ndim < 5: post = post[None]
        post_frac, _, _ = __import__("Model.ablation.frequency_uda", fromlist=["band_power_fractions"]).band_power_fractions(torch.as_tensor(post, dtype=torch.float32))
        if population == "S_train": post_source_fractions.append(post_frac[0].detach().cpu().numpy())
        elif population == "T_adapt": post_target_fractions.append(post_frac[0].detach().cpu().numpy())
    stability = bootstrap_discrepancy_stability(np.asarray(source_fractions), np.asarray(target_fractions), n_bootstrap, seed)
    post_stability = bootstrap_discrepancy_stability(np.asarray(post_source_fractions), np.asarray(post_target_fractions), n_bootstrap, seed)
    provenance = {"direction": direction, "seed": int(seed), "device": str(device), "n_bootstrap": int(n_bootstrap), "target_labels_read": False, "target_metrics_read": False, "target_test_accessed": False, "prior_source_population": {"scope": "S_train", "count": len(source_rows), "subject_digest": _subject_digest(source_rows)}, "diagnostic_populations": {"S_val": {"count": len(source_val_rows), "subject_digest": _subject_digest(source_val_rows)}, "T_adapt": {"count": len(adapt_rows), "subject_digest": _subject_digest(adapt_rows)}}, "source_split": str(source_split), "target_split": str(target_split), "prior": str(prior), "source_original_capm_checkpoint": str(source_original_capm_checkpoint), "final_c4_checkpoint": str(final_c4_checkpoint), "hashes": {name: _sha256(path) for name, path in (("source_split", source_split), ("target_split", target_split), ("prior", prior), ("source_checkpoint", source_original_capm_checkpoint), ("final_checkpoint", final_c4_checkpoint))}}
    boundary = {"target_labels_read": False, "target_metrics_read": False, "target_test_accessed": False}
    provenance.update(boundary)
    (output / "prior_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    band_rows = []
    diagnostic_source_fractions = {"S_train": source_fractions, "S_val": [], "T_adapt": target_fractions}
    for population, row in [("S_val", row) for row in source_val_rows]:
        result = model_forward(row, True)
        values = np.asarray(result.get("pre_gate_features", result.get("pre_features", row["image"])))
        while values.ndim < 5: values = values[None]
        fractions, _, _ = __import__("Model.ablation.frequency_uda", fromlist=["band_power_fractions"]).band_power_fractions(torch.as_tensor(values, dtype=torch.float32))
        diagnostic_source_fractions[population].append(fractions[0].detach().cpu().numpy())
    for population, fractions in diagnostic_source_fractions.items():
        values = np.asarray(fractions, float)
        for index, band in enumerate(("low", "mid", "high")):
            band_rows.append({"population": population, "band": band, "count": int(len(values)), "mean": float(values[:, index].mean()) if len(values) else np.nan, "std": float(values[:, index].std()) if len(values) else np.nan, "fraction": float(values[:, index].mean()) if len(values) else np.nan})
    pd.DataFrame(band_rows).to_csv(output / "source_target_band_summary.csv", index=False)
    (output / "prior_bootstrap_stability.json").write_text(json.dumps({**boundary, **_jsonable(stability)}, indent=2))
    pd.DataFrame(gate_rows).to_csv(output / "gate_activity.csv", index=False)
    pd.DataFrame(spectrum_rows).to_csv(output / "pre_post_spectrum_summary.csv", index=False)
    comparison = compare_gate_forwards(model_forward, all_rows[0][1])
    comparison.update({**boundary, "checkpoint_mutated": False})
    (output / "c4_identity_forward_comparison.json").write_text(json.dumps(_jsonable(comparison), indent=2))
    env_rows = []
    environment_evaluator = resolved_config.get("environment_evaluator")
    if environment_evaluator is None:
        raise ValueError("resolved_config must provide source environment evaluator")
    for environment in ("original", "lowpass", "downsample_resample", "mild_blur"):
        values = dict(environment_evaluator(source_rows, environment))
        values["worst_group_risk"] = None
        values["worst_group_risk_source"] = "not_available_without_training_group_state"
        values["groupdro_weight"] = values.get("groupdro_weight")
        values["groupdro_weight_source"] = values.get("groupdro_weight_source", "not_available_from_frozen_checkpoint" if values["groupdro_weight"] is None else "replayed_training_history")
        env_rows.append({"population": "source", "environment": environment, **values, "evaluated": True})
    pd.DataFrame(env_rows).to_csv(output / "source_environment_audit.csv", index=False)
    source_covariates = [{k: row.get(k) for k in ("age", "sex", "education")} for population, row in all_rows if population in {"S_train", "S_val"}]
    target_covariates = [{k: row.get(k) for k in ("age", "sex", "education")} for population, row in all_rows if population == "T_adapt"]
    summarize_covariate_support(source_covariates, target_covariates, sex_impute=int(source_manifest.get("covariate_preprocessor", {}).get("sex_impute", 0))).to_csv(output / "covariate_support_audit.csv", index=False)
    (output / "pre_post_discrepancy.json").write_text(json.dumps({**boundary, "prior_source_scope": "S_train", "pre": _jsonable(stability), "post": _jsonable(post_stability), "source_count": len(source_rows), "source_val_count": len(source_val_rows), "target_adapt_count": len(adapt_rows), "diagnostic_populations": {"S_val": {"count": len(source_val_rows), "subject_digest": _subject_digest(source_val_rows)}}, "prior_source_subject_digest": _subject_digest(source_rows)}, indent=2))
    names = {path.name for path in output.iterdir() if path.is_file()}
    required = set(_REQUIRED_AUDIT_ARTIFACTS)
    if names != required:
        raise RuntimeError(f"audit artifact contract violation: expected={sorted(required)}, found={sorted(names)}")
    return {**boundary, "artifacts": list(_REQUIRED_AUDIT_ARTIFACTS)}



def _checkpoint_model(config: dict[str, Any], prior_path: str | Path, source_checkpoint: str | Path, final_checkpoint: str | Path, device: str):
    import torch
    from Model.ablation.frequency_uda import FrequencyGuidedScaleTable3D, FrequencyPrior
    prior_obj = FrequencyPrior.load(prior_path)
    ablation = config.get("scale_table_ablation") or {}
    model = FrequencyGuidedScaleTable3D(
        prior=prior_obj, preset="layer5_pixel", num_classes=len(set(config["task"]["label_mapping"].values())),
        layers=tuple(int(v) for v in ablation.get("layers", (2, 2, 2, 2))),
        spatial_shape=tuple(int(v) for v in ablation.get("spatial_shape", (4, 4, 4))),
        transformer_dim=int(ablation.get("transformer_dim", 128)), num_heads=int(ablation.get("num_heads", 4)),
        transformer_dropout=float(ablation.get("transformer_dropout", .1)), classifier_dropout=float(ablation.get("classifier_dropout", .3)),
        gate_init=float(ablation.get("gate_init", .95)), input_shape=tuple(int(v) for v in ablation.get("input_shape", (160, 196, 160))),
    ).to(device)
    model.load_source_baseline(source_checkpoint, map_location=device)
    payload = torch.load(final_checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state", payload)
    incompatible = model.load_state_dict(state, strict=False)
    missing = {k for k in incompatible.missing_keys if not k.startswith("frequency_gate.")}
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"final C4 checkpoint incompatible: missing={sorted(missing)}, unexpected={sorted(incompatible.unexpected_keys)}")
    model.eval()
    return model


def build_target_adapt_rows(dataset: Any, indices: Iterable[int], adapt_ids: set[str]) -> list[dict[str, Any]]:
    output = []
    for index in indices:
        record = dataset.records[int(index)]
        subject_id = record["subject_id"]
        if str(subject_id) not in adapt_ids:
            continue
        output.append({"image": dataset._load_image(record["path"]).numpy(), "subject_id": [subject_id], "age": [record.get("age")], "sex": [record.get("sex")], "education": [record.get("education")]})
    return _earliest_rows(output)


def forward_with_optional_gate(model: Any, image: Any, covariates: Any, *, gate_enabled: bool) -> dict[str, Any]:
    import torch
    pre_features = model.extract_layer4(image)
    previous_audit = getattr(model.frequency_gate, "last_audit", None)
    try:
        if gate_enabled:
            gated_features, frequency_audit = model.frequency_gate(pre_features, return_audit=True)
        else:
            gated_features, frequency_audit = pre_features, {"gate_bypassed": torch.ones(1, device=pre_features.device)}
        post_features = model.layer5(gated_features)
        post_features, capm_audit = model._apply_calibrator(post_features, covariates, False, True)
        logits = model.fc(model.dropout(model.pool(post_features).flatten(1)))
        audit = {f"frequency_{key}": value for key, value in frequency_audit.items()}
        audit.update(capm_audit)
        return {"pre_features": pre_features.detach().cpu().numpy(), "pre_gate_features": pre_features.detach().cpu().numpy(), "post_gate_features": gated_features.detach().cpu().numpy(), "post_features": post_features.detach().cpu().numpy(), "features": post_features.detach().cpu().numpy(), "logits": logits.detach().cpu().numpy(), "audit": {key: value.detach().cpu().numpy() for key, value in audit.items()}}
    finally:
        if hasattr(model.frequency_gate, "last_audit"):
            model.frequency_gate.last_audit = previous_audit


def build_cli_runtime(config: dict[str, Any], source_split: str | Path, target_split: str | Path, prior: str | Path, source_checkpoint: str | Path, final_checkpoint: str | Path, direction: str, device: str):
    import torch
    from experiments.train_journal import _dataset, _load_frozen_split
    source_name, target_name = direction.split("_to_")
    source, target = _dataset(config, source_name), _dataset(config, target_name)
    source_train, source_val, _, _ = _load_frozen_split(source, str(source_split))
    metadata = _read_json(target_split); adapt_ids = {str(x) for x in metadata["target_adapt_subjects"]}
    def rows(dataset, indices, label_allowed):
        output = []
        for index in indices:
            record = dataset.records[int(index)]
            if str(record["subject_id"]) not in adapt_ids and not label_allowed: continue
            item = {"image": dataset._load_image(record["path"]).numpy(), "subject_id": [record["subject_id"]], "age": [record.get("age")], "sex": [record.get("sex")], "education": [record.get("education")]}
            if label_allowed: item["label"] = [int(record["label"])]
            output.append(item)
        return _earliest_rows(output)
    target_indices = [i for i, record in enumerate(target.records) if str(record["subject_id"]) in adapt_ids]
    model = _checkpoint_model(config, prior, source_checkpoint, final_checkpoint, device)
    def forward(row, gate_enabled):
        image = torch.as_tensor(row["image"], dtype=torch.float32, device=device)
        if image.ndim == 4: image = image.unsqueeze(0)
        cov = torch.as_tensor(row["covariates"], dtype=torch.float32, device=device)
        with torch.no_grad():
            return forward_with_optional_gate(model, image, cov, gate_enabled=gate_enabled)
    def env_eval(rows_, environment):
        from training.frequency_environments import FrequencyEnvironmentAugment3D
        from sklearn.metrics import balanced_accuracy_score, log_loss
        augment = FrequencyEnvironmentAugment3D(); values = []
        for row in rows_:
            image = torch.as_tensor(row["image"], dtype=torch.float32, device=device); image = image.unsqueeze(0) if image.ndim == 4 else image
            eid = ("original", "lowpass", "downsample_resample", "mild_blur").index(environment)
            transformed, _ = augment(image, environment_ids=torch.tensor([eid], device=device))
            result = forward({**row, "image": transformed.cpu().numpy()}, True); values.append((int(row["label"][0]), result["logits"][0]))
        labels, logits = zip(*values); probs = torch.softmax(torch.tensor(np.asarray(logits)), 1).numpy()
        return {"cross_entropy": float(log_loss(labels, probs, labels=list(range(probs.shape[1])))), "balanced_accuracy": float(balanced_accuracy_score(labels, np.argmax(probs, 1))), "worst_group_risk": None, "worst_group_risk_source": "not_available_without_training_group_state", "groupdro_weight": None, "groupdro_weight_source": "not_available_from_frozen_checkpoint"}
    return {"source_loader_factory": lambda split="S_train": rows(source, source_train, True), "source_val_loader_factory": lambda split="S_val": rows(source, source_val, True), "target_adapt_loader_factory": lambda split="target_adapt": build_target_adapt_rows(target, target_indices, adapt_ids), "model_forward": forward, "environment_evaluator": env_eval}


def main(argv=None) -> None:
    import argparse, yaml
    parser = argparse.ArgumentParser(description="Leakage-safe frozen C4 mechanism audit")
    for name in ("config", "source-split", "target-split", "prior", "source-original-capm-checkpoint", "final-c4-checkpoint", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--direction", required=True, choices=("ADNI_to_NACC", "NACC_to_ADNI")); parser.add_argument("--seed", required=True, type=int); parser.add_argument("--device", default="cpu"); parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    runtime = build_cli_runtime(config, args.source_split, args.target_split, args.prior, args.source_original_capm_checkpoint, args.final_c4_checkpoint, args.direction, args.device)
    print(json.dumps(run_c4_mechanism_audit(resolved_config=runtime, source_split=args.source_split, target_split=args.target_split, prior=args.prior, source_original_capm_checkpoint=args.source_original_capm_checkpoint, final_c4_checkpoint=args.final_c4_checkpoint, output_dir=args.output_dir, seed=args.seed, direction=args.direction, device=args.device, n_bootstrap=args.n_bootstrap), indent=2))


if __name__ == "__main__":
    main()
