import json
import numpy as np
import pandas as pd
import pytest
from experiments.audit_frequency_uda_c4 import bootstrap_discrepancy_stability, summarize_covariate_support, summarize_gate_audit

def test_bootstrap_discrepancy_is_deterministic_and_reports_rank_probabilities():
    source = np.array([[.80,.20],[.75,.25],[.70,.30]])
    target = np.array([[.20,.80],[.25,.75],[.30,.70]])
    first = bootstrap_discrepancy_stability(source, target, n_bootstrap=1000, seed=43)
    assert first == bootstrap_discrepancy_stability(source, target, n_bootstrap=1000, seed=43)
    assert first["n_bootstrap"] == 1000
    assert np.allclose(first["discrepancy"], [.5,-.5])
    assert np.allclose(np.asarray(first["rank_probabilities"]).sum(axis=1), [1., 1.])
    assert np.isfinite(np.asarray(first["ci_low"] + first["ci_high"])).all()

def test_gate_summary_is_finite_for_scalar_audits():
    result = summarize_gate_audit([{"attenuation":.8,"strength":.1},{"attenuation":.6,"strength":.3}])
    assert result["count"] == 2
    assert result["fields"]["attenuation"]["mean"] == pytest.approx(.7)
    assert result["fields"]["strength"]["max"] == pytest.approx(.3)
    assert np.isfinite(result["fields"]["attenuation"]["std"])

def test_covariate_support_reports_missingness_and_numeric_ranges():
    result = summarize_covariate_support([{"age":70,"weight":60.},{"age":None,"weight":72.},{"age":74,"weight":np.nan}])
    assert isinstance(result, pd.DataFrame)
    assert list(result["covariate"]) == ["age","weight"]
    indexed = result.set_index("covariate")
    assert indexed.loc["age","missing_fraction"] == pytest.approx(1/3)
    assert indexed.loc["weight","missing_fraction"] == pytest.approx(1/3)
    assert indexed.loc["age","min"] == pytest.approx(70)
    assert indexed.loc["age","max"] == pytest.approx(74)
    assert np.isfinite(result[["missing_fraction","min","max"]].to_numpy()).all()


def test_target_split_contract_rejects_any_target_read_flag():
    from experiments.audit_frequency_uda_c4 import validate_target_split_contract
    with pytest.raises(ValueError, match="target_labels_read"):
        validate_target_split_contract({"target_labels_read": True, "target_metrics_read": False})
    with pytest.raises(ValueError, match="target_metrics_read"):
        validate_target_split_contract({"target_labels_read": False, "target_metrics_read": True})
    validate_target_split_contract({"target_labels_read": False, "target_metrics_read": False})

def test_target_subject_disjointness_uses_only_subject_membership():
    from experiments.audit_frequency_uda_c4 import validate_target_subject_disjointness
    assert validate_target_subject_disjointness({"target_adapt_subjects": ["a", "b"], "target_test_subjects": ["c"]}) == {"a", "b"}
    with pytest.raises(ValueError, match="disjoint"):
        validate_target_subject_disjointness({"target_adapt_subjects": ["a"], "target_test_subjects": ["a"]})

def test_gate_forwards_pair_same_inputs_and_finite_differences():
    from experiments.audit_frequency_uda_c4 import compare_gate_forwards
    calls = []
    def forward(model_inputs, gate_enabled):
        calls.append((model_inputs, gate_enabled))
        value = np.asarray(model_inputs, dtype=float) + (1.0 if gate_enabled else 0.0)
        return {"features": value, "logits": value * 2.0}
    result = compare_gate_forwards(forward, np.array([[1.0, 2.0]]))
    assert calls[0][1] is True and calls[1][1] is False
    assert np.array_equal(calls[0][0], calls[1][0])
    assert result["paired_input_equal"] is True and result["finite"] is True
    assert result["feature_abs_diff"] > 0 and result["logit_abs_diff"] > 0

def test_runner_writes_exact_nine_artifacts_without_target_test_loader(tmp_path):
    from experiments.audit_frequency_uda_c4 import run_c4_mechanism_audit
    target_split = tmp_path / "target.json"
    target_split.write_text('{"target_labels_read": false, "target_metrics_read": false, "target_adapt_subjects": ["ta"], "target_test_subjects": ["tt"]}')
    seen = []
    def source_loader():
        seen.append("source")
        return [{"image": np.ones((1, 2)), "subject_id": ["s"], "label": [1]}]
    def adapt_loader():
        seen.append("adapt")
        return [{"image": np.ones((1, 2)), "subject_id": ["ta"]}]
    def forbidden_test_loader():
        raise AssertionError("T_test loader was accessed")
    def model_forward(row, gate_enabled):
        x = np.asarray(row["image"], dtype=float); features = x if gate_enabled else x * 0.99
        return {"pre_features": x, "features": features, "logits": np.zeros((1, 2)), "audit": {"effective_strength": 0.01, "attenuation": np.ones((1, 3)) * 0.99, "identity_loss": 0.01}}
    def environment_evaluator(rows, environment):
        return {"cross_entropy": 0.5, "balanced_accuracy": 0.5, "worst_group_risk": 0.5, "groupdro_weight": 1.0}
    result = run_c4_mechanism_audit(resolved_config={"source_loader_factory": source_loader, "target_adapt_loader_factory": adapt_loader, "target_test_loader_factory": forbidden_test_loader, "model_forward": model_forward, "environment_evaluator": environment_evaluator}, source_split=tmp_path / "source.json", target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=tmp_path / "out", seed=43, direction="ADNI_to_NACC", device="cpu", n_bootstrap=8)
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"prior_provenance.json", "source_target_band_summary.csv", "prior_bootstrap_stability.json", "gate_activity.csv", "pre_post_spectrum_summary.csv", "c4_identity_forward_comparison.json", "source_environment_audit.csv", "covariate_support_audit.csv", "pre_post_discrepancy.json"}
    assert set(seen) == {"source", "adapt"} and result["target_test_accessed"] is False


def test_runner_rejects_placeholder_artifacts_and_records_real_mechanism_values(tmp_path):
    from experiments.audit_frequency_uda_c4 import run_c4_mechanism_audit
    target_split = tmp_path / "target.json"
    target_split.write_text(json.dumps({"target_labels_read": False, "target_metrics_read": False, "target_adapt_subjects": ["ta"], "target_test_subjects": ["tt"]}))
    rows = [
        {"image": np.ones((1, 4, 4, 4)), "subject_id": ["s"], "label": [1], "age": [70], "sex": ["female"], "education": [12]},
    ]
    adapt = [{"image": np.ones((1, 4, 4, 4)) * 2, "subject_id": ["ta"], "age": [72], "sex": ["male"], "education": [14]}]
    def model_forward(batch, gate_enabled):
        x = np.asarray(batch["image"], dtype=float)
        factor = 1.0 if gate_enabled else 0.5
        features = x * factor
        return {"pre_features": x, "features": features, "logits": np.column_stack((features.mean(axis=tuple(range(1, features.ndim))), features.mean(axis=tuple(range(1, features.ndim))) + 1)), "audit": {"effective_strength": factor, "attenuation": np.array([[factor, factor + .1, factor + .2]]), "identity_loss": abs(1-factor)}}
    out = tmp_path / "out"
    run_c4_mechanism_audit(resolved_config={"source_loader_factory": lambda: rows, "target_adapt_loader_factory": lambda: adapt, "model_forward": model_forward, "environment_evaluator": lambda rows, environment: {"cross_entropy": 0.5, "balanced_accuracy": 0.5, "worst_group_risk": 0.5, "groupdro_weight": 1.0}}, source_split=tmp_path / "source.json", target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=out, seed=43, direction="ADNI_to_NACC", n_bootstrap=8)
    gate = pd.read_csv(out / "gate_activity.csv")
    spectra = pd.read_csv(out / "pre_post_spectrum_summary.csv")
    identity = json.loads((out / "c4_identity_forward_comparison.json").read_text())
    assert len(gate) >= 1 and gate["effective_strength"].notna().all()
    assert len(spectra) >= 6 and spectra["mean"].notna().all()
    assert identity["feature_abs_diff"] > 0 and identity["logit_abs_diff"] > 0



def test_bootstrap_reports_plan_raw_and_normalized_discrepancies():
    source = np.array([[.80, .20], [.60, .40], [.70, .30]])
    target = np.array([[.20, .80], [.40, .60], [.30, .70]])
    result = bootstrap_discrepancy_stability(source, target, n_bootstrap=100, seed=43)
    signed = source.mean(0) - target.mean(0)
    pooled = np.sqrt((source.std(0) ** 2 + target.std(0) ** 2) / 2)
    raw_d = np.abs(signed) / pooled
    assert np.allclose(result["signed_mean_delta"], signed)
    assert np.allclose(result["raw_d"], raw_d)
    assert np.allclose(result["normalized_discrepancy"], raw_d / raw_d.max())
    assert np.isfinite(np.asarray(result["ci_low"] + result["ci_high"])).all()


def test_covariate_support_separates_populations_and_calculates_smd():
    result = summarize_covariate_support(
        [{"age": [70], "sex": [0], "education": [12]}, {"age": [74], "sex": [1], "education": [16]}],
        [{"age": [72], "sex": [1], "education": [14]}, {"age": [None], "sex": [0], "education": [14]}],
    )
    assert set(result["population"]) == {"source", "T_adapt"}
    age = result[result["covariate"] == "age"]
    assert set(age["missing_fraction"]) == {0.0, 0.5}
    assert age["standardized_mean_difference"].notna().all()


def test_runner_emits_band_statistics_and_json_boundary_metadata(tmp_path):
    from experiments.audit_frequency_uda_c4 import run_c4_mechanism_audit
    target_split = tmp_path / "target.json"
    target_split.write_text(json.dumps({"target_labels_read": False, "target_metrics_read": False, "target_adapt_subjects": ["ta"], "target_test_subjects": ["tt"]}))
    source = [{"image": np.ones((1, 4, 4, 4)), "subject_id": ["s"], "label": [1], "age": [70], "sex": [0], "education": [12]}]
    target = [{"image": np.ones((1, 4, 4, 4)) * 2, "subject_id": ["ta"], "age": [72], "sex": [1], "education": [14]}]
    def forward(row, enabled):
        x = np.asarray(row["image"], dtype=float)
        return {"pre_features": x, "features": x * (1 if enabled else .9), "logits": np.zeros((1, 2)), "audit": {"effective_strength": .1}}
    output = tmp_path / "out"
    run_c4_mechanism_audit(resolved_config={"source_loader_factory": lambda: source, "target_adapt_loader_factory": lambda: target, "model_forward": forward, "environment_evaluator": lambda rows, environment: {"cross_entropy": .2, "balanced_accuracy": .8, "worst_group_risk": .3}}, source_split=tmp_path / "source.json", target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=output, seed=43, direction="ADNI_to_NACC", n_bootstrap=8)
    bands = pd.read_csv(output / "source_target_band_summary.csv")
    assert set(bands["population"]) == {"S_train", "S_val", "T_adapt"}
    assert set(bands["band"]) == {"low", "mid", "high"}
    assert {"mean", "std", "fraction"}.issubset(bands.columns)
    for name in ("prior_provenance.json", "prior_bootstrap_stability.json", "c4_identity_forward_comparison.json", "pre_post_discrepancy.json"):
        payload = json.loads((output / name).read_text())
        assert payload["target_labels_read"] is False
        assert payload["target_metrics_read"] is False
        assert payload["target_test_accessed"] is False
    environments = pd.read_csv(output / "source_environment_audit.csv")
    assert set(environments["environment"]) == {"original", "lowpass", "downsample_resample", "mild_blur"}
    assert set(("cross_entropy", "balanced_accuracy", "worst_group_risk", "groupdro_weight", "groupdro_weight_source")).issubset(environments.columns)
    assert environments["groupdro_weight"].isna().all()
    assert set(environments["groupdro_weight_source"]) == {"not_available_from_frozen_checkpoint"}



def test_target_adapt_row_builder_never_accesses_target_labels():
    from experiments.audit_frequency_uda_c4 import build_target_adapt_rows
    class TargetRecord(dict):
        def __getitem__(self, key):
            if key == "label":
                raise AssertionError("target label was accessed")
            return super().__getitem__(key)
        def get(self, key, default=None):
            if key == "label":
                raise AssertionError("target label was accessed")
            return super().get(key, default)
    class TargetDataset:
        records = [TargetRecord(path="target.nii", subject_id="ta", age=72, sex="male", education=14)]
        def _load_image(self, path):
            assert path == "target.nii"
            return __import__("torch").ones(1, 4, 4, 4)
        def __getitem__(self, index):
            raise AssertionError("labeled dataset item was accessed")
    rows = build_target_adapt_rows(TargetDataset(), [0], {"ta"})
    assert set(rows[0]) == {"image", "subject_id", "age", "sex", "education"}


def test_optional_gate_forward_does_not_mutate_gate_or_model_state():
    import torch
    from experiments.audit_frequency_uda_c4 import forward_with_optional_gate
    class Gate(torch.nn.Module):
        def forward(self, x, return_audit=False):
            output = x + 1
            audit = {"strength": torch.tensor([.1])}
            return (output, audit) if return_audit else output
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frequency_gate = Gate()
            self.layer5 = torch.nn.Identity()
            self.pool = torch.nn.AdaptiveAvgPool3d(1)
            self.dropout = torch.nn.Identity()
            self.fc = torch.nn.Linear(1, 2)
        def extract_layer4(self, image): return image
        def _apply_calibrator(self, features, table, force_capm, return_audit): return features, {"calibrated": torch.tensor([1.])}
    model = Model().eval()
    image, covariates = torch.ones(1, 1, 2, 2, 2), torch.zeros(1, 3)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_forward = model.frequency_gate.forward
    enabled = forward_with_optional_gate(model, image, covariates, gate_enabled=True)
    disabled = forward_with_optional_gate(model, image, covariates, gate_enabled=False)
    assert enabled["pre_features"].shape == disabled["pre_features"].shape
    assert not np.array_equal(enabled["post_features"], disabled["post_features"])
    assert model.frequency_gate.forward.__func__ is original_forward.__func__
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_prior_reproduction_uses_only_s_train_and_reports_s_val_diagnostics(tmp_path):
    from experiments.audit_frequency_uda_c4 import run_c4_mechanism_audit

    target_split = tmp_path / "target.json"
    target_split.write_text(json.dumps({"target_labels_read": False, "target_metrics_read": False, "target_adapt_subjects": ["ta"], "target_test_subjects": ["tt"]}))
    source_split = tmp_path / "source.json"
    source_split.write_text(json.dumps({"covariate_preprocessor": {"age_median": 70, "education_median": 12, "age_mean": 70, "age_scale": 10, "education_mean": 12, "education_scale": 2, "sex_impute": 0}}))
    train = [{"image": np.ones((1, 4, 4, 4)), "subject_id": ["st"], "label": [1], "age": [70], "sex": ["female"], "education": [12]}]
    val = [{"image": np.arange(64, dtype=float).reshape(1, 4, 4, 4), "subject_id": ["sv"], "label": [0], "age": [80], "sex": ["male"], "education": [20]}]
    adapt = [{"image": np.ones((1, 4, 4, 4)) * 2, "subject_id": ["ta"], "age": [72], "sex": ["male"], "education": [14]}]
    def forward(row, enabled):
        x = np.asarray(row["image"], dtype=float)
        return {"pre_features": x, "features": x, "logits": np.zeros((1, 2)), "audit": {"effective_strength": .1}}
    config = {"source_loader_factory": lambda: train, "source_val_loader_factory": lambda: val, "target_adapt_loader_factory": lambda: adapt, "model_forward": forward, "environment_evaluator": lambda rows, environment: {"cross_entropy": .2, "balanced_accuracy": .8}}
    first = tmp_path / "first"
    run_c4_mechanism_audit(resolved_config=config, source_split=source_split, target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=first, seed=43, direction="ADNI_to_NACC", n_bootstrap=8)
    baseline = tmp_path / "baseline"
    run_c4_mechanism_audit(resolved_config={**config, "source_val_loader_factory": lambda: []}, source_split=source_split, target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=baseline, seed=43, direction="ADNI_to_NACC", n_bootstrap=8)
    assert json.loads((first / "prior_bootstrap_stability.json").read_text()) == json.loads((baseline / "prior_bootstrap_stability.json").read_text())
    assert json.loads((first / "pre_post_discrepancy.json").read_text())["pre"] == json.loads((baseline / "pre_post_discrepancy.json").read_text())["pre"]
    bands = pd.read_csv(first / "source_target_band_summary.csv")
    assert set(bands.population) == {"S_train", "S_val", "T_adapt"}
    assert bands.loc[bands.population == "S_train", "count"].eq(1).all()
    provenance = json.loads((first / "prior_provenance.json").read_text())
    assert provenance["prior_source_population"]["scope"] == "S_train"
    assert provenance["diagnostic_populations"]["S_val"]["count"] == 1


def test_runner_transforms_covariates_with_frozen_source_preprocessor(tmp_path):
    from experiments.audit_frequency_uda_c4 import run_c4_mechanism_audit

    target_split = tmp_path / "target.json"
    target_split.write_text(json.dumps({"target_labels_read": False, "target_metrics_read": False, "target_adapt_subjects": ["ta"], "target_test_subjects": ["tt"]}))
    source_split = tmp_path / "source.json"
    source_split.write_text(json.dumps({"covariate_preprocessor": {"age_median": 70, "education_median": 12, "age_mean": 60, "age_scale": 5, "education_mean": 10, "education_scale": 2, "sex_impute": 1, "scale_continuous": True, "impute_unknown_sex": True}}))
    source = [{"image": np.ones((1, 4, 4, 4)), "subject_id": ["s"], "label": [1], "age": [70], "sex": ["unknown"], "education": [None]}]
    adapt = [{"image": np.ones((1, 4, 4, 4)), "subject_id": ["ta"], "age": [50], "sex": ["female"], "education": [14]}]
    received = []
    def forward(row, enabled):
        received.append(np.asarray(row["covariates"], dtype=float).copy())
        x = np.asarray(row["image"], dtype=float)
        return {"pre_features": x, "features": x, "logits": np.zeros((1, 2)), "audit": {"effective_strength": .1}}
    run_c4_mechanism_audit(resolved_config={"source_loader_factory": lambda: source, "target_adapt_loader_factory": lambda: adapt, "model_forward": forward, "environment_evaluator": lambda rows, environment: {"cross_entropy": .2, "balanced_accuracy": .8}}, source_split=source_split, target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=tmp_path / "out", seed=43, direction="ADNI_to_NACC", n_bootstrap=8)
    assert any(np.allclose(value, [[2., 1., 1.]]) for value in received)
    assert any(np.allclose(value, [[-2., 0., 2.]]) for value in received)


def test_source_environment_worst_group_risk_is_unavailable_without_group_state(tmp_path):
    from experiments.audit_frequency_uda_c4 import run_c4_mechanism_audit

    target_split = tmp_path / "target.json"
    target_split.write_text(json.dumps({"target_labels_read": False, "target_metrics_read": False, "target_adapt_subjects": ["ta"], "target_test_subjects": ["tt"]}))
    source_split = tmp_path / "source.json"
    source_split.write_text(json.dumps({"covariate_preprocessor": {"age_median": 70, "education_median": 12, "age_mean": 70, "age_scale": 1, "education_mean": 12, "education_scale": 1, "sex_impute": 0}}))
    row = {"image": np.ones((1, 4, 4, 4)), "subject_id": ["s"], "label": [1], "age": [70], "sex": ["female"], "education": [12]}
    def forward(row, enabled):
        x = np.asarray(row["image"], dtype=float)
        return {"pre_features": x, "features": x, "logits": np.zeros((1, 2)), "audit": {"effective_strength": .1}}
    run_c4_mechanism_audit(resolved_config={"source_loader_factory": lambda: [row], "target_adapt_loader_factory": lambda: [{**row, "subject_id": ["ta"]}], "model_forward": forward, "environment_evaluator": lambda rows, environment: {"cross_entropy": .2, "balanced_accuracy": .8, "worst_group_risk": .2}}, source_split=source_split, target_split=target_split, prior=tmp_path / "prior.json", source_original_capm_checkpoint=tmp_path / "source.pt", final_c4_checkpoint=tmp_path / "final.pt", output_dir=tmp_path / "out", seed=43, direction="ADNI_to_NACC", n_bootstrap=8)
    environments = pd.read_csv(tmp_path / "out" / "source_environment_audit.csv")
    assert environments.worst_group_risk.isna().all()
    assert set(environments.worst_group_risk_source) == {"not_available_without_training_group_state"}


def test_spectrum_uses_gate_level_post_features_not_layer5_features():
    from experiments.audit_frequency_uda_c4 import _spectrum_rows
    row = {"image": np.ones((1, 1, 4, 4, 4)), "subject_id": ["s"]}
    def forward(_, enabled):
        gate = np.ones((1, 1, 4, 4, 4)) * 2
        return {"pre_gate_features": np.ones_like(gate), "post_gate_features": gate,
                "features": np.indices(gate.shape).sum(axis=0) % 2, "logits": np.zeros((1, 2)), "audit": {}}
    spectra, _ = _spectrum_rows([("S_train", row)], forward)
    post = [item for item in spectra if item["stage"] == "post"]
    assert post and post[0]["mean"] == pytest.approx(1.0)


def test_optional_gate_forward_exposes_gate_level_features_and_disabled_identity():
    import torch
    from experiments.audit_frequency_uda_c4 import forward_with_optional_gate
    class Gate(torch.nn.Module):
        def forward(self, x, return_audit=False):
            result = x + 1
            return (result, {}) if return_audit else result
    class Model(torch.nn.Module):
        frequency_gate = Gate()
        layer5 = torch.nn.Identity()
        pool = torch.nn.AdaptiveAvgPool3d(1)
        dropout = torch.nn.Identity()
        fc = torch.nn.Linear(1, 2)
        def extract_layer4(self, image): return image
        def _apply_calibrator(self, features, table, force_capm, return_audit): return features * 3, {}
    model = Model().eval()
    image = torch.ones(1, 1, 2, 2, 2)
    enabled = forward_with_optional_gate(model, image, torch.zeros(1, 3), gate_enabled=True)
    disabled = forward_with_optional_gate(model, image, torch.zeros(1, 3), gate_enabled=False)
    assert np.array_equal(enabled["post_gate_features"], np.ones((1, 1, 2, 2, 2)) * 2)
    assert np.array_equal(disabled["post_gate_features"], disabled["pre_gate_features"])


def test_covariate_support_encodes_sex_categories_and_reports_support():
    from experiments.audit_frequency_uda_c4 import summarize_covariate_support
    result = summarize_covariate_support(
        [{"sex": "female"}, {"sex": "male"}],
        [{"sex": "male"}, {"sex": "unknown"}],
    )
    sex = result[result["covariate"] == "sex"].set_index("population")
    assert sex.loc["source", "missing_count"] == 0
    assert sex.loc["T_adapt", "missing_count"] == 0
    assert sex.loc["source", "category_counts"] == '{"female": 1, "male": 1}'
    assert sex.loc["T_adapt", "category_counts"] == '{"male": 1, "unknown": 1}'
    assert np.isfinite(sex["encoded_mean"].to_numpy(float)).all()
