import torch

from Model.ablation.fmm_baseline import FMMNet, GradientReversal
from training.fmm_frequency import (
    fft_amplitude_phase,
    ifft_amplitude_phase,
    mix_amplitude,
    sample_lambda,
)
from training.fmm_protocol import (
    FMMDatasetView,
    SyntheticFMMDataset,
    split_source_indices,
    split_target_indices,
)
from experiments.train_fmm_baseline import _variant_flags


def test_fft_round_trip_and_endpoint_contracts():
    torch.manual_seed(3)
    source = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn_like(source)
    amplitude, phase = fft_amplitude_phase(source)
    reconstructed = ifft_amplitude_phase(amplitude, phase)
    assert torch.allclose(reconstructed, source, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mix_amplitude(source, target, 0.0), target, atol=1e-5, rtol=1e-5)
    target_amplitude, target_phase = fft_amplitude_phase(target)
    mixed = mix_amplitude(source, target, 1.0)
    mixed_amplitude, mixed_phase = fft_amplitude_phase(mixed)
    assert torch.allclose(mixed_amplitude, fft_amplitude_phase(source)[0], atol=1e-5, rtol=1e-5)
    # Phase is only meaningful modulo 2*pi; compare the reconstructed complex
    # spectrum instead of raw angle values at zeros.
    assert torch.allclose(
        torch.polar(mixed_amplitude, mixed_phase),
        torch.polar(fft_amplitude_phase(source)[0], target_phase),
        atol=1e-5,
        rtol=1e-5,
    )
    assert target_amplitude.shape == target_phase.shape


def test_gradient_reversal_changes_only_backward_sign():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    GradientReversal(0.25)(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.25))


def test_fmm_net_variable_shape_and_heads():
    model = FMMNet(channels=(2, 2, 4, 4, 8, 8, 16, 16, 32, 32), pool_shape=(1, 1, 1), dropout=0.0)
    x = torch.randn(2, 1, 32, 32, 32)
    output = model(x, return_attention=True)
    assert output["logits"].shape == (2, 2)
    assert output["features"].shape == (2, model.feature_dim)
    assert output["attention"].shape[:2] == (2, 1)
    domain = model.domain_logits(output["features"], coefficient=1.0)
    intensity = model.domain_logits(output["features"], coefficient=1.0, head="intensity")
    assert domain.shape == (2,)
    assert intensity.shape == (2,)


def test_lambda_sampling_is_bounded_and_reproducible():
    generator = torch.Generator().manual_seed(12)
    first = sample_lambda(4, low=0.2, high=0.7, device="cpu", generator=generator)
    generator = torch.Generator().manual_seed(12)
    second = sample_lambda(4, low=0.2, high=0.7, device="cpu", generator=generator)
    assert torch.equal(first, second)
    assert float(first.min()) >= 0.2
    assert float(first.max()) <= 0.7


def test_subject_split_and_label_blind_target_view():
    images = torch.randn(12, 1, 8, 8, 8)
    source = SyntheticFMMDataset(images, [0, 1] * 6, "S")
    target = SyntheticFMMDataset(images[:8], None, "T")
    train, val, test = split_source_indices(source, 0.5, 0.25, 0.25, 7)
    source_sets = [set(source.subject_ids[index].tolist()) for index in (train, val, test)]
    assert not source_sets[0] & source_sets[1]
    assert not source_sets[0] & source_sets[2]
    assert not source_sets[1] & source_sets[2]
    adapt, holdout = split_target_indices(target, 0.5, 7)
    target_view = FMMDatasetView(target, adapt, include_label=False)
    assert "label" not in target_view[0]
    assert not set(target.subject_ids[adapt].tolist()) & set(target.subject_ids[holdout].tolist())


def test_registered_variant_flags_match_ablation_contract():
    assert _variant_flags("b0_ref") == {
        "source_stage": False,
        "inter_stage": False,
        "source_fft": False,
        "attention": False,
        "grl": False,
    }
    assert _variant_flags("b1c_no_grl")["source_stage"]
    assert not _variant_flags("b1c_no_grl")["grl"]
    assert not _variant_flags("b1b_no_attention")["attention"]


def test_ds038_factorial_variant_flags_are_independent():
    from experiments.train_fmm_baseline import _variant_flags

    assert _variant_flags("g0_no_grl") == {
        "source_stage": True, "inter_stage": True, "source_fft": True,
        "attention": True, "domain_grl": False, "intensity_grl": False,
    }
    assert _variant_flags("g1_domain_only")["domain_grl"]
    assert not _variant_flags("g1_domain_only")["intensity_grl"]
    assert not _variant_flags("g2_intensity_only")["domain_grl"]
    assert _variant_flags("g2_intensity_only")["intensity_grl"]
    assert _variant_flags("g3_both_grl")["domain_grl"]
    assert _variant_flags("g3_both_grl")["intensity_grl"]


def test_ds038_head_diagnostics_have_discriminator_and_gradient_fields():
    from experiments.train_fmm_baseline import _head_diagnostics

    logits = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    result = _head_diagnostics(logits, labels, gradient_norm=0.25, coefficient=1.0)
    assert set(("loss", "accuracy", "balanced_accuracy", "auc", "gradient_norm", "grl_coefficient")) <= result.keys()
    assert result["gradient_norm"] == 0.25
    assert result["grl_coefficient"] == 1.0


def test_ds038_mechanism_record_is_checkpoint_bound():
    from experiments.train_fmm_baseline import _mechanism_record

    record = _mechanism_record("domain", epoch=2, step=5, is_best_checkpoint=True, metrics={"loss": 0.4})
    assert record == {"head": "domain", "epoch": 2, "step": 5, "is_best_checkpoint": True, "metrics": {"loss": 0.4}}


def test_ds038_frozen_probe_is_label_blind_and_reports_alignment_metrics():
    from experiments.train_fmm_baseline import _frozen_feature_domain_probe

    source = torch.zeros(8, 4)
    target = torch.ones(8, 4)
    result = _frozen_feature_domain_probe(source, target, seed=7, epochs=3)
    assert {"balanced_accuracy", "auc", "mmd", "source_norm", "target_norm"} <= result.keys()
    assert result["mmd"] > 0.0


def test_ds038_required_artifacts_detect_incomplete_run(tmp_path):
    from experiments.train_fmm_baseline import _required_artifacts_complete

    assert not _required_artifacts_complete(tmp_path)
    for name in ("best.pt", "summary.json", "audit.json", "config.yaml", "predictions.json"):
        (tmp_path / name).write_text("{}")
    assert _required_artifacts_complete(tmp_path)


def test_ds038_grl_gradient_probe_reverses_encoder_only():
    from experiments.train_fmm_baseline import _grl_gradient_probe

    result = _grl_gradient_probe(coefficient=1.0)
    assert result["encoder_gradient_with_grl"] == -result["encoder_gradient_without_grl"]
    assert result["discriminator_gradient_with_grl"] == result["discriminator_gradient_without_grl"]
