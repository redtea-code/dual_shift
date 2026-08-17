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
