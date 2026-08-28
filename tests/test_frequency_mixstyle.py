import pytest
import torch

from Model.ablation.frequency_mixstyle import (
    FrequencyMixStyle3D,
    FrequencyMixStyleScaleTable3D,
)
from experiments.train_journal import (
    FREQUENCY_MIXSTYLE_VARIANTS,
    _logits,
    _make_model,
    run,
)


def _phase_distance(left, right):
    left_spectrum = torch.fft.rfftn(left.float(), dim=(-3, -2, -1), norm="ortho")
    right_spectrum = torch.fft.rfftn(right.float(), dim=(-3, -2, -1), norm="ortho")
    valid = (left_spectrum.abs() > 1e-5) & (right_spectrum.abs() > 1e-5)
    return torch.angle(right_spectrum[valid] * left_spectrum[valid].conj()).abs().max()


def test_bandwise_mixstyle_changes_amplitude_but_preserves_recipient_phase_and_class_donors():
    torch.manual_seed(5)
    module = FrequencyMixStyle3D(mode="bandwise_statistics", probability=1.0, alpha=0.5)
    module.train()
    features = torch.randn(4, 3, 5, 6, 7, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    mixed, audit = module(features, style_labels=labels, return_audit=True)
    assert torch.isfinite(mixed).all()
    assert audit["applied_mask"].all()
    assert torch.equal(labels, labels[audit["partner_indices"]])
    assert audit["donor_eligible_fraction"].item() == 1.0
    assert audit["amplitude_relative_delta"].gt(0).all()
    assert audit["band_amplitude_relative_delta"].shape == (4, 3)
    assert audit["mix_lambda"].shape == (4, 3)
    assert audit["feature_relative_delta"].gt(0).all()
    assert _phase_distance(features.detach(), mixed.detach()) < 2e-4
    mixed.square().mean().backward()
    assert torch.isfinite(features.grad).all()


def test_singleton_class_and_evaluation_are_identity_fallbacks():
    torch.manual_seed(9)
    module = FrequencyMixStyle3D(mode="full_amplitude", probability=1.0, alpha=0.3)
    features = torch.randn(3, 2, 4, 5, 6)
    labels = torch.tensor([0, 0, 1])
    module.train()
    mixed, audit = module(features, style_labels=labels, return_audit=True)
    assert audit["applied_mask"].tolist().count(False) == 1
    assert audit["donor_eligible_mask"].tolist() == [True, True, False]
    assert torch.equal(mixed[2], features[2])
    module.eval()
    identity, eval_audit = module(features, return_audit=True)
    assert torch.equal(identity, features)
    assert eval_audit["applied_fraction"].item() == 0.0


def test_frequency_mixstyle_scale_table_model_and_journal_builder_accept_source_labels():
    torch.manual_seed(12)
    model = FrequencyMixStyleScaleTable3D(
        mix_mode="bandwise_statistics",
        probability=1.0,
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
    )
    model.train()
    images = torch.randn(4, 1, 32, 32, 32)
    tables = torch.tensor([[65.0, 0.0, 12.0], [66.0, 0.0, 13.0], [80.0, 1.0, 18.0], [81.0, 1.0, 17.0]])
    logits, audit = model(images, tables, style_labels=torch.tensor([0, 0, 1, 1]), return_audit=True)
    assert logits.shape == (4, 2)
    assert audit["frequency_mixstyle_applied_fraction"].item() > 0.0
    assert model.experiment_signature()["frequency_mixstyle_insert_stage"] == "layer3"

    config = {
        "model": {},
        "task": {"label_mapping": {2: 0, 3: 1}},
        "training": {"image_shape": (32, 32, 32)},
        "scale_table_ablation": {
            "layers": (1, 1, 1, 1), "input_shape": (32, 32, 32), "spatial_shape": (2, 2, 2),
            "transformer_dim": 16, "num_heads": 4, "transformer_dropout": 0.0, "classifier_dropout": 0.0,
        },
        "frequency_mixstyle": {"probability": 1.0, "alpha": 0.3, "mix_stage": "layer3"},
    }
    variant = "frequency_mixstyle_full"
    assert variant in FREQUENCY_MIXSTYLE_VARIANTS
    journal_model = _make_model(config, 2, variant).train()
    journal_logits = _logits(journal_model, {"image": images, "covariates": tables, "label": torch.tensor([0, 0, 1, 1])}, False, variant=variant)
    assert journal_logits.shape == (4, 2)


def test_frequency_mixstyle_journal_rejects_target_construction_before_data_loading(tmp_path):
    with pytest.raises(ValueError, match="source-only"):
        run(
            {"seed": 0, "variants": ["frequency_mixstyle_full"]},
            "ADNI_to_NACC",
            str(tmp_path),
            device="cpu",
            source_only=False,
        )
