import pytest
import torch

from Model.ablation.raw_frequency_skip import (
    RawFrequencySkip3D,
    RawFrequencySkipScaleTable3D,
    extract_raw_frequency_features,
)
from experiments.train_journal import (
    RAW_FREQUENCY_SKIP_VARIANTS,
    _logits,
    _make_model,
    run,
)


def test_raw_frequency_descriptor_and_skip_are_finite_differentiable_and_nonidentity():
    torch.manual_seed(21)
    images = torch.randn(3, 1, 12, 14, 16)
    descriptor = extract_raw_frequency_features(images, spectral_grid=(3, 4, 5))
    assert descriptor.shape == (3, 3, 3, 4, 5)

    module = RawFrequencySkip3D(
        image_channels=1,
        target_channels=4,
        spectral_grid=(3, 4, 5),
        hidden_channels=8,
        max_residual=0.2,
        gate_init=0.8,
    )
    features = torch.randn(3, 4, 4, 5, 6, requires_grad=True)
    output, audit = module(images, features, return_audit=True)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert audit["raw_band_fractions"].shape == (3, 3)
    assert audit["residual_relative_rms"].gt(0).all()
    assert audit["nonidentity_fraction"].item() == 1.0
    output.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(module.raw_gate.grad)


def test_raw_frequency_skip_scale_table_model_and_journal_builder():
    torch.manual_seed(22)
    model = RawFrequencySkipScaleTable3D(
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
        spectral_grid=(3, 3, 3),
        hidden_channels=8,
    ).train()
    images = torch.randn(2, 1, 32, 32, 32)
    tables = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    logits, audit = model(images, tables, return_audit=True)
    assert logits.shape == (2, 2)
    assert audit["raw_frequency_skip_nonidentity_fraction"].item() == 1.0
    assert model.experiment_signature()["raw_frequency_skip_stage"] == "layer3"

    config = {
        "model": {},
        "task": {"label_mapping": {2: 0, 3: 1}},
        "training": {"image_shape": (32, 32, 32)},
        "scale_table_ablation": {
            "layers": (1, 1, 1, 1), "input_shape": (32, 32, 32), "spatial_shape": (2, 2, 2),
            "transformer_dim": 16, "num_heads": 4, "transformer_dropout": 0.0, "classifier_dropout": 0.0,
        },
        "raw_frequency_skip": {"spectral_grid": (3, 3, 3), "hidden_channels": 8},
    }
    variant = "raw_frequency_skip"
    assert variant in RAW_FREQUENCY_SKIP_VARIANTS
    journal_model = _make_model(config, 2, variant).train()
    journal_logits = _logits(
        journal_model,
        {"image": images, "covariates": tables, "label": torch.tensor([0, 1])},
        False,
        variant=variant,
    )
    assert journal_logits.shape == (2, 2)


def test_raw_frequency_skip_requires_source_only_before_dataset_loading(tmp_path):
    with pytest.raises(ValueError, match="source-only"):
        run(
            {"seed": 0, "variants": ["raw_frequency_skip"]},
            "ADNI_to_NACC",
            str(tmp_path),
            device="cpu",
            source_only=False,
        )
