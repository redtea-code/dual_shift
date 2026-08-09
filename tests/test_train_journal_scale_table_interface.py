import torch
import pytest

try:
    from sklearn.model_selection import train_test_split  # noqa: F401
except ModuleNotFoundError as error:
    pytest.skip(
        f"train_journal import requires a working scikit-learn/SciPy installation: {error}",
        allow_module_level=True,
    )

from experiments.train_journal import (
    SCALE_TABLE_VARIANTS,
    VARIANT_STAGES,
    _logits,
    _make_model,
)


def _config(preset="layer4_pixel"):
    return {
        "model": {"name": "scale_table_ablation"},
        "training": {"image_shape": [32, 32, 32]},
        "scale_table_ablation": {
            "preset": preset,
            "input_shape": [32, 32, 32],
            "layers": [1, 1, 1, 1],
            "spatial_shape": [2, 2, 2],
            "transformer_dim": 16,
            "num_heads": 4,
            "transformer_dropout": 0.0,
            "classifier_dropout": 0.0,
        },
    }


def test_scale_table_variants_are_registered_and_use_the_training_interface():
    expected = {
        "image_only",
        "capm",
        "conv_gate",
        "original_capm",
        "transformer_self",
        "transformer_cross",
    }
    assert SCALE_TABLE_VARIANTS == expected
    assert expected.issubset(VARIANT_STAGES)

    batch = {
        "image": torch.randn(2, 1, 32, 32, 32),
        "covariates": torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]]),
    }
    for variant in expected:
        model = _make_model(_config(), num_classes=2, variant=variant)
        logits = _logits(model, batch, spatial=False, variant=variant)
        assert logits.shape == (2, 2)
        assert model.experiment_signature()["preset"] == "layer4_pixel"


def test_scale_table_preset_is_selected_from_the_task_yaml():
    model = _make_model(_config("layer3_patch2"), num_classes=2, variant="capm")
    assert model.experiment_signature()["preset"] == "layer3_patch2"
