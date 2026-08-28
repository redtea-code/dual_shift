import json

import torch
import numpy as np
from types import SimpleNamespace
import pytest

try:
    from sklearn.model_selection import train_test_split  # noqa: F401
except ModuleNotFoundError as error:
    pytest.skip(
        f"train_journal import requires a working scikit-learn/SciPy installation: {error}",
        allow_module_level=True,
    )

from experiments.train_journal import (
    FREQUENCY_UDA_VARIANTS,
    SCALE_TABLE_VARIANTS,
    VARIANT_STAGES,
    _logits,
    _load_frequency_uda_target_test_indices,
    _make_model,
    _run_epoch,
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



def test_frequency_uda_variants_require_and_load_a_frozen_original_capm_checkpoint(tmp_path):
    from Model.ablation import FrequencyPrior, build_scale_table_ablation

    config = _config("layer5_pixel")
    prior = FrequencyPrior.from_summaries(
        {"count": 4, "mean": [0.5, 0.3, 0.2], "std": [0.02, 0.03, 0.02]},
        {"count": 4, "mean": [0.4, 0.3, 0.3], "std": [0.03, 0.02, 0.03]},
    )
    prior_path = tmp_path / "prior.json"
    prior.save(prior_path)
    baseline = build_scale_table_ablation(
        preset="layer5_pixel",
        interaction="original_capm",
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
    )
    checkpoint_path = tmp_path / "original_capm.pt"
    torch.save({"model_state": baseline.state_dict()}, checkpoint_path)
    config["frequency_uda"] = {
        "prior_path": str(prior_path),
        "base_checkpoint": str(checkpoint_path),
        "max_strength": 0.4,
        "init_strength": 0.04,
    }
    batch = {
        "image": torch.randn(2, 1, 32, 32, 32),
        "covariates": torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]]),
    }
    for variant in FREQUENCY_UDA_VARIANTS:
        model = _make_model(config, num_classes=2, variant=variant)
        logits = _logits(model, batch, spatial=False, variant=variant)
        assert logits.shape == (2, 2)
        if variant in {"frequency_uda_baseline", "frequency_uda_env_dro"}:
            assert model.experiment_signature()["interaction"] == "original_capm"
            assert torch.equal(model.conv1.weight, baseline.conv1.weight)
        else:
            assert model.experiment_signature()["model_family"] == "frequency_uda"


def test_frequency_uda_training_uses_frequency_groups_and_identity_regularization(tmp_path):
    from Model.ablation import FrequencyPrior, build_scale_table_ablation
    from training.frequency_environments import FrequencyEnvironmentAugment3D
    from training.group_dro import GroupDRO

    config = _config("layer5_pixel")
    prior = FrequencyPrior.from_summaries(
        {"count": 4, "mean": [0.5, 0.3, 0.2], "std": [0.02, 0.03, 0.02]},
        {"count": 4, "mean": [0.4, 0.3, 0.3], "std": [0.03, 0.02, 0.03]},
    )
    prior_path = tmp_path / "prior.json"
    prior.save(prior_path)
    baseline = build_scale_table_ablation(
        preset="layer5_pixel",
        interaction="original_capm",
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
    )
    checkpoint_path = tmp_path / "original_capm.pt"
    torch.save({"model_state": baseline.state_dict()}, checkpoint_path)
    config["frequency_uda"] = {
        "prior_path": str(prior_path),
        "base_checkpoint": str(checkpoint_path),
        "identity_weight": 0.01,
    }
    model = _make_model(config, num_classes=2, variant="frequency_uda")
    batch = {
        "image": torch.randn(2, 1, 32, 32, 32),
        "covariates": torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]]),
        "label": torch.tensor([0, 1]),
        "environment_id": torch.tensor([0, 1]),
        "subject_id": ["s1", "s2"],
        "folder": ["f1", "f2"],
    }
    result = _run_epoch(
        model,
        [batch],
        torch.device("cpu"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        dro=GroupDRO(4),
        config=config,
        variant="frequency_uda",
        frequency_augmenter=FrequencyEnvironmentAugment3D(),
        frequency_identity_weight=0.01,
    )
    assert result["frequency_identity_loss"] is not None
    assert result["frequency_effective_strength"] is not None
    assert result["frequency_effective_strength"] > 0
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_frequency_uda_evaluation_loader_excludes_target_adaptation_subjects(tmp_path):
    from experiments.frequency_uda import save_target_adaptation_split

    subject_ids = ["t1", "t1", "t2", "t3", "t4"]
    split_path = tmp_path / "target_split.json"
    save_target_adaptation_split(
        split_path,
        subject_ids,
        adaptation_indices=[0, 1, 2],
        test_indices=[3, 4],
        direction="ADNI_to_NACC",
        adaptation_fraction=0.5,
        seed=43,
    )
    target = SimpleNamespace(subject_ids=np.asarray(subject_ids, dtype=object))
    test_indices, payload = _load_frequency_uda_target_test_indices(
        target,
        np.arange(len(subject_ids)),
        str(split_path),
        direction="ADNI_to_NACC",
    )
    assert test_indices.tolist() == [3, 4]
    assert payload["target_labels_read"] is False

    payload["target_metrics_read"] = True
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target_metrics_read=false"):
        _load_frequency_uda_target_test_indices(
            target,
            np.arange(len(subject_ids)),
            str(split_path),
            direction="ADNI_to_NACC",
        )
