import torch
import pytest

from Model.ablation import (
    ABLATION_PRESETS,
    OriginalPatchwiseCAPM,
    ScaleTableInteractionAblation3D,
    TransformerCalibratedCAPM,
    demographic_var_specs,
)


def _table(batch_size=2):
    return torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])[:batch_size]


def test_equal_token_budget_presets_on_128_geometry():
    specs = demographic_var_specs()
    shallow = TransformerCalibratedCAPM(
        specs,
        feature_dim=8,
        patch_size=2,
        interaction_mode="table_cross",
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        dropout=0.0,
    )
    deep = TransformerCalibratedCAPM(
        specs,
        feature_dim=8,
        patch_size=1,
        interaction_mode="table_cross",
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        dropout=0.0,
    )
    _, shallow_audit = shallow(
        torch.randn(2, 8, 8, 8, 8), _table(), return_audit=True
    )
    _, deep_audit = deep(
        torch.randn(2, 8, 4, 4, 4), _table(), return_audit=True
    )
    assert shallow_audit["token_count"].item() == 64
    assert deep_audit["token_count"].item() == 64
    assert ABLATION_PRESETS["layer3_patch2"].expected_tokens_128 == 64
    assert ABLATION_PRESETS["layer4_pixel"].expected_tokens_128 == 64
    assert ABLATION_PRESETS["layer3_patch2"].expected_tokens_160x196x160 == 175
    assert ABLATION_PRESETS["layer4_pixel"].expected_tokens_160x196x160 == 175


def test_cross_attention_shapes_and_exact_capm_control():
    module = TransformerCalibratedCAPM(
        demographic_var_specs(),
        feature_dim=8,
        patch_size=1,
        interaction_mode="table_cross",
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        dropout=0.0,
    )
    features = torch.randn(2, 8, 3, 3, 3)
    output, audit = module(features, _table(), return_audit=True)
    assert output.shape == features.shape
    assert audit["attention"].shape == (2, 4, 27, 3)
    controlled, controlled_audit = module(
        features, _table(), force_capm=True, return_audit=True
    )
    assert torch.equal(
        controlled_audit["gates"], torch.ones_like(controlled_audit["gates"])
    )
    expected = features + module._normalize_residual(
        features
        * (1.0 - torch.sigmoid(controlled_audit["raw_fields"].sum(1, keepdim=True)))
    )
    torch.testing.assert_close(controlled, expected)


def test_transformer_pads_partial_patches_and_skips_attention_without_audit():
    module = TransformerCalibratedCAPM(
        demographic_var_specs(),
        feature_dim=8,
        patch_size=2,
        interaction_mode="image_self",
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        dropout=0.0,
    )
    padded_output = module(torch.randn(2, 8, 3, 3, 3), _table())
    assert padded_output.shape == (2, 8, 3, 3, 3)
    assert module.last_audit is not None
    assert module.last_audit["right_padding"].tolist() == [1, 1, 1]

    module(torch.randn(2, 8, 4, 4, 4), _table())
    assert module.last_audit is not None
    assert "attention" not in module.last_audit


def test_original_patchwise_capm_matches_source_residual_formula():
    module = OriginalPatchwiseCAPM(
        txt_dim=3,
        patch_size=2,
        expected_feature_shape=(4, 4, 4),
        table_dim=8,
    )
    for parameter in module.z_to_patch.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 4, 4, 4, 4)
    output, audit = module(x, _table(), return_audit=True)
    torch.testing.assert_close(output, 2.0 * x)
    assert audit["gamma"].shape == (2, 8)
    assert audit["patch_count"].item() == 8
    with pytest.raises(ValueError, match="batch sizes"):
        module(x, _table(batch_size=1))


def test_ablation_backbone_interface_and_losses():
    model = ScaleTableInteractionAblation3D(
        preset="layer3_patch2",
        interaction="transformer_cross",
        layers=(1, 1, 1, 1),
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        transformer_dropout=0.0,
    )
    logits, audit = model(
        torch.randn(2, 1, 32, 32, 32), _table(), return_audit=True
    )
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert audit["token_count"].item() == 1
    assert model.experiment_signature()["table_variables"] == (
        "age", "sex", "education"
    )
    assert all(torch.isfinite(value) for value in model.regularization_losses().values())
    assert model.get_regularization_losses().keys() == model.regularization_losses().keys()


def test_original_capm_ablation_uses_fixed_source_geometry():
    model = ScaleTableInteractionAblation3D(
        preset="layer3_patch2",
        interaction="original_capm",
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
    )
    logits, audit = model(
        torch.randn(2, 1, 32, 32, 32), _table(), return_audit=True
    )
    assert logits.shape == (2, 2)
    assert audit["patch_count"].item() == 1
    signature = model.experiment_signature()
    assert signature["input_shape"] == (32, 32, 32)
    assert signature["expected_tokens_160x196x160"] == 175


def test_original_capm_uses_explicit_padding_for_native_journal_shape():
    model = ScaleTableInteractionAblation3D(
        preset="layer3_patch2",
        interaction="original_capm",
        layers=(1, 1, 1, 1),
    )
    assert model.original_feature_shape == (10, 13, 10)
    assert model.calibrator.expected_feature_shape == (10, 14, 10)
    assert model.calibrator.patch_count == 175


def test_original_capm_right_pads_and_crops_odd_feature_dimension():
    module = OriginalPatchwiseCAPM(
        txt_dim=3,
        patch_size=2,
        expected_feature_shape=(4, 6, 4),
        table_dim=8,
    )
    for parameter in module.z_to_patch.parameters():
        torch.nn.init.zeros_(parameter)
    features = torch.randn(2, 4, 4, 5, 4)
    output, audit = module(features, _table(), return_audit=True)
    assert output.shape == features.shape
    torch.testing.assert_close(output, 2.0 * features)
    assert audit["right_padding"].tolist() == [0, 1, 0]


def test_image_only_mode_does_not_require_table():
    model = ScaleTableInteractionAblation3D(
        preset="layer4_pixel",
        interaction="image_only",
        layers=(1, 1, 1, 1),
    )
    logits = model(torch.randn(2, 1, 32, 32, 32))
    assert logits.shape == (2, 2)
