import torch

from Model.backbone.evidence_calibrated_capm import (
    EvidenceCalibratedCAPM,
    ResNetEvidenceCalibratedCAPMBackbone,
)


SPECS = [
    {"name": "age", "type": "continuous", "min_val": 50, "max_val": 100, "n_bases": 2},
    {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 1},
]


def test_gate_one_is_exact_capm_control():
    torch.manual_seed(4)
    module = EvidenceCalibratedCAPM(SPECS, feature_dim=4, spatial_shape=(2, 2, 2))
    x = torch.randn(2, 4, 3, 3, 3)
    z = torch.tensor([[65.0, 0.0], [80.0, 1.0]])
    _, audit = module(x, z, force_capm=True, return_audit=True)
    assert torch.equal(audit["gates"], torch.ones_like(audit["gates"]))
    expected = x + module._normalize_residual(x * (1.0 - torch.sigmoid(audit["raw_fields"].sum(1, keepdim=True))))
    controlled = module(x, z, force_capm=True)
    torch.testing.assert_close(controlled, expected)


def test_gate_has_no_tabular_input_and_losses_are_finite():
    module = EvidenceCalibratedCAPM(SPECS, feature_dim=4, spatial_shape=(2, 2, 2))
    assert module.evidence_gate[0].in_channels == 4
    assert module.evidence_gate[-1].out_channels == len(SPECS)
    x = torch.randn(2, 4, 3, 3, 3)
    z = torch.tensor([[65.0, 0.0], [80.0, 1.0]])
    module(x, z)
    for loss in module.regularization_losses().values():
        assert torch.isfinite(loss)


def test_backbone_exposes_audit_and_regularizers():
    model = ResNetEvidenceCalibratedCAPMBackbone(
        txt_dim=2,
        num_classes=3,
        var_specs=SPECS,
        layers=(1, 1, 1, 1),
        spatial_shape=(2, 2, 2),
        evidence_hidden=4,
    )
    logits, audit = model(
        torch.randn(2, 1, 32, 32, 32),
        torch.tensor([[65.0, 0.0], [80.0, 1.0]]),
        return_audit=True,
    )
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert set(audit) == {"layer1", "layer2", "layer3", "layer4"}
    assert set(model.regularization_losses()) == {
        "basis_tv", "basis_orth", "gate_anchor", "gate_floor", "modulation_preservation"
    }
