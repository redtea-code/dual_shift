import torch

from Model.ablation.scale_table_transformer import build_scale_table_ablation
from Model.ablation.target_style_transport import (
    TargetStyleCAPM,
    TargetStyleFeatureTransport3D,
)


def test_target_style_transport_identity_and_audit():
    source = torch.randn(2, 4, 5, 6, 7)
    target = torch.randn_like(source)
    transport = TargetStyleFeatureTransport3D(strength=0.0)
    output, audit = transport(source, target, return_audit=True)
    assert torch.equal(output, source)
    assert float(audit["strength"]) == 0.0
    assert float(audit["target_detached"]) == 1.0


def test_target_style_transport_preserves_source_shape_and_detaches_target():
    source = torch.randn(2, 4, 5, 6, 7, requires_grad=True)
    target = torch.randn(2, 4, 3, 4, 5, requires_grad=True)
    transport = TargetStyleFeatureTransport3D(strength=0.5)
    output = transport(source, target)
    assert output.shape == source.shape
    output.square().mean().backward()
    assert source.grad is not None
    assert target.grad is None


def test_target_style_capm_has_clean_and_mixed_paths():
    backbone = build_scale_table_ablation(
        preset="layer4_pixel",
        interaction="capm",
        num_classes=2,
        layers=(1, 1, 1, 1),
        spatial_shape=(2, 2, 2),
        input_shape=(32, 32, 32),
    )
    model = TargetStyleCAPM(backbone, transport_strength=0.25).eval()
    image = torch.randn(2, 1, 32, 32, 32)
    target = torch.randn_like(image)
    # CAPM expects the preprocessed [age, sex-code, education] contract;
    # sex is a categorical embedding and cannot receive arbitrary floats.
    covariates = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    output = model(image, covariates, target_image=target, return_audit=True)
    assert output["clean_logits"].shape == (2, 2)
    assert output["mixed_logits"].shape == (2, 2)
    assert output["mixed_features"].shape == output["source_features"].shape
    assert "transport_audit" in output
    assert torch.isfinite(output["mixed_logits"]).all()
