import torch
from torch import nn

from Model.ablation.capm_frequency_grl import (
    CAPMFrequencyGRL3D,
    PROJECTOR_SCHEMA,
    TaskSupportProjector,
    compute_capm_frequency_losses,
    make_frequency_batch,
)


def test_task_projector_is_orthogonal_and_serializable(tmp_path):
    torch.manual_seed(4)
    features = torch.randn(8, 6)
    labels = torch.tensor([0, 1] * 4)
    classifier = nn.Linear(6, 2)
    projector = TaskSupportProjector.fit_from_pooled_features(features, labels, classifier, rank=3)
    assert projector.rank == 3
    assert projector.metadata["schema"] == PROJECTOR_SCHEMA
    spatial = features.view(8, 6, 1, 1, 1)
    projected = projector.project(spatial)
    residual = projector.residual(spatial)
    assert projected.shape == spatial.shape
    assert torch.allclose(projector.project(residual), torch.zeros_like(residual), atol=1e-5)
    path = tmp_path / "projector.json"
    projector.save(str(path))
    restored = TaskSupportProjector.load(str(path))
    assert torch.allclose(restored.basis, projector.basis)
    assert restored.metadata["target_labels_read"] is False


def test_frequency_batch_is_shape_preserving_and_target_label_free():
    source = torch.randn(2, 1, 8, 8, 8)
    target = torch.randn_like(source)
    batch = make_frequency_batch(source, target, generator=torch.Generator().manual_seed(2))
    assert batch.source_image.shape == source.shape
    assert batch.source_intensity.shape == source.shape
    assert batch.target_style.shape == source.shape


def test_full_and_residual_grl_losses_have_separate_inputs():
    torch.manual_seed(9)
    source = torch.randn(2, 1, 32, 32, 32)
    target = torch.randn_like(source)
    table = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    labels = torch.tensor([0, 1])
    full = CAPMFrequencyGRL3D(domain_grl=True, intensity_grl=True, input_shape=(32, 32, 32), layers=(1, 1, 1, 1))
    with torch.no_grad():
        _, pooled = full(source, table, return_features=True)
    projector = TaskSupportProjector.fit_from_pooled_features(pooled.mean(dim=(-3, -2, -1)), labels, full.classifier, rank=4)
    residual = CAPMFrequencyGRL3D(projector=projector, grl_mode="residual", domain_grl=True, intensity_grl=True, input_shape=(32, 32, 32), layers=(1, 1, 1, 1))
    generated = make_frequency_batch(source, target, generator=torch.Generator().manual_seed(3))
    loss, parts = compute_capm_frequency_losses(residual, generated, table, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"classification", "domain", "intensity", "attention", "anchor", "domain_accuracy", "intensity_accuracy", "domain_auc", "intensity_auc", "adversarial_feature_rms", "full_feature_rms", "full_feature_mean_shift", "adversarial_feature_mean_shift", "full_feature_mmd_proxy", "adversarial_feature_mmd_proxy", "source_amplitude_mean", "target_style_amplitude_mean"} == set(parts)
    assert residual.domain_discriminator[0].weight.grad is not None
    assert residual.experiment_signature()["grl_mode"] == "residual"


def test_residual_mode_requires_projector():
    try:
        CAPMFrequencyGRL3D(grl_mode="residual", input_shape=(16, 16, 16), layers=(1, 1, 1, 1))
    except ValueError as error:
        assert "projector" in str(error)
    else:
        raise AssertionError("residual mode accepted without a projector")
