import pytest
import torch

from Model.ablation.capm_frequency_grl import TaskSupportProjector
from Model.ablation.capm_residual_distribution_alignment import (
    CAPMResidualDistributionAlignment3D,
    RESIDUAL_DISTRIBUTION_STATS_SCHEMA,
    ResidualDistributionStats,
    ResidualDistributionTransport3D,
    SubjectResidualAccumulator,
    apply_bounded_intensity_perturbation,
    apply_fourier_amplitude_perturbation,
    audit_synthetic_perturbation,
    build_residual_distribution_stats_from_loaders,
    load_task_support_projector_artifact,
    save_residual_distribution_target_split,
)


def _stats(channels: int = 4, components: int = 1) -> ResidualDistributionStats:
    source = torch.cat((torch.randn(12, channels) * 0.5, torch.randn(12, channels) * 0.5 + 1.0))
    target = source + 0.4
    return ResidualDistributionStats.fit(source, target, components=components, iterations=10)


def _model_stats() -> ResidualDistributionStats:
    return _stats(512, components=1)


def _projector(channels: int = 512, rank: int = 8) -> TaskSupportProjector:
    basis = torch.eye(channels, dtype=torch.float32)[:, :rank]
    return TaskSupportProjector(basis, source_count=16, metadata={"schema": "dualshift_capm_task_projector_v1"})


def test_stats_fit_is_deterministic_and_roundtrips(tmp_path):
    torch.manual_seed(3)
    source = torch.cat((torch.randn(16, 4) * 0.2 - 1.0, torch.randn(16, 4) * 0.2 + 1.0))
    target = torch.cat((torch.randn(16, 4) * 0.2 - 0.5, torch.randn(16, 4) * 0.2 + 1.5))
    first = ResidualDistributionStats.fit(source, target, components=2, iterations=12)
    second = ResidualDistributionStats.fit(source, target, components=2, iterations=12)
    assert first.to_dict() == second.to_dict()
    path = tmp_path / "stats.json"
    first.save(path)
    restored = ResidualDistributionStats.load(path)
    assert restored.to_dict() == first.to_dict()
    assert restored.metadata["schema"] == RESIDUAL_DISTRIBUTION_STATS_SCHEMA


def test_projector_loader_accepts_nested_ds040_report(tmp_path):
    projector = _projector(channels=8, rank=2)
    path = tmp_path / "ds040_report.json"
    path.write_text(
        __import__("json").dumps({"schema": "dualshift_ds040_report_v1", "projector": projector.to_dict()}),
        encoding="utf-8",
    )
    restored = load_task_support_projector_artifact(path)
    assert torch.allclose(restored.basis, projector.basis)


def test_subject_accumulator_is_subject_level_and_rejects_duplicates():
    accumulator = SubjectResidualAccumulator()
    accumulator.update(torch.ones(2, 3, 2, 2, 2), ["s1", "s2"])
    with pytest.raises(ValueError, match="duplicate IDs"):
        accumulator.update(torch.ones(2, 3, 2, 2, 2), ["s2", "s3"])
    with pytest.raises(ValueError, match="duplicate IDs in batch"):
        accumulator.update(torch.ones(2, 3, 2, 2, 2), ["s3", "s3"])
    assert accumulator.pooled().shape == (2, 3)


def test_transport_zero_strength_is_exact_identity_and_gmm_is_finite():
    torch.manual_seed(4)
    features = torch.randn(3, 4, 2, 2, 2)
    stats = _stats(4, components=2)
    identity = ResidualDistributionTransport3D(stats, max_strength=0.0)
    output, audit = identity(features, return_audit=True)
    assert torch.equal(output, features)
    assert audit["finite"].item() == 1.0
    adapted = ResidualDistributionTransport3D(stats, max_strength=0.25)(features)
    assert adapted.shape == features.shape
    assert torch.isfinite(adapted).all()


def test_residual_scope_preserves_task_component():
    torch.manual_seed(5)
    projector = _projector()
    model = CAPMResidualDistributionAlignment3D(
        stats=_model_stats(),
        projector=projector,
        transport_scope="residual",
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        classifier_dropout=0.0,
    )
    image = torch.randn(2, 1, 32, 32, 32)
    table = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    control, _ = model.extract_features(image, table, apply_transport=False)
    adapted, _ = model.extract_features(image, table, apply_transport=True)
    task_delta = projector.project(adapted - control)
    assert torch.allclose(task_delta, torch.zeros_like(task_delta), atol=1e-5)
    assert model.experiment_signature()["transport_scope"] == "residual"


def test_statistics_builder_rejects_target_labels():
    class DummyModel:
        training = False

        def eval(self):
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def extract_layer4(self, image):
            return image.repeat(1, 4, 1, 1, 1)

    loader = [{"image": torch.ones(1, 1, 1, 1, 1), "subject_id": ["s1"]}]
    with pytest.raises(ValueError, match="only image and subject_id"):
        build_residual_distribution_stats_from_loaders(
            DummyModel(),
            loader,
            [{"image": torch.ones(1, 1, 1, 1, 1), "subject_id": ["t1"], "label": torch.tensor([0])}],
            device="cpu",
            output_path="unused.json",
        )


def test_statistics_builder_persists_subject_level_artifact(tmp_path):
    class DummyModel:
        training = True

        def eval(self):
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def extract_layer4(self, image):
            return image.repeat(1, 4, 1, 1, 1)

    source_loader = [
        {
            "image": torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1),
            "subject_id": ["s1", "s2"],
        }
    ]
    target_loader = [
        {
            "image": torch.tensor([2.0, 3.0]).reshape(2, 1, 1, 1, 1),
            "subject_id": ["t1", "t2"],
        }
    ]
    output = tmp_path / "stats.json"
    stats = build_residual_distribution_stats_from_loaders(
        DummyModel(),
        source_loader,
        target_loader,
        device="cpu",
        output_path=output,
        components=1,
        gmm_iterations=4,
        metadata={"direction": "ADNI_to_NACC"},
    )
    assert output.exists()
    assert stats.source_count == 2
    assert stats.target_count == 2
    assert stats.metadata["statistic_unit"] == "subject_level_gap_residual"
    assert stats.metadata["target_labels_read"] is False


def test_split_is_label_blind_and_subject_disjoint(tmp_path):
    payload = save_residual_distribution_target_split(
        tmp_path / "target_split.json",
        ["t1", "t1", "t2", "t3"],
        [0, 1, 2],
        [3],
        direction="ADNI_to_NACC",
        adaptation_fraction=0.5,
        seed=43,
    )
    assert payload["target_labels_read"] is False
    assert payload["target_metrics_read"] is False
    assert set(payload["target_adapt_subjects"]).isdisjoint(payload["target_test_subjects"])


def test_synthetic_perturbation_audit_is_finite():
    torch.manual_seed(6)
    model = CAPMResidualDistributionAlignment3D(
        stats=_model_stats(),
        projector=_projector(rank=16),
        transport_scope="residual",
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        classifier_dropout=0.0,
    )
    clean = torch.randn(2, 1, 32, 32, 32)
    perturbed = apply_bounded_intensity_perturbation(clean, scale=1.05, bias=0.02)
    perturbed = apply_fourier_amplitude_perturbation(perturbed, amplitude_scale=1.01)
    table = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    audit = audit_synthetic_perturbation(model, clean, perturbed, table)
    assert audit["finite"] is True
    assert 0.0 <= audit["clean_prediction_agreement"] <= 1.0


def test_fourier_perturbation_preserves_dc_and_changes_high_frequency_content():
    image = torch.randn(1, 1, 8, 8, 8)
    perturbed = apply_fourier_amplitude_perturbation(image, amplitude_scale=1.2)
    before = torch.fft.rfftn(image, dim=(-3, -2, -1), norm="ortho")
    after = torch.fft.rfftn(perturbed, dim=(-3, -2, -1), norm="ortho")
    assert torch.allclose(before[..., 0, 0, 0], after[..., 0, 0, 0], atol=1e-5)
    assert not torch.allclose(before, after)
