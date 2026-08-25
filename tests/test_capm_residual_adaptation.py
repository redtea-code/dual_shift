import pytest
import torch

from Model.ablation.residual_adaptation import (
    CAPMResidualAdaptation3D,
    CAPMResidualAdapter3D,
    CAPMResidualStats,
    FeatureMomentAccumulator,
    build_capm_residual_stats_from_loaders,
    save_capm_residual_target_split,
)


def _stats(channels=512):
    source = {"count": 4, "mean": [0.0] * channels, "std": [1.0] * channels}
    target = {"count": 4, "mean": [1.0] + [0.0] * (channels - 1), "std": [1.0] * channels}
    return CAPMResidualStats.from_summaries(source, target, metadata={"target_labels_read": False})


def test_stats_are_serializable_and_discrepancy_is_bounded(tmp_path):
    stats = _stats(4)
    path = tmp_path / "stats.json"
    stats.save(path)
    restored = CAPMResidualStats.load(path)
    assert restored.channels == 4
    assert restored.discrepancy[0] == pytest.approx(1.0)
    assert max(restored.discrepancy) <= 1.0


def test_adapter_only_moves_flagged_residual_channels():
    adapter = CAPMResidualAdapter3D(_stats(4), max_strength=0.25)
    features = torch.zeros(2, 4, 2, 2, 2)
    corrected, audit = adapter(features, return_audit=True)
    assert corrected[:, 0].mean().item() == pytest.approx(-0.25)
    assert corrected[:, 1:].abs().max().item() == pytest.approx(0.0)
    assert audit["effective_strength"].item() == pytest.approx(0.25)


def test_feature_moment_accumulator_counts_samples_and_elements():
    accumulator = FeatureMomentAccumulator()
    accumulator.update(torch.ones(2, 3, 2, 2, 2))
    summary = accumulator.summary()
    assert summary["count"] == 2
    assert summary["elements"] == 16
    assert summary["mean"] == [1.0, 1.0, 1.0]
    assert summary["std"] == [0.0, 0.0, 0.0]


def test_residual_model_preserves_original_capm_interface():
    model = CAPMResidualAdaptation3D(
        stats=_stats(),
        preset="layer4_pixel",
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        classifier_dropout=0.0,
    )
    image = torch.randn(2, 1, 32, 32, 32)
    table = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    logits = model(image, table, apply_adaptation=True)
    control = model(image, table, apply_adaptation=False)
    assert logits.shape == (2, 2)
    assert control.shape == logits.shape
    assert model.experiment_signature()["residual_stage"] == "post_capm_layer4_map"


def test_statistics_builder_rejects_target_labels():
    class DummyModel:
        training = False

        def eval(self):
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def extract_layer4(self, image):
            return image.repeat(1, 512, 1, 1, 1)

    loader = [{"image": torch.ones(1, 1, 1, 1, 1), "subject_id": ["s1"]}]
    with pytest.raises(ValueError, match="only image and subject_id"):
        build_capm_residual_stats_from_loaders(
            DummyModel(),
            loader,
            [{"image": torch.ones(1, 1, 1, 1, 1), "subject_id": ["t1"], "label": torch.tensor([0])}],
            device="cpu",
            output_path="unused.json",
        )


def test_target_split_is_subject_disjoint_and_label_blind(tmp_path):
    payload = save_capm_residual_target_split(
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


def test_statistics_builder_uses_capm_output_for_standard_scale_model(tmp_path):
    from Model.ablation import build_scale_table_ablation

    model = build_scale_table_ablation(
        preset="layer4_pixel",
        interaction="original_capm",
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        classifier_dropout=0.0,
    )
    loader = [{"image": torch.randn(1, 1, 32, 32, 32), "subject_id": ["s1"]}]
    stats = build_capm_residual_stats_from_loaders(
        model,
        loader,
        loader,
        device="cpu",
        output_path=tmp_path / "stats.json",
        reference_table=torch.tensor([65.0, 0.0, 12.0]),
    )
    assert stats.channels == 512
    assert stats.metadata["method"] == "label_free_capm_adjusted_channel_moment_residual"
