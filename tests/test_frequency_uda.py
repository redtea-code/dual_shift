import torch
from torch import nn

from experiments.frequency_uda import (
    ImageSubjectSubset,
    build_frequency_prior_from_loaders,
    earliest_subject_indices,
    save_target_adaptation_split,
    split_target_adaptation_indices,
)
from Model.ablation.frequency_uda import (
    DomainGuidedFrequencyGate3D,
    FeatureSpectrumAccumulator,
    FrequencyGuidedScaleTable3D,
    FrequencyPrior,
    radial_band_masks_rfft,
)


def _prior():
    return FrequencyPrior.from_summaries(
        {"count": 4, "mean": [0.55, 0.30, 0.15], "std": [0.02, 0.03, 0.02]},
        {"count": 5, "mean": [0.40, 0.32, 0.28], "std": [0.03, 0.02, 0.04]},
        metadata={"target_labels_read": False, "test": True},
    )


def test_radial_masks_partition_rfft_space():
    masks = radial_band_masks_rfft((3, 4, 5))
    assert masks.shape == (3, 3, 4, 3)
    assert torch.allclose(masks.sum(dim=0), torch.ones_like(masks[0]))


def test_frequency_prior_and_gate_are_label_free_and_differentiable(tmp_path):
    torch.manual_seed(7)
    accumulator = FeatureSpectrumAccumulator()
    accumulator.update(torch.randn(3, 4, 3, 4, 5))
    summary = accumulator.summary()
    assert summary["count"] == 3
    prior = _prior()
    path = tmp_path / "frequency_prior.json"
    prior.save(path)
    loaded = FrequencyPrior.load(path)
    assert loaded.metadata["target_labels_read"] is False
    gate = DomainGuidedFrequencyGate3D(loaded, max_strength=0.4, init_strength=0.04)
    features = torch.randn(2, 4, 3, 4, 5, requires_grad=True)
    output, audit = gate(features, return_audit=True)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert torch.all((audit["attenuation"] > 0) & (audit["attenuation"] <= 1))
    loss = output.square().mean() + gate.regularization_losses()["frequency_identity"]
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert gate.raw_strength.grad is not None


def test_frequency_uda_model_preserves_layer5_capm_contract():
    torch.manual_seed(11)
    model = FrequencyGuidedScaleTable3D(
        prior=_prior(),
        num_classes=2,
        layers=(1, 1, 1, 1),
        input_shape=(32, 32, 32),
        spatial_shape=(2, 2, 2),
        transformer_dim=16,
        num_heads=4,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
    )
    images = torch.randn(2, 1, 32, 32, 32)
    tables = torch.tensor([[65.0, 0.0, 12.0], [80.0, 1.0, 18.0]])
    logits, audit = model(images, tables, return_audit=True)
    assert logits.shape == (2, 2)
    assert model.extract_layer4(images).shape[1] == 512
    assert "frequency_attenuation" in audit
    assert model.experiment_signature()["frequency_insert_stage"] == "layer4_to_layer5"
    (logits.square().mean() + model.get_regularization_losses()["frequency_identity"]).backward()
    assert model.frequency_gate.raw_strength.grad is not None


def test_target_adapt_split_and_prior_builder_are_subject_disjoint_and_label_free(tmp_path):
    subject_ids = ["a", "a", "b", "c", "d", "d"]
    adapt, test = split_target_adaptation_indices(
        subject_ids, adaptation_fraction=0.5, seed=3
    )
    assert set(adapt).isdisjoint(set(test))
    assert {subject_ids[index] for index in adapt}.isdisjoint(
        {subject_ids[index] for index in test}
    )

    class Layer4Only(nn.Module):
        def extract_layer4(self, image):
            return image

    model = Layer4Only().train()
    source_loader = [{"image": torch.randn(2, 1, 4, 4, 4), "label": [1, 0], "subject_id": ["s1", "s2"]}]
    target_loader = [{"image": torch.randn(3, 1, 4, 4, 4), "label": [0, 1, 1], "subject_id": ["t1", "t2", "t3"]}]
    prior = build_frequency_prior_from_loaders(
        model,
        source_loader,
        target_loader,
        device="cpu",
        output_path=tmp_path / "prior.json",
    )
    assert prior.source_count == 2
    assert prior.target_count == 3
    assert prior.metadata["target_labels_read"] is False
    assert model.training
    split = save_target_adaptation_split(
        tmp_path / "target_split.json",
        subject_ids,
        adapt,
        test,
        direction="ADNI_to_NACC",
        adaptation_fraction=0.5,
        seed=3,
    )
    assert split["target_labels_read"] is False
    assert set(split["target_adapt_subjects"]).isdisjoint(split["target_test_subjects"])


def test_prior_selection_uses_one_earliest_visit_per_subject():
    subject_ids = ["s1", "s1", "s2", "s2"]
    records = [
        {"scan_date": "2021-02-01", "folder": "later"},
        {"scan_date": "2020-01-01", "folder": "earlier"},
        {"scan_date": "2020-03-01", "folder": "z"},
        {"scan_date": "2020-03-01", "folder": "a"},
    ]
    selected = earliest_subject_indices(subject_ids, [0, 1, 2, 3], records)
    assert selected.tolist() == [1, 3]


def test_image_subject_subset_never_calls_label_bearing_dataset_getitem():
    class LabelBearingDataset:
        records = [{"path": "path-1", "subject_id": "s1", "label": 1}]

        def _load_image(self, path):
            assert path == "path-1"
            return torch.ones(1, 4, 4, 4)

        def __getitem__(self, index):
            raise AssertionError("frequency-prior extraction must not request a labeled sample")

    item = ImageSubjectSubset(LabelBearingDataset(), [0])[0]
    assert set(item) == {"image", "subject_id"}
    assert item["subject_id"] == "s1"
