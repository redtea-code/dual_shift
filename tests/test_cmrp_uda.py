import numpy as np
import torch
from torch.utils.data import DataLoader

from data.journal_dataset import CovariatePreprocessor, UnlabeledJournalSubset
from experiments.cmrp_uda import split_target_adaptation_indices
from Model.ablation.cmrp_uda import build_cmrp_uda_model, coral_loss, paired_relation_loss
from training.cmrp_uda_loop import run_cmrp_train_epoch


def _model():
    return build_cmrp_uda_model(
        input_shape=(32, 32, 32),
        layers=(1, 1, 1, 1),
        representation_dim=16,
        table_hidden_dim=8,
    )


def _batch(subject_prefix, *, labeled):
    rows = []
    for index in range(4):
        row = {
            "image": torch.randn(1, 32, 32, 32),
            "covariates": torch.tensor([float(index), float(index % 2), 12.0]),
            "age_missing": torch.tensor(0.0),
            "sex_missing": torch.tensor(0.0),
            "education_missing": torch.tensor(0.0),
            "subject_id": f"{subject_prefix}{index}",
            "folder": f"folder-{subject_prefix}{index}",
        }
        if labeled:
            row["label"] = index % 2
            row["environment_id"] = 0
        rows.append(row)
    return rows


def test_cmrp_model_exposes_representations_and_gradients():
    model = _model()
    batch = _batch("s", labeled=True)
    output = model.forward_with_repr(
        torch.stack([row["image"] for row in batch]),
        torch.stack([row["covariates"] for row in batch]),
    )
    assert output["logits"].shape == (4, 2)
    assert output["joint"].shape == (4, 16)
    loss = output["logits"].square().mean() + paired_relation_loss(
        output["mri_shared"], output["table_shared"]
    )
    loss.backward()
    assert model.classifier.weight.grad is not None


def test_coral_is_finite_and_singletons_are_identity():
    source = torch.randn(3, 5)
    target = torch.randn(2, 5)
    assert torch.isfinite(coral_loss(source, target))
    assert coral_loss(source[:1], target[:1]).item() == 0.0


def test_target_split_is_subject_disjoint_without_labels():
    subjects = np.asarray(["a", "a", "b", "c", "c", "d"], dtype=object)
    adapt, test, payload = split_target_adaptation_indices(
        subjects, np.arange(len(subjects)), adaptation_fraction=0.5, seed=7
    )
    assert set(subjects[adapt]).isdisjoint(set(subjects[test]))
    assert payload["target_labels_read"] is False
    assert payload["target_metrics_read"] is False


def test_unlabeled_subset_does_not_materialize_label():
    class FakeDataset:
        raw_age = np.asarray([60.0, 61.0])
        raw_sex = np.asarray(["female", "male"], dtype=object)
        raw_education = np.asarray([12.0, 14.0])
        records = [
            {"path": "a", "subject_id": "a", "folder": "fa", "label": 1},
            {"path": "b", "subject_id": "b", "folder": "fb", "label": 0},
        ]

        def _load_image(self, path):
            return torch.zeros(1, 2, 2, 2) + (0 if path == "a" else 1)

    processor = CovariatePreprocessor().fit(
        FakeDataset.raw_age, FakeDataset.raw_sex, FakeDataset.raw_education
    )
    item = UnlabeledJournalSubset(FakeDataset(), [0, 1], processor)[0]
    assert "label" not in item
    assert "environment_id" not in item
    assert item["image"].shape == (1, 2, 2, 2)


def test_cmrp_train_epoch_uses_unlabeled_target_batch():
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = run_cmrp_train_epoch(
        model,
        DataLoader(_batch("s", labeled=True), batch_size=2),
        DataLoader(_batch("t", labeled=False), batch_size=2),
        torch.device("cpu"),
        optimizer=optimizer,
        variant="cmrp_uda",
        config={"cmrp_uda": {"target_aug_noise": 0.0}},
    )
    assert result["loss"] > 0.0
    assert result["alignment"] >= 0.0
