import torch

from experiments.train_target_style_uda import _UnlabeledTargetImages


class _FakeImageDataset:
    records = [
        {"path": "first", "label": 0, "age": 70.0},
        {"path": "second", "label": 1, "age": 72.0},
    ]

    def _load_image(self, path):
        return torch.full((1, 2, 2, 2), 1.0 if path == "first" else 2.0)


def test_unlabeled_target_images_bypass_record_fields():
    view = _UnlabeledTargetImages(_FakeImageDataset(), [1])
    item = view[0]
    assert set(item) == {"image"}
    assert torch.all(item["image"] == 2.0)
