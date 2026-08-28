import torch

from training.frequency_environments import (
    FREQUENCY_ENVIRONMENTS,
    FrequencyEnvironmentAugment3D,
)


def test_frequency_environments_keep_one_view_per_sample_and_are_deterministic_when_assigned():
    augmenter = FrequencyEnvironmentAugment3D()
    images = torch.zeros(4, 1, 8, 8, 8)
    images[1, :, 3, 3, 3] = 1.0
    images[2] = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
    images[3, :, 2:6, 2:6, 2:6] = 1.0
    ids = torch.tensor([0, 1, 2, 3])
    output, actual_ids = augmenter(images, environment_ids=ids)
    assert tuple(FREQUENCY_ENVIRONMENTS) == (
        "original",
        "lowpass",
        "downsample_resample",
        "mild_blur",
    )
    assert torch.equal(actual_ids, ids)
    assert output.shape == images.shape
    assert torch.equal(output[0], images[0])
    assert not torch.equal(output[1], images[1])
    assert not torch.equal(output[2], images[2])
    assert not torch.equal(output[3], images[3])
    assert torch.isfinite(output).all()


def test_frequency_environment_sampling_stays_in_registered_range():
    augmenter = FrequencyEnvironmentAugment3D()
    generator = torch.Generator().manual_seed(9)
    images = torch.randn(12, 2, 8, 8, 8)
    _, ids = augmenter(images, generator=generator)
    assert ids.shape == (12,)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < augmenter.num_environments
