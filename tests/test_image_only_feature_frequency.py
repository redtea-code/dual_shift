import numpy as np
import pandas as pd
import pytest
import torch

from experiments.analyze_image_only_feature_frequency import (
    capture_selected_features,
    classifier_feature_columns,
    feature_power_summary,
    selected_feature_layer,
)


def test_selected_feature_layer_uses_frozen_preset_contract():
    assert selected_feature_layer("layer3_patch2") == "layer3"
    assert selected_feature_layer("layer4_pixel") == "layer4"
    assert selected_feature_layer("layer5_pixel") == "layer5"


def test_feature_power_summary_averages_channel_power_not_channels():
    feature_map = torch.zeros((2, 8, 8, 8))
    feature_map[0, 3:5, 3:5, 3:5] = 1.0
    feature_map[1, :, :, :] = 2.0
    summary = feature_power_summary(feature_map)
    assert 0.0 <= summary["high_fraction"] <= 1.0
    assert np.isclose(sum(summary[f"{band}_fraction"] for band in ("low", "mid", "high")), 1.0)


def test_feature_power_summary_rejects_non_4d_feature_map():
    with pytest.raises(ValueError, match=r"\[C, D, H, W\]"):
        feature_power_summary(torch.zeros((8, 8, 8)))



def test_capture_selected_features_returns_hook_output_and_removes_hook():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.Conv3d(1, 2, kernel_size=1)
        def forward(self, image, covariates=None):
            return self.layer4(image).mean(dim=(2, 3, 4))

    model = Model().eval()
    features = capture_selected_features(model, "layer4", torch.zeros((1, 1, 4, 4, 4)), None)
    assert features.shape == (1, 2, 4, 4, 4)
    assert not model.layer4._forward_hooks



def test_classifier_feature_columns_exclude_provenance_and_labels():
    frame = pd.DataFrame({
        "cohort": ["ADNI", "NACC"], "subject_id": ["a", "b"],
        "checkpoint_id": ["x", "x"], "label": [0, 1],
        "high_fraction": [0.1, 0.2],
    })
    assert classifier_feature_columns(frame) == ["high_fraction"]
