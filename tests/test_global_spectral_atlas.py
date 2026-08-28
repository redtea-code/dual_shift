import numpy as np
import pandas as pd
import torch
from torch import nn

from Model.ablation.frequency_uda import FrequencyPrior
from experiments.global_spectral_atlas import (
    atlas_bootstrap_report,
    bootstrap_band_stability,
    compare_frequency_priors,
    collect_subject_band_fractions,
    summarize_band_fractions,
)


class _TinySpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=1, bias=False)
        self.bn1 = nn.Identity()
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.layer5 = nn.Identity()


def test_bootstrap_band_stability_is_deterministic_and_reports_rank_probabilities():
    source = np.array([[0.70, 0.20, 0.10], [0.72, 0.19, 0.09], [0.68, 0.21, 0.11]])
    target = np.array([[0.40, 0.30, 0.30], [0.42, 0.29, 0.29], [0.38, 0.31, 0.31]])
    first = bootstrap_band_stability(source, target, n_bootstrap=200, seed=43)
    second = bootstrap_band_stability(source, target, n_bootstrap=200, seed=43)
    assert first == second
    assert set(first["highest_band_probability"]) == {"low", "mid", "high"}
    assert np.isclose(sum(first["highest_band_probability"].values()), 1.0)
    assert np.isfinite(np.asarray(first["ci_low"] + first["ci_high"])).all()


def test_collect_and_summarize_subject_band_fractions_without_labels():
    torch.manual_seed(3)
    model = _TinySpatialBackbone().train()
    loader = [{"image": torch.randn(2, 1, 5, 6, 7), "subject_id": ["s1", "s2"]}]
    rows = collect_subject_band_fractions(model, loader, stages=("raw", "layer3"), population="S_train")
    assert model.training
    assert len(rows) == 12
    assert set(rows.band) == {"low", "mid", "high"}
    summary = summarize_band_fractions(rows)
    assert set(summary.stage) == {"raw", "layer3"}

    target_rows = rows.copy()
    target_rows["population"] = "T_adapt"
    target_rows["fraction"] = target_rows["fraction"] * np.array([0.8, 1.1, 1.2] * 4)
    report = atlas_bootstrap_report(
        pd.concat([rows, target_rows], ignore_index=True), n_bootstrap=20, seed=7
    )
    assert set(report) == {"layer3", "raw"}


def test_prior_comparison_requires_exact_provenance_and_band_agreement():
    source = {"count": 3, "mean": [0.6, 0.3, 0.1], "std": [0.1, 0.05, 0.03]}
    target = {"count": 4, "mean": [0.5, 0.3, 0.2], "std": [0.08, 0.04, 0.06]}
    metadata = {
        "direction": "ADNI_to_NACC",
        "source_split_manifest_sha256": "source",
        "target_split_sha256": "target",
        "source_checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "target_test_subject_digest": "heldout",
        "target_labels_read": False,
        "target_metrics_read": False,
    }
    reference = FrequencyPrior.from_summaries(source, target, metadata=metadata)
    recomputed = FrequencyPrior.from_summaries(source, target, metadata=metadata)
    report = compare_frequency_priors(
        reference, recomputed, expected_metadata={key: metadata[key] for key in metadata if key.startswith(("direction", "source_", "target_", "config_")) and key not in {"target_labels_read", "target_metrics_read"}}
    )
    assert report["passed"]
    failed = compare_frequency_priors(
        reference,
        recomputed.with_discrepancy((1.0, 0.0, 0.0)),
        expected_metadata={key: metadata[key] for key in metadata if key.startswith(("direction", "source_", "target_", "config_")) and key not in {"target_labels_read", "target_metrics_read"}},
    )
    assert not failed["passed"]
