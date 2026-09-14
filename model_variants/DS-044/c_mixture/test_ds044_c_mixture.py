from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ds044_c_mixture import aggregate_subject_embeddings, run_c_mixture_diagnostics


def _toy_data(seed=7):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(60, 6))
    target_subjects = np.repeat([f"t-{index:02d}" for index in range(20)], 2)
    target = np.vstack(
        [rng.normal(loc=-2.0, scale=0.2, size=(20, 6)), rng.normal(loc=2.0, scale=0.2, size=(20, 6))]
    )
    probabilities = np.vstack([np.tile([0.8, 0.2], (10, 1)), np.tile([0.2, 0.8], (10, 1))])
    return source, target, target_subjects, probabilities


def test_subject_pooling_is_deterministic():
    result = aggregate_subject_embeddings(
        np.asarray([[2.0, 0.0], [0.0, 2.0], [4.0, 0.0]]), ["b", "a", "b"]
    )
    assert result.subject_ids.tolist() == ["a", "b"]
    np.testing.assert_allclose(result.embeddings, [[0.0, 2.0], [3.0, 0.0]])
    assert result.row_counts.tolist() == [1, 2]


def test_c_diagnostics_is_label_blind_and_selects_two_clusters():
    source, target, subjects, probabilities = _toy_data()
    result = run_c_mixture_diagnostics(
        source,
        target,
        subjects,
        target_probabilities=probabilities,
        candidate_k=(2, 3, 4),
        n_bootstrap=20,
        random_state=42,
        min_subjects_per_cluster=8,
    )
    assert result["protocol"] == "DS-044-C1-C2"
    assert result["target_labels_read"] is False
    assert result["selected_k"] == 2
    assert result["target_subject_count"] == 20
    assert len(result["selected_cluster_summary"]) == 2
    assert result["entropy_by_cluster"] is not None


def test_cli_round_trip(tmp_path):
    source, target, subjects, probabilities = _toy_data()
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "summary.json"
    np.savez(
        input_path,
        source_train_embeddings=source,
        target_embeddings=target,
        target_subject_ids=subjects,
        target_probabilities=probabilities,
    )
    runner = Path(__file__).with_name("run_ds044_c_mixture.py")
    subprocess.run(
        [sys.executable, str(runner), "--input", str(input_path), "--output", str(output_path), "--bootstrap", "10"],
        check=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["target_labels_read"] is False
    assert payload["selected_k"] == 2
