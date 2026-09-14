"""Leakage-safe DS-044 C1/C2 target-mixture diagnostics.

The module operates on frozen P0 outputs. It never consumes target labels and
fits the PCA reference space on source-train embeddings only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass(frozen=True)
class SubjectEmbeddingSet:
    subject_ids: np.ndarray
    embeddings: np.ndarray
    row_counts: np.ndarray


def _as_2d_float(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def aggregate_subject_embeddings(
    embeddings: np.ndarray,
    subject_ids: Sequence[Any],
) -> SubjectEmbeddingSet:
    """Mean-pool scan embeddings by subject in deterministic ID order."""
    values = _as_2d_float(embeddings, "embeddings")
    ids = np.asarray([str(value) for value in subject_ids], dtype=object)
    if ids.ndim != 1 or len(ids) != len(values):
        raise ValueError("subject_ids must be a 1D array aligned with embeddings")
    order = np.argsort(ids.astype(str), kind="mergesort")
    sorted_ids = ids[order]
    unique_ids, starts = np.unique(sorted_ids, return_index=True)
    pooled = []
    counts = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(sorted_ids)
        pooled.append(values[order[start:stop]].mean(axis=0))
        counts.append(stop - start)
    return SubjectEmbeddingSet(
        subject_ids=unique_ids.astype(object),
        embeddings=np.asarray(pooled, dtype=np.float64),
        row_counts=np.asarray(counts, dtype=np.int64),
    )


def fit_source_pca(source_train_embeddings: np.ndarray, n_components: int = 16) -> PCA:
    """Fit PCA using source-train subjects only."""
    source = _as_2d_float(source_train_embeddings, "source_train_embeddings")
    components = min(int(n_components), source.shape[0], source.shape[1])
    if components < 1:
        raise ValueError("PCA requires at least one component")
    return PCA(n_components=components, svd_solver="full", random_state=0).fit(source)


def _cluster_once(values: np.ndarray, k: int, random_state: int) -> tuple[KMeans, np.ndarray]:
    if len(values) < k:
        raise ValueError(f"k={k} exceeds the number of target subjects ({len(values)})")
    model = KMeans(n_clusters=k, n_init=50, random_state=int(random_state))
    return model, model.fit_predict(values)


def bootstrap_cluster_stability(
    values: np.ndarray,
    labels: np.ndarray,
    k: int,
    n_bootstrap: int = 200,
    random_state: int = 0,
) -> np.ndarray:
    """Return bootstrap ARI values against a reference clustering.

    Each bootstrap refit is unsupervised and uses only target embeddings.
    """
    rng = np.random.default_rng(int(random_state))
    scores = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(values), size=len(values))
        _, boot_labels = _cluster_once(values[indices], k, int(rng.integers(0, 2**31 - 1)))
        scores.append(adjusted_rand_score(labels[indices], boot_labels))
    return np.asarray(scores, dtype=np.float64)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return -(probs * np.log(probs)).sum(axis=1)


def _cluster_summary(
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    subject_ids: np.ndarray,
    row_counts: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, Any]]:
    summaries = []
    entropy = _entropy(probabilities) if probabilities is not None else None
    max_probability = (
        np.asarray(probabilities, dtype=np.float64).max(axis=1)
        if probabilities is not None
        else None
    )
    positive_rate = (
        np.asarray(probabilities, dtype=np.float64)[:, -1]
        if probabilities is not None and np.asarray(probabilities).shape[1] == 2
        else None
    )
    for cluster in sorted(np.unique(labels).tolist()):
        mask = labels == cluster
        summaries.append(
            {
                "cluster": int(cluster),
                "subject_count": int(mask.sum()),
                "scan_count": int(row_counts[mask].sum()),
                "subject_ids": [str(item) for item in subject_ids[mask]],
                "support_distance_mean": float(np.mean(distances[mask])),
                "support_distance_median": float(np.median(distances[mask])),
                "entropy_mean": None if entropy is None else float(np.mean(entropy[mask])),
                "entropy_median": None if entropy is None else float(np.median(entropy[mask])),
                "max_probability_mean": (
                    None if max_probability is None else float(np.mean(max_probability[mask]))
                ),
                "positive_prediction_rate": (
                    None if positive_rate is None else float(np.mean(positive_rate[mask]))
                ),
            }
        )
    return summaries


def run_c_mixture_diagnostics(
    source_train_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    target_subject_ids: Sequence[Any],
    *,
    source_train_subject_ids: Sequence[Any] | None = None,
    target_probabilities: np.ndarray | None = None,
    target_scan_probabilities: np.ndarray | None = None,
    candidate_k: Iterable[int] = (2, 3, 4),
    pca_components: int = 16,
    n_bootstrap: int = 200,
    random_state: int = 0,
    min_subjects_per_cluster: int = 8,
) -> dict[str, Any]:
    """Run C1/C2 on frozen P0 pooled embeddings.

    ``source_train_embeddings`` may be scan-level when
    ``source_train_subject_ids`` is supplied; otherwise it must already be
    subject-level. Target probabilities must be subject-level. Scan-level
    probabilities are aggregated by subject for the variance diagnostic.
    """
    source_values = _as_2d_float(source_train_embeddings, "source_train_embeddings")
    if source_train_subject_ids is not None:
        source = aggregate_subject_embeddings(source_values, source_train_subject_ids)
    else:
        source = SubjectEmbeddingSet(
            subject_ids=np.asarray([str(i) for i in range(len(source_values))], dtype=object),
            embeddings=source_values,
            row_counts=np.ones(len(source_values), dtype=np.int64),
        )
    target = aggregate_subject_embeddings(target_embeddings, target_subject_ids)
    if target.embeddings.shape[1] != source.embeddings.shape[1]:
        raise ValueError("source and target embeddings must have the same feature dimension")

    pca = fit_source_pca(source.embeddings, pca_components)
    source_pca = pca.transform(source.embeddings)
    target_pca = pca.transform(target.embeddings)
    source_center = source_pca.mean(axis=0)
    source_scale = source_pca.std(axis=0)
    source_scale[source_scale < 1e-8] = 1.0
    standardized_source = (source_pca - source_center) / source_scale
    standardized_target = (target_pca - source_center) / source_scale
    source_reference = np.linalg.norm(standardized_source, axis=1)
    target_distance = np.linalg.norm(standardized_target, axis=1)

    probabilities = None
    if target_probabilities is not None:
        probabilities = np.asarray(target_probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or len(probabilities) != len(target.subject_ids):
            raise ValueError("target_probabilities must be subject-level and aligned with target_subject_ids")

    scan_probabilities = None
    scan_subject_probability_summary = None
    if target_scan_probabilities is not None:
        scan_probabilities = np.asarray(target_scan_probabilities, dtype=np.float64)
        if len(scan_probabilities) != len(target_subject_ids):
            raise ValueError("target_scan_probabilities must align with target scan rows")
        scan_subject_probability_summary = []
        for subject in target.subject_ids:
            values = scan_probabilities[np.asarray(target_subject_ids, dtype=object) == subject]
            scan_subject_probability_summary.append(
                {"subject_id": str(subject), "scan_count": int(len(values)), "probability_mean": values.mean(axis=0).tolist()}
            )

    candidate_results: dict[str, Any] = {}
    for k in sorted({int(value) for value in candidate_k}):
        if k < 2 or k > len(target.subject_ids):
            continue
        model, labels = _cluster_once(standardized_target, k, random_state + k)
        bootstrap = bootstrap_cluster_stability(
            standardized_target, labels, k, n_bootstrap=n_bootstrap, random_state=random_state + 1000 + k
        )
        silhouette = None
        if len(np.unique(labels)) > 1 and len(labels) > k:
            silhouette = float(silhouette_score(standardized_target, labels))
        sizes = np.bincount(labels, minlength=k)
        candidate_results[str(k)] = {
            "k": int(k),
            "cluster_sizes": sizes.tolist(),
            "minimum_cluster_subjects": int(sizes.min()),
            "bootstrap_ari_mean": float(bootstrap.mean()),
            "bootstrap_ari_median": float(np.median(bootstrap)),
            "bootstrap_ari_p05": float(np.quantile(bootstrap, 0.05)),
            "bootstrap_ari_p95": float(np.quantile(bootstrap, 0.95)),
            "silhouette": silhouette,
            "eligible_by_min_cluster_size": bool(sizes.min() >= int(min_subjects_per_cluster)),
            "labels": labels.tolist(),
            "centers": model.cluster_centers_.tolist(),
            "bootstrap_ari": bootstrap.tolist(),
        }

    if not candidate_results:
        raise ValueError("candidate_k contains no valid values for the target subject count")
    eligible = [item for item in candidate_results.values() if item["eligible_by_min_cluster_size"]]
    if not eligible:
        eligible = list(candidate_results.values())
    selected = sorted(eligible, key=lambda item: (-item["bootstrap_ari_mean"], item["k"]))[0]
    selected_labels = np.asarray(selected["labels"], dtype=np.int64)
    selected_summary = _cluster_summary(
        selected_labels, probabilities, target.subject_ids, target.row_counts, target_distance
    )
    entropy_values = None if probabilities is None else _entropy(probabilities)
    entropy_by_cluster = None
    if entropy_values is not None:
        entropy_by_cluster = {
            str(cluster["cluster"]): float(entropy_values[selected_labels == cluster["cluster"]].mean())
            for cluster in selected_summary
        }
    return {
        "protocol": "DS-044-C1-C2",
        "target_labels_read": False,
        "pca_fit_scope": "source_train_subject_embeddings_only",
        "pca_components": int(pca.n_components_),
        "source_subject_count": int(len(source.subject_ids)),
        "target_subject_count": int(len(target.subject_ids)),
        "target_scan_count": int(len(target_subject_ids)),
        "source_reference_distance_mean": float(source_reference.mean()),
        "source_reference_distance_p95": float(np.quantile(source_reference, 0.95)),
        "target_distance_mean": float(target_distance.mean()),
        "target_distance_p95": float(np.quantile(target_distance, 0.95)),
        "candidate_k": sorted(candidate_results),
        "selected_k": int(selected["k"]),
        "candidates": candidate_results,
        "selected_cluster_summary": selected_summary,
        "entropy_by_cluster": entropy_by_cluster,
        "scan_subject_probability_summary": scan_subject_probability_summary,
        "target_subject_ids": [str(item) for item in target.subject_ids],
        "target_cluster_labels": selected_labels.tolist(),
    }


def json_ready(value: Any) -> Any:
    """Convert numpy scalars/arrays recursively for JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
