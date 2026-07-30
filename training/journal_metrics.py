"""Publication-oriented classification metrics and reporting helpers."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def as_probabilities(
    predictions: Sequence,
    input_type: str = "auto",
) -> np.ndarray:
    """Convert binary/multiclass logits or probabilities to an ``N x C`` array."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim not in (1, 2) or len(values) == 0:
        raise ValueError("predictions must be a non-empty 1D or 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictions must be finite")
    if input_type not in {"auto", "logits", "probabilities"}:
        raise ValueError("input_type must be auto, logits, or probabilities")

    if values.ndim == 1 or values.shape[1] == 1:
        positive = values.reshape(-1)
        is_probability = np.all((positive >= 0.0) & (positive <= 1.0))
        if input_type == "logits" or (input_type == "auto" and not is_probability):
            positive = 1.0 / (1.0 + np.exp(-np.clip(positive, -709, 709)))
        elif input_type == "probabilities" and not is_probability:
            raise ValueError("Probability values must lie in [0, 1]")
        return np.column_stack((1.0 - positive, positive))

    rows_are_probabilities = (
        np.all((values >= 0.0) & (values <= 1.0))
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
    )
    if input_type == "probabilities":
        if not rows_are_probabilities:
            raise ValueError("Probability rows must be in [0, 1] and sum to one")
        return values
    if input_type == "auto" and rows_are_probabilities:
        return values
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(labels, probabilities[:, 1]))
        return float(
            roc_auc_score(
                labels,
                probabilities,
                labels=np.arange(probabilities.shape[1]),
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        return float("nan")


def _ece(labels: np.ndarray, probabilities: np.ndarray, n_bins: int) -> float:
    if n_bins < 1:
        raise ValueError("n_bins must be at least one")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(confidence, edges[1:-1]), n_bins - 1)
    result = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(result)


def _sensitivity_specificity(
    labels: np.ndarray,
    predicted: np.ndarray,
    *,
    positive_class: int = 1,
) -> Dict[str, float]:
    """AD-positive SEN/SPE (not macro-averaged)."""
    positive = labels == positive_class
    negative = ~positive
    tp = int(np.sum(positive & (predicted == positive_class)))
    fn = int(np.sum(positive & (predicted != positive_class)))
    tn = int(np.sum(negative & (predicted != positive_class)))
    fp = int(np.sum(negative & (predicted == positive_class)))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_class": int(positive_class),
    }


def _overall_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
    *,
    positive_class: int = 1,
) -> Dict[str, float]:
    num_classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(num_classes, dtype=float)[labels]
    if num_classes == 2:
        brier = np.mean((probabilities[:, 1] - labels) ** 2)
    else:
        brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    metrics = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "auc": _auc(labels, probabilities),
        "brier": float(brier),
        "ece": _ece(labels, probabilities, n_bins),
    }
    if num_classes == 2:
        metrics.update(
            _sensitivity_specificity(
                labels, predicted, positive_class=positive_class
            )
        )
    else:
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = float("nan")
        metrics["positive_class"] = int(positive_class)
    return metrics


def aggregate_subject_predictions(
    probabilities: Sequence,
    labels: Sequence[int],
    subject_ids: Sequence[object],
    environment_ids: Optional[Sequence[object]] = None,
    *,
    input_type: str = "auto",
    strategy: str = "mean_probability",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate scan-level predictions to one row per subject.

    ``strategy='mean_probability'`` averages probabilities within a subject and
    uses the majority label (ties broken by the first scan). Environment IDs
    use the majority environment for that subject.
    """
    if strategy != "mean_probability":
        raise ValueError("Only strategy='mean_probability' is currently supported")
    probs = as_probabilities(probabilities, input_type=input_type)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    subjects = np.asarray(subject_ids, dtype=object).reshape(-1)
    environments = (
        np.zeros(len(targets), dtype=object)
        if environment_ids is None
        else np.asarray(environment_ids, dtype=object).reshape(-1)
    )
    if not (len(probs) == len(targets) == len(subjects) == len(environments)):
        raise ValueError("All inputs must have equal length")
    if len(probs) == 0:
        raise ValueError("Cannot aggregate empty predictions")

    unique_subjects = []
    seen = set()
    for subject in subjects.tolist():
        if subject not in seen:
            unique_subjects.append(subject)
            seen.add(subject)

    agg_probs, agg_labels, agg_envs, agg_subjects = [], [], [], []
    for subject in unique_subjects:
        mask = subjects == subject
        subject_probs = probs[mask].mean(axis=0)
        subject_labels = targets[mask]
        counts = np.bincount(subject_labels, minlength=probs.shape[1])
        majority = int(np.argmax(counts))
        env_values, env_counts = np.unique(environments[mask], return_counts=True)
        majority_env = env_values[int(np.argmax(env_counts))]
        agg_probs.append(subject_probs)
        agg_labels.append(majority)
        agg_envs.append(majority_env)
        agg_subjects.append(subject)
    return (
        np.asarray(agg_probs, dtype=float),
        np.asarray(agg_labels, dtype=int),
        np.asarray(agg_envs, dtype=object),
        np.asarray(agg_subjects, dtype=object),
    )


def field_strength_bin(value: object) -> str:
    """Map numeric Tesla values to claim strata labels."""
    try:
        tesla = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(tesla):
        return "unknown"
    if tesla < 2.0:
        return "1.5T"
    return "3T"


def compute_metrics_by_field_strength(
    predictions: Sequence,
    labels: Sequence[int],
    field_strengths: Sequence[object],
    environment_ids: Optional[Sequence[object]] = None,
    *,
    input_type: str = "auto",
    subject_ids: Optional[Sequence[object]] = None,
    aggregate: str = "subject_mean",
    positive_class: int = 1,
) -> Dict[str, Dict]:
    """Compute journal metrics for 1.5T / 3T / unknown strata."""
    bins = np.asarray(
        [field_strength_bin(value) for value in field_strengths], dtype=object
    )
    labels_arr = np.asarray(labels)
    env_arr = (
        None
        if environment_ids is None
        else np.asarray(environment_ids, dtype=object)
    )
    subject_arr = (
        None if subject_ids is None else np.asarray(subject_ids, dtype=object)
    )
    probabilities = as_probabilities(predictions, input_type=input_type)
    out: Dict[str, Dict] = {}
    for name in ("1.5T", "3T", "unknown"):
        mask = bins == name
        if not np.any(mask):
            continue
        out[name] = compute_journal_metrics(
            probabilities[mask],
            labels_arr[mask],
            None if env_arr is None else env_arr[mask],
            input_type="probabilities",
            subject_ids=None if subject_arr is None else subject_arr[mask],
            aggregate=aggregate,
            positive_class=positive_class,
        )
    return out


def compute_journal_metrics(
    predictions: Sequence,
    labels: Sequence[int],
    environment_ids: Optional[Sequence[object]] = None,
    *,
    input_type: str = "auto",
    ece_bins: int = 10,
    subject_ids: Optional[Sequence[object]] = None,
    aggregate: str = "none",
    positive_class: int = 1,
) -> Dict:
    """Compute overall and demographic-group publication metrics.

    AUC is binary AUC for two classes and macro one-vs-rest AUC otherwise.
    Groups with only one observed class receive ``NaN`` AUC.

    ``aggregate='subject_mean'`` averages probabilities within each subject
    before computing the primary metrics.
    """
    if aggregate not in {"none", "subject_mean"}:
        raise ValueError("aggregate must be 'none' or 'subject_mean'")
    if aggregate == "subject_mean":
        if subject_ids is None:
            raise ValueError("subject_ids are required for subject_mean aggregation")
        probabilities, targets, environments, _ = aggregate_subject_predictions(
            predictions,
            labels,
            subject_ids,
            environment_ids,
            input_type=input_type,
        )
        input_type = "probabilities"
    else:
        probabilities = as_probabilities(predictions, input_type=input_type)
        targets = np.asarray(labels, dtype=int).reshape(-1)
        environments = (
            None
            if environment_ids is None
            else np.asarray(environment_ids, dtype=object).reshape(-1)
        )
    if len(targets) != len(probabilities):
        raise ValueError("labels and predictions must have equal length")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("labels must index prediction columns")

    result = _overall_metrics(
        targets, probabilities, ece_bins, positive_class=positive_class
    )
    result["num_classes"] = int(probabilities.shape[1])
    result["n_subjects"] = (
        int(len(np.unique(np.asarray(subject_ids, dtype=object))))
        if subject_ids is not None
        else int(result["n"])
    )
    result["aggregation"] = aggregate
    result["per_group"] = {}
    if environments is None:
        result["worst_group_auc"] = float("nan")
        result["group_gap"] = float("nan")
        return result

    if len(environments) != len(targets):
        raise ValueError("environment_ids and labels must have equal length")
    for environment in sorted(set(environments.tolist()), key=str):
        mask = environments == environment
        result["per_group"][str(environment)] = _overall_metrics(
            targets[mask],
            probabilities[mask],
            ece_bins,
            positive_class=positive_class,
        )
    group_aucs = np.asarray(
        [metrics["auc"] for metrics in result["per_group"].values()], dtype=float
    )
    valid_aucs = group_aucs[np.isfinite(group_aucs)]
    result["worst_group_auc"] = (
        float(np.min(valid_aucs)) if len(valid_aucs) else float("nan")
    )
    result["group_gap"] = (
        float(np.max(valid_aucs) - np.min(valid_aucs))
        if len(valid_aucs) >= 2
        else float("nan")
    )
    return result


def _prepare_bootstrap_rows(
    probabilities: np.ndarray,
    targets: np.ndarray,
    environments: Optional[np.ndarray],
    subjects: Optional[np.ndarray],
    *,
    aggregate: str,
    cluster_by_subject: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str]:
    """Collapse to subject rows before clustered resampling when requested."""
    if aggregate == "subject_mean":
        if subjects is None:
            raise ValueError("subject_ids are required for subject_mean aggregation")
        probs, labs, envs, subs = aggregate_subject_predictions(
            probabilities,
            targets,
            subjects,
            environments,
            input_type="probabilities",
        )
        return probs, labs, envs, subs, "none"
    if cluster_by_subject:
        if subjects is None:
            raise ValueError("subject_ids are required for cluster bootstrap")
        return probabilities, targets, environments, subjects, "none"
    return probabilities, targets, environments, subjects, "none"


def _resample_row_indices(
    n_rows: int,
    subjects: Optional[np.ndarray],
    rng: np.random.Generator,
    *,
    cluster_by_subject: bool,
) -> np.ndarray:
    """Draw bootstrap row indices; duplicate subjects keep duplicate rows."""
    if not cluster_by_subject or subjects is None:
        return rng.integers(0, n_rows, size=n_rows)
    # Subjects are already one-row-per-subject after aggregation.
    return rng.integers(0, n_rows, size=n_rows)


def _cluster_resample_indices(
    subject_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deprecated scan-level helper kept for test compatibility.

    Prefer aggregate-then-resample via ``bootstrap_confidence_intervals``.
    """
    subjects = np.asarray(subject_ids, dtype=object).reshape(-1)
    unique = []
    seen = set()
    for subject in subjects.tolist():
        if subject not in seen:
            unique.append(subject)
            seen.add(subject)
    unique = np.asarray(unique, dtype=object)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(subjects == subject) for subject in chosen]
    return np.concatenate(pieces) if pieces else np.asarray([], dtype=int)


def bootstrap_confidence_intervals(
    predictions: Sequence,
    labels: Sequence[int],
    environment_ids: Optional[Sequence[object]] = None,
    *,
    metrics: Sequence[str] = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
        "brier",
        "ece",
        "worst_group_auc",
        "group_gap",
        "sensitivity",
        "specificity",
    ),
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    input_type: str = "auto",
    ece_bins: int = 10,
    subject_ids: Optional[Sequence[object]] = None,
    cluster_by_subject: bool = True,
    aggregate: str = "subject_mean",
    positive_class: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Return percentile bootstrap CIs.

    For ``aggregate='subject_mean'`` + ``cluster_by_subject=True`` the procedure is:
    1. aggregate scans to one row per subject;
    2. resample those subject rows with replacement (duplicates retained);
    3. compute metrics on the resampled subject table without re-collapsing IDs.
    """
    probabilities = as_probabilities(predictions, input_type=input_type)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    environments = (
        None
        if environment_ids is None
        else np.asarray(environment_ids, dtype=object).reshape(-1)
    )
    subjects = (
        None
        if subject_ids is None
        else np.asarray(subject_ids, dtype=object).reshape(-1)
    )
    if len(targets) != len(probabilities):
        raise ValueError("labels and predictions must have equal length")
    if environments is not None and len(environments) != len(targets):
        raise ValueError("environment_ids and labels must have equal length")
    if subjects is not None and len(subjects) != len(targets):
        raise ValueError("subject_ids and labels must have equal length")
    if cluster_by_subject and subjects is None:
        raise ValueError("subject_ids are required for cluster bootstrap")
    if n_bootstrap < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_bootstrap >= 1 and 0 < confidence < 1 are required")

    estimate = compute_journal_metrics(
        probabilities,
        targets,
        environments,
        input_type="probabilities",
        ece_bins=ece_bins,
        subject_ids=subjects,
        aggregate=aggregate,
        positive_class=positive_class,
    )
    unknown = set(metrics).difference(estimate)
    if unknown:
        raise ValueError("Unknown or non-scalar metrics: {}".format(sorted(unknown)))

    boot_probs, boot_labs, boot_envs, boot_subs, boot_agg = _prepare_bootstrap_rows(
        probabilities,
        targets,
        environments,
        subjects,
        aggregate=aggregate,
        cluster_by_subject=cluster_by_subject,
    )
    samples = {name: [] for name in metrics}
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        indices = _resample_row_indices(
            len(boot_labs),
            boot_subs,
            rng,
            cluster_by_subject=cluster_by_subject,
        )
        # Keep duplicate subject rows; do not re-aggregate by ID.
        boot = compute_journal_metrics(
            boot_probs[indices],
            boot_labs[indices],
            None if boot_envs is None else boot_envs[indices],
            input_type="probabilities",
            ece_bins=ece_bins,
            subject_ids=None,
            aggregate=boot_agg,
            positive_class=positive_class,
        )
        for name in metrics:
            samples[name].append(boot[name])

    alpha = (1.0 - confidence) / 2.0
    result = {}
    for name in metrics:
        values = np.asarray(samples[name], dtype=float)
        valid = values[np.isfinite(values)]
        result[name] = {
            "estimate": float(estimate[name]),
            "lower": (
                float(np.quantile(valid, alpha)) if len(valid) else float("nan")
            ),
            "upper": (
                float(np.quantile(valid, 1.0 - alpha))
                if len(valid)
                else float("nan")
            ),
            "n_valid": int(len(valid)),
            "bootstrap": (
                "subject_row"
                if cluster_by_subject and aggregate == "subject_mean"
                else ("subject_cluster" if cluster_by_subject else "scan")
            ),
            "aggregation": aggregate,
        }
    return result


def paired_bootstrap_difference(
    predictions_a: Sequence,
    predictions_b: Sequence,
    labels: Sequence[int],
    environment_ids: Optional[Sequence[object]] = None,
    *,
    metric: str = "auc",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    input_type: str = "auto",
    ece_bins: int = 10,
    subject_ids: Optional[Sequence[object]] = None,
    cluster_by_subject: bool = True,
    aggregate: str = "subject_mean",
    positive_class: int = 1,
) -> Dict[str, float]:
    """Estimate paired ``model A - model B`` difference and percentile CI."""
    probabilities_a = as_probabilities(predictions_a, input_type=input_type)
    probabilities_b = as_probabilities(predictions_b, input_type=input_type)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    environments = (
        None
        if environment_ids is None
        else np.asarray(environment_ids, dtype=object).reshape(-1)
    )
    subjects = (
        None
        if subject_ids is None
        else np.asarray(subject_ids, dtype=object).reshape(-1)
    )
    if not (len(probabilities_a) == len(probabilities_b) == len(targets)):
        raise ValueError("Both predictions and labels must have equal length")
    if environments is not None and len(environments) != len(targets):
        raise ValueError("environment_ids and labels must have equal length")
    if subjects is not None and len(subjects) != len(targets):
        raise ValueError("subject_ids and labels must have equal length")
    if cluster_by_subject and subjects is None:
        raise ValueError("subject_ids are required for cluster bootstrap")
    if n_bootstrap < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_bootstrap >= 1 and 0 < confidence < 1 are required")

    estimate_a = compute_journal_metrics(
        probabilities_a,
        targets,
        environments,
        input_type="probabilities",
        ece_bins=ece_bins,
        subject_ids=subjects,
        aggregate=aggregate,
        positive_class=positive_class,
    )
    estimate_b = compute_journal_metrics(
        probabilities_b,
        targets,
        environments,
        input_type="probabilities",
        ece_bins=ece_bins,
        subject_ids=subjects,
        aggregate=aggregate,
        positive_class=positive_class,
    )
    if metric not in estimate_a or not np.isscalar(estimate_a[metric]):
        raise ValueError("metric must name a scalar journal metric")
    estimate = float(estimate_a[metric]) - float(estimate_b[metric])

    rows_a, labs, envs, _, boot_agg = _prepare_bootstrap_rows(
        probabilities_a,
        targets,
        environments,
        subjects,
        aggregate=aggregate,
        cluster_by_subject=cluster_by_subject,
    )
    rows_b, labs_b, _, _, _ = _prepare_bootstrap_rows(
        probabilities_b,
        targets,
        environments,
        subjects,
        aggregate=aggregate,
        cluster_by_subject=cluster_by_subject,
    )
    if not np.array_equal(labs, labs_b):
        raise ValueError("Paired models must share identical subject labels")

    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(n_bootstrap):
        indices = _resample_row_indices(
            len(labs),
            None,
            rng,
            cluster_by_subject=cluster_by_subject,
        )
        report_a = compute_journal_metrics(
            rows_a[indices],
            labs[indices],
            None if envs is None else envs[indices],
            input_type="probabilities",
            ece_bins=ece_bins,
            subject_ids=None,
            aggregate=boot_agg,
            positive_class=positive_class,
        )
        report_b = compute_journal_metrics(
            rows_b[indices],
            labs[indices],
            None if envs is None else envs[indices],
            input_type="probabilities",
            ece_bins=ece_bins,
            subject_ids=None,
            aggregate=boot_agg,
            positive_class=positive_class,
        )
        differences.append(float(report_a[metric]) - float(report_b[metric]))
    valid = np.asarray(differences, dtype=float)
    valid = valid[np.isfinite(valid)]
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "estimate": float(estimate),
        "lower": float(np.quantile(valid, alpha)) if len(valid) else float("nan"),
        "upper": (
            float(np.quantile(valid, 1.0 - alpha))
            if len(valid)
            else float("nan")
        ),
        "n_valid": int(len(valid)),
        "bootstrap": (
            "subject_row"
            if cluster_by_subject and aggregate == "subject_mean"
            else ("subject_cluster" if cluster_by_subject else "scan")
        ),
        "aggregation": aggregate,
    }


def save_predictions_csv(
    path: str,
    predictions: Sequence,
    labels: Sequence[int],
    environment_ids: Optional[Sequence[object]] = None,
    subject_ids: Optional[Sequence[object]] = None,
    *,
    input_type: str = "auto",
) -> None:
    """Save one row per sample with labels, predictions, and probabilities."""
    probabilities = as_probabilities(predictions, input_type=input_type)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    if len(targets) != len(probabilities):
        raise ValueError("labels and predictions must have equal length")
    environments = (
        np.full(len(targets), "", dtype=object)
        if environment_ids is None
        else np.asarray(environment_ids, dtype=object).reshape(-1)
    )
    subjects = (
        np.arange(len(targets), dtype=object)
        if subject_ids is None
        else np.asarray(subject_ids, dtype=object).reshape(-1)
    )
    if len(environments) != len(targets) or len(subjects) != len(targets):
        raise ValueError("environment_ids and subject_ids must match labels")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "subject_id",
        "label",
        "predicted_label",
        "environment_id",
    ] + ["probability_{}".format(i) for i in range(probabilities.shape[1])]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            row = {
                "subject_id": subjects[index],
                "label": int(targets[index]),
                "predicted_label": int(np.argmax(probability)),
                "environment_id": environments[index],
            }
            row.update(
                {
                    "probability_{}".format(class_index): float(value)
                    for class_index, value in enumerate(probability)
                }
            )
            writer.writerow(row)


def save_json_summary(path: str, summary: Dict) -> None:
    """Save a strict JSON report, representing non-finite values as null."""

    def sanitise(value):
        if isinstance(value, dict):
            return {str(key): sanitise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [sanitise(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        return value

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitise(summary), handle, indent=2, sort_keys=True, allow_nan=False)
