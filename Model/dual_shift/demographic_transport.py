"""Continuous Demographic Transport (CDT) controller."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch


class ContinuousDemographicTransport:
    """Epoch-level sample weights from continuous demographic risk surfaces.

    Loss memory and weight recomputation are aggregated at the subject level,
    then broadcast to all scans belonging to that subject.
    """

    def __init__(
        self,
        *,
        age_bandwidth: float = 1.0,
        education_bandwidth: float = 1.0,
        cross_sex_rho: float = 0.5,
        temperature: float = 1.0,
        ema_beta: float = 0.9,
        ess_ratio_min: float = 0.2,
        max_weight_factor: float = 10.0,
        kl_lambda: float = 0.05,
    ):
        self.age_bandwidth = float(age_bandwidth)
        self.education_bandwidth = float(education_bandwidth)
        self.cross_sex_rho = float(cross_sex_rho)
        self.temperature = float(temperature)
        self.ema_beta = float(ema_beta)
        self.ess_ratio_min = float(ess_ratio_min)
        self.max_weight_factor = float(max_weight_factor)
        self.kl_lambda = float(kl_lambda)
        self.sample_ids: list[str] = []
        self.subject_ids: list[str] = []
        self.labels: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.age: np.ndarray = np.zeros((0,), dtype=float)
        self.sex: np.ndarray = np.zeros((0,), dtype=float)
        self.education: np.ndarray = np.zeros((0,), dtype=float)
        self.loss_memory: np.ndarray = np.zeros((0,), dtype=float)
        self.weights: np.ndarray = np.zeros((0,), dtype=float)
        self.id_to_index: Dict[str, int] = {}
        self.subject_to_indices: Dict[str, List[int]] = {}
        self.unique_subjects: list[str] = []
        self.subject_index: Dict[str, int] = {}
        self.subject_labels: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.subject_age: np.ndarray = np.zeros((0,), dtype=float)
        self.subject_sex: np.ndarray = np.zeros((0,), dtype=float)
        self.subject_education: np.ndarray = np.zeros((0,), dtype=float)
        self.subject_loss_memory: np.ndarray = np.zeros((0,), dtype=float)
        self.subject_weights: np.ndarray = np.zeros((0,), dtype=float)
        self.enabled = False

    def initialize(
        self,
        sample_ids: Sequence[str],
        subject_ids: Sequence[str],
        labels: Sequence[int],
        age: Sequence[float],
        sex: Sequence[float],
        education: Sequence[float],
    ) -> None:
        self.sample_ids = [str(value) for value in sample_ids]
        self.subject_ids = [str(value) for value in subject_ids]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.age = np.asarray(age, dtype=float)
        self.sex = np.asarray(sex, dtype=float)
        self.education = np.asarray(education, dtype=float)
        n = len(self.sample_ids)
        self.loss_memory = np.zeros(n, dtype=float)
        self.weights = np.ones(n, dtype=float)
        self.id_to_index = {
            sample_id: index for index, sample_id in enumerate(self.sample_ids)
        }
        self.subject_to_indices = defaultdict(list)
        for index, subject_id in enumerate(self.subject_ids):
            self.subject_to_indices[subject_id].append(index)
        self.unique_subjects = list(self.subject_to_indices.keys())
        self.subject_index = {
            subject_id: index for index, subject_id in enumerate(self.unique_subjects)
        }
        n_subj = len(self.unique_subjects)
        self.subject_labels = np.zeros(n_subj, dtype=np.int64)
        self.subject_age = np.zeros(n_subj, dtype=float)
        self.subject_sex = np.zeros(n_subj, dtype=float)
        self.subject_education = np.zeros(n_subj, dtype=float)
        self.subject_loss_memory = np.zeros(n_subj, dtype=float)
        self.subject_weights = np.ones(n_subj, dtype=float)
        for subject_id, indices in self.subject_to_indices.items():
            sidx = self.subject_index[subject_id]
            self.subject_labels[sidx] = int(self.labels[indices[0]])
            self.subject_age[sidx] = float(np.mean(self.age[indices]))
            self.subject_sex[sidx] = float(self.sex[indices[0]])
            self.subject_education[sidx] = float(np.mean(self.education[indices]))
        self.recompute_weights(force_uniform=True)

    def update_losses(
        self,
        sample_ids: Sequence[str],
        clean_losses: Sequence[float],
        apis_losses: Optional[Sequence[float]] = None,
    ) -> None:
        per_subject: Dict[str, List[float]] = defaultdict(list)
        for i, sample_id in enumerate(sample_ids):
            index = self.id_to_index.get(str(sample_id))
            if index is None:
                continue
            clean = float(clean_losses[i])
            apis = float(apis_losses[i]) if apis_losses is not None else clean
            observed = max(clean, apis)
            subject_id = self.subject_ids[index]
            per_subject[subject_id].append(observed)
        for subject_id, values in per_subject.items():
            sidx = self.subject_index.get(subject_id)
            if sidx is None:
                continue
            observed = float(np.mean(values))
            prev = self.subject_loss_memory[sidx]
            if prev <= 0:
                self.subject_loss_memory[sidx] = observed
            else:
                self.subject_loss_memory[sidx] = (
                    self.ema_beta * prev + (1.0 - self.ema_beta) * observed
                )
            # Broadcast subject memory to all scans of that subject.
            for sample_index in self.subject_to_indices[subject_id]:
                self.loss_memory[sample_index] = self.subject_loss_memory[sidx]

    def _kernel_matrix(self, indices: np.ndarray) -> np.ndarray:
        age = self.subject_age[indices]
        edu = self.subject_education[indices]
        sex = self.subject_sex[indices]
        age_term = np.exp(
            -((age[:, None] - age[None, :]) ** 2) / (2.0 * self.age_bandwidth**2 + 1e-8)
        )
        edu_term = np.exp(
            -((edu[:, None] - edu[None, :]) ** 2)
            / (2.0 * self.education_bandwidth**2 + 1e-8)
        )
        sex_term = np.where(
            sex[:, None] == sex[None, :],
            1.0,
            self.cross_sex_rho,
        )
        return age_term * edu_term * sex_term

    def recompute_weights(self, force_uniform: bool = False) -> Dict[str, float]:
        n_subj = len(self.unique_subjects)
        if n_subj == 0:
            return {"ess_min": 0.0, "kl": 0.0}
        subject_weights = np.zeros(n_subj, dtype=float)
        kl_total = 0.0
        ess_values = []
        classes = np.unique(self.subject_labels)
        for label in classes:
            indices = np.where(self.subject_labels == label)[0]
            n_c = len(indices)
            if n_c == 0:
                continue
            if force_uniform or not self.enabled:
                class_weights = np.ones(n_c, dtype=float) / n_c
            else:
                kernel = self._kernel_matrix(indices)
                memory = self.subject_loss_memory[indices]
                risk = (kernel @ memory) / (kernel.sum(axis=1) + 1e-8)
                temperature = self.temperature
                ess = float(n_c)
                for _ in range(20):
                    logits = risk / max(temperature, 1e-6)
                    logits = logits - logits.max()
                    class_weights = np.exp(logits)
                    class_weights = class_weights / class_weights.sum()
                    cap = self.max_weight_factor / n_c
                    class_weights = np.minimum(class_weights, cap)
                    class_weights = class_weights / class_weights.sum()
                    ess = 1.0 / (np.sum(class_weights**2) + 1e-12)
                    if ess >= self.ess_ratio_min * n_c:
                        break
                    temperature *= 1.5
                ess_values.append(float(ess))
                uniform = np.ones(n_c, dtype=float) / n_c
                kl_total += float(
                    np.sum(
                        class_weights
                        * (np.log(class_weights + 1e-12) - np.log(uniform))
                    )
                )
            subject_weights[indices] = class_weights
        subject_weights = subject_weights * (
            n_subj / max(subject_weights.sum(), 1e-8)
        )
        self.subject_weights = subject_weights
        # Broadcast subject weights to scans.
        weights = np.ones(len(self.sample_ids), dtype=float)
        for subject_id, indices in self.subject_to_indices.items():
            sidx = self.subject_index[subject_id]
            for sample_index in indices:
                weights[sample_index] = subject_weights[sidx]
        self.weights = weights
        return {
            "ess_min": float(min(ess_values) if ess_values else float(n_subj)),
            "kl": float(kl_total),
            "weight_max": float(weights.max()) if len(weights) else 0.0,
            "weight_min": float(weights.min()) if len(weights) else 0.0,
            "n_subjects": float(n_subj),
            "weight_entropy": float(
                -np.sum(
                    (subject_weights / max(subject_weights.sum(), 1e-8))
                    * np.log(
                        subject_weights / max(subject_weights.sum(), 1e-8) + 1e-12
                    )
                )
            ),
        }

    def batch_weights(self, sample_ids: Sequence[str]) -> torch.Tensor:
        values = []
        for sample_id in sample_ids:
            index = self.id_to_index.get(str(sample_id))
            values.append(1.0 if index is None else float(self.weights[index]))
        return torch.tensor(values, dtype=torch.float32)

    def kl_penalty(self) -> float:
        if len(self.subject_weights) == 0:
            return 0.0
        kl = 0.0
        for label in np.unique(self.subject_labels):
            indices = np.where(self.subject_labels == label)[0]
            w = self.subject_weights[indices]
            w = w / max(w.sum(), 1e-8)
            u = np.ones_like(w) / len(w)
            kl += float(np.sum(w * (np.log(w + 1e-12) - np.log(u))))
        return kl

    def state_dict(self) -> Dict[str, object]:
        return {
            "sample_ids": self.sample_ids,
            "subject_ids": self.subject_ids,
            "labels": self.labels,
            "age": self.age,
            "sex": self.sex,
            "education": self.education,
            "loss_memory": self.loss_memory,
            "weights": self.weights,
            "subject_loss_memory": self.subject_loss_memory,
            "subject_weights": self.subject_weights,
            "unique_subjects": self.unique_subjects,
            "enabled": self.enabled,
            "hyper": {
                "age_bandwidth": self.age_bandwidth,
                "education_bandwidth": self.education_bandwidth,
                "cross_sex_rho": self.cross_sex_rho,
                "temperature": self.temperature,
                "ema_beta": self.ema_beta,
                "ess_ratio_min": self.ess_ratio_min,
                "max_weight_factor": self.max_weight_factor,
                "kl_lambda": self.kl_lambda,
            },
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        self.initialize(
            sample_ids=list(payload["sample_ids"]),  # type: ignore[arg-type]
            subject_ids=list(payload["subject_ids"]),  # type: ignore[arg-type]
            labels=np.asarray(payload["labels"], dtype=np.int64),
            age=np.asarray(payload["age"], dtype=float),
            sex=np.asarray(payload["sex"], dtype=float),
            education=np.asarray(payload["education"], dtype=float),
        )
        self.loss_memory = np.asarray(payload["loss_memory"], dtype=float)
        self.weights = np.asarray(payload["weights"], dtype=float)
        if "subject_loss_memory" in payload:
            self.subject_loss_memory = np.asarray(
                payload["subject_loss_memory"], dtype=float
            )
        if "subject_weights" in payload:
            self.subject_weights = np.asarray(payload["subject_weights"], dtype=float)
        self.enabled = bool(payload.get("enabled", False))
        hyper = payload.get("hyper") or {}
        for key, value in hyper.items():  # type: ignore[union-attr]
            if hasattr(self, key):
                setattr(self, key, value)
