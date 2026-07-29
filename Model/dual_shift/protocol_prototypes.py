"""Source-only protocol prototype bank for APIS."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


def _empty_pending() -> Dict[str, Dict[str, Any]]:
    return {}


class ProtocolPrototypeBank:
    """Epoch-frozen source protocol prototypes with pending epoch accumulators."""

    def __init__(
        self,
        min_subjects: int = 8,
        ema_beta: float = 0.9,
        distance_lo_quantile: float = 0.05,
        distance_hi_quantile: float = 0.95,
    ):
        self.min_subjects = int(min_subjects)
        self.ema_beta = float(ema_beta)
        self.distance_lo_quantile = float(distance_lo_quantile)
        self.distance_hi_quantile = float(distance_hi_quantile)
        self.prototypes: Dict[str, Dict[str, Any]] = {}
        self.pending: Dict[str, Dict[str, Any]] = _empty_pending()
        self.frozen = False
        self.distance_lo: Optional[float] = None
        self.distance_hi: Optional[float] = None
        self.last_selection_audit: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.prototypes = {}
        self.pending = _empty_pending()
        self.frozen = False
        self.distance_lo = None
        self.distance_hi = None
        self.last_selection_audit = []

    def begin_epoch(self) -> None:
        """Start an epoch: keep frozen bank for sampling, clear pending stats."""
        self.pending = _empty_pending()
        self.frozen = True
        self.last_selection_audit = []

    def freeze_for_epoch(self) -> None:
        """Mark bank read-only for sampling and clear pending accumulators."""
        self.frozen = True
        self.pending = _empty_pending()

    def _fallback_keys(self, key: str) -> List[str]:
        parts = key.split("|")
        if len(parts) != 4:
            return [key]
        manufacturer, field, _model, sequence = parts
        return [
            key,
            f"{manufacturer}|{field}|*|{sequence}",
            f"{manufacturer}|{field}|*|*",
            f"{manufacturer}|*|*|*",
        ]

    def resolve_key(self, key: str) -> Optional[str]:
        for candidate in self._fallback_keys(key):
            if (
                candidate in self.prototypes
                and self._subject_count(candidate) >= self.min_subjects
            ):
                return candidate
        return None

    def _subject_count(self, key: str) -> int:
        proto = self.prototypes.get(key)
        if proto is None:
            return 0
        return len(proto.get("subjects") or [])

    @torch.no_grad()
    def collect_epoch_statistics(
        self,
        domain_keys: Sequence[str],
        subject_ids: Sequence[str],
        acquisition_embeddings: torch.Tensor,
        layer1: torch.Tensor,
        layer2: torch.Tensor,
    ) -> None:
        """Accumulate detached statistics for the current epoch (does not alter frozen bank)."""
        from Model.dual_shift.apis import channel_stats

        mean1, std1 = channel_stats(layer1)
        mean2, std2 = channel_stats(layer2)
        mean1 = mean1.flatten(1)
        std1 = std1.flatten(1)
        mean2 = mean2.flatten(1)
        std2 = std2.flatten(1)
        grouped: Dict[str, List[int]] = defaultdict(list)
        for index, key in enumerate(domain_keys):
            grouped[key].append(index)
        for key, indices in grouped.items():
            subjects = {str(subject_ids[i]) for i in indices}
            emb = acquisition_embeddings[indices].mean(dim=0).detach().cpu()
            m1 = mean1[indices].mean(dim=0).detach().cpu()
            s1 = std1[indices].mean(dim=0).detach().cpu()
            m2 = mean2[indices].mean(dim=0).detach().cpu()
            s2 = std2[indices].mean(dim=0).detach().cpu()
            if key not in self.pending:
                self.pending[key] = {
                    "embedding": emb,
                    "mean1": m1,
                    "std1": s1,
                    "mean2": m2,
                    "std2": s2,
                    "subjects": set(subjects),
                    "n_updates": 1,
                }
            else:
                entry = self.pending[key]
                n = int(entry["n_updates"])
                w_old = n / (n + 1)
                w_new = 1.0 / (n + 1)
                for name, value in (
                    ("embedding", emb),
                    ("mean1", m1),
                    ("std1", s1),
                    ("mean2", m2),
                    ("std2", s2),
                ):
                    entry[name] = w_old * entry[name] + w_new * value
                entry["subjects"].update(subjects)
                entry["n_updates"] = n + 1

    def update_from_batch(self, *args, **kwargs) -> None:
        """Back-compat alias: routes to pending-only collection."""
        self.collect_epoch_statistics(*args, **kwargs)

    def _rebuild_fallbacks(self) -> None:
        exact_items = [
            (key, proto)
            for key, proto in self.prototypes.items()
            if "*" not in key and len(proto["subjects"]) >= 1
        ]
        self.prototypes = {
            key: proto for key, proto in self.prototypes.items() if "*" not in key
        }
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for key, proto in exact_items:
            manufacturer, field, _model, sequence = key.split("|")
            buckets[f"{manufacturer}|{field}|*|{sequence}"].append(proto)
            buckets[f"{manufacturer}|{field}|*|*"].append(proto)
            buckets[f"{manufacturer}|*|*|*"].append(proto)
        for key, protos in buckets.items():
            subjects = set()
            for proto in protos:
                subjects.update(proto["subjects"])
            if len(subjects) < self.min_subjects:
                continue
            self.prototypes[key] = {
                "embedding": torch.stack([p["embedding"] for p in protos]).mean(0),
                "mean1": torch.stack([p["mean1"] for p in protos]).mean(0),
                "std1": torch.stack([p["std1"] for p in protos]).mean(0),
                "mean2": torch.stack([p["mean2"] for p in protos]).mean(0),
                "std2": torch.stack([p["std2"] for p in protos]).mean(0),
                "subjects": subjects,
                "n_updates": int(sum(p["n_updates"] for p in protos)),
            }

    def end_epoch_update(self) -> None:
        """Merge pending epoch stats into the bank and freeze for the next epoch."""
        beta = self.ema_beta
        for key, pending in self.pending.items():
            if key not in self.prototypes:
                self.prototypes[key] = {
                    "embedding": pending["embedding"].clone(),
                    "mean1": pending["mean1"].clone(),
                    "std1": pending["std1"].clone(),
                    "mean2": pending["mean2"].clone(),
                    "std2": pending["std2"].clone(),
                    "subjects": set(pending["subjects"]),
                    "n_updates": int(pending["n_updates"]),
                }
            else:
                proto = self.prototypes[key]
                for name in ("embedding", "mean1", "std1", "mean2", "std2"):
                    proto[name] = beta * proto[name] + (1.0 - beta) * pending[name]
                proto["subjects"].update(pending["subjects"])
                proto["n_updates"] = int(proto["n_updates"]) + int(pending["n_updates"])
        self._rebuild_fallbacks()
        self.fit_distance_bounds_from_bank()
        self.pending = _empty_pending()
        self.frozen = True

    def eligible_keys(self) -> List[str]:
        return [
            key
            for key, proto in self.prototypes.items()
            if len(proto.get("subjects") or []) >= self.min_subjects
        ]

    def fit_distance_bounds_from_bank(self) -> None:
        eligible = self.eligible_keys()
        if len(eligible) < 2:
            self.distance_lo = None
            self.distance_hi = None
            return
        embeddings = torch.stack(
            [self.prototypes[key]["embedding"].float().reshape(-1) for key in eligible]
        )
        dists = torch.cdist(embeddings, embeddings, p=2)
        mask = ~torch.eye(len(eligible), dtype=torch.bool)
        values = dists[mask].detach().cpu()
        if values.numel() == 0:
            self.distance_lo = None
            self.distance_hi = None
            return
        self.distance_lo = float(
            torch.quantile(values, self.distance_lo_quantile).item()
        )
        self.distance_hi = float(
            torch.quantile(values, self.distance_hi_quantile).item()
        )

    def _protocol_parts(self, key: str) -> Optional[Tuple[str, str, str, str]]:
        parts = key.split("|")
        if len(parts) != 4:
            return None
        return parts[0], parts[1], parts[2], parts[3]

    def _is_same_protocol(self, source_key: str, candidate_key: str) -> bool:
        src = self._protocol_parts(source_key)
        cand = self._protocol_parts(candidate_key)
        if src is None or cand is None:
            return source_key == candidate_key
        return src[0] == cand[0] and src[1] == cand[1]

    def _embedding_distance(self, source_key: str, candidate_key: str) -> Optional[float]:
        resolved = self.resolve_key(source_key)
        src_proto = self.prototypes.get(resolved) if resolved else None
        if src_proto is None:
            src_proto = self.prototypes.get(source_key)
        cand_proto = self.prototypes.get(candidate_key)
        if src_proto is None or cand_proto is None:
            return None
        a = src_proto["embedding"].float().reshape(-1)
        b = cand_proto["embedding"].float().reshape(-1)
        return float(torch.norm(a - b, p=2).item())

    def _filter_candidates(self, source_key: str, eligible: Sequence[str]) -> List[str]:
        current = self.resolve_key(source_key)
        filtered = []
        for name in eligible:
            if name == current:
                continue
            if self._is_same_protocol(source_key, name):
                continue
            distance = self._embedding_distance(source_key, name)
            if (
                distance is not None
                and self.distance_lo is not None
                and self.distance_hi is not None
            ):
                if distance < self.distance_lo or distance > self.distance_hi:
                    continue
            filtered.append(name)
        return filtered

    def sample_targets(
        self,
        domain_keys: Sequence[str],
        *,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        eligible = self.eligible_keys()
        batch = len(domain_keys)
        if not eligible:
            return None, None, None, None, torch.full((batch,), -1, device=device)

        mean1 = []
        std1 = []
        mean2 = []
        std2 = []
        selected = []
        audit = []
        for key in domain_keys:
            candidates = self._filter_candidates(key, eligible)
            if not candidates:
                mean1.append(None)
                std1.append(None)
                mean2.append(None)
                std2.append(None)
                selected.append(-1)
                audit.append(
                    {
                        "source_key": key,
                        "chosen": None,
                        "reason": "no_valid_candidate",
                        "n_candidates": 0,
                    }
                )
                continue
            idx = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
            chosen = candidates[idx]
            proto = self.prototypes[chosen]
            mean1.append(proto["mean1"].to(device))
            std1.append(proto["std1"].to(device))
            mean2.append(proto["mean2"].to(device))
            std2.append(proto["std2"].to(device))
            selected.append(eligible.index(chosen) if chosen in eligible else -1)
            audit.append(
                {
                    "source_key": key,
                    "chosen": chosen,
                    "distance": self._embedding_distance(key, chosen),
                    "fallback_level": chosen.count("*"),
                    "n_subjects": len(proto.get("subjects") or []),
                    "n_candidates": len(candidates),
                }
            )
        self.last_selection_audit = audit
        if any(item is None for item in mean1):
            return None, None, None, None, torch.full((batch,), -1, device=device)
        return (
            torch.stack(mean1, dim=0),
            torch.stack(std1, dim=0),
            torch.stack(mean2, dim=0),
            torch.stack(std2, dim=0),
            torch.tensor(selected, device=device, dtype=torch.long),
        )

    def state_dict(self) -> Dict[str, Any]:
        payload = {
            "min_subjects": self.min_subjects,
            "ema_beta": self.ema_beta,
            "distance_lo": self.distance_lo,
            "distance_hi": self.distance_hi,
            "frozen": self.frozen,
            "items": {},
            "pending": {},
        }
        for key, proto in self.prototypes.items():
            payload["items"][key] = {
                "embedding": proto["embedding"],
                "mean1": proto["mean1"],
                "std1": proto["std1"],
                "mean2": proto["mean2"],
                "std2": proto["std2"],
                "subjects": sorted(map(str, proto["subjects"])),
                "n_updates": proto["n_updates"],
            }
        for key, proto in self.pending.items():
            payload["pending"][key] = {
                "embedding": proto["embedding"],
                "mean1": proto["mean1"],
                "std1": proto["std1"],
                "mean2": proto["mean2"],
                "std2": proto["std2"],
                "subjects": sorted(map(str, proto["subjects"])),
                "n_updates": proto["n_updates"],
            }
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.min_subjects = int(payload.get("min_subjects", self.min_subjects))
        self.ema_beta = float(payload.get("ema_beta", self.ema_beta))
        self.distance_lo = payload.get("distance_lo")
        self.distance_hi = payload.get("distance_hi")
        self.frozen = bool(payload.get("frozen", False))
        self.prototypes = {}
        for key, proto in (payload.get("items") or {}).items():
            self.prototypes[key] = {
                "embedding": proto["embedding"],
                "mean1": proto["mean1"],
                "std1": proto["std1"],
                "mean2": proto["mean2"],
                "std2": proto["std2"],
                "subjects": set(proto.get("subjects") or []),
                "n_updates": int(proto.get("n_updates", 1)),
            }
        self.pending = {}
        for key, proto in (payload.get("pending") or {}).items():
            self.pending[key] = {
                "embedding": proto["embedding"],
                "mean1": proto["mean1"],
                "std1": proto["std1"],
                "mean2": proto["mean2"],
                "std2": proto["std2"],
                "subjects": set(proto.get("subjects") or []),
                "n_updates": int(proto.get("n_updates", 1)),
            }
