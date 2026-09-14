"""Structured outputs for DualShiftResNet3D."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class DualShiftOutput:
    clean_logits: torch.Tensor
    shifted_logits: Optional[torch.Tensor] = None
    clean_embedding: Optional[torch.Tensor] = None
    shifted_embedding: Optional[torch.Tensor] = None
    demographic_embedding: Optional[torch.Tensor] = None
    shift_strength: Optional[torch.Tensor] = None
    selected_protocol_index: Optional[torch.Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def primary_logits(self) -> torch.Tensor:
        return self.clean_logits
