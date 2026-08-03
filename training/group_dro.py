"""Numerically stable GroupDRO objective."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GroupDRO(nn.Module):
    """Exponentiated-gradient GroupDRO over integer environment IDs.

    Group weights are stored in log space and therefore participate in the
    regular ``state_dict`` mechanism.  Groups absent from a minibatch keep
    their previous weight and are excluded from that minibatch's weighted
    mean.
    """

    def __init__(
        self,
        num_groups: int,
        step_size: float = 0.01,
        *,
        weight_floor: float = 1e-12,
    ):
        super().__init__()
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if step_size < 0:
            raise ValueError("step_size must be non-negative")
        self.num_groups = int(num_groups)
        self.step_size = float(step_size)
        self.weight_floor = float(weight_floor)
        self.register_buffer("log_group_weights", torch.zeros(num_groups))

    @property
    def group_weights(self) -> Tensor:
        """Normalized detached view of the current adversarial weights."""
        return torch.softmax(self.log_group_weights, dim=0)

    def reset(self) -> None:
        self.log_group_weights.zero_()

    def compute_group_losses(
        self, sample_losses: Tensor, environment_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        if sample_losses.ndim != 1:
            raise ValueError("sample_losses must have shape [B]")
        environment_ids = environment_ids.reshape(-1).long()
        if environment_ids.numel() != sample_losses.numel():
            raise ValueError("environment_ids and sample_losses must have equal length")
        if environment_ids.numel() and (
            environment_ids.min() < 0 or environment_ids.max() >= self.num_groups
        ):
            raise ValueError(
                f"environment_ids must be in [0, {self.num_groups - 1}]"
            )

        losses = sample_losses.new_zeros(self.num_groups)
        present = torch.zeros(
            self.num_groups, dtype=torch.bool, device=sample_losses.device
        )
        for group_id in range(self.num_groups):
            mask = environment_ids == group_id
            if mask.any():
                losses[group_id] = sample_losses[mask].mean()
                present[group_id] = True
        return losses, present

    @torch.no_grad()
    def update(self, group_losses: Tensor, present: Tensor) -> None:
        if group_losses.shape != self.log_group_weights.shape:
            raise ValueError("group_losses has an incompatible shape")
        present = present.to(device=self.log_group_weights.device, dtype=torch.bool)
        increments = self.step_size * group_losses.detach().to(
            self.log_group_weights.device
        )
        self.log_group_weights[present] += increments[present]

        # A common shift leaves softmax unchanged and prevents overflow.
        self.log_group_weights -= self.log_group_weights.max()
        if self.weight_floor > 0:
            minimum = torch.log(
                self.log_group_weights.new_tensor(self.weight_floor)
            )
            self.log_group_weights.clamp_(min=minimum)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        environment_ids: Tensor,
        *,
        update: bool = True,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        sample_losses = F.cross_entropy(logits, targets.long(), reduction="none")
        group_losses, present = self.compute_group_losses(
            sample_losses, environment_ids.to(logits.device)
        )
        if update:
            self.update(group_losses, present)

        weights = self.group_weights.to(logits.device)
        present_weights = weights * present.to(weights.dtype)
        denominator = present_weights.sum().clamp_min(
            torch.finfo(present_weights.dtype).tiny
        )
        objective = (present_weights * group_losses).sum() / denominator
        if not return_details:
            return objective
        return objective, {
            "sample_losses": sample_losses,
            "group_losses": group_losses,
            "present_groups": present,
            "group_weights": weights,
        }


__all__ = ["GroupDRO"]
