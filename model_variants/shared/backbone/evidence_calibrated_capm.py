"""Image-evidence calibrated CAPM (IE-CAPM).

This module keeps CAPM's variable-specific, spatial modulation but only lets a
variable act strongly at locations whose *image features* support that action.
The evidence gate never receives the raw tabular vector.  Consequently, it is
not another tabular shortcut: it can attenuate a CAPM modulation, but cannot
invent one from a demographic value alone.

For feature map ``F`` and CAPM's pre-sigmoid variable fields ``r_v(z_v)``:

    g_v(F) = sigmoid(q_v(F))
    m(F, z) = sigmoid(sum_v g_v(F) r_v(z_v))
    F_out = F + Norm((1 - m(F, z)) F)

``g_v == 1`` recovers the corresponding SCA-CAPM computation exactly.  The
factory at the bottom of this file therefore gives an explicit CAPM control by
passing ``force_capm=True`` to ``forward``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.backbone.film_backbone import BasicBlock


VarSpec = Mapping[str, Any]


class RBFEmbedding(nn.Module):
    """Fixed radial-basis expansion used for one continuous table variable."""

    def __init__(self, n_centers: int, var_min: float, var_max: float) -> None:
        super().__init__()
        if n_centers < 2:
            raise ValueError("n_centers must be at least 2")
        centers = torch.linspace(float(var_min), float(var_max), n_centers)
        self.register_buffer("centers", centers)
        spacing = float(var_max - var_min) / (n_centers - 1)
        self.sigma = max(spacing, 1e-6)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        delta = values.float().reshape(-1, 1) - self.centers.reshape(1, -1)
        return torch.exp(-0.5 * (delta / self.sigma).square())


class EvidenceCalibratedCAPM(nn.Module):
    """CAPM with image-only, per-variable spatial evidence gates.

    ``var_specs`` follows the existing CAPM convention.  Each specification
    needs ``name`` and ``type``; continuous variables also need ``min_val`` and
    ``max_val``, categorical variables need ``n_cats``.  ``n_total_bases`` is
    inferred from the individual variable allocations by default, preventing
    the silent base-slice mismatch present when those two settings diverge.
    """

    def __init__(
        self,
        var_specs: Sequence[VarSpec],
        feature_dim: int,
        spatial_shape: tuple[int, int, int] = (4, 4, 4),
        emb_dim: int = 16,
        evidence_hidden: int = 16,
        gate_init: float = 0.95,
        min_gate_mean: float = 0.65,
        min_effective_ratio: float = 0.60,
        n_total_bases: int | None = None,
    ) -> None:
        super().__init__()
        if not var_specs:
            raise ValueError("var_specs must contain at least one variable")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between zero and one")
        self.var_specs = [dict(spec) for spec in var_specs]
        self.feature_dim = feature_dim
        self.n_vars = len(self.var_specs)
        self.spatial_shape = spatial_shape
        self.min_gate_mean = min_gate_mean
        self.min_effective_ratio = min_effective_ratio

        self.n_bases_per_var = [int(spec.get("n_bases", 4)) for spec in self.var_specs]
        required_bases = sum(self.n_bases_per_var)
        self.n_total_bases = required_bases if n_total_bases is None else n_total_bases
        if self.n_total_bases != required_bases:
            raise ValueError(
                "n_total_bases must equal sum(spec['n_bases']) for a "
                "variable-specific basis allocation"
            )

        self.embedders = nn.ModuleList()
        self.coefficient_nets = nn.ModuleList()
        for spec, n_bases in zip(self.var_specs, self.n_bases_per_var):
            if spec["type"] == "continuous":
                n_centers = int(spec.get("n_centers", 8))
                embedder: nn.Module = nn.Sequential(
                    RBFEmbedding(n_centers, spec["min_val"], spec["max_val"]),
                    nn.Linear(n_centers, emb_dim),
                )
            elif spec["type"] == "categorical":
                embedder = nn.Embedding(int(spec["n_cats"]), emb_dim)
            else:
                raise ValueError("variable type must be 'continuous' or 'categorical'")
            self.embedders.append(embedder)
            net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, n_bases),
            )
            nn.init.normal_(net[-1].weight, std=0.001)
            nn.init.zeros_(net[-1].bias)
            self.coefficient_nets.append(net)

        self.spatial_bases = nn.Parameter(
            torch.empty(self.n_total_bases, 1, *spatial_shape)
        )
        nn.init.normal_(self.spatial_bases, std=0.02)

        # The gate takes only F.  Its final layer starts near a CAPM identity.
        self.evidence_gate = nn.Sequential(
            nn.Conv3d(feature_dim, evidence_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(evidence_hidden, self.n_vars, kernel_size=1),
        )
        nn.init.zeros_(self.evidence_gate[-1].weight)
        nn.init.constant_(self.evidence_gate[-1].bias, math.log(gate_init / (1.0 - gate_init)))
        self.last_audit: dict[str, torch.Tensor] | None = None

    def _split_tabular(self, z: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(z, torch.Tensor):
            if z.ndim != 2 or z.shape[1] != self.n_vars:
                raise ValueError(
                    f"Expected z with shape [B, {self.n_vars}], got {tuple(z.shape)}"
                )
            values = [z[:, index] for index in range(self.n_vars)]
        else:
            if len(z) != self.n_vars:
                raise ValueError(f"Expected {self.n_vars} tabular variables, got {len(z)}")
            values = list(z)
        return values

    def _variable_fields(self, z: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        fields: list[torch.Tensor] = []
        offset = 0
        for spec, value, embedder, coefficient_net, n_bases in zip(
            self.var_specs,
            self._split_tabular(z),
            self.embedders,
            self.coefficient_nets,
            self.n_bases_per_var,
        ):
            if spec["type"] == "categorical":
                value = value.long()
            else:
                value = value.float()
            coefficients = coefficient_net(embedder(value))
            bases = self.spatial_bases[offset : offset + n_bases].reshape(n_bases, -1)
            field = coefficients @ bases
            fields.append(field.reshape(value.shape[0], 1, *self.spatial_shape))
            offset += n_bases
        return torch.cat(fields, dim=1)

    @staticmethod
    def _normalize_residual(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        # ``unbiased=False`` remains defined at the final 1x1x1 ResNet stage.
        std = x.std(dim=(-3, -2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | Sequence[torch.Tensor],
        evidence_features: torch.Tensor | None = None,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Adjust ``x`` and optionally return non-detached mechanism tensors."""
        raw_fields = self._variable_fields(z)
        raw_fields = F.interpolate(
            raw_fields, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        evidence_source = x if evidence_features is None else evidence_features
        if evidence_source.shape != x.shape:
            raise ValueError("evidence_features must have the same shape as x")
        gates = torch.ones_like(raw_fields) if force_capm else torch.sigmoid(self.evidence_gate(evidence_source))
        effective_fields = gates * raw_fields
        modulation = torch.sigmoid(effective_fields.sum(dim=1, keepdim=True))
        adjusted = x * (1.0 - modulation)
        output = x + self._normalize_residual(adjusted)

        audit = {
            "gates": gates,
            "raw_fields": raw_fields,
            "effective_fields": effective_fields,
            "modulation": modulation,
        }
        self.last_audit = audit
        if return_audit:
            return output, audit
        return output

    def regularization_losses(self) -> dict[str, torch.Tensor]:
        """Anti-collapse and spatial-complexity terms for the latest forward pass."""
        zero = self.spatial_bases.new_zeros(())
        if self.last_audit is None:
            return {
                "basis_tv": zero,
                "basis_orth": zero,
                "gate_anchor": zero,
                "gate_floor": zero,
                "modulation_preservation": zero,
            }

        bases = self.spatial_bases
        basis_tv = (
            (bases[:, :, 1:] - bases[:, :, :-1]).abs().mean()
            + (bases[:, :, :, 1:] - bases[:, :, :, :-1]).abs().mean()
            + (bases[:, :, :, :, 1:] - bases[:, :, :, :, :-1]).abs().mean()
        ) / 3.0
        flat = F.normalize(bases.flatten(1), dim=1)
        gram = flat @ flat.T
        basis_orth = (gram - torch.eye(self.n_total_bases, device=bases.device)).square().mean()

        gates = self.last_audit["gates"]
        raw = self.last_audit["raw_fields"]
        effective = self.last_audit["effective_fields"]
        gate_anchor = (gates - 1.0).square().mean()
        gate_floor = F.relu(self.min_gate_mean - gates.mean()).square()
        raw_energy = raw.abs().mean().detach()
        effective_ratio = effective.abs().mean() / raw_energy.clamp_min(1e-6)
        active = (raw_energy > 1e-5).to(effective_ratio.dtype)
        modulation_preservation = active * F.relu(
            self.min_effective_ratio - effective_ratio
        ).square()
        return {
            "basis_tv": basis_tv,
            "basis_orth": basis_orth,
            "gate_anchor": gate_anchor,
            "gate_floor": gate_floor,
            "modulation_preservation": modulation_preservation,
        }


def default_adni_var_specs() -> list[dict[str, Any]]:
    """Default nine-column ADNI/NACC-compatible table contract for IE-CAPM."""
    return [
        {"name": "age", "type": "continuous", "n_centers": 8, "min_val": 55, "max_val": 95, "n_bases": 6},
        {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 2},
        {"name": "education", "type": "continuous", "n_centers": 6, "min_val": 0, "max_val": 22, "n_bases": 4},
        {"name": "MMSE", "type": "continuous", "n_centers": 8, "min_val": 0, "max_val": 30, "n_bases": 6},
        {"name": "APOE4", "type": "categorical", "n_cats": 3, "n_bases": 3},
        {"name": "CDRSB", "type": "continuous", "n_centers": 6, "min_val": 0, "max_val": 18, "n_bases": 4},
        {"name": "ADAS11", "type": "continuous", "n_centers": 6, "min_val": 0, "max_val": 70, "n_bases": 4},
        {"name": "FAQ", "type": "continuous", "n_centers": 6, "min_val": 0, "max_val": 30, "n_bases": 3},
        {"name": "site", "type": "categorical", "n_cats": 10, "n_bases": 2},
    ]


class ResNetEvidenceCalibratedCAPMBackbone(nn.Module):
    """3D ResNet whose CAPM modulations are calibrated by MRI evidence.

        During a gated forward pass, a shared-weight table-free visual stream is
        propagated alongside the conditioned stream.  It is the sole source of
        evidence gates, so a later gate cannot receive an earlier CAPM output.
        """

    def __init__(
        self,
        txt_dim: int = 9,
        num_classes: int = 3,
        spatial_shape: tuple[int, int, int] = (4, 4, 4),
        block: type[BasicBlock] = BasicBlock,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        var_specs: Sequence[VarSpec] | None = None,
        evidence_hidden: int = 16,
        gate_init: float = 0.95,
    ) -> None:
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes
        self.var_specs = default_adni_var_specs() if var_specs is None else list(var_specs)
        if len(self.var_specs) != txt_dim:
            raise ValueError("txt_dim must equal len(var_specs)")

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        stage_channels = {
            "layer1": 64 * block.expansion,
            "layer2": 128 * block.expansion,
            "layer3": 256 * block.expansion,
            "layer4": 512 * block.expansion,
        }
        self.calibrators = nn.ModuleDict({
            name: EvidenceCalibratedCAPM(
                var_specs=self.var_specs,
                feature_dim=channels,
                spatial_shape=spatial_shape,
                evidence_hidden=evidence_hidden,
                gate_init=gate_init,
            )
            for name, channels in stage_channels.items()
        })
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_backbone_weights()

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        modules = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        modules.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*modules)

    def _init_backbone_weights(self) -> None:
        calibrator_parameters = {id(parameter) for calibrator in self.calibrators.values() for parameter in calibrator.parameters()}
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                if all(id(parameter) not in calibrator_parameters for parameter in module.parameters()):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                if all(id(parameter) not in calibrator_parameters for parameter in module.parameters()):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _forward_stage(
        self,
        name: str,
        features: torch.Tensor,
        z: torch.Tensor | None,
        evidence_features: torch.Tensor | None,
        force_capm: bool,
        return_audit: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        stage = getattr(self, name)
        features = stage(features)
        if evidence_features is not None:
            evidence_features = stage(evidence_features)
        if z is None:
            return features, evidence_features, None
        result = self.calibrators[name](
            features,
            z,
            evidence_features=evidence_features,
            force_capm=force_capm,
            return_audit=return_audit,
        )
        if return_audit:
            output, audit = result  # type: ignore[misc]
            return output, evidence_features, audit
        return result, evidence_features, None  # type: ignore[return-value]

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Return logits; ``force_capm=True`` is the exact ungated CAPM control."""
        features = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # Avoid an unnecessary second path for image-only and CAPM-control runs.
        evidence_features = features if z is not None and not force_capm else None
        audits: dict[str, dict[str, torch.Tensor]] = {}
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features, evidence_features, audit = self._forward_stage(
                name, features, z, evidence_features, force_capm, return_audit
            )
            if audit is not None:
                audits[name] = audit
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audits
        return logits

    def regularization_losses(self) -> dict[str, torch.Tensor]:
        """Aggregate the named IE-CAPM constraints from all stages."""
        losses: dict[str, torch.Tensor] = {}
        for calibrator in self.calibrators.values():
            for name, value in calibrator.regularization_losses().items():
                losses[name] = losses.get(name, value.new_zeros(())) + value
        return losses


def resnet18_ie_capm(
    txt_dim: int = 9,
    num_classes: int = 3,
    var_specs: Sequence[VarSpec] | None = None,
    **kwargs: Any,
) -> ResNetEvidenceCalibratedCAPMBackbone:
    """Instantiate the journal candidate with the ResNet-18 depth schedule."""
    return ResNetEvidenceCalibratedCAPMBackbone(
        txt_dim=txt_dim,
        num_classes=num_classes,
        var_specs=var_specs,
        layers=(2, 2, 2, 2),
        **kwargs,
    )
