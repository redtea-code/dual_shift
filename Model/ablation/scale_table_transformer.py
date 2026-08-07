"""Scale and patch-table interaction ablations for IE-CAPM.

The module separates two variables that are easily conflated:

1. feature scale: which ResNet stage supplies the spatial representation;
2. token granularity: how many feature-map cells form one transformer token.

For a feature map ``F_s`` and patch size ``p``, the token count is the product
of the three patch-grid dimensions.  The provided ``layer3_patch2`` and
``layer4_pixel`` presets both produce 64 tokens for a 128-cubed input, while
using different semantic depths.  ``layer5_pixel`` is the coarse 8-token
extreme.

The transformer variant does not replace CAPM.  It predicts a variable-wise
gate over CAPM's pre-sigmoid spatial fields.  Setting every gate to one gives
the exact CAPM computation implemented by the same module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.backbone.evidence_calibrated_capm import EvidenceCalibratedCAPM
from Model.backbone.film_backbone import BasicBlock


VarSpec = Mapping[str, Any]


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        if len(value) != 3:
            raise ValueError("patch_size must be an int or a length-three sequence")
        result = tuple(int(item) for item in value)
    if any(item < 1 for item in result):
        raise ValueError("patch_size entries must be positive")
    return result


def demographic_var_specs() -> list[dict[str, Any]]:
    """The frozen age/sex/education contract in the aligned protocol."""
    return [
        {
            "name": "age",
            "type": "continuous",
            "n_centers": 8,
            "min_val": 55,
            "max_val": 95,
            "n_bases": 6,
        },
        {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 2},
        {
            "name": "education",
            "type": "continuous",
            "n_centers": 6,
            "min_val": 0,
            "max_val": 22,
            "n_bases": 4,
        },
    ]


@dataclass(frozen=True)
class AblationPreset:
    selected_stage: str
    patch_size: tuple[int, int, int]
    expected_tokens_128: int
    expected_tokens_160x196x160: int


ABLATION_PRESETS: dict[str, AblationPreset] = {
    "layer3_patch2": AblationPreset("layer3", (2, 2, 2), 64, 175),
    "layer4_pixel": AblationPreset("layer4", (1, 1, 1), 64, 175),
    "layer5_pixel": AblationPreset("layer5", (1, 1, 1), 8, 36),
}


def _ceil_stride2(size: int) -> int:
    """Output size of the stride-two convolution/pooling used by this backbone."""
    return (size + 1) // 2


def _selected_feature_shape(
    input_shape: Sequence[int], selected_stage: str
) -> tuple[int, int, int]:
    """Infer the fixed source geometry required by original patchwise CAPM."""
    shape = tuple(int(size) for size in input_shape)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("input_shape must contain three positive spatial dimensions")
    # Conv1 then maxpool, followed by the stride-two starts of layers 2--5.
    for _ in range(2):
        shape = tuple(_ceil_stride2(size) for size in shape)
    stage_downsamples = {"layer3": 2, "layer4": 3, "layer5": 4}
    if selected_stage not in stage_downsamples:
        raise ValueError(f"Unsupported selected stage {selected_stage!r}")
    for _ in range(stage_downsamples[selected_stage]):
        shape = tuple(_ceil_stride2(size) for size in shape)
    return shape


def _right_padded_shape(
    feature_shape: Sequence[int], patch_size: int | Sequence[int]
) -> tuple[int, int, int]:
    """Return the smallest patch-divisible shape using right-side zero padding."""
    patch = _triple(patch_size)
    return tuple(
        size + (step - size % step) % step
        for size, step in zip((int(value) for value in feature_shape), patch)
    )


class OriginalPatchwiseCAPM(nn.Module):
    """Source-faithful patchwise CAPM from ``redtea-code/Causal_fusion``.

    The external source (Apache-2.0, commit d1a37d9) implements
    ``PatchwiseBackdoorBlock`` after its visual patch extractor.  For visual
    patches ``X[b,p]`` and table embedding ``z_c`` it computes:

        gamma = MLP(z_c) in R^(B x P)
        X_adj[b,p] = X[b,p] - gamma[b,p] X[b,p] + X[b,p]

    Its code flattens the patch axis into the batch axis before broadcasting;
    this implementation has the identical numerical operation while retaining
    the patch axis for auditing.  The source used a fixed patch count, so this
    module intentionally requires a fixed expected feature geometry.
    """

    def __init__(
        self,
        txt_dim: int,
        patch_size: int | Sequence[int],
        expected_feature_shape: Sequence[int],
        table_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.expected_feature_shape = tuple(int(size) for size in expected_feature_shape)
        if len(self.expected_feature_shape) != 3:
            raise ValueError("expected_feature_shape must have three dimensions")
        if any(
            size % patch != 0
            for size, patch in zip(self.expected_feature_shape, self.patch_size)
        ):
            raise ValueError(
                "Original patchwise CAPM requires feature dimensions divisible by patch_size"
            )
        self.patch_grid = tuple(
            size // patch
            for size, patch in zip(self.expected_feature_shape, self.patch_size)
        )
        self.patch_count = math.prod(self.patch_grid)
        # Same two-layer table encoder and per-patch gamma head as Ours.py.
        self.table_encoder = nn.Sequential(
            nn.Linear(txt_dim, 2 * table_dim),
            nn.ReLU(),
            nn.Linear(2 * table_dim, table_dim),
        )
        self.z_to_patch = nn.Sequential(
            nn.Linear(table_dim, table_dim),
            nn.ReLU(inplace=True),
            nn.Linear(table_dim, self.patch_count),
        )
        self.last_audit: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        *,
        return_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actual_feature_shape = tuple(x.shape[2:])
        if any(actual > expected for actual, expected in zip(actual_feature_shape, self.expected_feature_shape)):
            raise ValueError(
                "Original patchwise CAPM was initialized for feature shape "
                f"up to {self.expected_feature_shape}, received {actual_feature_shape}"
            )
        if z.ndim != 2 or z.shape[1] != self.table_encoder[0].in_features:
            raise ValueError(
                f"Expected table shape [B, {self.table_encoder[0].in_features}], "
                f"got {tuple(z.shape)}"
            )
        batch_size, channels, depth, height, width = x.shape
        if z.shape[0] != batch_size:
            raise ValueError(
                f"Image and table batch sizes must match, got {batch_size} and {z.shape[0]}"
            )
        pad_d = self.expected_feature_shape[0] - depth
        pad_h = self.expected_feature_shape[1] - height
        pad_w = self.expected_feature_shape[2] - width
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        depth, height, width = self.expected_feature_shape
        patch_d, patch_h, patch_w = self.patch_size
        grid_d, grid_h, grid_w = self.patch_grid
        patches = x.unfold(2, patch_d, patch_d).unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        patches = patches.view(batch_size, self.patch_count, channels, patch_d, patch_h, patch_w)

        gamma = self.z_to_patch(self.table_encoder(z.float()))
        # Equivalent to source ``gamma.flatten(0)[:, None, None, None, None]``.
        flat_patches = patches.reshape(batch_size * self.patch_count, channels, patch_d, patch_h, patch_w)
        flat_gamma = gamma.reshape(batch_size * self.patch_count, 1, 1, 1, 1)
        flat_adjusted = flat_patches - flat_gamma * flat_patches
        flat_adjusted = flat_adjusted + flat_patches

        adjusted = flat_adjusted.view(
            batch_size, grid_d, grid_h, grid_w, channels, patch_d, patch_h, patch_w
        )
        output = adjusted.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        output = output.view(batch_size, channels, depth, height, width)
        if pad_d or pad_h or pad_w:
            output = output[:, :, : depth - pad_d, : height - pad_h, : width - pad_w]
        audit = {
            "gamma": gamma,
            "effective_scale": 2.0 - gamma,
            "patch_grid": torch.tensor(self.patch_grid, device=x.device),
            "patch_count": torch.tensor(self.patch_count, device=x.device),
            "right_padding": torch.tensor((pad_d, pad_h, pad_w), device=x.device),
        }
        self.last_audit = audit
        if return_audit:
            return output, audit
        return output

    def regularization_losses(self) -> dict[str, torch.Tensor]:
        # The original block does not define an auxiliary regularizer.
        return {}


class TransformerCalibratedCAPM(EvidenceCalibratedCAPM):
    """CAPM whose variable fields are gated by feature-token attention.

    ``interaction_mode='image_self'`` is a transformer-capacity control: the
    gate sees image tokens only.  ``interaction_mode='table_cross'`` lets image
    patch queries attend to the three value-aware table tokens.  Both modes use
    the same gate head and CAPM field generator.
    """

    def __init__(
        self,
        var_specs: Sequence[VarSpec],
        feature_dim: int,
        patch_size: int | Sequence[int] = 1,
        interaction_mode: str = "table_cross",
        spatial_shape: tuple[int, int, int] = (4, 4, 4),
        table_emb_dim: int = 16,
        transformer_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        min_gate_mean: float = 0.65,
        min_effective_ratio: float = 0.60,
    ) -> None:
        if interaction_mode not in {"image_self", "table_cross"}:
            raise ValueError("interaction_mode must be 'image_self' or 'table_cross'")
        if transformer_dim % num_heads != 0:
            raise ValueError("transformer_dim must be divisible by num_heads")
        super().__init__(
            var_specs=var_specs,
            feature_dim=feature_dim,
            spatial_shape=spatial_shape,
            emb_dim=table_emb_dim,
            evidence_hidden=max(transformer_dim // 4, 4),
            gate_init=gate_init,
            min_gate_mean=min_gate_mean,
            min_effective_ratio=min_effective_ratio,
        )
        # Remove the convolutional gate inherited from the baseline.  Keeping
        # unused parameters would invalidate the capacity comparison.
        self.evidence_gate = nn.Identity()
        self.patch_size = _triple(patch_size)
        self.interaction_mode = interaction_mode
        self.transformer_dim = transformer_dim

        self.patch_embedding = nn.Conv3d(
            feature_dim,
            transformer_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.position_conv = nn.Conv3d(
            transformer_dim,
            transformer_dim,
            kernel_size=3,
            padding=1,
            groups=transformer_dim,
            bias=False,
        )
        self.image_norm = nn.LayerNorm(transformer_dim)
        self.attention = nn.MultiheadAttention(
            transformer_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(transformer_dim)
        hidden_dim = max(int(transformer_dim * ffn_ratio), transformer_dim)
        self.ffn = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, transformer_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(transformer_dim)

        if interaction_mode == "table_cross":
            self.table_projections = nn.ModuleList(
                nn.Linear(table_emb_dim, transformer_dim) for _ in self.var_specs
            )
            self.variable_identity = nn.Parameter(
                torch.empty(1, self.n_vars, transformer_dim)
            )
            nn.init.normal_(self.variable_identity, std=0.02)
        else:
            self.table_projections = nn.ModuleList()
            self.register_parameter("variable_identity", None)

        self.gate_head = nn.Linear(transformer_dim, self.n_vars)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(
            self.gate_head.bias, math.log(gate_init / (1.0 - gate_init))
        )

    def _table_tokens(
        self, z: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor:
        values = self._split_tabular(z)
        tokens: list[torch.Tensor] = []
        for spec, value, embedder, projection in zip(
            self.var_specs, values, self.embedders, self.table_projections
        ):
            value = value.long() if spec["type"] == "categorical" else value.float()
            tokens.append(projection(embedder(value)))
        return torch.stack(tokens, dim=1) + self.variable_identity

    def _attention_gates(
        self,
        evidence_features: torch.Tensor,
        z: torch.Tensor | Sequence[torch.Tensor],
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spatial_size = evidence_features.shape[2:]
        if any(size < patch for size, patch in zip(spatial_size, self.patch_size)):
            raise ValueError(
                f"Feature size {tuple(spatial_size)} is smaller than patch size "
                f"{self.patch_size}"
            )
        right_padding = tuple(
            (patch - size % patch) % patch
            for size, patch in zip(spatial_size, self.patch_size)
        )
        if any(right_padding):
            evidence_features = F.pad(evidence_features, (0, right_padding[2], 0, right_padding[1], 0, right_padding[0]))
        padded_spatial_size = evidence_features.shape[2:]
        patch_map = self.patch_embedding(evidence_features)
        patch_map = patch_map + self.position_conv(patch_map)
        token_grid = patch_map.shape[2:]
        image_tokens = patch_map.flatten(2).transpose(1, 2)
        image_tokens = self.image_norm(image_tokens)

        if self.interaction_mode == "table_cross":
            context_tokens = self._table_tokens(z)
        else:
            context_tokens = image_tokens
        attention_output, attention_weights = self.attention(
            image_tokens,
            context_tokens,
            context_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        fused = self.attention_norm(image_tokens + attention_output)
        fused = self.ffn_norm(fused + self.ffn(fused))

        gate_logits = self.gate_head(fused)
        token_gates = torch.sigmoid(gate_logits)
        gate_map = token_gates.transpose(1, 2).reshape(
            evidence_features.shape[0], self.n_vars, *token_grid
        )
        gates = F.interpolate(
            gate_map,
            size=padded_spatial_size,
            mode="trilinear",
            align_corners=False,
        )
        audit = {
            "token_gates": token_gates,
            "token_grid": torch.tensor(token_grid, device=evidence_features.device),
            "token_count": torch.tensor(
                image_tokens.shape[1], device=evidence_features.device
            ),
        }
        if return_attention:
            if attention_weights is None:
                raise RuntimeError("Attention weights were requested but not returned")
            audit["attention"] = attention_weights
        gates = gates[
            :, :, : spatial_size[0], : spatial_size[1], : spatial_size[2]
        ]
        audit["right_padding"] = torch.tensor(
            right_padding, device=evidence_features.device
        )
        return gates, audit

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | Sequence[torch.Tensor],
        evidence_features: torch.Tensor | None = None,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        evidence_source = x if evidence_features is None else evidence_features
        if evidence_source.shape != x.shape:
            raise ValueError("evidence_features must have the same shape as x")

        raw_fields = self._variable_fields(z)
        raw_fields = F.interpolate(
            raw_fields, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        transformer_audit: dict[str, torch.Tensor] = {}
        if force_capm:
            gates = torch.ones_like(raw_fields)
        else:
            gates, transformer_audit = self._attention_gates(
                evidence_source, z, return_attention=return_audit
            )

        effective_fields = gates * raw_fields
        modulation = torch.sigmoid(effective_fields.sum(dim=1, keepdim=True))
        output = x + self._normalize_residual(x * (1.0 - modulation))
        audit = {
            "gates": gates,
            "raw_fields": raw_fields,
            "effective_fields": effective_fields,
            "modulation": modulation,
            **transformer_audit,
        }
        self.last_audit = audit
        if return_audit:
            return output, audit
        return output


class ScaleTableInteractionAblation3D(nn.Module):
    """A capacity-controlled 3D ResNet ablation harness.

    The selected feature stage is still table-free when its calibrator is
    called.  CAPM/IE-CAPM is applied once, so a later gate cannot read an
    earlier table-conditioned feature map.
    """

    _INTERACTIONS = {
        "image_only",
        "capm",
        "conv_gate",
        "original_capm",
        "transformer_self",
        "transformer_cross",
    }

    def __init__(
        self,
        preset: str = "layer4_pixel",
        interaction: str = "transformer_cross",
        num_classes: int = 2,
        var_specs: Sequence[VarSpec] | None = None,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        block: type[BasicBlock] = BasicBlock,
        spatial_shape: tuple[int, int, int] = (4, 4, 4),
        transformer_dim: int = 128,
        num_heads: int = 4,
        transformer_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        gate_init: float = 0.95,
        input_shape: tuple[int, int, int] = (160, 196, 160),
    ) -> None:
        super().__init__()
        if preset not in ABLATION_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(ABLATION_PRESETS)}")
        if interaction not in self._INTERACTIONS:
            raise ValueError(
                f"Unknown interaction {interaction!r}; choose from {sorted(self._INTERACTIONS)}"
            )
        self.preset_name = preset
        self.preset = ABLATION_PRESETS[preset]
        self.interaction = interaction
        self.input_shape = tuple(int(size) for size in input_shape)
        if len(self.input_shape) != 3 or any(size < 1 for size in self.input_shape):
            raise ValueError("input_shape must contain three positive spatial dimensions")
        self.var_specs = demographic_var_specs() if var_specs is None else list(var_specs)
        self.txt_dim = len(self.var_specs)

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if self.preset.selected_stage == "layer5":
            self.layer5: nn.Module = self._make_layer(block, 512, 1, stride=2)
        else:
            self.layer5 = nn.Identity()

        stage_channels = {"layer3": 256, "layer4": 512, "layer5": 512}
        feature_dim = stage_channels[self.preset.selected_stage]
        if interaction == "image_only":
            self.calibrator: nn.Module | None = None
        elif interaction == "original_capm":
            self.original_feature_shape = _selected_feature_shape(
                self.input_shape, self.preset.selected_stage
            )
            self.calibrator = OriginalPatchwiseCAPM(
                txt_dim=self.txt_dim,
                patch_size=self.preset.patch_size,
                expected_feature_shape=_right_padded_shape(
                    self.original_feature_shape, self.preset.patch_size
                ),
            )
        elif interaction in {"transformer_self", "transformer_cross"}:
            mode = "image_self" if interaction == "transformer_self" else "table_cross"
            self.calibrator = TransformerCalibratedCAPM(
                var_specs=self.var_specs,
                feature_dim=feature_dim,
                patch_size=self.preset.patch_size,
                interaction_mode=mode,
                spatial_shape=spatial_shape,
                transformer_dim=transformer_dim,
                num_heads=num_heads,
                dropout=transformer_dropout,
                gate_init=gate_init,
            )
        else:
            self.calibrator = EvidenceCalibratedCAPM(
                var_specs=self.var_specs,
                feature_dim=feature_dim,
                spatial_shape=spatial_shape,
                gate_init=gate_init,
            )
        if interaction != "original_capm":
            self.original_feature_shape: tuple[int, int, int] | None = None

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(512, num_classes)
        self._init_non_calibrator_weights()

    def _make_layer(
        self, block: type[BasicBlock], planes: int, blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        modules = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        modules.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*modules)

    def _init_non_calibrator_weights(self) -> None:
        calibrator_ids = set()
        if self.calibrator is not None:
            calibrator_ids = {id(parameter) for parameter in self.calibrator.parameters()}
        for module in self.modules():
            if any(id(parameter) in calibrator_ids for parameter in module.parameters()):
                continue
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_calibrator(
        self,
        features: torch.Tensor,
        z: torch.Tensor | None,
        force_capm: bool,
        return_audit: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.calibrator is None:
            return features, {}
        if z is None:
            raise ValueError(f"interaction={self.interaction!r} requires a table tensor")
        if self.interaction == "original_capm":
            if force_capm:
                raise ValueError(
                    "original_capm has no gate=1 CAPM control; compare it as a separate historical variant"
                )
            result = self.calibrator(features, z, return_audit=return_audit)
            if return_audit:
                output, audit = result  # type: ignore[misc]
                return output, audit
            return result, {}  # type: ignore[return-value]
        use_capm = force_capm or self.interaction == "capm"
        result = self.calibrator(
            features,
            z,
            evidence_features=features,
            force_capm=use_capm,
            return_audit=return_audit,
        )
        if return_audit:
            output, audit = result  # type: ignore[misc]
            return output, audit
        return result, {}  # type: ignore[return-value]

    def forward(
        self,
        image: torch.Tensor,
        table: torch.Tensor | None = None,
        *,
        force_capm: bool = False,
        return_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        audit: dict[str, torch.Tensor] = {}
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(self, name)(features)
            if name == self.preset.selected_stage:
                features, audit = self._apply_calibrator(
                    features, table, force_capm, return_audit
                )
        if self.preset.selected_stage == "layer5":
            features = self.layer5(features)
            features, audit = self._apply_calibrator(
                features, table, force_capm, return_audit
            )
        logits = self.fc(self.dropout(self.pool(features).flatten(1)))
        if return_audit:
            return logits, audit
        return logits

    def regularization_losses(self) -> dict[str, torch.Tensor]:
        if self.calibrator is None:
            return {"total": self.fc.weight.new_zeros(())}
        return self.calibrator.regularization_losses()  # type: ignore[no-any-return]

    def get_regularization_losses(self) -> dict[str, torch.Tensor]:
        """Match the regularizer accessor used by the journal training loop."""
        return self.regularization_losses()

    def experiment_signature(self) -> dict[str, Any]:
        """Return immutable settings that must accompany every result row."""
        return {
            "preset": self.preset_name,
            "selected_stage": self.preset.selected_stage,
            "patch_size": self.preset.patch_size,
            "expected_tokens_128": self.preset.expected_tokens_128,
            "expected_tokens_160x196x160": self.preset.expected_tokens_160x196x160,
            "input_shape": self.input_shape,
            "original_feature_shape": self.original_feature_shape,
            "interaction": self.interaction,
            "table_variables": tuple(spec["name"] for spec in self.var_specs),
        }


def build_scale_table_ablation(
    preset: str = "layer4_pixel",
    interaction: str = "transformer_cross",
    **kwargs: Any,
) -> ScaleTableInteractionAblation3D:
    """Public factory for the scale and patch-table ablation model."""
    return ScaleTableInteractionAblation3D(
        preset=preset,
        interaction=interaction,
        **kwargs,
    )
