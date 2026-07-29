
"""
ResNet3D + Patchwise Back-door Adjustment Backbone.

Design: After ResNet encoder extracts feature maps, we patchify the
spatial grid and apply back-door adjustment per patch using tabular
confounders. This removes confounding effects at a fine spatial
granularity before aggregation.

Architecture (default, use_class_head=False):
    MRI [B,1,D,H,W]                      Tabular [B,txt_dim]
         |                                      |
    ResNet3D (Stem+Layer1-4)              ConfounderEncoder → z [B,z_dim]
         |                                      |
    Patchify → [B,P,C]                          |
         |                                      |
    PatchwiseBackdoorBlock ←────────────────────┘
         |
    Unpatchify → AdaptiveAvgPool3d → Dropout → Linear → logits

Architecture (use_class_head=True):
    ... PatchwiseBackdoorBlock ...
         |
    ClassAttentionHead (cross-attn: K class queries attend to P patches)
         |
    → logits [B,num_classes] + attention weights [B,K,P]

Patchwise back-door: X'_p = X_p - g_p(z) · X_p
Each spatial position has its own adjustment gate g_p learned from z.

Interface aligned with ViTBackdoorBackbone:
    use_class_head, class_head_kwargs, get_feature,
    get_class_attention(), get_class_specific_importance(),
    patch_grid_size for backdoor spatial regularization.
"""

import torch
import torch.nn as nn

from Model.backbone.film_backbone import BasicBlock
from Model.causal import PatchwiseBackdoorBlock


# ═══════════════════════════════════════════════════════════
# Patchify / Unpatchify utilities
# ═══════════════════════════════════════════════════════════

def spatial_to_patches(x):
    """Reshape [B,C,D,H,W] → [B,P,C] treating each spatial pos as a patch."""
    B, C, D, H, W = x.shape
    P = D * H * W
    return x.permute(0, 2, 3, 4, 1).reshape(B, P, C)


def patches_to_spatial(x, D, H, W):
    """Reverse: [B,P,C] → [B,C,D,H,W]."""
    B, P, C = x.shape
    return x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)


def _conv3d_spatial_out(size, kernel, stride, padding):
    """Match PyTorch Conv3d output spatial size (single dim)."""
    return (size + 2 * padding - kernel) // stride + 1


def _maxpool3d_spatial_out(size, kernel=3, stride=2, padding=1):
    return (size + 2 * padding - kernel) // stride + 1


def compute_resnet_patch_grid(input_shape=(193, 229, 193)):
    """Compute layer4 feature map grid by simulating this backbone's ResNet3D.

    Do NOT use stem_size // 8: padding makes per-layer sizes differ from
    integer division (e.g. H=196 → 7, not 6).
    """
    d, h, w = input_shape

    d = _conv3d_spatial_out(d, kernel=7, stride=2, padding=3)
    h = _conv3d_spatial_out(h, kernel=7, stride=2, padding=3)
    w = _conv3d_spatial_out(w, kernel=7, stride=2, padding=3)

    d = _maxpool3d_spatial_out(d)
    h = _maxpool3d_spatial_out(h)
    w = _maxpool3d_spatial_out(w)

    # layer1 stride=1 (no change); layer2/3/4 first block stride=2
    for _ in range(3):
        d = _conv3d_spatial_out(d, kernel=3, stride=2, padding=1)
        h = _conv3d_spatial_out(h, kernel=3, stride=2, padding=1)
        w = _conv3d_spatial_out(w, kernel=3, stride=2, padding=1)

    patch_grid = (d, h, w)
    patch_num = d * h * w
    return patch_grid, patch_num


# ═══════════════════════════════════════════════════════════
# Confounder Encoder
# ═══════════════════════════════════════════════════════════

class ConfounderEncoder(nn.Module):
    """Encode tabular features into a confounder embedding z."""

    def __init__(self, txt_dim, z_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, txt):
        """txt: [B, txt_dim] → z: [B, z_dim]"""
        return self.net(txt)


# ═══════════════════════════════════════════════════════════
# Class-Aware Attention Aggregation Head
# ═══════════════════════════════════════════════════════════

class ClassAttentionHead(nn.Module):
    """Class-aware attention aggregation head.

    Replaces global pooling with learnable class queries that attend to
    patch tokens, producing class-specific evidence aggregation.

    Args:
        embed_dim: Patch feature dimension (C)
        num_classes: Number of output classes (K)
        num_heads: Number of attention heads
        dropout: Dropout rate in attention and MLP
    """

    def __init__(self, embed_dim, num_classes, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.class_queries = nn.Parameter(
            torch.randn(1, num_classes, embed_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens):
        """
        Args:
            tokens: [B, P, C] adjusted patch features

        Returns:
            logits:       [B, num_classes]
            attn_weights: [B, num_classes, P]
        """
        B = tokens.size(0)
        q = self.class_queries.expand(B, -1, -1)

        class_tokens, attn_weights = self.cross_attn(
            query=q,
            key=tokens,
            value=tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        class_tokens = self.norm(class_tokens + q)
        logits = self.mlp(class_tokens).squeeze(-1)
        return logits, attn_weights


# ═══════════════════════════════════════════════════════════
# Main Backbone
# ═══════════════════════════════════════════════════════════

class ResNetBackdoorBackbone(nn.Module):
    """ResNet3D encoder + patch-level back-door adjustment.

    Usage:
        backbone = ResNetBackdoorBackbone(txt_dim=9, num_classes=3)
        logits = backbone(mri_volume, tabular_data)
    """

    def __init__(self, txt_dim=9, num_classes=3,
                 block=BasicBlock, layers=(2, 2, 2, 2),
                 z_dim=128, pretrained_weights=None,
                 patch_num=None,
                 input_shape=(160, 160, 96),
                 backdoor_kwargs=None,
                 get_feature=False,
                 use_class_head=False,
                 class_head_kwargs=None):
        """
        Args:
            use_class_head: If True, replace global pool + Linear with
                            ClassAttentionHead on patch tokens.
            class_head_kwargs: Dict of kwargs for ClassAttentionHead
                              (num_heads, dropout). Only used if
                              use_class_head=True.
            get_feature: If True, return feature vector instead of logits.
        """
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes
        self.get_feature = get_feature
        self.use_class_head = use_class_head

        # ── ResNet stem ──
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ── ResNet layers ──
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        final_planes = 512 * block.expansion

        # ── Confounder encoder ──
        self.confounder_encoder = ConfounderEncoder(
            txt_dim=txt_dim, z_dim=z_dim, hidden_dim=64,
        )

        # ── Patch grid / count (estimate; synced on first forward) ──
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            input_shape = tuple(int(v) for v in input_shape)
        self.input_shape = input_shape
        self.patch_grid, computed_patch_num = compute_resnet_patch_grid(input_shape)
        if patch_num is None:
            patch_num = computed_patch_num
        self.num_patches = patch_num
        self.z_dim = z_dim
        self._backdoor_user_kwargs = dict(backdoor_kwargs) if backdoor_kwargs else {}
        self.backdoor = None
        self._build_backdoor(self.patch_grid, self.num_patches)

        # ── Classifier head ──
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.1)
        if not get_feature:
            if use_class_head:
                _ch_kwargs = dict(num_heads=4, dropout=0.1)
                if class_head_kwargs:
                    _ch_kwargs.update(class_head_kwargs)
                self.class_head = ClassAttentionHead(
                    embed_dim=final_planes,
                    num_classes=num_classes,
                    **_ch_kwargs,
                )
                self.fc = None
            else:
                self.fc = nn.Linear(final_planes, num_classes)
                self.class_head = None
        else:
            self.fc = None
            self.class_head = None

        # ── Metadata ──
        self.final_planes = final_planes
        self.embed_dim = final_planes
        self._spatial_shape = None
        self._last_gamma = None
        self._last_gamma_spatial_shape = None
        self._last_class_attn = None
        self.shuffle_tabular = bool(self._backdoor_user_kwargs.get('shuffle_tabular', False))
        self.shuffle_seed = int(self._backdoor_user_kwargs.get('shuffle_seed', 0))
        import numpy as np
        self._shuffle_rng = np.random.RandomState(self.shuffle_seed)

        self._init_weights()

        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

    def _build_backdoor(self, patch_grid, patch_num):
        """Create / replace PatchwiseBackdoorBlock for a spatial grid."""
        _bd_kwargs = dict(
            return_gamma=True,
            patch_grid_size=patch_grid,
        )
        user_bd = dict(self._backdoor_user_kwargs)
        user_bd.pop('patch_grid_size', None)
        user_bd.pop('shuffle_tabular', None)
        user_bd.pop('shuffle_seed', None)
        _bd_kwargs.update(user_bd)
        self.backdoor = PatchwiseBackdoorBlock(
            self.z_dim, patch_dim=patch_num, **_bd_kwargs,
        )
        self.patch_grid = patch_grid
        self.num_patches = patch_num

    def _sync_backdoor_to_feature_map(self, spatial_shape):
        """Verify backdoor patch_dim matches actual layer4 feature map size."""
        grid = tuple(spatial_shape)
        patch_num = grid[0] * grid[1] * grid[2]
        if self.patch_grid == grid and self.num_patches == patch_num:
            return

        est_grid, est_p = compute_resnet_patch_grid(self.input_shape)
        raise RuntimeError(
            f"ResNetBackdoorBackbone: layer4 feature map grid {grid} (P={patch_num}) "
            f"does not match backdoor patch_grid {self.patch_grid} (P={self.num_patches}). "
            f"Ensure config img_sz / model input_shape matches training data spatial size. "
            f"Current input_shape={self.input_shape} estimates grid {est_grid} (P={est_p})."
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_pretrained(self, path):
        if path is True:
            path = r'D:/cyh/Causal Infer/weights/resnet_18_23dataset.pth'
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        own = self.state_dict()
        loaded = {}
        for k, v in state.items():
            for prefix in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
                if prefix in k:
                    loaded[k] = v
                    break
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        print(f"ResNetBackdoorBackbone: loaded {len(loaded)} pretrained keys")

    def extract_features(self, x):
        """Extract feature map before backdoor adjustment.

        Returns:
            feat_map: [B, C, D, H, W] feature map from layer4
        """
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        return f

    def forward(self, x, txt=None):
        """
        Args:
            x: [B, 1, D, H, W] 3D MRI volume
            txt: [B, txt_dim] tabular data (if None, skip backdoor)

        Returns:
            logits [B, num_classes] or features (see get_feature)
        """
        feat = self.extract_features(x)
        B, C, D, H, W = feat.shape
        self._spatial_shape = (D, H, W)

        patches = spatial_to_patches(feat)

        if txt is not None:
            self._sync_backdoor_to_feature_map((D, H, W))
            txt_use = txt
            if self.shuffle_tabular and txt is not None and txt.size(0) > 1:
                perm = torch.from_numpy(
                    self._shuffle_rng.permutation(txt.size(0)),
                ).long().to(txt.device)
                txt_use = txt[perm]
            z = self.confounder_encoder(txt_use)
            patches, self._last_gamma = self.backdoor(patches, z)
            self._last_gamma_spatial_shape = (B, D * H * W, D, H, W)
        else:
            self._last_gamma = None
            self._last_gamma_spatial_shape = None

        if self.get_feature:
            pooled = patches.mean(dim=2)
            pooled = self.dropout(pooled)
            return pooled

        if self.use_class_head and self.class_head is not None:
            logits, self._last_class_attn = self.class_head(patches)
            return logits

        feat_adj = patches_to_spatial(patches, D, H, W)
        pooled = self.pool(feat_adj).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

    def get_backdoor_regularization_loss(self):
        """计算 backdoor 门控 gamma 的正则化损失（单值，向后兼容）。"""
        if self._last_gamma is None:
            return None
        return self._last_gamma.abs().mean()

    def get_regularization_losses(self):
        """计算所有 backdoor 正则化损失（字典形式）。"""
        if self._last_gamma is None:
            return {}

        losses = {}

        if (hasattr(self.backdoor, '_gamma_pre_dropout')
                and self.backdoor._gamma_pre_dropout is not None):
            losses['backdoor_sparsity'] = self.backdoor._gamma_pre_dropout.abs().mean()

        if (hasattr(self.backdoor, 'last_reg_loss')
                and self.backdoor.last_reg_loss is not None
                and self.backdoor.last_reg_loss.abs().item() > 0):
            losses['backdoor_smoothness'] = self.backdoor.last_reg_loss
        elif self._last_gamma_spatial_shape is not None:
            B, P, D, H, W = self._last_gamma_spatial_shape
            gamma = self._last_gamma
            try:
                gamma_spatial = gamma.reshape(B, D, H, W)
                tv_d = (gamma_spatial[:, 1:, :, :] - gamma_spatial[:, :-1, :, :]).abs().mean()
                tv_h = (gamma_spatial[:, :, 1:, :] - gamma_spatial[:, :, :-1, :]).abs().mean()
                tv_w = (gamma_spatial[:, :, :, 1:] - gamma_spatial[:, :, :, :-1]).abs().mean()
                losses['backdoor_smoothness'] = (tv_d + tv_h + tv_w) / 3.0
            except RuntimeError:
                pass

        return losses

    def get_gamma(self):
        """返回最近一次 forward 的 gamma 值（供可视化使用）。"""
        if self._last_gamma is None:
            return None, None
        if self._last_gamma_spatial_shape is not None:
            _, _, D, H, W = self._last_gamma_spatial_shape
            return self._last_gamma, (D, H, W)
        return self._last_gamma, None

    def get_class_attention(self):
        """返回最近一次 forward 的 class attention 权重。

        仅在 use_class_head=True 时有值。

        Returns:
            class_attn: (B, num_classes, P) tensor，若未启用则返回 None
        """
        return self._last_class_attn

    def get_class_specific_importance(self):
        """结合 patch-wise gate gamma 和 class attention 权重。

        Returns:
            importance: (B, num_classes, P) tensor
                        若 gamma 或 class_attn 为 None 则返回 None
        """
        gamma = self._last_gamma
        attn = self._last_class_attn
        if gamma is None or attn is None:
            return None
        return gamma.unsqueeze(1) * attn


# ═══════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════

def _resolve_backdoor_factory_kwargs(kwargs):
    """Map shared factory kwargs (img_size, feature) to backbone args."""
    resolved = {}
    if 'input_shape' in kwargs:
        resolved['input_shape'] = kwargs['input_shape']
    elif 'img_size' in kwargs:
        resolved['input_shape'] = kwargs['img_size']
    if 'patch_num' in kwargs:
        resolved['patch_num'] = kwargs['patch_num']

    feature = kwargs.get('feature')
    get_feature = kwargs.get('get_feature')
    if get_feature is not None:
        resolved['get_feature'] = get_feature
    elif feature:
        resolved['get_feature'] = True
    return resolved


def resnet18_backdoor(txt_dim=9, num_classes=3, z_dim=128,
                      pretrained_weights=None, feature=False,
                      get_feature=None, backdoor_kwargs=None,
                      use_class_head=False, class_head_kwargs=None,
                      **kwargs):
    """ResNet-18 + Patchwise Backdoor backbone."""
    factory_kwargs = _resolve_backdoor_factory_kwargs(kwargs)
    if get_feature is not None:
        factory_kwargs['get_feature'] = get_feature
    elif feature:
        factory_kwargs['get_feature'] = True
    return ResNetBackdoorBackbone(
        txt_dim=txt_dim, num_classes=num_classes, z_dim=z_dim,
        block=BasicBlock, layers=(2, 2, 2, 2),
        pretrained_weights=pretrained_weights,
        backdoor_kwargs=backdoor_kwargs,
        use_class_head=use_class_head,
        class_head_kwargs=class_head_kwargs,
        **factory_kwargs,
    )


def resnet34_backdoor(txt_dim=9, num_classes=3, z_dim=128,
                      pretrained_weights=None, feature=False,
                      get_feature=None, backdoor_kwargs=None,
                      use_class_head=False, class_head_kwargs=None,
                      **kwargs):
    """ResNet-34 + Patchwise Backdoor backbone."""
    factory_kwargs = _resolve_backdoor_factory_kwargs(kwargs)
    if get_feature is not None:
        factory_kwargs['get_feature'] = get_feature
    elif feature:
        factory_kwargs['get_feature'] = True
    return ResNetBackdoorBackbone(
        txt_dim=txt_dim, num_classes=num_classes, z_dim=z_dim,
        block=BasicBlock, layers=(3, 4, 6, 3),
        pretrained_weights=pretrained_weights,
        backdoor_kwargs=backdoor_kwargs,
        use_class_head=use_class_head,
        class_head_kwargs=class_head_kwargs,
        **factory_kwargs,
    )


def resnet10_backdoor(txt_dim=9, num_classes=3, z_dim=128,
                      pretrained_weights=None, feature=False,
                      get_feature=None, backdoor_kwargs=None,
                      use_class_head=False, class_head_kwargs=None,
                      **kwargs):
    """ResNet-10 + Patchwise Backdoor backbone (lightweight)."""
    factory_kwargs = _resolve_backdoor_factory_kwargs(kwargs)
    if get_feature is not None:
        factory_kwargs['get_feature'] = get_feature
    elif feature:
        factory_kwargs['get_feature'] = True
    return ResNetBackdoorBackbone(
        txt_dim=txt_dim, num_classes=num_classes, z_dim=z_dim,
        block=BasicBlock, layers=(1, 1, 1, 1),
        pretrained_weights=pretrained_weights,
        backdoor_kwargs=backdoor_kwargs,
        use_class_head=use_class_head,
        class_head_kwargs=class_head_kwargs,
        **factory_kwargs,
    )
