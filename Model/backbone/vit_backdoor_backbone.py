"""
ViT-3D + Patchwise Backdoor Adjustment Backbone.

Architecture (default, use_class_head=False):
    MRI [B,1,D,H,W]                   tabular [B,txt_dim]
         |                                    |
    ViT Encoder → [B,N,C] tokens    ConfounderEncoder → z [B,z_dim]
         |                                    |
    PatchwiseBackdoor ←──────────────────────┘
         |
    Mean Pool → [B,C]
         |
    Dropout → Linear → logits [B,num_classes]

Architecture (use_class_head=True):
    MRI [B,1,D,H,W]                   tabular [B,txt_dim]
         |                                    |
    ViT Encoder → [B,N,C] tokens    ConfounderEncoder → z [B,z_dim]
         |                                    |
    PatchwiseBackdoor ←──────────────────────┘
         |
    ClassAttentionHead (cross-attn: K class queries attend to N patches)
         |
    → logits [B,num_classes] + attention weights [B,K,N]

Backdoor: X'_p = X_p - g_p(z) * X_p
Each ViT token (patch) has its own adjustment gate g_p learned from
confounder embedding z. This removes confounding effects at a
fine spatial (patch) granularity before aggregation.

Equivalent to: tokens_adj = (1 - gate(z)) * tokens for each token.

Key differences from ViTFiLMBackbone / ViTDAFTBackbone:
    - Backdoor subtracts confounder projection (causal adjustment)
    - Per-token (patch-level) granularity vs per-channel
    - Uses existing PatchwiseBackdoorBlock from Model.causal
    - Optional ClassAttentionHead replaces mean pool for class-aware
      aggregation, preserving patch→class interpretability
"""

import torch
import torch.nn as nn
from .vit_film_backbone import ViTEncoder3D
from .backdoor_backbone import ConfounderEncoder, ClassAttentionHead

class ViTBackdoorBackbone(nn.Module):
    """ViT-3D encoder + patch-level back-door adjustment.

    ViT tokens are treated as spatial patches. Each token gets its
    own confounder gate, implementing fine-grained causal adjustment.

    Usage:
        backbone = ViTBackdoorBackbone(txt_dim=9, num_classes=3)
        logits = backbone(mri_volume, tabular_data)
    """

    def __init__(self, txt_dim=9, num_classes=3,
                 img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 embed_dim=192, depth=6, num_heads=3,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., z_dim=128,
                 pretrained_weights=None, get_feature=False,
                 backdoor_kwargs=None,
                 use_class_head=False,
                 class_head_kwargs=None):
        """
        Args:
            use_class_head: If True, replace mean pool + Linear with
                            ClassAttentionHead for class-aware aggregation.
                            Preserves patch→class interpretability and
                            prevents background patch dilution.
                            Default False for backward compatibility.
            class_head_kwargs: Dict of kwargs for ClassAttentionHead
                              (num_heads, dropout). Only used if
                              use_class_head=True.
        """
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes
        self.get_feature = get_feature
        self.use_class_head = use_class_head

        # ── ViT Encoder ──
        self.vit = ViTEncoder3D(
            img_size=img_size, patch_size=patch_size, in_chans=1,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = embed_dim
        self.final_planes = embed_dim
        self.num_patches = self.vit.num_patches

        # ── Confounder encoder ──
        self.confounder_encoder = ConfounderEncoder(
            txt_dim=txt_dim, z_dim=z_dim, hidden_dim=64)

        # ── Backdoor gate per token ──
        # We reuse the PatchwiseBackdoorBlock from Model.causal.
        # patch_dim = number of ViT tokens for per-token gating.
        from Model.causal import PatchwiseBackdoorBlock

        # Compute token grid for spatial smoothness
        self.token_grid = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        _bd_kwargs = dict(
            return_gamma=True,
            patch_grid_size=self.token_grid,
        )
        if backdoor_kwargs:
            _bd_kwargs.update(backdoor_kwargs)
        self.backdoor = PatchwiseBackdoorBlock(z_dim, patch_dim=self.num_patches, **_bd_kwargs)

        # ── Classifier head ──
        self.dropout = nn.Dropout(0.1)
        if not get_feature:
            if use_class_head:
                _ch_kwargs = dict(num_heads=4, dropout=0.1)
                if class_head_kwargs:
                    _ch_kwargs.update(class_head_kwargs)
                self.class_head = ClassAttentionHead(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    **_ch_kwargs,
                )
                self.fc = None  # not used; kept for attr compatibility
            else:
                self.fc = nn.Linear(self.num_patches, num_classes)
                self.class_head = None
        else:
            self.fc = None
            self.class_head = None

        # ── Attention storage (for visualization) ──
        self._last_class_attn = None

        if pretrained_weights:
            self._load_pretrained(pretrained_weights)

    def _load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        own = self.state_dict()
        loaded = {k: v for k, v in state.items()
                  if k in own and v.shape == own[k].shape}
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        print(f"ViTBackdoorBackbone: loaded {len(loaded)} pretrained keys")

    def forward(self, x, txt=None):
        """
        Args:
            x:   [B, 1, D, H, W] 3D MRI volume
            txt: [B, txt_dim] tabular data (or None → skip backdoor)
        Returns:
            logits [B, num_classes] or features [B, embed_dim]
        """
        tokens = self.vit(x)                   # [B, N, C]

        if txt is not None:
            z = self.confounder_encoder(txt)   # [B, z_dim]
            # PatchwiseBackdoorBlock expects tokens as [B, P, C]
            tokens, self._last_gamma = self.backdoor(tokens, z)  # [B,N,C], [B,N]
        else:
            self._last_gamma = None

        if self.get_feature:
            pooled = tokens.mean(dim=2)            # [B, C]
            pooled = self.dropout(pooled)
            return pooled

        if self.use_class_head and self.class_head is not None:
            logits, self._last_class_attn = self.class_head(tokens)
            return logits
        else:
            pooled = tokens.mean(dim=2)            # [B, C]
            pooled = self.dropout(pooled)
            return self.fc(pooled)

    def get_backdoor_regularization_loss(self):
        """计算 backdoor 门控 gamma 的正则化损失（单值，向后兼容）。

        推荐使用 get_regularization_losses() 获取更细粒度的损失字典。

        返回 None 表示当前 batch 无 backdoor 路径（txt=None）。
        """
        if self._last_gamma is None:
            return None
        return self._last_gamma.abs().mean()

    def get_regularization_losses(self):
        """计算所有 backdoor 正则化损失（字典形式）。

        在 forward() 之后调用。返回：
        - backdoor_sparsity (L1):  |gamma_pre_dropout|.mean()，使用 pre-dropout gamma
        - backdoor_smoothness:     模块内置空间平滑损失（若启用）

        Returns:
            dict: {loss_name: scalar_tensor}，若无 backdoor 路径则返回空 dict
        """
        if self._last_gamma is None:
            return {}

        losses = {}

        # ── L1 Sparsity: 使用 pre-dropout gamma ──
        if (hasattr(self.backdoor, '_gamma_pre_dropout')
                and self.backdoor._gamma_pre_dropout is not None):
            losses['backdoor_sparsity'] = self.backdoor._gamma_pre_dropout.abs().mean()

        # ── 空间平滑（来自模块内置计算）──
        if (hasattr(self.backdoor, 'last_reg_loss')
                and self.backdoor.last_reg_loss is not None
                and self.backdoor.last_reg_loss.abs().item() > 0):
            losses['backdoor_smoothness'] = self.backdoor.last_reg_loss

        return losses

    def get_gamma(self):
        """返回最近一次 forward 的 gamma 值（供可视化使用）。

        Returns:
            gamma: (B, P) tensor，若 None 则无 backdoor 路径
            spatial_shape: (D, H, W) tuple，ViT 下通常为 None
        """
        if self._last_gamma is None:
            return None, None
        return self._last_gamma, None

    def get_class_attention(self):
        """返回最近一次 forward 的 class attention 权重。

        仅在 use_class_head=True 时有值，用于可视化 class-patch 对应关系。

        可构造类别相关解释图：
            importance_k = gamma[b, :] * class_attn[b, k, :]

        Returns:
            class_attn: (B, num_classes, N) tensor，若未启用则返回 None
        """
        return self._last_class_attn

    def get_class_specific_importance(self):
        """计算 class-specific adjusted importance.

        结合 patch-wise gate gamma 和 class attention 权重，
        得到每个类别对每个 patch 的综合重要性。

        Returns:
            importance: (B, num_classes, N) tensor
                        若 gamma 或 class_attn 为 None 则返回 None
        """
        gamma = self._last_gamma
        attn = self._last_class_attn
        if gamma is None or attn is None:
            return None
        # gamma: [B, N] → [B, 1, N], attn: [B, K, N]
        return gamma.unsqueeze(1) * attn  # [B, K, N]


# ═══════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════

def vit_tiny_backdoor(txt_dim=9, num_classes=3, z_dim=128,
                      pretrained_weights=None, get_feature=False,
                      backdoor_kwargs=None,
                      use_class_head=False,
                      class_head_kwargs=None,
                      **kwargs):
    """Tiny ViT + Backdoor: embed_dim=192, depth=6, num_heads=3."""
    return ViTBackdoorBackbone(
        txt_dim=txt_dim, num_classes=num_classes, z_dim=z_dim,
        img_size=kwargs.get('img_size', (160, 196, 160)),
        patch_size=kwargs.get('patch_size', (16, 16, 8)),
        embed_dim=192, depth=6, num_heads=3,
        mlp_ratio=4., drop_path_rate=0.0,
        pretrained_weights=pretrained_weights,
        get_feature=get_feature,
        backdoor_kwargs=backdoor_kwargs,
        use_class_head=use_class_head,
        class_head_kwargs=class_head_kwargs,
    )


def vit_small_backdoor(txt_dim=9, num_classes=3, z_dim=128,
                       pretrained_weights=None, get_feature=False,
                       backdoor_kwargs=None,
                       use_class_head=False,
                       class_head_kwargs=None,
                       **kwargs):
    """Small ViT + Backdoor: embed_dim=384, depth=12, num_heads=6."""
    return ViTBackdoorBackbone(
        txt_dim=txt_dim, num_classes=num_classes, z_dim=z_dim,
        img_size=kwargs.get('img_size', (160, 196, 160)),
        patch_size=kwargs.get('patch_size', (16, 16, 8)),
        embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4., drop_path_rate=0.1,
        pretrained_weights=pretrained_weights,
        get_feature=get_feature,
        backdoor_kwargs=backdoor_kwargs,
        use_class_head=use_class_head,
        class_head_kwargs=class_head_kwargs,
    )
