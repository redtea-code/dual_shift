"""
ViT-3D + DAFT (Dynamic Affine Feature Map Transform) Backbone.

Reference:
    Polsterl et al. "Combining 3D Image and Tabular Data via the
    Dynamic Affine Feature Map Transform." MICCAI 2021.

Architecture:
    MRI [B,1,D,H,W]                   tabular [B,txt_dim]
         |                                    |
    ViT Encoder → [B,N,C] tokens             |
         |                                    |
    DAFT per-channel ←───────────────────────┘
         |
    Mean Pool → [B,C]
         |
    Dropout → Linear → logits [B,num_classes]

DAFT: out = (1.0 + gamma) * features + beta
      (gamma, beta) = conditioning_MLP(tabular)
      where conditioning uses a bottleneck architecture.

Key differences from ViTFiLMBackbone:
    - DAFT uses deeper conditioning with bottleneck hidden dim
    - More stable zero-init for the conditioning network
    - Designed specifically for medical imaging + tabular fusion
"""

import torch
import torch.nn as nn
from .vit_film_backbone import ViTEncoder3D


# ═══════════════════════════════════════════════════════════
# Token-level DAFT (per-channel for ViT tokens)
# ═══════════════════════════════════════════════════════════

class TokenDAFTLayer(nn.Module):
    """Per-channel DAFT conditioning for ViT token sequences.

    Equivalent to DAFTBlock but operates on [B, N, C] tokens
    instead of [B, C, D, H, W] feature maps.

    conditioning: tabular → MLP(bottleneck) → (gamma, beta)
    out = (1.0 + gamma) * tokens + beta
    """

    def __init__(self, tabular_dim, feature_channels, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(feature_channels // 4, 16)

        self.conditioning = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels * 2),
        )
        # Zero-init second Linear → gamma≈0, beta≈0 → near identity
        nn.init.zeros_(self.conditioning[-1].weight)
        nn.init.zeros_(self.conditioning[-1].bias)
        self.C = feature_channels

    def forward(self, tokens, tabular):
        """
        Args:
            tokens:  [B, N, C] ViT token sequence
            tabular: [B, T] tabular features
        Returns:
            transformed: [B, N, C]
        """
        params = self.conditioning(tabular)  # [B, 2C]
        gamma = params[:, :self.C]           # [B, C]
        beta = params[:, self.C:]            # [B, C]

        # Broadcast over token dimension
        return (1.0 + gamma).unsqueeze(1) * tokens + beta.unsqueeze(1)


# ═══════════════════════════════════════════════════════════
# ViTDAFTBackbone
# ═══════════════════════════════════════════════════════════

class ViTDAFTBackbone(nn.Module):
    """ViT-3D encoder + DAFT tabular fusion backbone.

    Usage:
        backbone = ViTDAFTBackbone(txt_dim=9, num_classes=3)
        logits = backbone(mri_volume, tabular_data)
    """

    def __init__(self, txt_dim=9, num_classes=3,
                 img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 embed_dim=192, depth=6, num_heads=3,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., daft_hidden=None,
                 pretrained_weights=None, get_feature=False):
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes
        self.get_feature = get_feature

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

        # ── DAFT fusion ──
        self.daft = TokenDAFTLayer(txt_dim, embed_dim, hidden_dim=daft_hidden)

        # ── Classifier head ──
        self.dropout = nn.Dropout(0.3)
        if not get_feature:
            self.fc = nn.Linear(embed_dim, num_classes)

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
        print(f"ViTDAFTBackbone: loaded {len(loaded)} pretrained keys")

    def forward(self, x, txt=None):
        """
        Args:
            x:   [B, 1, D, H, W] 3D MRI volume
            txt: [B, txt_dim] tabular data (or None → skip DAFT)
        Returns:
            logits [B, num_classes] or features [B, embed_dim]
        """
        tokens = self.vit(x)                   # [B, N, C]

        if txt is not None:
            tokens = self.daft(tokens, txt)    # [B, N, C]

        pooled = tokens.mean(dim=1)            # [B, C]
        pooled = self.dropout(pooled)

        if self.get_feature:
            return pooled
        return self.fc(pooled)


# ═══════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════

def vit_tiny_daft(txt_dim=9, num_classes=3, pretrained_weights=None,
                  get_feature=False, **kwargs):
    """Tiny ViT + DAFT: embed_dim=192, depth=6, num_heads=3."""
    return ViTDAFTBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        img_size=kwargs.get('img_size', (160, 160, 96)),
        patch_size=kwargs.get('patch_size', (16, 16, 8)),
        embed_dim=192, depth=6, num_heads=3,
        mlp_ratio=4., drop_path_rate=0.0,
        pretrained_weights=pretrained_weights,
        get_feature=get_feature,
    )


def vit_small_daft(txt_dim=9, num_classes=3, pretrained_weights=None,
                   get_feature=False, **kwargs):
    """Small ViT + DAFT: embed_dim=384, depth=12, num_heads=6."""
    return ViTDAFTBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        img_size=kwargs.get('img_size', (160, 160, 96)),
        patch_size=kwargs.get('patch_size', (16, 16, 8)),
        embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4., drop_path_rate=0.1,
        pretrained_weights=pretrained_weights,
        get_feature=get_feature,
    )
