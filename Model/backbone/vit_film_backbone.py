"""
ViT-3D + FiLM Backbone.

Architecture:
    MRI [B,1,D,H,W]                   tabular [B,txt_dim]
         |                                    |
    ViT Encoder (no CLS) → [B,N,C] tokens    |
         |                                    |
    FiLM per-channel ←───────────────────────┘
         |
    Mean Pool → [B,C]
         |
    Dropout → Linear → logits [B,num_classes]

FiLM: out = (1 + gamma) * features + beta
      (gamma, beta) = MLP(txt)

Design: ViT tokens already represent spatial patches. We apply
per-channel FiLM modulation directly to the token sequence, then
mean-pool all tokens for classification — no CLS token needed.

Shared ViT components (trunc_normal_, DropPath, Mlp, Attention, Block,
PatchEmbed3D, ViTEncoder3D) are defined here and imported by other
ViT fusion backbones.
"""

import math
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# Shared ViT Components (adapted from vt_dino)
# ═══════════════════════════════════════════════════════════

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (from DINO)."""
    def _norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = _norm_cdf((a - mean) / std)
        u = _norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm and stochastic depth."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding (Conv3d projection)."""
    def __init__(self, img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1],
                          img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: [B, C, D, H, W] → [B, N, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ═══════════════════════════════════════════════════════════
# ViT Encoder (no CLS token, outputs token sequence)
# ═══════════════════════════════════════════════════════════

class ViTEncoder3D(nn.Module):
    """3D ViT encoder that outputs token sequence [B, N, C].

    No CLS token — designed for fusion backbones that operate on
    the full token sequence (FiLM, DAFT, Backdoor) and then pool.

    Args:
        img_size: input 3D volume shape (D, H, W)
        patch_size: patch dimensions (pD, pH, pW)
        in_chans: input channels (1 for MRI)
        embed_dim: token embedding dimension
        depth: number of transformer blocks
        num_heads: attention heads per block
        mlp_ratio: hidden dim ratio for MLP
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
    """

    def __init__(self, img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 in_chans=1, embed_dim=384, depth=6, num_heads=6,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Position embedding (no CLS token → pure patch positions)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: [B, C_in, D, H, W] → tokens: [B, N, embed_dim]"""
        x = self.patch_embed(x)          # [B, N, C]
        x = x + self.pos_embed           # add position
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ═══════════════════════════════════════════════════════════
# ViT + FiLM Fusion
# ═══════════════════════════════════════════════════════════

class TokenFiLMLayer(nn.Module):
    """Per-channel FiLM for ViT token sequences.

    Given tabular features, predicts per-channel scale (gamma)
    and shift (beta) for the entire token tensor.

    out = (1.0 + gamma) * tokens + beta
    """

    def __init__(self, txt_dim, embed_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(txt_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim * 2),
        )
        # Initialize: gamma near 1, beta near 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens, txt):
        """
        Args:
            tokens: [B, N, C] token sequence
            txt:    [B, txt_dim] tabular features
        Returns:
            modulated: [B, N, C]
        """
        params = self.net(txt)            # [B, 2C]
        gamma = params[:, :params.size(1) // 2]  # [B, C]
        beta = params[:, params.size(1) // 2:]   # [B, C]
        gamma = 1.0 + gamma              # init near 1
        # Broadcast over token dimension
        return gamma.unsqueeze(1) * tokens + beta.unsqueeze(1)


# ═══════════════════════════════════════════════════════════
# ViTFiLMBackbone
# ═══════════════════════════════════════════════════════════

class ViTFiLMBackbone(nn.Module):
    """ViT-3D encoder + FiLM tabular fusion backbone.

    Usage:
        backbone = ViTFiLMBackbone(txt_dim=9, num_classes=3)
        logits = backbone(mri_volume, tabular_data)
    """

    def __init__(self, txt_dim=9, num_classes=3,
                 img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 embed_dim=192, depth=6, num_heads=3,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., film_hidden=64,
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

        # ── FiLM fusion ──
        self.film = TokenFiLMLayer(txt_dim, embed_dim, hidden=film_hidden)

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
        print(f"ViTFiLMBackbone: loaded {len(loaded)} pretrained keys")

    def forward(self, x, txt=None):
        """
        Args:
            x:   [B, 1, D, H, W] 3D MRI volume
            txt: [B, txt_dim] tabular data (or None → skip FiLM)
        Returns:
            logits [B, num_classes] or features [B, embed_dim]
        """
        tokens = self.vit(x)                   # [B, N, C]

        if txt is not None:
            tokens = self.film(tokens, txt)    # [B, N, C]

        # Mean pool over tokens
        pooled = tokens.mean(dim=1)            # [B, C]
        pooled = self.dropout(pooled)

        if self.get_feature:
            return pooled
        return self.fc(pooled)


# ═══════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════

def vit_tiny_film(txt_dim=9, num_classes=3, pretrained_weights=None,
                  get_feature=False, **kwargs):
    """Tiny ViT + FiLM: embed_dim=192, depth=6, num_heads=3."""
    return ViTFiLMBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        img_size=kwargs.get('img_size', (160, 160, 96)),
        patch_size=kwargs.get('patch_size', (16, 16, 8)),
        embed_dim=192, depth=6, num_heads=3,
        mlp_ratio=4., drop_path_rate=0.0,
        pretrained_weights=pretrained_weights,
        get_feature=get_feature,
    )


def vit_small_film(txt_dim=9, num_classes=3, pretrained_weights=None,
                   get_feature=False, **kwargs):
    """Small ViT + FiLM: embed_dim=384, depth=12, num_heads=6."""
    return ViTFiLMBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        img_size=kwargs.get('img_size', (160, 160, 96)),
        patch_size=kwargs.get('patch_size', (16, 16, 8)),
        embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4., drop_path_rate=0.1,
        pretrained_weights=pretrained_weights,
        get_feature=get_feature,
    )
