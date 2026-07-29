"""
3D Vision Transformer (ViT) Encoder for Medical Image Classification
Replaces ResNet18 backbone with ViT + conf_mask3 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# 3D Patch Embedding
# ============================================================================

class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=(160, 160, 96), patch_size=(16, 16, 8), in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # (B, N, embed_dim)
        x = self.norm(x)
        return x


# ============================================================================
# Transformer Components
# ============================================================================

class Attention3D(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-Norm"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# ViT Backbone (outputs spatial feature map for conf_mask3)
# ============================================================================

class ViTBackbone(nn.Module):
    """
    3D Vision Transformer Backbone
    Outputs: (B, embed_dim, D', H', W') spatial feature map
    """
    def __init__(self, img_size=(160, 160, 96), patch_size=(16, 16, 8), in_chans=1,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., 
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, 
                                        in_chans=in_chans, embed_dim=embed_dim)
        self.grid_size = self.patch_embed.grid_size  # (D', H', W')
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding (no cls token - we use mean pooling)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)  # (B, N, embed_dim)
        
        # Reshape back to spatial feature map (B, embed_dim, D', H', W')
        D, H, W = self.grid_size
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        return x


# ============================================================================
# conf_mask3 equivalent for ViT features
# ============================================================================

def patchify3D(E, patch_size):
    """
    E: [B, C, D, H, W]
    return: [B, N, C, pD, pH, pW]
    """
    b, c, d, h, w = E.size()
    patch_d, patch_h, patch_w = patch_size
    
    patches = E.unfold(2, patch_d, patch_d).unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    patches = patches.view(b, -1, c, patch_d, patch_h, patch_w)
    return patches


def mask_one_patch(patch, p):
    """Mask out patch p"""
    masked = patch.clone()
    masked[:, p] = 0.0
    return masked


class ResidualBlock3D(nn.Module):
    """Simple 3D residual block"""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ViTConfMask3(nn.Module):
    """
    conf_mask3 equivalent for ViT features
    Uses leave-one-out strategy (like original conf_mask3)
    """
    def __init__(self, patch_size, patch_num, in_dim=256):
        super().__init__()
        self.patch_size = patch_size
        
        # Causal score network (leave-one-out)
        self.causal_score = nn.Sequential(
            ResidualBlock3D(in_dim, in_dim, stride=1),
            nn.AdaptiveAvgPool3d(1)
        )
        self.score_head = nn.Linear(in_dim, 1)
        self.patch_head = nn.Linear(patch_num, 1)
    
    def forward(self, x, **kwargs):
        """
        x: (B, C, D, H, W) - ViT spatial feature map
        Returns: score, conf_patch, causal_patch
        """
        patch = patchify3D(x, self.patch_size)
        B, P, C, D, H, W = patch.shape
        scores = []
        
        # Leave-one-out scoring
        for p in range(P):
            masked_patch = mask_one_patch(patch, p)
            masked_feat = self.causal_score(masked_patch.reshape(B * P, C, D, H, W)).view(B, P, -1)
            delta = self.score_head(masked_feat).reshape(B, P)
            delta = self.patch_head(delta)
            scores.append(delta)
        
        score = torch.cat(scores, dim=-1)  # (B, P)
        score_mask = torch.sigmoid(score)
        
        score_mask = score_mask.view(B, P, 1, 1, 1, 1)
        conf_patch = patch * score_mask
        causal_patch = patch * (1.0 - score_mask)
        
        return score, conf_patch, causal_patch


class ViTConfMask6(nn.Module):
    """
    conf_mask equivalent for ViT features
    Uses learnable gating (conf_mask6 style) for efficiency
    """

    def __init__(self, patch_size, patch_num, in_dim=256):
        super().__init__()
        self.patch_size = patch_size

        # Score network: processes all patches at once
        self.causal_score = nn.Sequential(
            ResidualBlock3D(in_dim, in_dim, stride=1),
            nn.AdaptiveAvgPool3d(1)
        )
        self.score_head = nn.Linear(in_dim, 1)

    def forward(self, x, **kwargs):
        """
        x: (B, C, D, H, W) - ViT spatial feature map
        Returns: score, conf_patch, causal_patch
        """
        patch = patchify3D(x, self.patch_size)
        B, P, C, D, H, W = patch.shape

        # Process all patches: (B*P, C, D, H, W) -> (B*P, in_dim) -> (B, P, in_dim)
        feat = self.causal_score(patch.reshape(B * P, C, D, H, W)).view(B, P, -1)

        # Score per patch: (B, P, in_dim) -> (B, P, 1) -> (B, P)
        score = torch.sigmoid(self.score_head(feat)).squeeze(-1)  # (B, P)

        score_mask = score.view(B, P, 1, 1, 1, 1)
        conf_patch = patch * score_mask
        causal_patch = patch * (1.0 - score_mask)

        return score, conf_patch, causal_patch


# ============================================================================
# Feature Extractor (after conf_mask3)
# ============================================================================

class ViTFeatureExtractor(nn.Module):
    """Feature extractor after conf_mask3 (equivalent to MediatorExtractor2)"""
    def __init__(self, in_dim=256, out_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock3D(in_dim, out_dim, stride=1),
            ResidualBlock3D(out_dim, out_dim, stride=1),
            nn.AdaptiveAvgPool3d(1)
        )
    
    def forward(self, x):
        # x: (B*P, C, D, H, W)
        return self.layers(x)


# ============================================================================
# VisualEncoderVIT (main interface)
# ============================================================================

class VisualEncoderVIT(nn.Module):
    """
    ViT-based Visual Encoder
    Architecture: ViT Backbone -> conf_mask3 -> Feature Extractor -> FC
    Compatible with original VisualEncoder interface
    """
    def __init__(self, dim=256, img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 vit_patch_size=(8, 8, 4),  # ViT patch size (downsample factor)
                 depth=6, num_heads=8, mlp_ratio=4., drop_rate=0.1, num_classes=3):
        super().__init__()
        
        # ViT backbone (downsample to spatial feature map)
        self.vit_backbone = ViTBackbone(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_chans=1,
            embed_dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate
        )
        
        # Calculate patch numbers
        self.vit_grid_size = (
            img_size[0] // vit_patch_size[0],
            img_size[1] // vit_patch_size[1],
            img_size[2] // vit_patch_size[2]
        )
        patch_num = int((self.vit_grid_size[0] // patch_size[0]) * 
                        (self.vit_grid_size[1] // patch_size[1]) * 
                        (self.vit_grid_size[2] // patch_size[2]))
        
        # conf_mask3 on ViT features
        self.mask_attention = ViTConfMask6(
            patch_size=patch_size,
            patch_num=patch_num,
            in_dim=dim
        )
        
        # Feature extractors (conf and causal branches)
        feat_dim = 256  # 减小特征维度
        self.extractor2 = ViTFeatureExtractor(in_dim=dim, out_dim=feat_dim)
        self.extractor3 = ViTFeatureExtractor(in_dim=dim, out_dim=feat_dim)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # Output heads
        self.causal_fc = nn.Sequential(
            nn.Linear(patch_num * feat_dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim)
        )
        self.conf_fc = nn.Sequential(
            nn.Linear(patch_num * feat_dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x, **kwargs):
        """
        x: (B, 1, D, H, W)
        Returns: score, conf_out, causal_out
        """
        # ViT backbone -> spatial feature map
        E = self.vit_backbone(x)  # (B, dim, D', H', W')
        
        # conf_mask3
        score, conf_patch, causal_patch = self.mask_attention(E)
        
        B, P, C, D, H, W = conf_patch.shape
        
        # Feature extraction
        causal_patch = self.extractor2(causal_patch.reshape(B * P, C, D, H, W))
        conf_patch = self.extractor3(conf_patch.reshape(B * P, C, D, H, W))
        
        # Pool and reshape
        feat_dim = 256
        causal_feat = self.pool(causal_patch).reshape(B, P * feat_dim)
        conf_feat = self.pool(conf_patch).reshape(B, P * feat_dim)
        
        # Output
        causal_out = self.causal_fc(causal_feat)
        conf_out = self.conf_fc(conf_feat)
        conf_out = F.log_softmax(conf_out, dim=1)
        
        return score, conf_out, causal_out


# ============================================================================
# ADPC6_2_VIT (complete model)
# ============================================================================

class ADPC6_2_VIT(nn.Module):
    """ADPC6_2 with ViT backbone"""
    def __init__(self, txt_dim=9, dim=64, num_classes=3,
                 img_size=(160, 160, 96), 
                 vit_patch_size=(16, 16, 8),      # -> (10, 10, 12) = 1200 patches
                 conf_mask_patch_size=(5, 5, 6),  # -> (2, 2, 2) = 8 patches
                 vit_depth=2, vit_num_heads=2, vit_mlp_ratio=2., vit_dropout=0.1):
        super().__init__()
        
        self.visual_encoder = VisualEncoderVIT(
            dim=dim,
            img_size=img_size,
            patch_size=conf_mask_patch_size,
            vit_patch_size=vit_patch_size,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            drop_rate=vit_dropout,
            num_classes=num_classes
        )
        
        self.textual_encoder = TableEncoder(txt_dim, dim)
        
        self.backdoor = ResidualBackdoorBlock(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.active = nn.ReLU()
        self.classifier = nn.Linear(dim, num_classes)
    
    def forward(self, image, text_tokens):
        score, f_visual_conf, f_visual_causal = self.visual_encoder(image)
        f_text_causal, f_text_conf = self.textual_encoder(text_tokens)
        F_do = self.backdoor(f_visual_causal, f_text_causal)
        out = self.active(self.proj(F_do))
        out = self.classifier(out)
        return score, f_visual_conf, f_text_conf, out


# ============================================================================
# Helper modules (inline to avoid circular imports)
# ============================================================================

class TableEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, num_classes=3):
        super().__init__()
        self.causal_encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * out_dim), nn.ReLU(), nn.Linear(2 * out_dim, out_dim)
        )
        self.conf_encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * out_dim), nn.ReLU(), nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(), nn.Linear(out_dim, num_classes)
        )
    def forward(self, x):
        return self.causal_encoder(x), self.conf_encoder(x)


class ResidualBackdoorBlock(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: hidden_dim = x_dim
        self.z_to_x = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, x_dim)
        )
        self.norm = nn.LayerNorm(x_dim)
    def forward(self, x, z):
        x_hat = self.z_to_x(z)
        x_adj = x - x_hat
        x_adj = self.norm(x_adj + x)
        return x_adj


if __name__ == '__main__':
    batch_size = 2
    img = torch.rand(batch_size, 1, 160, 160, 96)
    text = torch.rand(batch_size, 7)
    
    model = ADPC6_2_VIT(txt_dim=7, dim=256)
    score, conf_out, text_conf, pred = model(img, text)
    
    print(f"Score shape: {score.shape}")
    print(f"Conf out shape: {conf_out.shape}")
    print(f"Text conf shape: {text_conf.shape}")
    print(f"Pred shape: {pred.shape}")
