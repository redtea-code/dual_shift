"""
Visual encoders for 3D medical images.
Includes: ResNet-based encoders (VisualEncoder_base, VisualEncoder2) and ViT-based encoder (VisualEncoderVIT).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict

from Model.backbone.resnet3d import resnet18
from Model.attention.conf_mask import conf_mask3
from Model.backbone.vit3d import ViTBackbone, PatchEmbed3D


# ============================================================
# Shared Utilities
# ============================================================

def patchify3D(E, patch_size):
    """Extract non-overlapping 3D patches: (B,C,D,H,W) -> (B,N,C,pD,pH,pW)."""
    b, c, d, h, w = E.size()
    patch_d, patch_h, patch_w = patch_size
    patches = E.unfold(2, patch_d, patch_d).unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    patches = patches.view(b, -1, c, patch_d, patch_h, patch_w)
    return patches


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class residual_basic_block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, change_channel=True):
        super().__init__()
        self.stride = stride
        self.change_channel = change_channel
        self.conv1 = conv3x3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = conv3x3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.change_channel_fuc = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))

    def forward(self, x):
        residual = x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.change_channel:
            residual = self.change_channel_fuc(residual)
        y += residual
        y = self.relu(y)
        return y


# ============================================================
# ResNet-based Mediator Extractors
# ============================================================

class MediatorExtractor(nn.Module):
    """ResNet18 backbone (conv1 ~ layer3) for visual feature extraction."""

    def __init__(self, backbone='resnet18', dim=256):
        super().__init__()
        resnet = resnet18(
            sample_input_D=128,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=3,
            shortcut_type='B'
        )
        state_dict = torch.load(r"D:\cyh\resnet_18_23dataset.pth")['state_dict']
        keys = list(state_dict.keys())
        keys = [key for key in keys if not key.startswith('module.layer4')]
        new_state_dict = OrderedDict((k, state_dict[k]) for k in keys)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        self.backbone.load_state_dict(new_state_dict, strict=False)
        self.proj = nn.Conv3d(256, dim, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat

    def set_param(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


class MediatorExtractor2(nn.Module):
    """ResNet18 layer4 for post-conf_mask feature extraction."""

    def __init__(self, backbone='resnet18', dim=256):
        super().__init__()
        resnet = resnet18(
            sample_input_D=128,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=3,
            shortcut_type='B'
        )
        state_dict = torch.load(r"D:\cyh\resnet_18_23dataset.pth")['state_dict']
        keys = list(state_dict.keys())
        new_state_dict = OrderedDict((k, state_dict[k]) for k in keys)
        self.backbone = nn.Sequential(resnet.layer4)
        self.backbone.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        return self.backbone(x)

    def set_param(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


# ============================================================
# ResNet-based Visual Encoders
# ============================================================

class VisualEncoder_base(nn.Module):
    """Base visual encoder (ResNet backbone only, no conf_mask)."""

    def __init__(self, dim=256, patch_size=(4, 4, 4), num_classes=3, patch_num=20):
        super().__init__()
        patch_num = int((20 // patch_size[0]) * (20 // patch_size[1]) * (12 // patch_size[2]))
        self.down_sample = MediatorExtractor()
        self.patch_size = patch_size

    def forward(self, x, k=8, **kwargs):
        E = self.down_sample(x)
        patch = patchify3D(E, self.patch_size)
        return patch


class VisualEncoder(nn.Module):
    """Visual encoder with conf_mask3 (leave-one-out scoring)."""

    def __init__(self, dim=256, patch_size=(4, 4, 4), num_classes=3, patch_num=None):
        super().__init__()
        self.down_sample = MediatorExtractor()
        self.mask_attention = conf_mask3(patch_size=patch_size, patch_num=patch_num)
        self.extractor2 = MediatorExtractor2()
        self.extractor3 = MediatorExtractor2()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conf_fc = None
        self._dim = dim
        self._num_classes = num_classes

    def forward(self, x, k=8, **kwargs):
        E = self.down_sample(x)
        score, conf_patch, causal_patch = self.mask_attention(E)
        B, P, C, D, H, W = conf_patch.shape
        causal_patch = self.extractor2(causal_patch.reshape(B * P, C, D, H, W))
        conf_patch = self.extractor3(conf_patch.reshape(B * P, C, D, H, W))

        if self.conf_fc is None:
            self.conf_fc = nn.Sequential(
                nn.Linear(P * 512, 2 * self._dim),
                nn.ReLU(),
                nn.Linear(2 * self._dim, self._dim),
                nn.ReLU(),
                nn.Linear(self._dim, self._num_classes),
            ).to(x.device)

        conf_out = self.conf_fc(self.pool(conf_patch).reshape(B, P * 512))
        return score, conf_out, causal_patch.reshape(B, P, 512, D, H, W)


class VisualEncoder2(nn.Module):
    """Visual encoder with conf_mask6 (gated scoring, more efficient)."""

    def __init__(self, dim=256, patch_size=(4, 4, 4), num_classes=3, patch_num=20):
        super().__init__()
        patch_num = int((20 // patch_size[0]) * (20 // patch_size[1]) * (12 // patch_size[2]))
        self.down_sample = MediatorExtractor()
        self.mask_attention = conf_mask3(patch_size=patch_size, patch_num=patch_num)
        self.extractor2 = MediatorExtractor2()
        self.extractor3 = MediatorExtractor2()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conf_fc = nn.Sequential(
            nn.Linear(patch_num * 512, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x, k=8, **kwargs):
        E = self.down_sample(x)
        score, conf_patch, causal_patch = self.mask_attention(E)
        B, P, C, D, H, W = conf_patch.shape
        causal_patch = self.extractor2(causal_patch.reshape(B * P, C, D, H, W))
        conf_patch = self.extractor3(conf_patch.reshape(B * P, C, D, H, W))
        conf_out = self.conf_fc(self.pool(conf_patch).reshape(B, P * 512))
        return score, conf_out, causal_patch.reshape(B, P, 512, D, H, W)


# ============================================================
# ViT-based Visual Encoder
# ============================================================

class ResidualBlock3D(nn.Module):
    """Simple 3D residual block for ViT feature processing."""

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


class ViTFeatureExtractor(nn.Module):
    """Feature extractor after conf_mask (equivalent to MediatorExtractor2 for ViT)."""

    def __init__(self, in_dim=256, out_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock3D(in_dim, out_dim, stride=1),
            ResidualBlock3D(out_dim, out_dim, stride=1),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):
        return self.layers(x)


class VisualEncoderVIT(nn.Module):
    """ViT-based Visual Encoder with conf_mask on ViT features."""

    def __init__(self, dim=256, img_size=(160, 160, 96), patch_size=(16, 16, 8),
                 vit_patch_size=(8, 8, 4), depth=6, num_heads=8, mlp_ratio=4.,
                 drop_rate=0.1, num_classes=3):
        super().__init__()
        self.vit_backbone = ViTBackbone(
            img_size=img_size, patch_size=vit_patch_size, in_chans=1,
            embed_dim=dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate
        )
        self.vit_grid_size = (
            img_size[0] // vit_patch_size[0],
            img_size[1] // vit_patch_size[1],
            img_size[2] // vit_patch_size[2]
        )
        patch_num = int((self.vit_grid_size[0] // patch_size[0]) *
                        (self.vit_grid_size[1] // patch_size[1]) *
                        (self.vit_grid_size[2] // patch_size[2]))
        self.mask_attention = ViTConfMask(
            patch_size=patch_size, patch_num=patch_num, in_dim=dim
        )
        feat_dim = 256
        self.extractor2 = ViTFeatureExtractor(in_dim=dim, out_dim=feat_dim)
        self.extractor3 = ViTFeatureExtractor(in_dim=dim, out_dim=feat_dim)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.causal_fc = nn.Sequential(
            nn.Linear(patch_num * feat_dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim)
        )
        self.conf_fc = nn.Sequential(
            nn.Linear(patch_num * feat_dim, 2 * dim), nn.ReLU(),
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, num_classes)
        )

    def forward(self, x, **kwargs):
        E = self.vit_backbone(x)
        score, conf_patch, causal_patch = self.mask_attention(E)
        B, P, C, D, H, W = conf_patch.shape
        causal_patch = self.extractor2(causal_patch.reshape(B * P, C, D, H, W))
        conf_patch = self.extractor3(conf_patch.reshape(B * P, C, D, H, W))
        feat_dim = 256
        causal_feat = self.pool(causal_patch).reshape(B, P * feat_dim)
        conf_feat = self.pool(conf_patch).reshape(B, P * feat_dim)
        causal_out = self.causal_fc(causal_feat)
        conf_out = self.conf_fc(conf_feat)
        conf_out = F.log_softmax(conf_out, dim=1)
        return score, conf_out, causal_out
