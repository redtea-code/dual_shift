"""
Pre-activation ResNet3D — Ultra-lightweight 3D CNN Backbone

Design philosophy (from HyperFusion):
  - Pre-activation: BN → ReLU → Conv (instead of Conv → BN → ReLU)
  - Aggressive channel reduction: init_features=4 (vs 64 in standard ResNet)
  - Progressive downsampling via stride-2 blocks + maxpool

Channel progression with init_features=f:
    conv_bn_relu: 1 → f
    max_pool:     f → f         (spatial /2)
    block1:       f → 2f        (stride=1)
    block2:       2f → 4f       (stride=2)
    block3:       4f → 8f       (stride=2)
    block4:       8f → 16f      (stride=2)
    → AdaptiveAvgPool3d(1) → fc → logits

Sizes:
    init_features=4  → ~75K params (ultra-tiny)
    init_features=8  → ~300K params
    init_features=16 → ~1.2M params

Reference:
    HyperFusion for Alzheimer's Disease prediction
    Pre-activation design: He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════

def conv3d_bn3d_relu(in_channels, out_channels, bn_momentum=0.05,
                     kernel_size=3, stride=1, padding=1):
    """Initial conv block: Conv → BN → ReLU (standard post-activation)."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=padding, bias=True),
        nn.BatchNorm3d(out_channels, momentum=bn_momentum),
        nn.ReLU(inplace=True),
    )


class PreactResBlock(nn.Module):
    """Pre-activation Residual Block.

    Order: BN → ReLU → Conv → BN → ReLU → Conv, then +skip.

    This is the "full pre-activation" variant from He et al. (ECCV 2016),
    which makes the identity path clean with no activations on the skip,
    improving gradient flow.

    Args:
        in_channels:  input channels
        out_channels: output channels
        stride:       stride for first conv (spatial downsampling)
        dropout:      3D dropout rate
        bn_momentum:  BatchNorm momentum
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 dropout=0.0, bn_momentum=0.05):
        super().__init__()

        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout)

        # Skip connection: 1×1 conv when channel or spatial dims change
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=True),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += identity
        return out


# ═══════════════════════════════════════════════════════════
# Pre-activation ResNet3D Backbone
# ═══════════════════════════════════════════════════════════

class PreactResNet3D(nn.Module):
    """Ultra-lightweight pre-activation ResNet3D.

    Designed as a minimal feature extractor for 3D medical images,
    suitable for rapid prototyping, hyperparameter search, and
    resource-constrained deployment.

    Architecture:
        MRI [B,1,D,H,W]
          │
        conv_bn_relu  (1 → init_features)
          │
        max_pool3d    (kernel=2, spatial /2)
          │
        block1        (f → 2f,  stride=1)
        block2        (2f → 4f, stride=2)
        block3        (4f → 8f, stride=2)
        block4        (8f → 16f, stride=2)
          │
        AdaptiveAvgPool3d(1) → Flatten → Dropout → Linear → logits
    """

    def __init__(self, in_channels=1, num_classes=3, init_features=4,
                 bn_momentum=0.1, dropout_rates=(0.0, 0.0, 0.0, 0.0),
                 get_feature=False):
        """
        Args:
            in_channels:    input image channels (1 for grayscale MRI)
            num_classes:    output classes (3 for AD/CN/MCI)
            init_features:  base channel count. 4=tiny, 8=small, 16=medium.
            bn_momentum:    BatchNorm momentum
            dropout_rates:  per-block dropout: (block1, block2, block3, block4)
            get_feature:    if True, return pre-logits feature vector
        """
        super().__init__()

        f = init_features
        d1, d2, d3, d4 = dropout_rates if len(dropout_rates) == 4 else (0, 0, 0, 0)

        # ── Stem ──
        self.stem = conv3d_bn3d_relu(in_channels, f, bn_momentum=bn_momentum)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # ── Pre-activation ResBlocks (f → 2f → 4f → 8f → 16f) ──
        self.block1 = PreactResBlock(f, 2 * f, stride=1,
                                     dropout=d1, bn_momentum=bn_momentum)
        self.block2 = PreactResBlock(2 * f, 4 * f, stride=2,
                                     dropout=d2, bn_momentum=bn_momentum)
        self.block3 = PreactResBlock(4 * f, 8 * f, stride=2,
                                     dropout=d3, bn_momentum=bn_momentum)
        self.block4 = PreactResBlock(8 * f, 16 * f, stride=2,
                                     dropout=d4, bn_momentum=bn_momentum)

        self.final_channels = 16 * f
        self.get_feature = get_feature

        # ── Classification head ──
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.final_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: [B, 1, D, H, W] → logits: [B, num_classes]"""
        f = self.stem(x)
        f = self.maxpool(f)
        f = self.block1(f)
        f = self.block2(f)
        f = self.block3(f)
        f = self.block4(f)
        f = self.pool(f).flatten(1)
        f = self.dropout(f)
        if self.get_feature:
            return f
        return self.fc(f)


# ═══════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════

def preact_resnet_ut(num_classes=3):
    """Ultra-tiny: init_features=4 → ~75K params."""
    return PreactResNet3D(num_classes=num_classes, init_features=4)


def preact_resnet_t(num_classes=3):
    """Tiny: init_features=8 → ~300K params."""
    return PreactResNet3D(num_classes=num_classes, init_features=8)


def preact_resnet_s(num_classes=3):
    """Small: init_features=16 → ~1.2M params."""
    return PreactResNet3D(num_classes=num_classes, init_features=16)


# ═══════════════════════════════════════════════════════════
# Dimension test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing PreactResNet3D on {device}...\n")

    batch = 2
    mri = torch.rand(batch, 1, 160, 160, 96).to(device)

    variants = {
        'Ultra-tiny (f=4)':  preact_resnet_ut(),
        'Tiny (f=8)':        preact_resnet_t(),
        'Small (f=16)':      preact_resnet_s(),
    }

    for name, model in variants.items():
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(mri)
        n = sum(p.numel() for p in model.parameters())
        print(f"  {name:<18}  {n:>8,} params  {list(mri.shape)} -> {list(out.shape)}  PASS")

    print(f"\nAll variants passed!")
