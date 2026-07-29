
"""
ResNet3D + DAFT (Dynamic Affine Feature Map Transform) Backbone.

Reference:
    Polsterl et al. "Combining 3D Image and Tabular Data via the Dynamic
    Affine Feature Map Transform." MICCAI 2021.

Architecture:
    MRI [B,1,D,H,W]                      Tabular [B,txt_dim]
         |                                      |
    ResNet3D Stem (conv1+bn+relu+mp)           /
         |                                    /
    layer1 ──── DAFT ────────────────────────┘
         |
    layer2 ──── DAFT ────────────────────────┘
         |
    layer3 ──── DAFT ────────────────────────┘
         |
    layer4 ──── DAFT ────────────────────────┘
         |
    AdaptiveAvgPool3d → Flatten → Dropout → Linear → [B,num_classes]

DAFT conditions each ResNet stage's feature maps on tabular clinical data,
dynamically rescaling and shifting channels for improved multimodal fusion.

Key differences from ResNetFilmBackbone:
    - DAFT uses deeper conditioning networks with bottleneck
    - More granular control: per-channel affine per spatial position
    - Designed specifically for 3D medical imaging + tabular fusion
    - Empirical results (MICCAI 2021): outperforms FiLM for AD diagnosis

Factory functions:
    resnet18_daft, resnet34_daft, resnet10_daft
"""

import torch
import torch.nn as nn

from Model.causal.daft import DAFTBlock
from Model.backbone.film_backbone import BasicBlock


class ResNetDAFTBackbone(nn.Module):
    """ResNet 3D feature extractor with DAFT tabular fusion.

    DAFT applies per-channel affine transformations to feature maps
    at each ResNet stage, conditioned on tabular clinical data.
    """

    def __init__(self, txt_dim=9, num_classes=3,
                 block=BasicBlock, layers=(2, 2, 2, 2),
                 pretrained_weights=None, daft_hidden=None,get_feature=True):
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes

        # ── Stem ──
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
        final_planes = 512 * block.expansion  # 512

        # ── DAFT conditioning at each stage ──
        stage_channels = {
            'layer1': 64 * block.expansion,   # 64
            'layer2': 128 * block.expansion,  # 128
            'layer3': 256 * block.expansion,  # 256
            'layer4': 512 * block.expansion,  # 512
        }
        self.daft_layers = nn.ModuleDict({
            name: DAFTBlock(txt_dim, ch, hidden_dim=daft_hidden)
            for name, ch in stage_channels.items()
        })

        # ── Classifier head ──
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(final_planes, num_classes)
        self.get_feature = get_feature

        if not get_feature:
            self.fc = nn.Linear(final_planes, num_classes)
        # ── Metadata ──
        self.final_planes = final_planes

        # ── Init ──
        self._init_weights()

        if pretrained_weights is not None:
            self._load_pretrained()

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
                # Skip DAFT conditioning Linear layers (already zero-init'd)
                if not any(
                    m is layer
                    for daft in self.daft_layers.values()
                    for layer in daft.conditioning
                ):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _load_pretrained(self, path=r'D:/cyh/resnet_18_23dataset.pth'):
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
        print(f"ResNetDAFTBackbone: loaded {len(loaded)} pretrained keys")

    def extract_features(self, x):
        """Extract feature maps through ResNet layers (no DAFT, no tabular)."""
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
            txt: [B, txt_dim] tabular clinical data (if None, skip DAFT)

        Returns:
            logits: [B, num_classes] — single tensor, no tuple
        """
        # ── Stem ──
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        # ── ResNet layers with DAFT conditioning ──
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        if txt is not None:
            f = self.daft_layers['layer4'](f, txt)

        # ── Global pooling + classification ──
        f = self.pool(f)
        f = f.flatten(1)
        f = self.dropout(f)
        if self.get_feature:
            return f
        logits = self.fc(f)

        return logits


# ═══════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════

def resnet18_daft(txt_dim=9, num_classes=3, pretrained_weights=None,feature=True):
    """ResNet-18 + DAFT backbone."""
    return ResNetDAFTBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(2, 2, 2, 2),
        pretrained_weights=pretrained_weights,get_feature=feature
    )


def resnet34_daft(txt_dim=9, num_classes=3, pretrained_weights=None):
    """ResNet-34 + DAFT backbone."""
    return ResNetDAFTBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(3, 4, 6, 3),
        pretrained_weights=pretrained_weights,
    )


def resnet10_daft(txt_dim=9, num_classes=3, pretrained_weights=None,
                  feature=False, **kwargs):
    """ResNet-10 + DAFT backbone (lightweight). Classification by default."""
    return ResNetDAFTBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(1, 1, 1, 1),
        pretrained_weights=pretrained_weights,
        get_feature=feature,
    )
