"""
Simplified ResNet3D + FiLM backbone for DualBranchClassifier.

Design goals:
1. No conf_mask / patch attention — pure ResNet feature extraction
2. FiLM (Feature-wise Linear Modulation) for tabular modality fusion
3. Single logits output (no tuple)
4. Compatible with DualBranchClassifier(feat_dim=num_classes)

Architecture:
    MRI [B,1,D,H,W]                   tabular [B,txt_dim]
        |                                     |
    conv1+bn1+relu+maxpool               FiLM_Conditioning
        |                                     |
    layer1 ──────⊙ FiLM(scale,shift) ─------─┘
        |                                     |
    layer2 ──────⊙ FiLM(scale,shift) ─------─┘
        |                                     |
    layer3 ──────⊙ FiLM(scale,shift) ─------─┘
        |                                     |
    layer4 ──────⊙ FiLM(scale,shift) ─------─┘
        |
    AdaptiveAvgPool3d(1) → Flatten → Dropout → Linear → logits [B,num_classes]

FiLM:
    out = gamma * features + beta
    (gamma, beta) = FiLM_net(txt)

Reference: FiLM — Perez et al., AAAI 2018
"""
import torch
import torch.nn as nn

try:
    from Model.E1 import SpatiallyCorrelatedCAPM
except Exception:  # pragma: no cover - optional legacy dependency
    SpatiallyCorrelatedCAPM = None


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ═══════════════════════════════════════════════════════════
# FiLM conditioning network
# ═══════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for a single ResNet stage.
    Given tabular features txt, predicts per-channel scale (gamma)
    and shift (beta) for the feature map.
    """

    def __init__(self, txt_dim, planes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(txt_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, planes * 2),
        )
        # Initialize: gamma near 1, beta near 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f, txt):
        """
        Args:
            f: [B, C, D, H, W] feature map
            txt: [B, txt_dim] tabular features
        Returns:
            modulated: [B, C, D, H, W]
        """
        params = self.net(txt)  # [B, 2C]
        gamma = params[:, :params.size(1) // 2]  # [B, C]
        beta = params[:, params.size(1) // 2:]
        # [B, C]
        gamma = 1.0 + gamma  # initialize near 1
        gamma = gamma.view(-1, gamma.size(1), 1, 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1, 1)
        return gamma * f + beta


class MultiStageFiLM(nn.Module):
    """FiLM conditioning applied at multiple ResNet stages."""

    def __init__(self, txt_dim, stage_planes=(64, 128, 256, 512), hidden=64):
        super().__init__()
        self.film_layers = nn.ModuleList([
            FiLMLayer(txt_dim, planes, hidden)
            for planes in stage_planes
        ])

    def forward(self, features, txt):
        """
        Args:
            features: list of [B, C_i, D_i, H_i, W_i] tensors
            txt: [B, txt_dim]
        Returns:
            list of modulated feature tensors
        """
        return [film(f, txt) for film, f in zip(self.film_layers, features)]


# ═══════════════════════════════════════════════════════════
# Simplified ResNet3D + FiLM backbone
# ═══════════════════════════════════════════════════════════

class ResNetFilmBackbone(nn.Module):
    """ResNet 3D feature extractor with FiLM tabular fusion.

    Core improvements over original ADPC models:
    - No conf_mask / patch attention
    - No score/conf/causal multi-branch output
    - Simple FiLM conditioning (no patch splitting)
    - Single logits output tensor

    Usage:
        backbone = ResNetFilmBackbone(
            txt_dim=9, num_classes=3, block=BasicBlock, layers=[2,2,2,2],
        )
        logits = backbone(mri_volume, tabular_data)
    """

    def __init__(self, txt_dim=9, num_classes=3,
                 block=BasicBlock, layers=(2, 2, 2, 2),
                 planes=None,
                 film_stages='all', pretrained_weights=None, get_feature=True):
        """
        Args:
            film_stages: which ResNet stages to apply FiLM.
                'all'  → all 4 stages (layer1-4)
                'last' → only final stage (layer4)
                'none' → no FiLM (pure ResNet)
            planes: channel widths for [stem, layer1, layer2, layer3, layer4].
                Default None = [64, 64, 128, 256, 512] (standard ResNet).
                Light: [32, 32, 64, 128, 256] halves the channels.
                Tiny:  [16, 16, 32, 64, 128] quarter channels.
        """
        super().__init__()

        # ── Channel configuration ──
        if planes is None:
            planes = [64, 64, 128, 256, 512]
        p_stem, p1, p2, p3, p4 = planes

        # ── Stem ──
        self.conv1 = nn.Conv3d(
            1, p_stem, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(p_stem)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ── ResNet layers ──
        self.inplanes = p_stem
        stage_planes = []
        stage_planes.append(self.inplanes)  # stem output channels

        self.layer1 = self._make_layer(block, p1, layers[0], stride=1)
        stage_planes.append(p1 * block.expansion)

        self.layer2 = self._make_layer(block, p2, layers[1], stride=2)
        stage_planes.append(p2 * block.expansion)

        self.layer3 = self._make_layer(block, p3, layers[2], stride=2)
        stage_planes.append(p3 * block.expansion)

        self.layer4 = self._make_layer(block, p4, layers[3], stride=2)
        stage_planes.append(p4 * block.expansion)

        # ── FiLM conditioning ──
        self.film_stages = film_stages
        all_stage_planes = stage_planes[1:]  # (64, 64, 128, 256, 512) → skip stem
        if film_stages == 'none':
            self.film_layers = nn.ModuleList()
        elif film_stages == 'last':
            # Only create FiLM for the final (layer4) output
            self.film_layers = nn.ModuleList([
                FiLMLayer(txt_dim, all_stage_planes[-1], hidden=64)
            ])
        else:  # 'all'
            self.film_layers = nn.ModuleList([
                FiLMLayer(txt_dim, planes, hidden=64)
                for planes in all_stage_planes
            ])
        # ── Classifier head ──
        final_planes = stage_planes[-1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.get_feature = get_feature
        self.fc = nn.Linear(final_planes, num_classes)
        # ── Init ──
        self._init_weights()
        self.final_planes = final_planes

        # Load pretrained weights if provided
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_pretrained(self, path=r'D:/cyh/resnet_18_23dataset.pth'):
        """Load pretrained ResNet weights (only stem + layers, skip FiLM + fc)."""
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']

        # Map conv1, bn1, layer1-4 keys
        own = self.state_dict()
        loaded = {}
        for k, v in state.items():
            # Try matching resnet layers
            for prefix in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
                if prefix in k:
                    loaded[k] = v
                    break
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        print(f"ResNetFilmBackbone: loaded {len(loaded)} pretrained keys")

    def forward(self, x, txt=None):
        """
        Args:
            x: [B, 1, D, H, W] 3D MRI volume
            txt: [B, txt_dim] tabular clinical data
        Returns:
            logits: [B, num_classes] — single tensor, no tuple
        """
        # Stem
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        # ResNet layers with FiLM
        f = self.layer1(f)
        if txt is not None and self.film_stages == 'all':
            f = self.film_layers[0](f, txt)

        f = self.layer2(f)
        if txt is not None and self.film_stages == 'all':
            f = self.film_layers[1](f, txt)

        f = self.layer3(f)
        if txt is not None and self.film_stages == 'all':
            f = self.film_layers[2](f, txt)

        f = self.layer4(f)
        if txt is not None and len(self.film_layers) > 0:
            idx = 3 if self.film_stages == 'all' else 0
            f = self.film_layers[idx](f, txt)

        # Global pooling + classification
        f = self.pool(f)
        f = f.flatten(1)
        f = self.dropout(f)
        if self.get_feature:
            return f
        return self.fc(f)


class ResNetE1Backbone(nn.Module):

    def __init__(self, txt_dim=9, num_classes=3,
                 block=BasicBlock, layers=(2, 2, 2, 2),
                 pretrained_weights=True, get_feature=True, spatial_shape=(4, 4, 4)):
        super().__init__()
        if SpatiallyCorrelatedCAPM is None:
            raise ImportError(
                "ResNetE1Backbone requires Model.E1.SpatiallyCorrelatedCAPM"
            )

        # ── Stem ──
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ── ResNet layers ──
        self.inplanes = 64
        stage_planes = []
        stage_planes.append(self.inplanes)  # 64

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        stage_planes.append(64 * block.expansion)  # 64

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        stage_planes.append(128 * block.expansion)  # 128

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        stage_planes.append(256 * block.expansion)  # 256

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        stage_planes.append(512 * block.expansion)  # 512
        # ── FiLM conditioning ──
        self.film_layers = SpatiallyCorrelatedCAPM(txt_dim, n_bases=16,
                                                   spatial_shape=spatial_shape, upsample_to=None,
                                                   max_eta=0.5, use_tv=True, use_sparse=False)
        # ── Classifier head ──
        final_planes = stage_planes[-1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.1)
        self.get_feature = get_feature

        if not get_feature:
            self.fc = nn.Linear(final_planes, num_classes)

        # ── Init ──
        self._init_weights()
        self.final_planes = final_planes

        # Load pretrained weights if provided
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_pretrained(self, path=r'D:/cyh/Causal Infer/weights/resnet_18_23dataset.pth'):
        """Load pretrained ResNet weights (only stem + layers, skip FiLM + fc)."""
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']

        # Map conv1, bn1, layer1-4 keys
        own = self.state_dict()
        loaded = {}
        for k, v in state.items():
            # Try matching resnet layers
            for prefix in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
                if prefix in k:
                    loaded[k] = v
                    break
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        print(f"ResNetFilmBackbone: loaded {len(loaded)} pretrained keys")

    def forward(self, x, txt=None):
        """
        Args:
            x: [B, 1, D, H, W] 3D MRI volume
            txt: [B, txt_dim] tabular clinical data
        Returns:
            logits: [B, num_classes] — single tensor, no tuple
        """
        # Stem
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        # ResNet layers
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        if txt is not None: f = self.film_layers(f, txt)

        # Global pooling + classification
        f = self.pool(f[0])
        f = f.flatten(1)
        f = self.dropout(f)
        if self.get_feature:
            return f
        logits = self.fc(f)
        return logits

    def compute_losses(self):
        return self.film_layers.compute_losses()


# ═══════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════

# ── Standard ResNet variants with FiLM ──

def resnet10_film(txt_dim=9, num_classes=3, film_stages='all',
                  pretrained_weights=None, feature=True, **kwargs):
    """ResNet-10 + FiLM backbone (lightest layers, standard channels)."""
    return ResNetFilmBackbone(
        txt_dim=txt_dim, num_classes=num_classes, film_stages=film_stages,
        block=BasicBlock, layers=(1, 1, 1, 1),
        pretrained_weights=pretrained_weights, get_feature=feature,
    )


def resnet10_ce_only(txt_dim=0, num_classes=3, pretrained_weights=None,
                     feature=False, **kwargs):
    """CE-only ResNet-10: no FiLM, 64-dim GAP features for dictionary MVP.

    ``feature=True`` returns pooled 64-d features; ``feature=False`` trains
    a linear classifier on the same 64-d representation (txt ignored).
    """
    return ResNetFilmBackbone(
        txt_dim=txt_dim, num_classes=num_classes, film_stages='none',
        block=BasicBlock, layers=(1, 1, 1, 1),
        planes=[32, 32, 32, 32, 64],
        pretrained_weights=pretrained_weights, get_feature=feature,
    )


def resnet18_film(txt_dim=9, num_classes=3, pretrained_weights=None, feature=True):
    """ResNet-18 + FiLM backbone."""
    return ResNetFilmBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(2, 2, 2, 2),
        pretrained_weights=pretrained_weights, get_feature=feature
    )


def resnet34_film(txt_dim=9, num_classes=3, film_stages='all',
                  pretrained_weights=None):
    """ResNet-34 + FiLM backbone."""
    return ResNetFilmBackbone(
        txt_dim=txt_dim, num_classes=num_classes, film_stages=film_stages,
        block=BasicBlock, layers=(3, 4, 6, 3),
        pretrained_weights=pretrained_weights,
    )


# ── Lightweight variants (halved channels) ──

def resnet_light_film(txt_dim=9, num_classes=3, film_stages='all',
                      pretrained_weights=None, feature=True):
    """Lightweight ResNet + FiLM: [1,1,1,1] layers, channels [32,32,64,128,256].
    
    ~2.5× fewer params than resnet18_film, suitable for quick experiments.
    """
    return ResNetFilmBackbone(
        txt_dim=txt_dim, num_classes=num_classes, film_stages=film_stages,
        block=BasicBlock, layers=(1, 1, 1, 1),
        planes=[32, 32, 64, 128, 256],
        pretrained_weights=pretrained_weights, get_feature=feature
    )


def resnet_tiny_film(txt_dim=9, num_classes=3, film_stages='all',
                     pretrained_weights=None, feature=True):
    """Tiny ResNet + FiLM: [1,1,1,1] layers, channels [16,16,32,64,128].
    
    ~10× fewer params than resnet18_film, for constrained settings.
    """
    return ResNetFilmBackbone(
        txt_dim=txt_dim, num_classes=num_classes, film_stages=film_stages,
        block=BasicBlock, layers=(1, 1, 1, 1),
        planes=[16, 16, 32, 64, 128],
        pretrained_weights=pretrained_weights, get_feature=feature
    )


def resnet18_e1(txt_dim=9, num_classes=3, pretrained_weights=None, feature=True):
    """ResNet-18 + FiLM backbone."""
    return ResNetE1Backbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(2, 2, 2, 2),
        pretrained_weights=pretrained_weights, get_feature=feature
    )
