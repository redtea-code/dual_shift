
"""
ResNet3D + SCA-CAPM Backbone.

SCA-CAPM: Spatially Continuous and Attribute-specific
          Confounder-Aware Patch Modulation.

Inserts SCA-CAPM or its variants after each ResNet stage to perform
spatially-continuous covariate adjustment on feature maps.

Default var_specs for ADNI-style tabular data (9 variables):
    age, sex, education, MMSE, APOE, CDR, ADAS, FAQ, site
    (adjust based on your actual data columns)

Architecture:
    same as ResNetDAFTBackbone, but with SCA-CAPM blocks instead of DAFT.
"""

import torch
import torch.nn as nn

from Model.causal.sca_capm import (
    SpatiallyCorrelatedCAPM,
    CovariateResidualAdjustment,
    VariableSpecificCAPM,
    SCACAPM,
)
from Model.causal.daft import DAFTBlock
from Model.backbone.film_backbone import BasicBlock


# ══════════════════════════════════════════════════════════════
# Default ADNI variable specs (adjust to your actual data)
# ══════════════════════════════════════════════════════════════

def default_adni_var_specs():
    """Return default variable specs for 9-tabular ADNI data.
    
    Typical ADNI table columns (used in New_project):
        age, sex, education, MMSE, APOE4, CDRSB, ADAS11, FAQ, site_id
    """
    return [
        {'name': 'age',       'type': 'continuous',  'n_centers': 8,  'min_val': 55, 'max_val': 95,  'n_bases': 6},
        {'name': 'sex',       'type': 'categorical',  'n_cats': 2,                                    'n_bases': 2},
        {'name': 'education', 'type': 'continuous',  'n_centers': 6,  'min_val': 0,  'max_val': 22,  'n_bases': 4},
        {'name': 'MMSE',      'type': 'continuous',  'n_centers': 8,  'min_val': 0,  'max_val': 30,  'n_bases': 6},
        {'name': 'APOE4',     'type': 'categorical',  'n_cats': 3,                                    'n_bases': 3},
        {'name': 'CDRSB',     'type': 'continuous',  'n_centers': 6,  'min_val': 0,  'max_val': 18,  'n_bases': 4},
        {'name': 'ADAS11',    'type': 'continuous',  'n_centers': 6,  'min_val': 0,  'max_val': 70,  'n_bases': 4},
        {'name': 'FAQ',       'type': 'continuous',  'n_centers': 6,  'min_val': 0,  'max_val': 30,  'n_bases': 3},
        {'name': 'site',      'type': 'categorical',  'n_cats': 10,                                   'n_bases': 2},
    ]


# ══════════════════════════════════════════════════════════════
# ResNet + SCA-CAPM Backbone
# ══════════════════════════════════════════════════════════════

class ResNetSCACAPMBackbone(nn.Module):
    """ResNet 3D with SCA-CAPM spatial covariate adjustment.

    Variant modes (select via `variant` param):
        'sc'    → SpatiallyCorrelatedCAPM at each stage (Direction 1)
        'csra'  → CovariateResidualAdjustment at each stage (Direction 2)
        'var'   → VariableSpecificCAPM at each stage (Direction 4)
        'sca'   → Full SCA-CAPM at each stage (Final unified)
        'daft'  → DAFT blocks at each stage (baseline comparison)
    """

    def __init__(self, txt_dim=9, num_classes=3, variant='sca',
                 spatial_shape=(4, 4, 4),
                 block=BasicBlock, layers=(2, 2, 2, 2),
                 var_specs=None):
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes
        self.variant = variant

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
        final_planes = 512 * block.expansion

        # ── Stage channels ──
        stage_channels = {
            'layer1': 64 * block.expansion,
            'layer2': 128 * block.expansion,
            'layer3': 256 * block.expansion,
            'layer4': 512 * block.expansion,
        }

        # ── Var specs (default ADNI if not provided) ──
        if var_specs is None:
            var_specs = default_adni_var_specs()
        self.var_specs = var_specs

        # ── Build adjustment modules per stage ──
        if variant == 'sc':
            # Direction 1: Spatially Correlated CAPM
            self.adj_modules = nn.ModuleDict({
                name: SpatiallyCorrelatedCAPM(
                    tabular_dim=txt_dim,
                    n_bases=8,
                    spatial_shape=spatial_shape,
                )
                for name in stage_channels
            })
        elif variant == 'csra':
            # Direction 2: Covariate Residual Adjustment
            self.adj_modules = nn.ModuleDict({
                name: CovariateResidualAdjustment(
                    tabular_dim=txt_dim,
                    feature_dim=ch,
                    n_bases=8,
                    spatial_shape=spatial_shape,
                )
                for name, ch in stage_channels.items()
            })
        elif variant == 'var':
            # Direction 4: Variable-specific CAPM
            self.adj_modules = nn.ModuleDict({
                name: VariableSpecificCAPM(
                    var_specs=var_specs,
                    spatial_shape=spatial_shape,
                    emb_dim=16,
                )
                for name in stage_channels
            })
        elif variant == 'sca':
            # Final: full SCA-CAPM
            n_total = 32  # total bases across all vars
            self.adj_modules = nn.ModuleDict({
                name: SCACAPM(
                    var_specs=var_specs,
                    n_total_bases=n_total,
                    spatial_shape=spatial_shape,
                    emb_dim=16,
                    feature_dim=ch,
                )
                for name, ch in stage_channels.items()
            })
        elif variant == 'daft':
            self.adj_modules = nn.ModuleDict({
                name: DAFTBlock(tabular_dim=txt_dim, feature_channels=ch)
                for name, ch in stage_channels.items()
            })
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # ── Classifier head ──
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(final_planes, num_classes)

        # ── Init ──
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers_list = []
        layers_list.append(block(self.inplanes, planes, stride=stride,
                                 downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes))
        return nn.Sequential(*layers_list)

    def _init_weights(self):
        # Collect all parameter ids belonging to adj_modules (already init'd)
        adj_param_ids = set()
        for adj in self.adj_modules.values():
            for p in adj.parameters():
                adj_param_ids.add(id(p))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Skip Linear layers that are part of adj_modules
                # (their weights are already near-zero initialized)
                if any(id(p) in adj_param_ids for p in m.parameters()):
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_stage(self, name, f, z):
        """Apply one ResNet stage + adjustment."""
        stage = getattr(self, name)
        adj = self.adj_modules[name]
        f = stage(f)
        if z is not None:
            if self.variant == 'sca':
                # SCACAPM returns (x_out, gamma, gamma_vars) — keep x_out only
                f, _, _ = adj(f, z)
            elif self.variant == 'daft':
                # DAFT returns single tensor
                f = adj(f, z)
            else:
                # sc, csra, var return (x_adj, gamma_or_residual)
                f, _ = adj(f, z)
        return f

    def forward(self, x, z=None):
        """
        Args:
            x: [B, 1, D, H, W] MRI volume
            z: [B, txt_dim] tabular data (None → skip adjustment)

        Returns:
            logits: [B, num_classes]
        """
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        f = self._forward_stage('layer1', f, z)
        f = self._forward_stage('layer2', f, z)
        f = self._forward_stage('layer3', f, z)
        f = self._forward_stage('layer4', f, z)

        f = self.pool(f)
        f = f.flatten(1)
        f = self.dropout(f)
        logits = self.fc(f)
        return logits

    def compute_mask_loss(self):
        """Aggregate regularization losses from all adjustment modules."""
        losses = {}
        for module in self.adj_modules.values():
            if hasattr(module, 'compute_losses'):
                mod_losses = module.compute_losses()
                for k, v in mod_losses.items():
                    losses[k] = losses.get(k, 0) + v
        return losses

    def adversarial_loss(self, x_adj, z):
        """Aggregate adversarial loss from CSRA modules."""
        total = 0.0
        for module in self.adj_modules.values():
            if hasattr(module, 'adversarial_loss'):
                total += module.adversarial_loss(x_adj, z)
        return total


# ══════════════════════════════════════════════════════════════
# Factory functions
# ══════════════════════════════════════════════════════════════

def _make_sca_capm(depth='18', variant='sca', txt_dim=9, num_classes=3,
                   var_specs=None):
    layer_map = {'10': (1,1,1,1), '18': (2,2,2,2), '34': (3,4,6,3)}
    return ResNetSCACAPMBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        variant=variant,
        block=BasicBlock, layers=layer_map[depth],
        var_specs=var_specs,
    )


def resnet18_sca(txt_dim=9, num_classes=3, var_specs=None):
    return _make_sca_capm('18', 'sca', txt_dim, num_classes, var_specs)


def resnet18_sc(txt_dim=9, num_classes=3, var_specs=None):
    return _make_sca_capm('18', 'sc', txt_dim, num_classes, var_specs)


def resnet18_csra(txt_dim=9, num_classes=3, var_specs=None):
    return _make_sca_capm('18', 'csra', txt_dim, num_classes, var_specs)


def resnet18_var(txt_dim=9, num_classes=3, var_specs=None):
    return _make_sca_capm('18', 'var', txt_dim, num_classes, var_specs)
