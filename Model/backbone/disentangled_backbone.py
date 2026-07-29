"""
ResNet3D + explicit disease / confounder subspace decomposition.

Phase 1 (M1/M2):  split F_d / F_c;  L = CE(Y, F_d) + λ_orth · ||F_d^T F_c||
                  optional: λ_age · MSE(age, head(F_c))  (use_age_prediction)
Phase 2 (M3):     γ(Z) → F_c,  F' = F_d + (1 - γ(Z)) F_c  (+ confounder supervision)
Phase 3–5:      intervention consistency & IRM hooks reserved on the same class.

Ablation chain (paper):
  M0 baseline ResNet  →  use film_backbone / resnet18_film
  M1 + feature split  →  this module, causal_phase=1, orth weight=0
  M2 + orthogonality  →  causal_phase=1, orth weight>0
  M3+                 →  causal_phase>=2
"""

import numpy as np
import torch
import torch.nn as nn

from Model.backbone.film_backbone import BasicBlock
from Model.backbone.backdoor_backbone import ConfounderEncoder
from Model.causal.losses import (
    orthogonality_loss,
    confounder_supervision_loss,
    intervention_consistency_loss,
    age_prediction_loss,
)
from Model.causal.grl import grad_reverse


class ConfounderAgeHead(nn.Module):
    """Predict scalar age from confounder subspace F_c (optional auxiliary task)."""

    def __init__(self, sub_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sub_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, F_c: torch.Tensor) -> torch.Tensor:
        """F_c [B, D] → age_pred [B]."""
        return self.net(F_c).squeeze(-1)


class ConditionalAgeAdversary(nn.Module):
    """Conditional age predictor on GRL(F_d) + Y — pushes F_d to shed age given label."""

    def __init__(self, sub_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        in_dim = sub_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, F_d_grl: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([F_d_grl, y_onehot], dim=1)).squeeze(-1)


class FeatureSplitHead(nn.Module):
    """Map pooled encoder features into disease (F_d) and confounder (F_c) subspaces."""

    def __init__(self, in_dim: int, sub_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_d = nn.Sequential(
            nn.Linear(in_dim, sub_dim),
            nn.BatchNorm1d(sub_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.proj_c = nn.Sequential(
            nn.Linear(in_dim, sub_dim),
            nn.BatchNorm1d(sub_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj_d(x), self.proj_c(x)


class ResNetDisentangledBackbone(nn.Module):
    """ResNet3D encoder with explicit F_d + F_c decomposition."""

    def __init__(
            self,
            txt_dim: int = 9,
            num_classes: int = 3,
            block=BasicBlock,
            layers=(2, 2, 2, 2),
            sub_dim: int = 256,
            z_dim: int = 128,
            pretrained_weights=None,
            input_shape=(160, 160, 96),
            causal_phase: int = 1,
            dropout: float = 0.1,
            get_feature: bool = False,
            use_age_prediction: bool = False,
            use_age_adversarial: bool = False,
            age_head_hidden: int = 64,
            age_adv_hidden: int = 64,
            grl_lambda: float = 1.0,
            gamma_mech_mode: str = 'learned',
            gamma_constant_value=0.5,
            gamma_shuffle_seed: int = 0,
            fusion_mode: str = 'additive_gate',
    ):
        super().__init__()
        self.txt_dim = txt_dim
        self.num_classes = num_classes
        self.sub_dim = sub_dim
        self.z_dim = z_dim
        self.causal_phase = causal_phase
        self.get_feature = get_feature
        self.use_age_prediction = use_age_prediction
        self.use_age_adversarial = use_age_adversarial
        self.grl_lambda = grl_lambda
        self.input_shape = tuple(int(v) for v in input_shape) if input_shape else (160, 160, 96)
        self.gamma_mech_mode = str(gamma_mech_mode or 'learned').lower()
        self.gamma_constant_value = gamma_constant_value
        self.gamma_shuffle_seed = int(gamma_shuffle_seed or 0)
        self.fusion_mode = str(fusion_mode or 'additive_gate').lower()
        self._gamma_shuffle_rng = np.random.RandomState(self.gamma_shuffle_seed)

        # ── ResNet stem (shared with backdoor_backbone) ──
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        final_planes = 512 * block.expansion

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.split_head = FeatureSplitHead(final_planes, sub_dim, dropout=dropout)

        # Phase 2+: tabular → confounder mapping g(Z) and γ(Z) gate
        self.confounder_encoder = None
        self.confounder_mapper = None
        self.gamma_head = None
        if causal_phase >= 2 and txt_dim > 0:
            self.confounder_encoder = ConfounderEncoder(
                txt_dim=txt_dim, z_dim=z_dim, hidden_dim=64,
            )
            self.confounder_mapper = nn.Sequential(
                nn.Linear(z_dim, sub_dim),
                nn.BatchNorm1d(sub_dim),
                nn.ReLU(inplace=True),
                nn.Linear(sub_dim, sub_dim),
            )
            self.gamma_head = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.ReLU(inplace=True),
                nn.Linear(z_dim, 1),
            )

        # Ablation gate parameters
        self.gamma_logit = None
        self.linear_age_gate = None
        if causal_phase >= 2 and self.fusion_mode == 'additive_gate':
            if self.gamma_mech_mode == 'constant':
                if gamma_constant_value is None:
                    self.gamma_logit = nn.Parameter(torch.zeros(1))
                else:
                    c = float(gamma_constant_value)
                    c = min(max(c, 1e-4), 1.0 - 1e-4)
                    logit = float(np.log(c / (1.0 - c)))
                    self.register_buffer(
                        '_fixed_gamma_logit',
                        torch.tensor([logit], dtype=torch.float32),
                    )
            elif self.gamma_mech_mode == 'linear_age':
                self.linear_age_gate = nn.Linear(1, 1)

        if not get_feature:
            if self.fusion_mode == 'concat_fc' and causal_phase >= 2:
                self.fc = nn.Linear(2 * sub_dim + 1, num_classes)
            else:
                self.fc = nn.Linear(sub_dim, num_classes)
        else:
            self.fc = None

        self.age_head = (
            ConfounderAgeHead(sub_dim, hidden_dim=age_head_hidden, dropout=dropout)
            if use_age_prediction else None
        )
        self.age_adv_head = (
            ConditionalAgeAdversary(
                sub_dim, num_classes, hidden_dim=age_adv_hidden, dropout=dropout,
            )
            if use_age_adversarial else None
        )

        self.final_planes = final_planes
        self._last_F_d = None
        self._last_F_c = None
        self._last_F = None
        self._last_z = None
        self._last_gamma = None
        self._last_F_d_cf = None
        self._last_F_c_cf = None
        self._last_age_pred = None
        self._last_age_adv_pred = None
        self._last_txt = None

        self._init_weights()
        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
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
        print(f"ResNetDisentangledBackbone: loaded {len(loaded)} pretrained keys")

    def extract_features(self, x):
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        return f

    def _task_features(self, F_d, F_c, F_composed, txt=None):
        """Features fed to the classification head (phase-dependent)."""
        if self.causal_phase <= 1:
            return F_d
        if self.fusion_mode == 'concat_fc':
            age = self._extract_age_scalar(txt, F_d)
            return torch.cat([F_d, F_c, age], dim=-1)
        return F_composed

    def _extract_age_scalar(self, txt, ref: torch.Tensor):
        """Use first tabular column as age; fallback zeros."""
        b = ref.size(0)
        if txt is None:
            return ref.new_zeros(b, 1)
        if txt.dim() == 1:
            return txt.view(b, 1).float()
        return txt[:, :1].float()

    def _maybe_shuffle_txt(self, txt):
        if txt is None or self.gamma_mech_mode != 'shuffle':
            return txt
        b = txt.size(0)
        if b <= 1:
            return txt
        perm = torch.from_numpy(
            self._gamma_shuffle_rng.permutation(b),
        ).long().to(txt.device)
        return txt[perm]

    def _compute_gamma(self, txt, F_d):
        """Return gamma [B, 1] according to gamma_mech_mode."""
        b = F_d.size(0)
        mode = self.gamma_mech_mode

        if mode == 'zeros':
            return F_d.new_zeros(b, 1)
        if mode == 'ones':
            return F_d.new_ones(b, 1)

        if mode == 'constant':
            if self.gamma_logit is not None:
                g = torch.sigmoid(self.gamma_logit).view(1, 1).expand(b, 1)
                return g
            logit = getattr(self, '_fixed_gamma_logit', None)
            if logit is not None:
                return torch.sigmoid(logit).view(1, 1).expand(b, 1)
            return F_d.new_full((b, 1), 0.5)

        if mode == 'linear_age' and self.linear_age_gate is not None:
            age = self._extract_age_scalar(txt, F_d)
            return torch.sigmoid(self.linear_age_gate(age))

        # learned / shuffle → Enc(txt) → gamma_head
        txt_use = self._maybe_shuffle_txt(txt)
        if txt_use is None or self.confounder_encoder is None or self.gamma_head is None:
            return F_d.new_zeros(b, 1)
        z = self.confounder_encoder(txt_use)
        self._last_z = z
        return torch.sigmoid(self.gamma_head(z))

    def _compose_representation(self, F_d, F_c, txt=None):
        """Build composed F for diagnostics / later phases.

        Phase 1: F = F_d + F_c (stored only; CE uses F_d alone).
        Phase 2+: F' = F_d + (1 - γ) F_c (unless concat_fc fusion).
        """
        F = F_d + F_c
        self._last_gamma = None
        self._last_z = None
        self._last_txt = txt

        if self.causal_phase >= 2 and self.fusion_mode == 'additive_gate':
            # Still run encoder for confounder_supervision when possible
            if (
                txt is not None
                and self.confounder_encoder is not None
                and self.gamma_mech_mode not in ('shuffle',)
            ):
                # For learned path, encoder is inside _compute_gamma; for others set z.
                if self.gamma_mech_mode not in ('learned',):
                    self._last_z = self.confounder_encoder(txt)

            gamma = self._compute_gamma(txt, F_d)
            self._last_gamma = gamma
            F = F_d + (1.0 - gamma) * F_c

            # Ensure _last_z for cs loss under learned/shuffle
            if self._last_z is None and txt is not None and self.confounder_encoder is not None:
                self._last_z = self.confounder_encoder(txt)

        elif self.causal_phase >= 2 and self.fusion_mode == 'concat_fc':
            if txt is not None and self.confounder_encoder is not None:
                self._last_z = self.confounder_encoder(txt)
            self._last_gamma = None

        self._last_F_d = F_d
        self._last_F_c = F_c
        self._last_F = F
        return F

    def forward(self, x, txt=None):
        """
        Args:
            x:   [B, 1, D, H, W]
            txt: [B, txt_dim] tabular (used from phase 2 onward)

        Returns:
            logits [B, num_classes] or task features if get_feature=True.
            Phase 1: logits / features from F_d only.
        """
        feat_map = self.extract_features(x)
        pooled = self.pool(feat_map).flatten(1)
        F_d, F_c = self.split_head(pooled)
        F = self._compose_representation(F_d, F_c, txt=txt)
        self._last_age_pred = self.predict_age(F_c)
        F_task = self._task_features(F_d, F_c, F, txt=txt)

        if self.get_feature:
            return F_task

        return self.fc(F_task)

    def get_gamma(self):
        """Return last forward gamma [B, 1] or None."""
        return self._last_gamma

    # ── Causal interfaces ────────────────────────────────────────────────────

    def predict_age(self, F_c: torch.Tensor = None):
        """Predict age from F_c. Returns None if age head is disabled."""
        if self.age_head is None:
            return None
        F_c = F_c if F_c is not None else self._last_F_c
        if F_c is None:
            return None
        return self.age_head(F_c)

    def get_disentanglement_features(self):
        """Return last forward pass subspace features (F_d, F_c, F)."""
        return self._last_F_d, self._last_F_c, self._last_F

    def get_age_prediction(self):
        """Return last forward pass age prediction from F_c (or None)."""
        return self._last_age_pred

    def predict_age_adversarial(self, F_d: torch.Tensor, y: torch.Tensor):
        """age ← [GRL(F_d), one_hot(Y)]; returns None if adversary disabled."""
        if self.age_adv_head is None or F_d is None or y is None:
            return None
        F_d_grl = grad_reverse(F_d, self.grl_lambda)
        y_onehot = torch.nn.functional.one_hot(
            y.long().clamp(min=0), num_classes=self.num_classes,
        ).float()
        return self.age_adv_head(F_d_grl, y_onehot)

    def get_age_adversarial_prediction(self):
        return self._last_age_adv_pred

    def get_causal_losses(self, phase=None, age: torch.Tensor = None, y: torch.Tensor = None):
        """Phase-aware auxiliary losses for the training loop."""
        phase = phase if phase is not None else self.causal_phase
        losses = {}

        if self._last_F_d is not None and self._last_F_c is not None:
            if phase >= 1:
                losses['orthogonality'] = orthogonality_loss(
                    self._last_F_d, self._last_F_c,
                )

        if age is not None and self._last_age_pred is not None:
            losses['age_prediction'] = age_prediction_loss(
                self._last_age_pred, age,
            )

        if age is not None and y is not None and self._last_F_d is not None:
            adv_pred = self.predict_age_adversarial(self._last_F_d, y)
            if adv_pred is not None:
                self._last_age_adv_pred = adv_pred
                losses['age_adversarial'] = age_prediction_loss(adv_pred, age)

        if phase >= 2 and self._last_F_c is not None and self._last_z is not None:
            if self.confounder_mapper is not None:
                g_z = self.confounder_mapper(self._last_z)
                losses['confounder_supervision'] = confounder_supervision_loss(
                    self._last_F_c, g_z,
                )

        if phase >= 3 and self._last_F_d is not None and self._last_F_d_cf is not None:
            losses['intervention_consistency'] = intervention_consistency_loss(
                self._last_F_d, self._last_F_d_cf,
            )

        # Phase 4 IRM: computed externally via get_irm_penalty(); hook reserved.
        return losses

    def forward_intervention(self, x, txt, txt_cf):
        """Phase 3 counterfactual forward: same MRI, different tabular Z → Z'.

        Stores F_d, F_c under (Z) and F_d_cf, F_c_cf under (Z') for consistency losses.
        Returns logits under factual Z.
        """
        if self.causal_phase < 3:
            raise RuntimeError("forward_intervention requires causal_phase >= 3")

        feat_map = self.extract_features(x)
        pooled = self.pool(feat_map).flatten(1)
        F_d, F_c = self.split_head(pooled)

        # Factual
        _ = self._compose_representation(F_d, F_c, txt=txt)
        F_d_fact = self._last_F_d
        F_c_fact = self._last_F_c

        # Counterfactual (same image features, different Z)
        if txt_cf is not None and self.confounder_encoder is not None:
            z_cf = self.confounder_encoder(txt_cf)
            F_d_cf, F_c_cf = F_d, F_c  # image-derived; Z only modulates via gamma in phase 2
            gamma_cf = torch.sigmoid(self.gamma_head(z_cf))
            _ = F_d_cf  # F_d invariant by construction in phase 3 design
            F_c_cf = F_c  # placeholder until phase 3 full path is wired
            self._last_F_d_cf = F_d_cf
            self._last_F_c_cf = F_c_cf
            self._last_F_d = F_d_fact
            self._last_F_c = F_c_fact

        F_task = self._task_features(self._last_F_d, self._last_F_c, self._last_F, txt=txt)
        return self.fc(F_task)

    def get_regularization_losses(self):
        """Trainer compatibility — delegate to causal losses."""
        return self.get_causal_losses(phase=self.causal_phase)


def _resolve_factory_kwargs(kwargs):
    resolved = {}
    if 'input_shape' in kwargs:
        resolved['input_shape'] = kwargs['input_shape']
    elif 'img_size' in kwargs:
        resolved['input_shape'] = kwargs['img_size']
    feature = kwargs.get('feature')
    get_feature = kwargs.get('get_feature')
    if get_feature is not None:
        resolved['get_feature'] = get_feature
    elif feature:
        resolved['get_feature'] = True
    for key in ('sub_dim', 'z_dim', 'causal_phase', 'dropout',
                'use_age_prediction', 'use_age_adversarial',
                'age_head_hidden', 'age_adv_hidden', 'grl_lambda',
                'gamma_mech_mode', 'gamma_constant_value', 'gamma_shuffle_seed',
                'fusion_mode'):
        if key in kwargs:
            resolved[key] = kwargs[key]
    return resolved


def resnet18_disentangled(txt_dim=9, num_classes=3, sub_dim=256, z_dim=128,
                          pretrained_weights=None, feature=False,
                          get_feature=None, causal_phase=1, **kwargs):
    factory_kwargs = _resolve_factory_kwargs(kwargs)
    if get_feature is not None:
        factory_kwargs['get_feature'] = get_feature
    elif feature:
        factory_kwargs['get_feature'] = True
    return ResNetDisentangledBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(2, 2, 2, 2),
        sub_dim=sub_dim, z_dim=z_dim,
        pretrained_weights=pretrained_weights,
        causal_phase=causal_phase,
        **factory_kwargs,
    )


def resnet10_disentangled(txt_dim=9, num_classes=3, sub_dim=256, z_dim=128,
                          pretrained_weights=None, feature=False,
                          get_feature=None, causal_phase=1, **kwargs):
    factory_kwargs = _resolve_factory_kwargs(kwargs)
    if get_feature is not None:
        factory_kwargs['get_feature'] = get_feature
    elif feature:
        factory_kwargs['get_feature'] = True
    return ResNetDisentangledBackbone(
        txt_dim=txt_dim, num_classes=num_classes,
        block=BasicBlock, layers=(1, 1, 1, 1),
        sub_dim=sub_dim, z_dim=z_dim,
        pretrained_weights=pretrained_weights,
        causal_phase=causal_phase,
        **factory_kwargs,
    )
