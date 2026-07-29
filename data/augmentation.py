"""
3D medical image augmentation transforms for MRI training.

Two modes:
  Per-sample  — RandomAffine3D, RandomIntensityScale, …   (legacy, debug)
  Batch-aware — BatchAugmentation + make_collate_fn       (production)

Batch-aware mode vectorises affine / intensity / noise / bias-field
across the entire batch in a single call, which is much faster than
per-sample __getitem__ transforms, especially for grid_sample.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  Per-sample transforms  (kept for backward-compat / debug)
# ═══════════════════════════════════════════════════════════

class RandomAffine3D:
    """Random 3D affine: rotation, translation, isotropic scaling.

    Operates on a **single** tensor of shape (D, H, W).
    """

    def __init__(self, rotation=5.0, translation=3.0, scale=(0.97, 1.03), p=1.0):
        self.rotation = float(rotation)
        self.translation = float(translation)
        self.scale = tuple(scale)
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        D, H, W = x.shape[-3:]
        rx = random.uniform(-self.rotation, self.rotation) * np.pi / 180.0
        ry = random.uniform(-self.rotation, self.rotation) * np.pi / 180.0
        rz = random.uniform(-self.rotation, self.rotation) * np.pi / 180.0
        s = random.uniform(*self.scale)
        tx_n = random.uniform(-self.translation, self.translation) / (W / 2.0)
        ty_n = random.uniform(-self.translation, self.translation) / (H / 2.0)
        tz_n = random.uniform(-self.translation, self.translation) / (D / 2.0)

        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)

        Rx = torch.tensor([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=torch.float32)
        Ry = torch.tensor([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=torch.float32)
        Rz = torch.tensor([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=torch.float32)
        R = Rz @ Ry @ Rx
        S = torch.diag(torch.tensor([s, s, s], dtype=torch.float32))
        A = R @ S

        theta = torch.zeros(3, 4, dtype=torch.float32)
        theta[:3, :3] = A
        theta[0, 3] = tx_n;  theta[1, 3] = ty_n;  theta[2, 3] = tz_n
        theta = theta.unsqueeze(0)

        need_squeeze = (x.dim() == 3)
        x_in = x.unsqueeze(0).unsqueeze(0) if need_squeeze else x.unsqueeze(0)
        grid = F.affine_grid(theta, x_in.size(), align_corners=False)
        out = F.grid_sample(x_in, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return out.squeeze(0).squeeze(0) if need_squeeze else out.squeeze(0)


class RandomIntensityScale:
    def __init__(self, scale=(0.9, 1.1), p=1.0):
        self.scale, self.p = tuple(scale), p

    def __call__(self, x):
        if random.random() > self.p: return x
        return x * random.uniform(*self.scale)


class RandomIntensityShift:
    def __init__(self, shift=(-0.1, 0.1), p=1.0):
        self.shift, self.p = tuple(shift), p

    def __call__(self, x):
        if random.random() > self.p: return x
        return x + random.uniform(*self.shift)


class GaussianNoise:
    def __init__(self, std=(0.01, 0.03), p=1.0):
        self.std, self.p = tuple(std), p

    def __call__(self, x):
        if random.random() > self.p: return x
        return x + torch.randn_like(x) * random.uniform(*self.std)


class RandomBiasField:
    def __init__(self, coefficients=0.5, p=0.3):
        self.coefficients, self.p = coefficients, p

    def __call__(self, x):
        if random.random() > self.p: return x
        D, H, W = x.shape[-3:]
        field = torch.randn(1, 1, 4, 4, 4) * self.coefficients + 1.0
        field = F.interpolate(field, size=(D, H, W), mode='trilinear', align_corners=False)
        return x * field.squeeze(0).squeeze(0)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms: x = t(x)
        return x


def build_train_augmentation(config: dict = None) -> Compose:
    """Legacy per-sample pipeline builder (kept for backward compat)."""
    if config is None:
        config = {}
    if not config.get('enabled', True):
        return Compose([])

    transforms = []

    af = config.get('random_affine', {})
    if af.get('enabled', True):
        transforms.append(RandomAffine3D(
            rotation=af.get('rotation', 5.0), translation=af.get('translation', 3.0),
            scale=tuple(af.get('scale', [0.97, 1.03])), p=af.get('p', 1.0)))

    sc = config.get('intensity_scale', {})
    if isinstance(sc, (list, tuple)): sc = {'range': list(sc)}
    if sc.get('enabled', True):
        transforms.append(RandomIntensityScale(
            scale=tuple(sc.get('range', [0.9, 1.1])), p=sc.get('p', 1.0)))

    sh = config.get('intensity_shift', {})
    if isinstance(sh, (list, tuple)): sh = {'range': list(sh)}
    if sh.get('enabled', True):
        transforms.append(RandomIntensityShift(
            shift=tuple(sh.get('range', [-0.1, 0.1])), p=sh.get('p', 1.0)))

    gn = config.get('gaussian_noise', {})
    gn_range = tuple(gn['std']) if isinstance(gn, dict) and 'std' in gn else (0.01, 0.03)
    gn_enabled = gn.get('enabled', True) if isinstance(gn, dict) else True
    if gn_enabled:
        transforms.append(GaussianNoise(
            std=gn_range, p=gn.get('p', 1.0) if isinstance(gn, dict) else 1.0))

    bf = config.get('bias_field', {})
    if bf.get('enabled', True):
        transforms.append(RandomBiasField(
            coefficients=bf.get('coefficients', 0.5), p=bf.get('probability', 0.3)))

    return Compose(transforms)


# ═══════════════════════════════════════════════════════════
#  Batch-level transforms  (vectorised — production path)
# ═══════════════════════════════════════════════════════════

def _batch_random_affine(
    x: torch.Tensor,
    rotation: float = 5.0,
    translation: float = 3.0,
    scale: tuple = (0.97, 1.03),
) -> torch.Tensor:
    """Batch-vectorised 3D affine.

    Args:
        x: (B, D, H, W)
    Returns:
        (B, D, H, W) with per-sample random affine.
    """
    B, D, H, W = x.shape
    deg2rad = np.pi / 180.0

    # --- per-sample random parameters (vectorised) ---
    rx = (torch.rand(B) * 2 - 1) * rotation * deg2rad
    ry = (torch.rand(B) * 2 - 1) * rotation * deg2rad
    rz = (torch.rand(B) * 2 - 1) * rotation * deg2rad

    tx = (torch.rand(B) * 2 - 1) * translation / (W / 2.0)
    ty = (torch.rand(B) * 2 - 1) * translation / (H / 2.0)
    tz = (torch.rand(B) * 2 - 1) * translation / (D / 2.0)

    s_min, s_max = scale
    scales = torch.rand(B) * (s_max - s_min) + s_min

    # --- per-sample rotation matrices (B, 3, 3) ---
    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)

    one = torch.ones(B); zero = torch.zeros(B)

    Rx = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, cos_x, -sin_x], dim=1),
        torch.stack([zero, sin_x, cos_x], dim=1),
    ], dim=1)  # (B, 3, 3)

    Ry = torch.stack([
        torch.stack([cos_y, zero, sin_y], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-sin_y, zero, cos_y], dim=1),
    ], dim=1)

    Rz = torch.stack([
        torch.stack([cos_z, -sin_z, zero], dim=1),
        torch.stack([sin_z, cos_z, zero], dim=1),
        torch.stack([zero, zero, one], dim=1),
    ], dim=1)

    R = Rz @ Ry @ Rx                               # (B, 3, 3)

    # scale matrix
    S = torch.diag_embed(scales.unsqueeze(-1).expand(-1, 3))  # (B, 3, 3)
    A = R @ S                                       # (B, 3, 3)

    # --- build (B, 3, 4) theta for affine_grid ---
    theta = torch.zeros(B, 3, 4)
    theta[:, :, :3] = A
    theta[:, 0, 3] = tx
    theta[:, 1, 3] = ty
    theta[:, 2, 3] = tz

    # single batched grid_sample
    x_in = x.unsqueeze(1)                           # (B, 1, D, H, W)
    grid = F.affine_grid(theta, x_in.size(), align_corners=False)
    out = F.grid_sample(x_in, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=False)
    return out.squeeze(1)                           # (B, D, H, W)


def _batch_intensity_scale(x: torch.Tensor, scale_range=(0.9, 1.1)) -> torch.Tensor:
    B = x.shape[0]
    factors = torch.empty(B, 1, 1, 1).uniform_(*scale_range)
    return x * factors


def _batch_intensity_shift(x: torch.Tensor, shift_range=(-0.1, 0.1)) -> torch.Tensor:
    B = x.shape[0]
    offsets = torch.empty(B, 1, 1, 1).uniform_(*shift_range)
    return x + offsets


def _batch_gaussian_noise(x: torch.Tensor, std_range=(0.01, 0.03)) -> torch.Tensor:
    B = x.shape[0]
    stds = torch.empty(B, 1, 1, 1).uniform_(*std_range)
    noise = torch.randn_like(x) * stds
    return x + noise


def _batch_bias_field(x: torch.Tensor, coefficients=0.5) -> torch.Tensor:
    """Per-sample bias fields, all vectorised in one interpolate call."""
    B, D, H, W = x.shape
    # (B, 1, 4, 4, 4) → (B, 1, D, H, W)
    field = torch.randn(B, 1, 4, 4, 4) * coefficients + 1.0
    field = F.interpolate(field, size=(D, H, W),
                          mode='trilinear', align_corners=False)
    return x * field.squeeze(1)


# ═══════════════════════════════════════════════════════════
#  BatchAugmentation  —  the main callable
# ═══════════════════════════════════════════════════════════

class BatchAugmentation:
    """Apply all configured augmentations to a collated batch at once.

    Input / output:  dict with 'image' key as (B, D, H, W) tensor.
    All intensity transforms are applied **after** affine so that
    they operate on already-warped voxels.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        cfg = config

        self.do_affine = cfg.get('random_affine', {}).get('enabled', True)
        if self.do_affine:
            af = cfg['random_affine']
            self.affine_rotation = af.get('rotation', 5.0)
            self.affine_translation = af.get('translation', 3.0)
            self.affine_scale = tuple(af.get('scale', [0.97, 1.03]))

        sc = cfg.get('intensity_scale', {})
        if isinstance(sc, (list, tuple)): sc = {'range': list(sc)}
        self.do_scale = sc.get('enabled', True)
        if self.do_scale:
            self.scale_range = tuple(sc.get('range', [0.9, 1.1]))

        sh = cfg.get('intensity_shift', {})
        if isinstance(sh, (list, tuple)): sh = {'range': list(sh)}
        self.do_shift = sh.get('enabled', True)
        if self.do_shift:
            self.shift_range = tuple(sh.get('range', [-0.1, 0.1]))

        gn = cfg.get('gaussian_noise', {})
        self.do_noise = gn.get('enabled', True) if isinstance(gn, dict) else True
        if self.do_noise:
            self.noise_std = tuple(gn['std']) if isinstance(gn, dict) and 'std' in gn else (0.01, 0.03)

        bf = cfg.get('bias_field', {})
        self.do_bias = bf.get('enabled', True)
        if self.do_bias:
            self.bias_prob = bf.get('probability', 0.3)
            self.bias_coeff = bf.get('coefficients', 0.5)

        # Also parse intensity scale/shift from flat format for backward compat
        if 'intensity_scale' in cfg and isinstance(cfg['intensity_scale'], (list, tuple)):
            self.do_scale = True
            self.scale_range = tuple(cfg['intensity_scale'])
        if 'intensity_shift' in cfg and isinstance(cfg['intensity_shift'], (list, tuple)):
            self.do_shift = True
            self.shift_range = tuple(cfg['intensity_shift'])

    def __call__(self, batch: dict) -> dict:
        """Apply augmentations to the 'image' tensor in *batch*."""
        imgs = batch['image']  # (B, D, H, W)

        # 1. affine (geometric — do first so intensity follows warp)
        if self.do_affine:
            imgs = _batch_random_affine(
                imgs, self.affine_rotation, self.affine_translation, self.affine_scale)

        # 2. intensity scale
        if self.do_scale:
            imgs = _batch_intensity_scale(imgs, self.scale_range)

        # 3. intensity shift
        if self.do_shift:
            imgs = _batch_intensity_shift(imgs, self.shift_range)

        # 4. bias field (multiplicative, before additive noise)
        if self.do_bias:
            # per-sample probability
            B = imgs.shape[0]
            mask = torch.rand(B) < self.bias_prob
            if mask.any():
                imgs[mask] = _batch_bias_field(imgs[mask], self.bias_coeff)

        # 5. gaussian noise (additive — last)
        if self.do_noise:
            imgs = _batch_gaussian_noise(imgs, self.noise_std)

        batch['image'] = imgs
        return batch


def make_augmentation_collate_fn(config: dict = None):
    """Return a collate_fn for DataLoader that applies batch augmentation.

    Usage::

        aug = make_augmentation_collate_fn(config.get('augmentation'))
        train_dl = DataLoader(train_ds, ..., collate_fn=aug)

    When *config* is None or ``enabled: false``, returns the PyTorch
    default collate function (no augmentation).
    """
    if config is None or not config.get('enabled', False):
        from torch.utils.data import default_collate
        return default_collate

    aug = BatchAugmentation(config)
    from torch.utils.data import default_collate as _dc

    def collate_fn(samples):
        batch = _dc(samples)
        return aug(batch)

    return collate_fn
