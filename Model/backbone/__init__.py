"""Backbone package with opportunistic exports.

Submodules that need optional third-party deps (for example einops) must not
prevent Dual-Shift from importing ``BasicBlock`` / journal ResNet helpers.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = []


def _try_export(module_name: str, names: tuple[str, ...]) -> None:
    try:
        module = import_module(module_name, package=__name__)
    except Exception:
        return
    for name in names:
        try:
            globals()[name] = getattr(module, name)
            __all__.append(name)
        except AttributeError:
            continue


_try_export(".resnet3d", ("resnet18",))
_try_export(".vit3d", ("ViTBackbone",))
_try_export(
    ".preact_resnet",
    (
        "PreactResNet3D",
        "PreactResBlock",
        "preact_resnet_ut",
        "preact_resnet_t",
        "preact_resnet_s",
    ),
)
_try_export(
    ".film_backbone",
    (
        "ResNetFilmBackbone",
        "BasicBlock",
        "resnet10_film",
        "resnet18_film",
        "resnet34_film",
        "resnet_light_film",
        "resnet_tiny_film",
        "resnet10_ce_only",
    ),
)
_try_export(
    ".backdoor_backbone",
    (
        "ResNetBackdoorBackbone",
        "ConfounderEncoder",
        "ClassAttentionHead",
        "resnet10_backdoor",
        "resnet18_backdoor",
        "resnet34_backdoor",
        "spatial_to_patches",
        "patches_to_spatial",
    ),
)
_try_export(
    ".disentangled_backbone",
    ("ResNetDisentangledBackbone", "resnet10_disentangled", "resnet18_disentangled"),
)
_try_export(
    ".daft_backbone",
    ("ResNetDAFTBackbone", "resnet10_daft", "resnet18_daft", "resnet34_daft"),
)
_try_export(
    ".sca_capm_backbone",
    (
        "ResNetSCACAPMBackbone",
        "resnet18_sca",
        "resnet18_sc",
        "resnet18_csra",
        "resnet18_var",
        "default_adni_var_specs",
    ),
)
_try_export(
    ".vit_film_backbone",
    ("ViTFiLMBackbone", "ViTEncoder3D", "vit_tiny_film", "vit_small_film"),
)
_try_export(
    ".vit_daft_backbone",
    ("ViTDAFTBackbone", "vit_tiny_daft", "vit_small_daft"),
)
_try_export(
    ".vit_backdoor_backbone",
    ("ViTBackdoorBackbone", "vit_tiny_backdoor", "vit_small_backdoor"),
)
_try_export(
    ".journal_resnet",
    (
        "JournalResNet3D",
        "LateStageSpatialModulation",
        "journal_resnet10",
        "journal_resnet18",
        "default_journal_var_specs",
    ),
)


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
