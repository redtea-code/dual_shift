"""Model package entry.

Legacy modules are imported opportunistically so Dual-Shift / APIS v2 code can
load even when older research modules are absent from the checkout.
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


_try_export(
    ".adpc",
    ("ADPC6_2", "ADPC6_4", "ADPC6_2_VIT"),
)
_try_export(".dfiv", ("DFIV", "DFIV_train"))
_try_export(
    ".freq_mask",
    (
        "FAMC3DWrapper",
        "create_famc3d_adpc2",
        "create_famc3d_adpc4",
        "DualBranchClassifier",
        "MSFAM3DWrapper",
        "PatchFreqClassifier3D",
        "LearnableDirectionalMask",
        "adversarial_artifact_augment",
        "feature_consistency_loss3d",
        "mask_budget_loss3d",
        "structural_protection_loss3d",
        "directional_tv_loss3d",
        "ADNIFrequencyLoss",
    ),
)
_try_export(
    ".backbone",
    (
        "ResNetFilmBackbone",
        "ResNetBackdoorBackbone",
        "ResNetDAFTBackbone",
        "ResNetSCACAPMBackbone",
        "ResNetEvidenceCalibratedCAPMBackbone",
        "resnet10_film",
        "resnet18_film",
        "resnet34_film",
        "resnet10_backdoor",
        "resnet18_backdoor",
        "resnet34_backdoor",
        "resnet10_daft",
        "resnet18_daft",
        "resnet34_daft",
        "resnet10_disentangled",
        "resnet18_disentangled",
        "resnet18_sca",
        "resnet18_sc",
        "resnet18_csra",
        "resnet18_var",
        "resnet18_ie_capm",
        "ViTFiLMBackbone",
        "ViTDAFTBackbone",
        "ViTBackdoorBackbone",
        "vit_tiny_film",
        "vit_small_film",
        "vit_tiny_daft",
        "vit_small_daft",
        "vit_tiny_backdoor",
        "vit_small_backdoor",
    ),
)
_try_export(
    ".backbone.journal_resnet",
    (
        "JournalResNet3D",
        "LateStageSpatialModulation",
        "journal_resnet10",
        "journal_resnet18",
    ),
)


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
