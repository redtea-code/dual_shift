"""Frozen names and modality policy for APIC v3 screening."""
from __future__ import annotations

import hashlib
import json


APIC_V3_VARIANT_SPECS = {
    "ce_x": {"base_variant": "clean", "modalities": "X", "use_demographics": False},
    "mixstyle_x": {
        "base_variant": "mixstyle",
        "modalities": "X",
        "use_demographics": False,
    },
    "apic_v3_x": {
        "base_variant": "v3_style_memory",
        "modalities": "X",
        "use_demographics": False,
    },
    "ce_xd": {"base_variant": "clean", "modalities": "X+D", "use_demographics": True},
    "mixstyle_xd": {
        "base_variant": "mixstyle",
        "modalities": "X+D",
        "use_demographics": True,
    },
    "apic_v3_xd": {
        "base_variant": "v3_style_memory",
        "modalities": "X+D",
        "use_demographics": True,
    },
}
APIC_V3_PRIMARY_VARIANTS = ("ce_x", "mixstyle_x", "apic_v3_x")
APIC_V3_SECONDARY_VARIANTS = ("ce_xd", "mixstyle_xd", "apic_v3_xd")
APIC_V3_SCREENING_VARIANTS = frozenset(
    APIC_V3_PRIMARY_VARIANTS + APIC_V3_SECONDARY_VARIANTS
)


def apic_v3_variant_spec(variant: str) -> dict | None:
    spec = APIC_V3_VARIANT_SPECS.get(str(variant))
    return None if spec is None else dict(spec)


def config_fingerprint(config: dict) -> str:
    def normalise(value):
        if isinstance(value, dict):
            return {str(key): normalise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        return value

    payload = json.dumps(normalise(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
