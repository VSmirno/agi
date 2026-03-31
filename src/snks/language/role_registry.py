"""Role Registry: structural role mappings between SKS predicates (Stage 28).

Defines analogical roles that map source predicates to target predicates.
Used by AnalogicalReasoner to detect structural similarity between skills.
"""

from __future__ import annotations

from snks.language.grid_perception import (
    SKS_KEY_PRESENT,
    SKS_KEY_HELD,
    SKS_DOOR_LOCKED,
    SKS_DOOR_OPEN,
)
from snks.language.grid_perception import (
    SKS_CARD_PRESENT,
    SKS_CARD_HELD,
    SKS_GATE_LOCKED,
    SKS_GATE_OPEN,
)

# Structural roles: role_name → (source_sks, target_sks)
# Source: key/door world. Target: card/gate world.
ROLE_REGISTRY: dict[str, tuple[int, int]] = {
    "instrument_present": (SKS_KEY_PRESENT, SKS_CARD_PRESENT),
    "instrument_held":    (SKS_KEY_HELD,    SKS_CARD_HELD),
    "blocker_locked":     (SKS_DOOR_LOCKED, SKS_GATE_LOCKED),
    "blocker_open":       (SKS_DOOR_OPEN,   SKS_GATE_OPEN),
}

# Word mapping: source word → target word
WORD_ROLE_MAPPING: dict[str, str] = {
    "key":  "card",
    "door": "gate",
}

# Reverse: source_sks → target_sks
SOURCE_TO_TARGET_SKS: dict[int, int] = {
    src: tgt for _, (src, tgt) in ROLE_REGISTRY.items()
}

# Reverse word: target → source
TARGET_TO_SOURCE_WORD: dict[str, str] = {
    v: k for k, v in WORD_ROLE_MAPPING.items()
}
