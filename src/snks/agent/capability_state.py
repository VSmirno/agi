"""Capability state derived from current inventory and textbook facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snks.agent.crafter_textbook import CrafterTextbook


@dataclass(frozen=True)
class CapabilityState:
    """Compact affordance summary exposed to goal selection and traces."""

    armed_melee: bool = False
    weapons: tuple[str, ...] = ()

    def to_trace(self) -> dict:
        return {
            "armed_melee": bool(self.armed_melee),
            "weapons": list(self.weapons),
        }


def extract_capability_state(
    inventory: dict[str, int],
    textbook: "CrafterTextbook | None" = None,
) -> CapabilityState:
    """Derive capabilities from inventory using textbook categories when present."""
    weapon_items = _weapon_items(textbook)
    owned = tuple(
        sorted(
            item for item in weapon_items
            if int(inventory.get(item, 0)) > 0
        )
    )
    return CapabilityState(
        armed_melee=bool(owned),
        weapons=owned,
    )


def _weapon_items(textbook: "CrafterTextbook | None") -> set[str]:
    if textbook is None:
        return set()
    weapons: set[str] = set()
    for entry in textbook.vocabulary:
        if not isinstance(entry, dict):
            continue
        if entry.get("category") == "weapon" and isinstance(entry.get("id"), str):
            weapons.add(str(entry["id"]))
    return weapons
