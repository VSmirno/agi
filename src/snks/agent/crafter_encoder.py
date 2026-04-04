"""Stage 63: Crafter encoder — situation → neocortex key.

Crafter situations have different schema from MiniGrid:
- near: what object/terrain is nearby
- action: craft/collect/place action
- has_X: inventory items
- missing: what's missing for failure cases
"""

from __future__ import annotations


def make_crafter_key(situation: dict[str, str], action: str) -> str:
    """Build neocortex key for Crafter transition.

    Note: 'missing' is NOT part of the key — it's an explanation in the outcome,
    not a situation feature. Query callers don't know what's missing; they only
    know what they have (inventory) and where they are (near).
    """
    near = situation.get("near", "empty")

    # Include key inventory items
    has_parts = []
    for k, v in sorted(situation.items()):
        if k.startswith("has_"):
            has_parts.append(f"{k}={v}")

    has_str = "_".join(has_parts) if has_parts else "noinv"

    return f"crafter_{near}_{has_str}_{action}"
