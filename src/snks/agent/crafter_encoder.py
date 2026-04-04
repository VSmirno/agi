"""Stage 63: Crafter encoder — situation → neocortex key.

Crafter situations have different schema from MiniGrid:
- near: what object/terrain is nearby
- action: craft/collect/place action
- has_X: inventory items
- missing: what's missing for failure cases
"""

from __future__ import annotations


def make_crafter_key(situation: dict[str, str], action: str) -> str:
    """Build neocortex key for Crafter transition."""
    near = situation.get("near", "empty")
    missing = situation.get("missing", "")

    # Include key inventory items
    has_parts = []
    for k, v in sorted(situation.items()):
        if k.startswith("has_"):
            has_parts.append(f"{k}={v}")

    has_str = "_".join(has_parts) if has_parts else "noinv"
    miss_str = f"_miss_{missing}" if missing else ""

    return f"crafter_{near}_{has_str}{miss_str}_{action}"
