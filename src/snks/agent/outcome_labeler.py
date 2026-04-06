"""Stage 69: OutcomeLabeler — infer near_label from inventory changes.

No info["semantic"] used. Only info["inventory"] (proprioception) + action name.

Principle: if "do" action causes wood+1, the player was near a tree.
This is outcome-supervised labeling — the world tells us via its state changes.
"""

from __future__ import annotations

# inventory GAIN after "do" → agent was near this object
DO_GAIN_TO_NEAR: dict[str, str] = {
    "wood": "tree",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    # "drink"/"food" are buffs, not inventory items — not detectable this way
}

# make_* action name → near was "table" (if action succeeded and item was gained)
MAKE_GAIN_TO_NEAR: dict[str, str] = {
    "make_wood_pickaxe": "table",
    "make_stone_pickaxe": "table",
    "make_iron_pickaxe": "table",
    "make_wood_sword": "table",
    "make_stone_sword": "table",
    "make_iron_sword": "table",
}

# inventory items that DECREASE for each place_* action → agent was near "empty"
PLACE_ACTION_COST: dict[str, dict[str, int]] = {
    "place_table": {"wood": 2},
    "place_furnace": {"stone": 4},
    "place_stone": {"stone": 1},
    "place_plant": {"sapling": 1},
}


def inv_diff(
    before: dict[str, int], after: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute inventory gains and losses.

    Returns (gains, losses) where gains[item]=n means item increased by n,
    losses[item]=n means item decreased by n.
    """
    all_keys = set(before) | set(after)
    gains: dict[str, int] = {}
    losses: dict[str, int] = {}
    for k in all_keys:
        delta = after.get(k, 0) - before.get(k, 0)
        if delta > 0:
            gains[k] = delta
        elif delta < 0:
            losses[k] = -delta
    return gains, losses


class OutcomeLabeler:
    """Infers near_label from inventory changes.

    Uses only proprioceptive info: inventory before/after + action name.
    No info["semantic"] required.

    Coverage: 15/17 Crafter rules (water/cow skipped — give buffs, not items).
    """

    def label(
        self,
        action: str,
        inv_before: dict[str, int],
        inv_after: dict[str, int],
    ) -> str | None:
        """Infer near_label from action outcome.

        Args:
            action: action name string (e.g. "do", "make_wood_pickaxe").
            inv_before: inventory before action.
            inv_after: inventory after action.

        Returns:
            near_str (e.g. "tree", "table", "empty") or None if unrecognizable.
        """
        gains, losses = inv_diff(inv_before, inv_after)

        if action == "do":
            # Check what was gained — maps to what was nearby
            for item, near in DO_GAIN_TO_NEAR.items():
                if gains.get(item, 0) > 0:
                    return near
            return None

        if action in MAKE_GAIN_TO_NEAR:
            # make_* succeeded → gained the crafted item → near was table
            item = action[len("make_"):]  # "make_wood_pickaxe" → "wood_pickaxe"
            if gains.get(item, 0) > 0:
                return "table"
            return None

        if action in PLACE_ACTION_COST:
            # place_* succeeded → lost the required items → near was empty
            cost = PLACE_ACTION_COST[action]
            if all(losses.get(k, 0) >= v for k, v in cost.items()):
                return "empty"
            return None

        return None
