"""Stage 71: ReactiveCheck — reactive behavior from causal rules.

Checks for:
1. Danger rules (zombie nearby → attack/flee)
2. Survival needs (food/drink/energy low → seek resource)

Priority: danger > survival > plan.

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

FLEE_STEPS = 4
_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]

# Survival thresholds — act when stat drops below this
FOOD_THRESHOLD = 4
DRINK_THRESHOLD = 4
ENERGY_THRESHOLD = 3


class ReactiveCheck:
    """Check for reactive rules (danger + survival needs).

    Usage:
        rc = ReactiveCheck(store)

        # Danger check
        override = rc.check("zombie", {"wood_sword": 1})
        # "do" (attack) or "flee" or None

        # Survival check
        need = rc.check_needs({"food": 2, "drink": 8, "energy": 5})
        # ("food", "cow") or ("drink", "water") or ("energy", "_sleep") or None
    """

    def __init__(self, store: Any) -> None:  # ConceptStore
        self.store = store

    def check(self, near: str, inventory: dict[str, int]) -> str | None:
        """Check if current near object triggers a danger reactive rule.

        Returns:
            "do" — attack (have required weapon)
            "flee" — run away (no weapon)
            None — no reactive rule, continue plan
        """
        concept = self.store.query_text(near)
        if concept is None:
            return None

        if not concept.attributes.get("dangerous", False):
            return None

        # Check if we can fight (have required weapon)
        combat_link = concept.find_causal(action="do", check_requires=inventory)
        if combat_link is not None:
            return "do"

        return "flee"

    def check_needs(
        self, inventory: dict[str, int]
    ) -> tuple[str, str] | None:
        """Check survival needs from inventory stats.

        Returns:
            (need_name, target_concept) — e.g. ("food", "cow"), ("drink", "water")
            or None if all needs satisfied.

        Priority: lowest stat first.
        """
        needs: list[tuple[int, str, str]] = []

        food = inventory.get("food", 9)
        if food < FOOD_THRESHOLD:
            needs.append((food, "food", "cow"))

        drink = inventory.get("drink", 9)
        if drink < DRINK_THRESHOLD:
            needs.append((drink, "drink", "water"))

        energy = inventory.get("energy", 9)
        if energy < ENERGY_THRESHOLD:
            needs.append((energy, "energy", "_sleep"))

        if not needs:
            return None

        # Most urgent first (lowest value)
        needs.sort(key=lambda x: x[0])
        _, need_name, target = needs[0]
        return (need_name, target)

    def check_all(
        self, near: str, inventory: dict[str, int]
    ) -> dict[str, Any]:
        """Combined check: danger + survival needs.

        Returns dict with:
            action: "do" | "flee" | "seek" | "sleep" | None
            target: concept to seek (for "seek") or None
            need: need name (for "seek"/"sleep") or None
            reason: "danger" | "survival" | None
        """
        # Priority 1: danger (immediate threat)
        danger = self.check(near, inventory)
        if danger is not None:
            return {
                "action": danger,
                "target": None,
                "need": None,
                "reason": "danger",
            }

        # Priority 2: survival needs
        need = self.check_needs(inventory)
        if need is not None:
            need_name, target = need
            if target == "_sleep":
                return {
                    "action": "sleep",
                    "target": None,
                    "need": need_name,
                    "reason": "survival",
                }

            # Check if resource is already nearby
            if near == target:
                return {
                    "action": "do",
                    "target": target,
                    "need": need_name,
                    "reason": "survival",
                }

            return {
                "action": "seek",
                "target": target,
                "need": need_name,
                "reason": "survival",
            }

        return {"action": None, "target": None, "need": None, "reason": None}

    @staticmethod
    def flee_action(
        env: Any,
        rng: np.random.RandomState,
        steps: int = FLEE_STEPS,
    ) -> None:
        """Move away from danger for a few steps."""
        last_dir: str | None = None
        for _ in range(steps):
            options = [d for d in _DIRECTIONS if d != last_dir]
            direction = options[rng.randint(0, len(options))]
            env.step(direction)
            opposites = {
                "move_up": "move_down",
                "move_down": "move_up",
                "move_left": "move_right",
                "move_right": "move_left",
            }
            last_dir = opposites.get(direction)
