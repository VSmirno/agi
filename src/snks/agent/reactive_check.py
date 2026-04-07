"""Stage 71: ReactiveCheck — reactive behavior from causal rules.

Checks if current perception matches a danger rule (e.g. zombie nearby)
and returns action override (attack or flee). Integrates into ScenarioRunner
as a priority layer above planned actions.

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

FLEE_STEPS = 4
_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


class ReactiveCheck:
    """Check for reactive rules (danger → attack/flee).

    Usage:
        rc = ReactiveCheck(store)
        override = rc.check("zombie", {"wood_sword": 1})
        # override = "do" (attack) or "flee" or None (continue plan)
    """

    def __init__(self, store: Any) -> None:  # ConceptStore
        self.store = store

    def check(self, near: str, inventory: dict[str, int]) -> str | None:
        """Check if current near object triggers a reactive rule.

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

    @staticmethod
    def flee_action(
        env: Any,
        rng: np.random.RandomState,
        steps: int = FLEE_STEPS,
    ) -> None:
        """Move away from danger for a few steps.

        Simple heuristic: random direction, avoid last direction (don't walk back).
        """
        last_dir: str | None = None
        for _ in range(steps):
            options = [d for d in _DIRECTIONS if d != last_dir]
            direction = options[rng.randint(0, len(options))]
            env.step(direction)
            # Reverse of current direction = walking back
            opposites = {
                "move_up": "move_down",
                "move_down": "move_up",
                "move_left": "move_right",
                "move_right": "move_left",
            }
            last_dir = opposites.get(direction)
