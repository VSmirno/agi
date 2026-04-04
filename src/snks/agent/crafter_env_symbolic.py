"""Stage 64: Symbolic Crafter environment for curiosity exploration.

Simulates Crafter tech tree without rendering. CRAFTER_RULES is the hidden
ground truth — the agent never sees rules directly, only observes outcomes.
"""

from __future__ import annotations

import random

from snks.agent.crafter_trainer import CRAFTER_RULES, CRAFTER_FAILURES


class CrafterSymbolicEnv:
    """Symbolic Crafter tech tree for curiosity exploration."""

    ALL_NEARBY = [
        "tree", "stone", "coal", "iron", "diamond",
        "water", "cow", "table", "furnace", "empty",
    ]

    ALL_ACTIONS = [
        "do", "place_table", "place_furnace", "place_plant",
        "place_stone", "make_wood_pickaxe", "make_stone_pickaxe",
        "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
        "make_iron_sword",
    ]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._all_rules = CRAFTER_RULES
        self._all_failures = CRAFTER_FAILURES
        self.inventory: dict[str, int] = {}
        self._nearby_order: list[str] = []
        self._nearby_idx = 0
        self.reset()

    def reset(self) -> dict[str, str]:
        """Reset inventory and randomize nearby target order."""
        self.inventory = {}
        self._nearby_order = list(self.ALL_NEARBY)
        self._rng.shuffle(self._nearby_order)
        self._nearby_idx = 0
        return self.observe()

    def next_target(self) -> None:
        """Cycle to next nearby target."""
        self._nearby_idx = (self._nearby_idx + 1) % len(self._nearby_order)

    @property
    def current_nearby(self) -> str:
        return self._nearby_order[self._nearby_idx]

    def observe(self) -> dict[str, str]:
        """Current situation as dict (what agent sees)."""
        situation: dict[str, str] = {
            "domain": "crafter",
            "near": self.current_nearby,
        }
        for item, count in sorted(self.inventory.items()):
            if count > 0:
                situation[f"has_{item}"] = str(count)
        return situation

    def available_actions(self) -> list[str]:
        """All possible actions (agent doesn't know which will succeed)."""
        return list(self.ALL_ACTIONS)

    def step(self, action: str) -> tuple[dict[str, str], float]:
        """Execute action. Returns (outcome, reward).

        On success: consume required items, add produced item.
        On failure: no inventory change.
        """
        situation = self.observe()

        # Check success rules
        for rule in self._all_rules:
            if self._matches_success(rule, action, situation):
                # Consume requirements
                for item, count in rule.get("requires", {}).items():
                    self.inventory[item] = self.inventory.get(item, 0) - count
                    if self.inventory[item] <= 0:
                        del self.inventory[item]
                # Produce item
                gives = rule["gives"]
                self.inventory[gives] = self.inventory.get(gives, 0) + 1
                return {"result": rule["result"], "gives": gives}, 1.0

        # Check failure rules
        for fail in self._all_failures:
            if self._matches_failure(fail, action, situation):
                return {"result": fail["result"]}, -1.0

        return {"result": "nothing_happened"}, 0.0

    def _matches_success(self, rule: dict, action: str,
                         situation: dict[str, str]) -> bool:
        """Check if a success rule matches current action + situation."""
        if rule["action"] != action:
            return False
        if rule["near"] != situation.get("near", "empty"):
            return False
        # Check all required items are in inventory
        for item, count in rule.get("requires", {}).items():
            has = int(situation.get(f"has_{item}", "0"))
            if has < count:
                return False
        return True

    def _matches_failure(self, fail: dict, action: str,
                         situation: dict[str, str]) -> bool:
        """Check if a failure rule matches (action + missing requirement)."""
        if fail["action"] != action:
            return False
        if fail["near"] != situation.get("near", "empty"):
            return False
        # Failure fires when the required item is MISSING
        missing = fail["requires_missing"]
        has = int(situation.get(f"has_{missing}", "0"))
        return has <= 0
