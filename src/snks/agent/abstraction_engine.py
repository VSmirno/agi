"""Stage 63: Abstraction Engine — discovers categories from neocortex rules.

Scans concrete rules in neocortex and groups objects by their behavior
patterns (action → outcome). Objects that behave the same way for the
same action belong to the same category.

Categories are multi-label: key = carryable + solid + activator.

Abstract rules are encoded in SDM via VSA:
  bind(category_vec, action_vec) → outcome_vec

This gives ~20-50 abstract rules that generalize to unseen objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from typing import TYPE_CHECKING

from snks.agent.vsa_world_model import SDMMemory, VSACodebook

if TYPE_CHECKING:
    from snks.agent.cls_world_model import Rule


# Well-known category names (auto-discovered, named for readability)
CATEGORY_NAMES = {
    ("pickup_empty", "picked_up"): "carryable",
    ("pickup_carrying", "failed_carrying"): "carryable_blocked",
    ("pickup_empty", "nothing_to_pickup"): "not_carryable",
    ("pickup_carrying", "nothing_to_pickup"): "not_carryable_carrying",
    ("toggle_closed", "door_opened"): "openable",
    ("toggle_locked_withkey", "door_unlocked"): "unlockable_withkey",
    ("toggle_locked_nokey", "door_still_locked"): "locked_no_key",
    ("toggle_open", "door_closed"): "closeable",
    ("toggle", "nothing_happened"): "not_toggleable",
    ("forward", "blocked"): "solid",
    ("forward", "moved"): "passable",
    ("drop", "dropped"): "droppable_target",
    ("drop", "nothing_to_drop"): "nothing_carried",
    ("drop", "drop_blocked"): "drop_blocked",
}


@dataclass
class Category:
    """An auto-discovered object category."""
    name: str
    action: str
    outcome: str
    members: set[str] = field(default_factory=set)


class AbstractionEngine:
    """Discovers categories from neocortex rules and builds abstract SDM."""

    def __init__(self, codebook: VSACodebook, dim: int = 2048,
                 n_locations: int = 1000, n_amplify: int = 10,
                 seed: int = 500):
        self.codebook = codebook
        self.n_amplify = n_amplify
        self.categories: dict[str, Category] = {}

        device = codebook.device
        self.sdm = SDMMemory(
            n_locations=n_locations, dim=dim,
            seed=seed, device=device,
        )
        self._zeros = torch.zeros(dim, device=device)

    def discover_categories(self, neocortex: dict[str, Rule]) -> dict[str, Category]:
        """Scan neocortex rules and group objects by (action, outcome) patterns."""
        # Group: (action, outcome_result) → set of objects
        patterns: dict[tuple[str, str], set[str]] = {}

        for key, rule in neocortex.items():
            # Parse situation key: "facing_color_state_carrying_carrycolor_action"
            parts = key.split("_")
            action = parts[-1]
            facing_obj = parts[0]
            # Extract state (3rd field) for state-sensitive categories
            obj_state = parts[2] if len(parts) > 2 else "none"

            result = rule.outcome.get("result", "")
            if not result:
                continue

            # Include state for toggle actions (locked vs closed matters)
            # Include carrying type for actions where it matters
            carrying = parts[3] if len(parts) > 3 else "nothing"

            if action == "toggle" and obj_state in ("locked", "closed", "open"):
                # For locked doors, carrying key vs not matters
                if obj_state == "locked" and carrying == "key":
                    pattern_key = (f"toggle_locked_withkey", result)
                elif obj_state == "locked":
                    pattern_key = (f"toggle_locked_nokey", result)
                else:
                    pattern_key = (f"toggle_{obj_state}", result)
            elif action == "pickup" and carrying != "nothing":
                pattern_key = ("pickup_carrying", result)
            elif action == "pickup":
                pattern_key = ("pickup_empty", result)
            else:
                pattern_key = (action, result)
            patterns.setdefault(pattern_key, set()).add(facing_obj)

        # Create named categories
        self.categories = {}
        for (action, result), members in patterns.items():
            name = CATEGORY_NAMES.get((action, result), f"cat_{action}_{result}")
            self.categories[name] = Category(
                name=name,
                action=action,
                outcome=result,
                members=members,
            )

        return self.categories

    def build_abstract_sdm(self) -> int:
        """Write abstract rules to SDM. Returns number of rules written."""
        n_rules = 0
        for cat in self.categories.values():
            key_vec = self._encode_abstract_key(cat.name, cat.action)
            val_vec = self.codebook.filler(f"abs_outcome_{cat.outcome}")

            for _ in range(self.n_amplify):
                reward = 1.0 if cat.outcome in (
                    "picked_up", "door_opened", "door_unlocked",
                    "moved", "dropped",
                ) else -1.0
                self.sdm.write(key_vec, self._zeros, val_vec, reward)

            n_rules += 1

        return n_rules

    def query_abstract(self, obj_type: str,
                       action: str,
                       obj_state: str = "none",
                       carrying: str = "nothing") -> tuple[str, float]:
        """Query abstract SDM for predicted outcome.

        Finds which category the object belongs to for this action+state,
        then returns the category's outcome.

        Returns (predicted_outcome, confidence).
        """
        # Build context-sensitive action key
        action_key = action
        if action == "toggle" and obj_state == "locked":
            if carrying == "key":
                action_key = "toggle_locked_withkey"
            else:
                action_key = "toggle_locked_nokey"
        elif action == "toggle" and obj_state in ("closed", "open"):
            action_key = f"toggle_{obj_state}"
        elif action == "pickup":
            action_key = "pickup_empty" if carrying == "nothing" else "pickup_carrying"

        # Known categories: direct lookup — lower than neocortex (category, not exact)
        for cat in self.categories.values():
            if cat.action != action_key:
                continue
            if obj_type in cat.members:
                return cat.outcome, 0.80

        # Unknown object: SDM generalization — scaled down
        for cat in self.categories.values():
            if cat.action != action_key:
                continue
            key_vec = self._encode_abstract_key(cat.name, action_key)
            result_vec, conf = self.sdm.read_next(key_vec, self._zeros)
            if conf > 0.01:
                outcome = self._decode_outcome(result_vec)
                return outcome, conf * 0.7

        return "unknown", 0.0

    def query_abstract_reward(self, obj_type: str, action: str,
                              obj_state: str = "none",
                              carrying: str = "nothing") -> float:
        """Query abstract SDM for reward signal."""
        # Build context-sensitive action key (same logic as query_abstract)
        action_key = action
        if action == "toggle" and obj_state == "locked":
            action_key = "toggle_locked_withkey" if carrying == "key" else "toggle_locked_nokey"
        elif action == "toggle" and obj_state in ("closed", "open"):
            action_key = f"toggle_{obj_state}"
        elif action == "pickup":
            action_key = "pickup_empty" if carrying == "nothing" else "pickup_carrying"

        for cat in self.categories.values():
            if cat.action != action_key:
                continue
            if obj_type in cat.members:
                # Use stored reward directly from category
                positive = cat.outcome in (
                    "picked_up", "door_opened", "door_unlocked",
                    "moved", "dropped",
                )
                return 1.0 if positive else -1.0
        return 0.0

    def get_categories_for_object(self, obj_type: str) -> list[str]:
        """Get all categories an object belongs to."""
        return [cat.name for cat in self.categories.values()
                if obj_type in cat.members]

    def assign_to_category(self, obj_type: str, action: str,
                           outcome: str) -> str | None:
        """Assign a new object to a category based on observed behavior."""
        for cat in self.categories.values():
            if cat.action == action and cat.outcome == outcome:
                cat.members.add(obj_type)
                return cat.name
        return None

    def _encode_abstract_key(self, category_name: str,
                             action: str) -> torch.Tensor:
        """Encode abstract SDM key: bind(category, action)."""
        cat_vec = self.codebook.filler(f"abscat_{category_name}")
        act_vec = self.codebook.filler(f"absact_{action}")
        return VSACodebook.bind(cat_vec, act_vec)

    def _decode_outcome(self, vec: torch.Tensor) -> str:
        """Decode SDM output to nearest outcome string."""
        outcomes = [
            "picked_up", "failed_carrying", "nothing_to_pickup",
            "door_opened", "door_unlocked", "door_still_locked",
            "door_closed", "nothing_happened",
            "blocked", "moved",
            "dropped", "nothing_to_drop", "drop_blocked",
        ]
        best = "unknown"
        best_sim = -1.0
        for out in outcomes:
            out_vec = self.codebook.filler(f"abs_outcome_{out}")
            sim = VSACodebook.similarity(vec, out_vec)
            if sim > best_sim:
                best_sim = sim
                best = out
        return best

    def get_stats(self) -> dict:
        return {
            "n_categories": len(self.categories),
            "categories": {
                name: {"action": cat.action, "outcome": cat.outcome,
                       "members": sorted(cat.members)}
                for name, cat in self.categories.items()
            },
            "sdm_writes": self.sdm.n_writes,
        }
