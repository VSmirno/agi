"""Stage 64: Symbolic MiniGrid environment for curiosity exploration.

Uses _compute_transition() from world_model_trainer as ground truth.
Agent explores object interactions without needing a full MiniGrid env.
"""

from __future__ import annotations

import random

from snks.agent.world_model_trainer import (
    _compute_transition, Transition,
    OBJ_TYPES, COLORS, ACTIONS, CARRYABLE, DOOR_STATES,
)


class MiniGridSymbolicEnv:
    """Symbolic MiniGrid interaction simulator."""

    def __init__(self, colors: list[str] | None = None, seed: int = 42):
        self._rng = random.Random(seed)
        self.colors = colors or COLORS
        self._reset_state()

    def _reset_state(self) -> None:
        self.facing_obj = "empty"
        self.obj_color = ""
        self.obj_state = "none"
        self.carrying = "nothing"
        self.carrying_color = ""

    def reset(self) -> dict[str, str]:
        """Reset and place random object in front of agent."""
        self._reset_state()
        self._randomize_facing()
        return self.observe()

    def _randomize_facing(self) -> None:
        """Place a random object in front of the agent."""
        self.facing_obj = self._rng.choice(OBJ_TYPES)
        self.obj_color = self._rng.choice(self.colors)
        if self.facing_obj == "door":
            self.obj_state = self._rng.choice(DOOR_STATES)
        elif self.facing_obj in ("empty", "goal"):
            self.obj_color = ""
            self.obj_state = "none"
        else:
            self.obj_state = "none"

    def observe(self) -> dict[str, str]:
        return {
            "facing_obj": self.facing_obj,
            "obj_color": self.obj_color,
            "obj_state": self.obj_state,
            "carrying": self.carrying,
            "carrying_color": self.carrying_color,
        }

    def available_actions(self) -> list[str]:
        return list(ACTIONS)

    def step(self, action: str) -> tuple[dict[str, str], float]:
        """Execute action using MiniGrid physics ground truth."""
        t = _compute_transition(
            self.facing_obj, self.obj_color, self.obj_state,
            self.carrying, self.carrying_color, action,
        )
        if t is None:
            return {"result": "nothing_happened"}, 0.0

        # Apply state changes
        outcome = t.outcome
        result = outcome.get("result", "")

        if result == "picked_up":
            self.carrying = self.facing_obj
            self.carrying_color = self.obj_color
            self.facing_obj = "empty"
            self.obj_color = ""
            self.obj_state = "none"
        elif result == "dropped":
            self.facing_obj = self.carrying
            self.obj_color = self.carrying_color
            self.carrying = "nothing"
            self.carrying_color = ""
        elif result in ("door_opened", "door_unlocked"):
            self.obj_state = "open"
            if result == "door_unlocked":
                self.carrying = "nothing"
                self.carrying_color = ""
        elif result == "door_closed":
            self.obj_state = "closed"
        elif result == "moved":
            self._randomize_facing()

        return outcome, t.reward

    def set_scenario(self, facing_obj: str, obj_color: str = "",
                     obj_state: str = "none",
                     carrying: str = "nothing",
                     carrying_color: str = "") -> None:
        """Set specific scenario for directed exploration."""
        self.facing_obj = facing_obj
        self.obj_color = obj_color or (self._rng.choice(self.colors)
                                        if facing_obj not in ("empty", "goal")
                                        else "")
        self.obj_state = obj_state
        self.carrying = carrying
        self.carrying_color = carrying_color
