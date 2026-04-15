"""Stage 85: GoalSelector — pure-function goal derivation from textbook.

Reads passive/action rules from CrafterTextbook to derive a priority-ordered
threat list. select(state) is a pure function re-evaluated every step.

Design: docs/superpowers/specs/2026-04-15-stage85-goal-selector-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from snks.agent.vector_sim import VectorTrajectory, VectorState
    from snks.agent.crafter_textbook import CrafterTextbook


@dataclass
class Goal:
    id: str
    requirements: dict = field(default_factory=dict)

    def progress(self, trajectory: "VectorTrajectory") -> float:
        """How much did this trajectory advance the goal? Returns float >= 0."""
        if self.id == "fight_zombie":
            return max(0.0, trajectory.vital_delta("health"))
        elif self.id == "fight_skeleton":
            return max(0.0, trajectory.vital_delta("health"))
        elif self.id == "find_cow":
            return max(0.0, trajectory.vital_delta("food"))
        elif self.id == "find_water":
            return max(0.0, trajectory.vital_delta("drink"))
        elif self.id == "sleep":
            return max(0.0, trajectory.vital_delta("energy"))
        elif self.id == "craft_wood_sword":
            return 1.0 if trajectory.item_gained("wood_sword") else 0.0
        elif self.id == "gather_wood":
            return max(0.0, trajectory.inventory_delta("wood"))
        elif self.id == "explore":
            if not trajectory.confidences:
                return 0.0
            # Self-action-only trajectories (sleep) are not exploration.
            # Sleeping in place when goal=explore gives false surprise from
            # textbook prior confidence (0.5), which would beat baseline (0.0).
            if trajectory.plan.steps and all(
                s.target == "self" for s in trajectory.plan.steps
            ):
                return 0.0
            return 1.0 - sum(trajectory.confidences) / len(trajectory.confidences)
        return 0.0


@dataclass
class _Threat:
    """Internal: one entry in the priority-ordered threat list."""
    active_fn: Callable   # (VectorState) -> bool
    response_fn: Callable  # (VectorState) -> Goal


class GoalSelector:
    def __init__(self, textbook: "CrafterTextbook"):
        self._threats = self._derive_threats(textbook)

    def select(self, state: "VectorState") -> Goal:
        """Pure function: current state → active goal. Called every step."""
        for threat in self._threats:
            if threat.active_fn(state):
                return threat.response_fn(state)
        return Goal("explore")

    def _derive_threats(self, textbook: "CrafterTextbook") -> list[_Threat]:
        """Build priority-ordered threat list from textbook passive + action rules.

        Priority order:
          1. Physical threats: entity nearby with negative health passive rule
             → response depends on inventory (has sword → fight, else → craft)
          2. Critical vitals: health < 2 → find_cow (recover health via food)
          3. Low vitals: food < 3 → find_cow, drink < 3 → find_water, energy < 3 → sleep
        """
        threats: list[_Threat] = []

        # 1. Parse passive spatial rules for dangerous entities
        for rule in textbook.rules:
            if rule.get("passive") == "spatial":
                entity = rule.get("entity")
                effect = rule.get("effect", {}) or {}
                body_effect = effect.get("body", {})
                if body_effect.get("health", 0) < 0 and entity:
                    threats.append(self._make_entity_threat(entity, textbook))

        # 2. Critical health
        threats.append(_Threat(
            active_fn=lambda s: s.body.get("health", 9) < 2,
            response_fn=lambda s: Goal("find_cow"),
        ))

        # 3. Low vitals
        for vital, goal_id in [("food", "find_cow"), ("drink", "find_water"), ("energy", "sleep")]:
            threats.append(self._make_vital_threat(vital, goal_id))

        return threats

    @staticmethod
    def _make_entity_threat(entity: str, textbook: "CrafterTextbook") -> _Threat:
        """Build a _Threat for a dangerous entity derived from textbook rules."""
        def active(state: "VectorState", _e: str = entity) -> bool:
            sm = state.spatial_map
            if sm is None:
                return False
            pos = sm.find_nearest(_e, state.player_pos)
            if pos is None:
                return False
            dist = abs(pos[0] - state.player_pos[0]) + abs(pos[1] - state.player_pos[1])
            return dist <= 3

        def response(state: "VectorState", _e: str = entity,
                     _tb: "CrafterTextbook" = textbook) -> Goal:
            for r in _tb.rules:
                if r.get("action") == "do" and r.get("target") == _e:
                    req = r.get("requires", {}) or {}
                    weapon = next(iter(req), None)
                    if weapon and state.inventory.get(weapon, 0) > 0:
                        return Goal(f"fight_{_e}", {})
                    elif weapon:
                        return Goal(f"craft_{weapon}", {"item": weapon})
            return Goal("explore")

        return _Threat(active_fn=active, response_fn=response)

    @staticmethod
    def _make_vital_threat(vital: str, goal_id: str) -> _Threat:
        return _Threat(
            active_fn=lambda s, _v=vital: s.body.get(_v, 9) < 3,
            response_fn=lambda s, _g=goal_id: Goal(_g),
        )
