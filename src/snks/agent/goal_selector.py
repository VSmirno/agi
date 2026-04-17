"""Stage 85: GoalSelector — pure-function goal derivation from textbook.

Reads passive/action rules from CrafterTextbook to derive a priority-ordered
threat list. select(state) is a pure function re-evaluated every step.

Design: docs/superpowers/specs/2026-04-15-stage85-goal-selector-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from snks.agent.vector_sim import VectorTrajectory, VectorState
    from snks.agent.crafter_textbook import CrafterTextbook


@dataclass
class Goal:
    id: str

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
            # Textbook-seeded sleep has confidence=0.5 → surprise=0.5, which would
            # beat baseline (confidences=[]) giving explore_progress=0.5. Guard: a
            # self-action-only plan is not exploration.
            if trajectory.plan.steps and all(
                s.target == "self" for s in trajectory.plan.steps
            ):
                return 0.0
            return trajectory.avg_surprise()
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
        dynamic_goal = self._dynamic_entity_goal(state)
        if dynamic_goal is not None:
            return dynamic_goal
        for threat in self._threats:
            if threat.active_fn(state):
                return threat.response_fn(state)
        return Goal("explore")

    @staticmethod
    def _dynamic_entity_goal(state: "VectorState") -> Goal | None:
        """Promote live dynamic threats into goal selection.

        Stage 89b: threat geometry already lives in `dynamic_entities`, so the
        goal layer must stop ignoring it. This does not hardcode a reflex; it
        only suppresses unrelated gather/craft goals when an active threat is
        present in the runtime world state.
        """
        present = {entity.concept_id for entity in state.dynamic_entities}
        if "arrow" in present or "skeleton" in present:
            return Goal("fight_skeleton")
        if "zombie" in present:
            return Goal("fight_zombie")
        return None

    def _derive_threats(self, textbook: "CrafterTextbook") -> list[_Threat]:
        """Build priority-ordered threat list from textbook passive + action rules.

        Priority order:
          1. Physical threats: entity nearby with negative health passive rule
             → response depends on inventory (has sword → fight, else → craft)
          2. Critical vitals: health < 2 → find_cow (recover health via food)
          3. Low vitals: food < 3 → find_cow, drink < 3 → find_water, energy < 3 → sleep
          4. Proactive crafting: missing weapon material → gather_material
             Derived from textbook crafting chain:
               dangerous entity → fight requires weapon → make weapon requires material
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

        # 4. Proactive crafting chain: derive from textbook
        #    For each fight rule (do entity, requires weapon):
        #      If no weapon AND make_weapon requires material:
        #        → gather_material goal when material is missing
        threats.extend(self._derive_proactive_crafting(textbook))

        return threats

    @staticmethod
    def _derive_proactive_crafting(textbook: "CrafterTextbook") -> list["_Threat"]:
        """Derive proactive crafting threats from textbook rules.

        Traverses: dangerous_entity → fight requires weapon → make weapon requires material.
        Adds lowest-priority threat: no_weapon AND material_count < chain_need → gather_material.

        chain_need: sum of this material's requirements across ALL rules in the textbook
        that use it (make + place). Ensures agent gathers enough for the full crafting chain,
        not just the first step.

        This is fully derived from textbook — no hardcoded item names or thresholds.
        Motivation: prepare for known threats before they appear (proactive survival).
        """
        result: list[_Threat] = []
        seen: set[tuple] = set()  # avoid duplicate (weapon, material) threats

        fight_rules: list[dict] = []
        make_rules: list[dict] = []
        material_chain_cost: dict[str, int] = {}

        for r in textbook.rules:
            action = r.get("action")
            if action == "do":
                fight_rules.append(r)
            elif action == "make":
                make_rules.append(r)
                for mat, qty in (r.get("requires", {}) or {}).items():
                    material_chain_cost[mat] = material_chain_cost.get(mat, 0) + int(qty)
            elif action == "place":
                for mat, qty in (r.get("requires", {}) or {}).items():
                    material_chain_cost[mat] = material_chain_cost.get(mat, 0) + int(qty)

        for fight_rule in fight_rules:
            fight_req = fight_rule.get("requires", {}) or {}
            weapon = next(iter(fight_req), None)
            if weapon is None:
                continue
            # Find make rule for this weapon
            for make_rule in make_rules:
                if make_rule.get("result") != weapon:
                    continue
                make_req = make_rule.get("requires", {}) or {}
                for material in make_req:
                    key = (weapon, material)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Gather until we have enough for the full chain
                    need = material_chain_cost.get(material, 1)
                    result.append(_Threat(
                        active_fn=lambda s, _w=weapon, _m=material, _n=need: (
                            s.inventory.get(_w, 0) < 1
                            and s.inventory.get(_m, 0) < _n
                        ),
                        response_fn=lambda s, _m=material: Goal(f"gather_{_m}"),
                    ))

        return result

    @staticmethod
    def _make_entity_threat(entity: str, textbook: "CrafterTextbook") -> _Threat:
        """Build a _Threat for a dangerous entity derived from textbook rules."""
        # Precompute at init — avoid re-scanning rules on every step the entity is nearby.
        weapon: str | None = None
        for r in textbook.rules:
            if r.get("action") == "do" and r.get("target") == entity:
                req = r.get("requires", {}) or {}
                weapon = next(iter(req), None)
                break

        def active(state: "VectorState", _e: str = entity) -> bool:
            sm = state.spatial_map
            if sm is None:
                return False
            pos = sm.find_nearest(_e, state.player_pos)
            if pos is None:
                return False
            dist = abs(pos[0] - state.player_pos[0]) + abs(pos[1] - state.player_pos[1])
            return dist <= 3

        def response(state: "VectorState", _e: str = entity, _w: str | None = weapon) -> Goal:
            if _w and state.inventory.get(_w, 0) > 0:
                return Goal(f"fight_{_e}")
            elif _w:
                return Goal(f"craft_{_w}")
            return Goal("explore")

        return _Threat(active_fn=active, response_fn=response)

    @staticmethod
    def _make_vital_threat(vital: str, goal_id: str) -> _Threat:
        return _Threat(
            active_fn=lambda s, _v=vital: s.body.get(_v, 9) < 3,
            response_fn=lambda s, _g=goal_id: Goal(_g),
        )
