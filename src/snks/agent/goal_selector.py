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
    parent_goal: str | None = None
    requested_capability: str | None = None
    blocked_by: str | None = None
    reason: str | None = None

    def to_trace(self) -> dict:
        return {
            "id": self.id,
            "parent_goal": self.parent_goal,
            "requested_capability": self.requested_capability,
            "blocked_by": self.blocked_by,
            "reason": self.reason,
        }

    def progress(self, trajectory: "VectorTrajectory") -> float:
        """How much did this trajectory advance the goal? Returns float >= 0."""
        if self.id.startswith("fight_"):
            target = self.id.removeprefix("fight_")
            if any(step.action == "do" and step.target == target for step in trajectory.plan.steps):
                return 1.0
            return 0.0
        elif self.id == "find_cow":
            return max(0.0, trajectory.vital_delta("food"))
        elif self.id == "find_water":
            return max(0.0, trajectory.vital_delta("drink"))
        elif self.id == "sleep":
            return max(0.0, trajectory.vital_delta("energy"))
        elif self.id.startswith("craft_"):
            return 1.0 if trajectory.item_gained(self.id.removeprefix("craft_")) else 0.0
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
    def __init__(
        self,
        textbook: "CrafterTextbook",
        allow_dynamic_entity_goals: bool = True,
    ):
        self._threats = self._derive_threats(textbook)
        self._allow_dynamic_entity_goals = allow_dynamic_entity_goals
        # entity -> weapon mapping derived from textbook fight rules.
        # Used by _dynamic_entity_goal to switch fight_X → craft_<weapon>
        # when the required weapon is missing from inventory.
        self._entity_weapons: dict[str, str] = {}
        for rule in textbook.rules:
            if rule.get("action") != "do":
                continue
            target = rule.get("target")
            req = rule.get("requires", {}) or {}
            weapon = next(iter(req), None)
            if target and weapon:
                self._entity_weapons[target] = weapon

    def select(self, state: "VectorState") -> Goal:
        """Pure function: current state → active goal. Called every step."""
        vital_goal = self._vital_goal(state)
        if vital_goal is not None:
            return vital_goal
        if self._allow_dynamic_entity_goals:
            dynamic_goal = self._dynamic_entity_goal(state)
            if dynamic_goal is not None:
                return dynamic_goal
        for threat in self._threats:
            if threat.active_fn(state):
                return threat.response_fn(state)
        return Goal("explore")

    def _dynamic_entity_goal(self, state: "VectorState") -> Goal | None:
        """Promote live dynamic threats into goal selection.

        Stage 89b: threat geometry already lives in `dynamic_entities`, so the
        goal layer must stop ignoring it. This does not hardcode a reflex; it
        only suppresses unrelated gather/craft goals when an active threat is
        present in the runtime world state.
        """
        present = {entity.concept_id for entity in state.dynamic_entities}

        def _goal_for(entity: str) -> Goal:
            # Without the required weapon, "fight" is futile — promote crafting
            # the weapon instead so plans that produce it earn goal_progress.
            weapon = self._entity_weapons.get(entity)
            if weapon and state.inventory.get(weapon, 0) <= 0:
                return Goal(
                    f"craft_{weapon}",
                    parent_goal=f"fight_{entity}",
                    requested_capability="armed_melee",
                    blocked_by=f"missing:{weapon}",
                    reason="required_weapon_missing",
                )
            return Goal(
                f"fight_{entity}",
                requested_capability="armed_melee" if weapon else None,
                reason="dynamic_threat_present",
            )

        if "arrow" in present or "skeleton" in present:
            return _goal_for("skeleton")
        if "zombie" in present:
            return _goal_for("zombie")
        return None

    @staticmethod
    def _vital_goal(state: "VectorState") -> Goal | None:
        """Critical body needs outrank optional engagement with threats."""
        if state.body.get("health", 9) < 2:
            return Goal("find_cow", reason="critical_health")
        for vital, goal_id in [("food", "find_cow"), ("drink", "find_water"), ("energy", "sleep")]:
            if state.body.get(vital, 9) < 3:
                return Goal(goal_id, reason=f"low_{vital}")
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

        # Proactive craft threat: when all materials for a weapon's `make`
        # rule are on hand and the weapon is missing, switch goal to
        # craft_<weapon>. Without this the agent stockpiles wood (gather
        # goal active) but never converts it into a sword — `do tree` keeps
        # winning on inventory_delta while crafting plans get zero goal
        # progress. The proactive craft threat is added at the END of the
        # threat list so dynamic-entity and vital-deficit threats still take
        # priority.
        proactive_craft_seen: set[str] = set()
        for fight_rule in fight_rules:
            fight_req = fight_rule.get("requires", {}) or {}
            weapon = next(iter(fight_req), None)
            if weapon is None or weapon in proactive_craft_seen:
                continue
            proactive_craft_seen.add(weapon)
            for make_rule in make_rules:
                if make_rule.get("result") != weapon:
                    continue
                make_req = {k: int(v) for k, v in (make_rule.get("requires", {}) or {}).items()}
                if not make_req:
                    continue
                result.append(_Threat(
                    active_fn=lambda s, _w=weapon, _r=make_req: (
                        s.inventory.get(_w, 0) < 1
                        and all(s.inventory.get(m, 0) >= q for m, q in _r.items())
                    ),
                    response_fn=lambda s, _w=weapon: Goal(f"craft_{_w}"),
                ))
                break

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
                return Goal(
                    f"fight_{_e}",
                    requested_capability="armed_melee",
                    reason="spatial_threat_present",
                )
            elif _w:
                return Goal(
                    f"craft_{_w}",
                    parent_goal=f"fight_{_e}",
                    requested_capability="armed_melee",
                    blocked_by=f"missing:{_w}",
                    reason="required_weapon_missing",
                )
            return Goal("explore")

        return _Threat(active_fn=active, response_fn=response)

    @staticmethod
    def _make_vital_threat(vital: str, goal_id: str) -> _Threat:
        return _Threat(
            active_fn=lambda s, _v=vital: s.body.get(_v, 9) < 3,
            response_fn=lambda s, _g=goal_id: Goal(_g),
        )
