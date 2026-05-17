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


# Tiebreaker awarded to a frontier-exploration plan whose target concept
# matches the active goal's `target_concept`. Small enough that any concrete
# plan with measurable goal_progress beats it, large enough to win over the
# zero-progress baseline plan in the lex-tuple sort. Documented as a
# tiebreaker, not a tunable weight — never raise it to "fix" survival numbers.
FRONTIER_PROGRESS_EPSILON: float = 0.05


@dataclass
class Goal:
    id: str
    parent_goal: str | None = None
    requested_capability: str | None = None
    blocked_by: str | None = None
    reason: str | None = None
    # Concept the goal wants the agent to physically locate or interact with
    # (e.g. find_water → "water", fight_zombie → "zombie"). Derived from
    # textbook by GoalSelector — Goal carries it so planner mechanism can
    # query the cognitive map without parsing the goal id string.
    target_concept: str | None = None

    def to_trace(self) -> dict:
        return {
            "id": self.id,
            "parent_goal": self.parent_goal,
            "requested_capability": self.requested_capability,
            "blocked_by": self.blocked_by,
            "reason": self.reason,
            "target_concept": self.target_concept,
        }

    def progress(self, trajectory: "VectorTrajectory") -> float:
        """How much did this trajectory advance the goal? Returns float >= 0."""
        if self.id.startswith("fight_"):
            target = self.id.removeprefix("fight_")
            if any(step.action == "do" and step.target == target for step in trajectory.plan.steps):
                return 1.0
            return self._frontier_epsilon(trajectory)
        elif self.id == "find_cow":
            v = trajectory.vital_delta("food")
            return v if v > 0 else self._frontier_epsilon(trajectory)
        elif self.id == "find_water":
            v = trajectory.vital_delta("drink")
            return v if v > 0 else self._frontier_epsilon(trajectory)
        elif self.id == "sleep":
            return max(0.0, trajectory.vital_delta("energy"))
        elif self.id.startswith("craft_"):
            return 1.0 if trajectory.item_gained(self.id.removeprefix("craft_")) else 0.0
        elif self.id == "gather_wood":
            v = trajectory.inventory_delta("wood")
            return v if v > 0 else self._frontier_epsilon(trajectory)
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

    def _frontier_epsilon(self, trajectory: "VectorTrajectory") -> float:
        """Tiebreaker for directed-exploration plans toward this goal's target.

        A frontier plan whose `target` matches the goal's `target_concept`
        earns `FRONTIER_PROGRESS_EPSILON` so the goal_prog slot of the score
        tuple beats baseline (which scores 0). Any concrete plan that
        actually advances the goal scores ≥ 1.0 (or the real vital delta),
        which dominates the tiebreaker.
        """
        if self.target_concept is None:
            return 0.0
        plan = trajectory.plan
        if not plan.steps:
            return 0.0
        first = plan.steps[0]
        if first.action == "frontier_seek" and first.target == self.target_concept:
            return FRONTIER_PROGRESS_EPSILON
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
        # goal-id -> target concept, derived from textbook `do` rules. Lets
        # the planner mechanism look at Goal.target_concept directly instead
        # of parsing the goal id string. Crafter facts stay in the textbook;
        # this method is the only place that translates them into goal labels.
        self._goal_targets: dict[str, str] = self._derive_goal_targets(textbook)

    @staticmethod
    def _derive_goal_targets(textbook: "CrafterTextbook") -> dict[str, str]:
        """Build `goal_id -> concept_id` from textbook `do` rules.

        Two key forms are emitted per matching rule so the existing
        resource-named goals (`find_cow`, `find_water`) and the more
        generic vital-named form (`find_food`, `find_drink`) both resolve
        to the same concept without parsing goal ids in Python:

        - rule `do <target>` with body delta `<vital>` > 0 →
            `find_<vital>: target` AND `find_<target>: target`
        - rule `do <target>` with `effect.remove_entity == target` →
            `fight_<target>: target`
        - rule `do <target>` with inventory delta `<item>` > 0 →
            `gather_<item>: target` AND `gather_<target>: target`

        First matching rule wins (subsequent duplicates ignored). Ambiguity
        should be resolved by editing the textbook, never by override logic
        here.
        """
        targets: dict[str, str] = {}
        for rule in textbook.rules:
            if rule.get("action") != "do":
                continue
            target = rule.get("target")
            if not target:
                continue
            effect = rule.get("effect", {}) or {}
            body_effect = effect.get("body", {}) or {}
            for vital, delta in body_effect.items():
                if isinstance(delta, (int, float)) and delta > 0:
                    targets.setdefault(f"find_{vital}", target)
                    targets.setdefault(f"find_{target}", target)
            inv_effect = effect.get("inventory", {}) or {}
            for item, delta in inv_effect.items():
                if isinstance(delta, (int, float)) and delta > 0:
                    targets.setdefault(f"gather_{item}", target)
                    targets.setdefault(f"gather_{target}", target)
            if effect.get("remove_entity") == target:
                targets.setdefault(f"fight_{target}", target)
        return targets

    def _attach_target(self, goal: Goal) -> Goal:
        """Populate `goal.target_concept` from the textbook-derived map.

        If the goal id is not in the map (e.g. `craft_*`, `sleep`,
        `explore`, or a `gather_<thing>` whose rule is absent), `target_concept`
        stays None and the planner falls back to its normal behaviour.
        """
        if goal.target_concept is None:
            goal.target_concept = self._goal_targets.get(goal.id)
        return goal

    def select(self, state: "VectorState") -> Goal:
        """Pure function: current state → active goal. Called every step."""
        vital_goal = self._vital_goal(state)
        if vital_goal is not None:
            return self._attach_target(vital_goal)
        if self._allow_dynamic_entity_goals:
            dynamic_goal = self._dynamic_entity_goal(state)
            if dynamic_goal is not None:
                return self._attach_target(dynamic_goal)
        for threat in self._threats:
            if threat.active_fn(state):
                return self._attach_target(threat.response_fn(state))
        return self._attach_target(Goal("explore"))

    def _dynamic_entity_goal(self, state: "VectorState") -> Goal | None:
        """Promote live dynamic threats into goal selection.

        Stage 89b: threat geometry already lives in `dynamic_entities`, so the
        goal layer must stop ignoring it. This does not hardcode a reflex; it
        only suppresses unrelated gather/craft goals when an active threat is
        present in the runtime world state.
        """
        # Collect hostile entities with their distances. Only entities with
        # a textbook "do <entity> requires {weapon}" rule are considered
        # hostile here — that's what `_entity_weapons` indexes. Arrows are
        # treated as the skeleton that fired them (no separate fight rule).
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

        px, py = state.player_pos
        hostiles: list[tuple[int, str]] = []
        for ent in state.dynamic_entities:
            cid = "skeleton" if ent.concept_id == "arrow" else ent.concept_id
            if cid not in self._entity_weapons:
                continue
            dist = abs(ent.position[0] - px) + abs(ent.position[1] - py)
            hostiles.append((dist, cid))

        if not hostiles:
            return None

        # Pick the nearest hostile. Ties broken by textbook iteration order
        # (sort key is just distance — stable, deterministic). Previously
        # the layer hardcoded skeleton>zombie priority regardless of distance,
        # so even when a zombie was adjacent and a skeleton was 5 tiles
        # away the agent would still emit fight_skeleton goal and never
        # face the zombie. seed 17 ep 0 (Phase 1 video) showed exactly that.
        hostiles.sort(key=lambda dc: dc[0])
        return _goal_for(hostiles[0][1])

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
