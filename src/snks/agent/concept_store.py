"""Stage 71: ConceptStore — unified omnimodal concept storage.

One concept = one entity with all modalities (visual, text, attributes,
causal links). Replaces GroundingMap + CausalWorldModel for the
text-visual integration pipeline.

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from snks.agent.forward_sim_types import (
    DynamicEntity,
    Failure,
    Plan,
    PlannedStep,
    RuleEffect,
    SimEvent,
    SimState,
    StatefulCondition,
    Trajectory,
)

if TYPE_CHECKING:
    from snks.agent.perception import HomeostaticTracker


@dataclass
class CausalLink:
    """One causal rule: action on this concept → result.

    Stage 77a adds `kind` discriminator and `effect` structured dispatch.
    The legacy `result: str` field is kept temporarily for backward compat
    with Stage 71-76 code during staged refactor (removed in Commit 8).

    Valid `kind` values:
      - "action_triggered" — fires on agent `do`/`make`/`place`/`sleep`
      - "passive_body_rate" — per-tick unconditional body delta
      - "passive_movement" — entity moves per tick per behavior
      - "passive_spatial" — adjacent damage / effect
      - "passive_stateful" — body delta while condition holds

    For action_triggered rules: `action` + `effect` describe what happens
    when the agent performs the action on `concept` (inherited from Concept).

    For passive rules: `effect` describes the per-tick update; `action` is
    typically None or "_passive".
    """

    action: str  # "do", "make", "place", "sleep", or "_passive" for passive rules
    requires: dict[str, int] = field(default_factory=dict)  # {wood_pickaxe: 1}
    condition: str | None = None  # "nearby", None — legacy
    confidence: float = 0.5  # starts at 0.5 (from textbook), grows with verification

    # Stage 77a — structured dispatch
    kind: str = "action_triggered"  # "action_triggered" | "passive_*"
    effect: "RuleEffect | None" = None  # structured effect — replaces legacy string `result`
    concept: str | None = None  # concept_id the rule is tied to (entity for passive rules)


MAX_OBSERVATIONS = 20  # raw feature samples per concept for metric learning


@dataclass
class Concept:
    """One concept with all modalities."""

    id: str
    visual: torch.Tensor | None = None  # center_feature prototype (EMA)
    text_sdr: torch.Tensor | None = None  # SDR from GroundedTokenizer
    attributes: dict[str, Any] = field(default_factory=dict)
    causal_links: list[CausalLink] = field(default_factory=list)
    confidence: float = 0.5
    observations: list[torch.Tensor] = field(default_factory=list)  # raw features

    def add_observation(self, feature: torch.Tensor) -> None:
        """Store raw feature from verified interaction (for metric learning)."""
        self.observations.append(feature.detach().clone())
        if len(self.observations) > MAX_OBSERVATIONS:
            self.observations.pop(0)

    def find_causal(
        self,
        action: str,
        check_requires: dict[str, int] | None = None,
    ) -> CausalLink | None:
        """Find a causal link by action, optionally checking inventory requirements.

        When multiple links match, prefers the most specific one (most requires
        items matching the inventory). This resolves ambiguity when e.g.
        make_wood_pickaxe (requires wood) and make_stone_pickaxe (requires
        wood + stone_item) both match an inventory with wood + stone_item.

        Args:
            action: action name to match ("do", "make", "place")
            check_requires: if provided, only return link if inventory
                           has all required items (>=).
        """
        candidates: list[tuple[int, CausalLink]] = []
        for link in self.causal_links:
            if link.action != action:
                continue
            if check_requires is not None and link.requires:
                if not all(
                    check_requires.get(item, 0) >= count
                    for item, count in link.requires.items()
                ):
                    continue
            # Score = number of requires items (more specific = higher)
            score = len(link.requires)
            candidates.append((score, link))
        if not candidates:
            return None
        # Return most specific match
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


# Stage 77a: PlannedStep is now imported from forward_sim_types (unified class
# with both new `rule` field and legacy `expected_gain`/`requires` for backward
# compat). The legacy duplicate that used to live here has been removed.


@dataclass
class SurpriseEvent:
    """An unexpected outcome not predicted by any rule."""

    outcome: str
    action: str
    near: str | None = None


@dataclass
class Prediction:
    """A prediction made before an action."""

    concept_id: str
    action: str
    expected: str
    confidence: float
    link: CausalLink | None = None


CONFIRM_DELTA = 0.15
REFUTE_DELTA = 0.15


class ConceptStore:
    """Unified omnimodal concept storage.

    Each concept stores visual embedding, text SDR, attributes, and
    causal links. Supports query by any modality, causal prediction,
    backward chaining planning, and experience-based verification.
    """

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.surprises: list[SurpriseEvent] = []
        # Stage 77a: passive rules (body_rate, movement, spatial, stateful).
        # Kept in a flat list separate from concept.causal_links — they're
        # not tied to an "action on a concept", they're per-tick world updates.
        self.passive_rules: list[CausalLink] = []

    # --- Registration ---

    def register(self, id: str, attributes: dict[str, Any] | None = None) -> Concept:
        """Register a new concept (or return existing)."""
        if id in self.concepts:
            if attributes:
                self.concepts[id].attributes.update(attributes)
            return self.concepts[id]
        concept = Concept(id=id, attributes=attributes or {})
        self.concepts[id] = concept
        return concept

    def add_causal(self, concept_id: str, link: CausalLink) -> None:
        """Add a causal link to a concept."""
        if concept_id not in self.concepts:
            self.register(concept_id)
        self.concepts[concept_id].causal_links.append(link)

    def add_passive_rule(self, link: CausalLink) -> None:
        """Add a passive rule (body_rate, movement, spatial, stateful).

        Stage 77a: passive rules are applied per sim tick by simulate_forward,
        not triggered by agent actions. They're stored in a flat list on the
        store rather than inside Concept.causal_links, because most of them
        aren't "actions on a concept" (body_rate, stateful are global).
        """
        self.passive_rules.append(link)
        # For entity-tied passive rules (movement, spatial), also register
        # the entity concept if not already known
        if link.concept and link.kind in ("passive_movement", "passive_spatial"):
            if link.concept not in self.concepts:
                self.register(link.concept)

    def body_rate_rules(self) -> list[CausalLink]:
        """All passive_body_rate rules (background per-tick decay)."""
        return [r for r in self.passive_rules if r.kind == "passive_body_rate"]

    def stateful_rules(self) -> list[CausalLink]:
        """All passive_stateful rules (per-tick when condition holds)."""
        return [r for r in self.passive_rules if r.kind == "passive_stateful"]

    def movement_rule_for(self, concept_id: str) -> CausalLink | None:
        """The (single) movement rule for a given entity concept, if any."""
        for r in self.passive_rules:
            if r.kind == "passive_movement" and r.concept == concept_id:
                return r
        return None

    def spatial_rules_for(self, concept_id: str) -> list[CausalLink]:
        """All spatial rules tied to a given entity concept."""
        return [
            r for r in self.passive_rules
            if r.kind == "passive_spatial" and r.concept == concept_id
        ]

    # --- Grounding ---

    def ground_visual(self, id: str, z_real: torch.Tensor) -> None:
        """Bind a visual embedding to a concept."""
        if id not in self.concepts:
            self.register(id)
        self.concepts[id].visual = z_real.detach().clone()

    def ground_text(self, id: str, text_sdr: torch.Tensor) -> None:
        """Bind a text SDR to a concept."""
        if id not in self.concepts:
            self.register(id)
        self.concepts[id].text_sdr = text_sdr.detach().clone()

    # --- Query ---

    def query_text(self, word: str) -> Concept | None:
        """Find concept by text label (exact match)."""
        return self.concepts.get(word)

    def query_visual(self, z_real: torch.Tensor) -> Concept | None:
        """Find closest concept by visual embedding (cosine similarity)."""
        concept, _ = self.query_visual_scored(z_real)
        return concept

    def query_visual_scored(
        self, z_real: torch.Tensor
    ) -> tuple[Concept | None, float]:
        """Find closest concept by visual embedding, returning similarity score.

        Returns (best_concept, best_similarity). If no concepts have visual
        embeddings, returns (None, -1.0).
        """
        best_concept = None
        best_sim = -1.0
        z_norm = torch.nn.functional.normalize(z_real.unsqueeze(0), dim=1)
        for concept in self.concepts.values():
            if concept.visual is None:
                continue
            c_norm = torch.nn.functional.normalize(
                concept.visual.unsqueeze(0), dim=1
            )
            sim = (z_norm @ c_norm.T).item()
            if sim > best_sim:
                best_sim = sim
                best_concept = concept
        return best_concept, best_sim

    # --- Causal reasoning ---

    def predict(
        self, concept_id: str, action: str, inventory: dict[str, int]
    ) -> CausalLink | None:
        """Predict outcome of action on concept given inventory."""
        concept = self.query_text(concept_id)
        if concept is None:
            return None
        return concept.find_causal(action=action, check_requires=inventory)

    # Legacy `plan(goal_id)` removed in Commit 8.
    # Use `plan_toward_rule(rule, inventory)` instead — see below.

    # --- Verification ---

    def verify(
        self, concept_id: str, action: str, actual_outcome: str | None
    ) -> None:
        """Update confidence based on actual outcome vs rule effect.

        Stage 77a: checks `effect.inventory_delta` / `effect.body_delta` /
        `effect.scene_remove` to determine if the outcome matches what the
        rule would produce. No more string `result` comparison.
        """
        concept = self.query_text(concept_id)
        if concept is None:
            return

        for link in concept.causal_links:
            if link.action != action:
                continue
            if link.effect is None:
                continue
            matched = self._effect_matches_outcome(link.effect, actual_outcome)
            if matched:
                link.confidence = min(1.0, link.confidence + CONFIRM_DELTA)
            else:
                link.confidence = max(0.0, link.confidence - REFUTE_DELTA)
            return

    @staticmethod
    def _effect_matches_outcome(
        effect: RuleEffect, actual_outcome: str | None,
    ) -> bool:
        """Check if an observed outcome label matches this rule effect."""
        if actual_outcome is None:
            return False
        # Gather/craft — any inventory item with positive delta
        for item, delta in effect.inventory_delta.items():
            if delta > 0 and actual_outcome == item:
                return True
        # Consume/self — restore_<var> labels
        if actual_outcome.startswith("restore_"):
            var = actual_outcome[len("restore_"):]
            if effect.body_delta.get(var, 0) > 0:
                return True
        # Remove (combat)
        if effect.scene_remove and actual_outcome == f"kill_{effect.scene_remove}":
            return True
        # Place
        if effect.world_place and actual_outcome == effect.world_place[0]:
            return True
        return False

    def record_surprise(self, outcome: str, action: str, near: str | None = None) -> None:
        """Log an unexpected event (no matching rule)."""
        self.surprises.append(SurpriseEvent(outcome=outcome, action=action, near=near))

    def verify_after_action(
        self,
        prediction: Prediction | None,
        action: str,
        actual_outcome: str | None,
        near: str | None = None,
    ) -> None:
        """Verify a prediction against actual outcome, update confidence."""
        if prediction is None:
            if actual_outcome is not None:
                self.record_surprise(actual_outcome, action, near=near)
            return

        if prediction.link is not None:
            if actual_outcome == prediction.expected:
                prediction.link.confidence = min(
                    1.0, prediction.link.confidence + CONFIRM_DELTA
                )
            else:
                prediction.link.confidence = max(
                    0.0, prediction.link.confidence - REFUTE_DELTA
                )

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Save/load removed in Commit 8.

        The legacy JSON format serialized `link.result` as a string, but
        Stage 77a uses structured `RuleEffect` objects which don't have a
        natural JSON form without a dedicated serializer. Since save/load
        was only used by legacy Stage 72-74 experiments that no longer
        run, the methods are removed. Re-introduce when needed with
        RuleEffect serialization.
        """
        raise NotImplementedError(
            "ConceptStore.save/load removed in Stage 77a Commit 8 "
            "(structured RuleEffect needs new serializer)"
        )

    def load(self, path: str) -> None:
        raise NotImplementedError(
            "ConceptStore.save/load removed in Stage 77a Commit 8"
        )

    # =========================================================================
    # Stage 77a: Forward simulation API
    # =========================================================================

    # Confidence threshold — rules below this are considered "broken" and
    # do not fire in simulate_forward. Tunable, documented magic number.
    # TODO(77b): replace with probabilistic weighted firing via multi-rollout.
    CONFIDENCE_THRESHOLD = 0.1

    def find_remedies(self, failure: Failure) -> list[CausalLink]:
        """Query the world model: which rules counteract this failure?

        For var_depleted failures, return rules whose effect raises the
        depleted variable (direct body_delta, stateful regen).
        For attributed_to failures, return rules that remove the causing
        entity from the scene (combat rules).
        """
        remedies: list[CausalLink] = []
        seen: set[int] = set()

        # Action-triggered rules in concept.causal_links
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if id(link) in seen:
                    continue
                if _rule_prevents(link, failure):
                    remedies.append(link)
                    seen.add(id(link))

        # Passive rules (stateful regen can prevent var_depleted)
        for link in self.passive_rules:
            if id(link) in seen:
                continue
            if _rule_prevents(link, failure):
                remedies.append(link)
                seen.add(id(link))

        return remedies

    def _find_rule_producing_item(self, item: str) -> CausalLink | None:
        """Find a rule whose effect adds `item` to inventory.

        Prefers action_triggered rules (gather, craft) because they're
        what the agent can actually do. Returns None if nothing produces
        the item.
        """
        # Prefer rules with fewer requires (simpler to execute)
        candidates: list[tuple[int, CausalLink]] = []
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if not link.effect:
                    continue
                if link.effect.inventory_delta.get(item, 0) > 0:
                    candidates.append((len(link.requires), link))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])  # fewest requires first
        return candidates[0][1]

    def _find_rule_producing_world_place(self, item: str) -> CausalLink | None:
        """Find a place rule that puts `item` in the world (e.g., table)."""
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if not link.effect or link.effect.kind != "place":
                    continue
                if link.effect.world_place and link.effect.world_place[0] == item:
                    return link
        return None

    def plan_toward_rule(
        self,
        target_rule: CausalLink,
        inventory: dict[str, int],
    ) -> list[PlannedStep]:
        """Backward chain from target_rule to a sequence of PlannedSteps.

        Recursively resolves prerequisites (required items + near-concept
        world state) until all are satisfied by current inventory. The
        final step is the execution of target_rule itself.
        """
        steps: list[PlannedStep] = []
        visited: set[tuple[str, str]] = set()
        working_inv = dict(inventory)
        self._plan_for_rule(target_rule, steps, visited, working_inv)
        return steps

    def _plan_for_rule(
        self,
        rule: CausalLink,
        steps: list[PlannedStep],
        visited: set[tuple[str, str]],
        inventory: dict[str, int],
    ) -> None:
        """Recursive backward-chaining helper.

        Stage 77a Attempt 3:
        1. Handles quantity > 1: loops producer calls until inventory
           satisfied. Leaf gather rules can be added multiple times.
        2. Order: spatial prereqs BEFORE item prereqs, so item prereqs
           see the post-consumption inventory. Fixes bug where plan
           had 5 steps for combat chain but 3rd wood was missing
           (place_table consumed both wood before make_sword got its 1).
        3. After adding a step, updates inventory with the rule's net
           effect (production + consumption) so the caller sees the
           post-step state.
        """
        # Leaf gather rules (no requires) can be repeated; everything else
        # uses the visited set to avoid cycles.
        is_leaf_gather = (
            rule.action == "do"
            and not rule.requires
            and rule.effect
            and rule.effect.kind in ("gather",)
        )
        if not is_leaf_gather:
            key = (rule.concept or "", rule.action or "_passive")
            if key in visited:
                return
            visited.add(key)

        # Resolve spatial preconditions FIRST (they may consume resources,
        # which the item-prereq loop must see). For `make X near Y`, Y
        # must be placeable — which consumes items via a `place` rule.
        if rule.action == "make" and rule.concept:
            producer = self._find_rule_producing_world_place(rule.concept)
            if producer is not None:
                self._plan_for_rule(producer, steps, visited, inventory)

        # Resolve item prerequisites — loop until quantity satisfied.
        # inventory has been updated by any spatial prereqs (consumption
        # reflected), so this sees the accurate pre-execution state.
        for item, count in rule.requires.items():
            max_retries = 20  # safety against infinite loops
            retries = 0
            while inventory.get(item, 0) < count and retries < max_retries:
                retries += 1
                producer = self._find_rule_producing_item(item)
                if producer is None:
                    break
                self._plan_for_rule(producer, steps, visited, inventory)
                # plan_for_rule already updated inventory with producer's
                # net effect — no manual update here.

        # Add this rule's execution as the final step
        steps.append(PlannedStep(
            action=rule.action or "do",
            target=rule.concept,
            near=rule.concept if rule.action in ("make", "place") else None,
            rule=rule,
            requires=dict(rule.requires),
        ))

        # Update inventory with THIS rule's net effect (production + consumption).
        # The caller will see the post-step state, so further prerequisites
        # in the caller are evaluated correctly.
        if rule.effect:
            for k, v in rule.effect.inventory_delta.items():
                inventory[k] = inventory.get(k, 0) + v

    def simulate_forward(
        self,
        plan: Plan,
        initial_state: SimState,
        tracker: "HomeostaticTracker",
        horizon: int = 20,
    ) -> Trajectory:
        """Roll out a plan through causal rules for `horizon` ticks.

        Returns a Trajectory with per-tick body values, events, and
        termination info. Deterministic: each tick applies all passive
        rules then the agent action's effects (if any).

        Phases per tick (fixed order, see _apply_tick docstring):
          1. Dynamic entities move
          2. Player moves (if primitive is move_*)
          3. Background body rates
          4. Stateful passive rules
          5. Spatial adjacency rules
          6. Action-triggered effects (do/make/place/sleep)
        """
        sim = initial_state.copy()
        traj = Trajectory(
            plan=plan,
            body_series={var: [] for var in sim.body},
            events=[],
            final_state=sim,
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        plan_cursor = 0

        for tick in range(horizon):
            # Termination check BEFORE acting (dead agents don't act)
            if sim.is_dead(tracker.vital_mins):
                traj.terminated = True
                traj.terminated_reason = "body_dead"
                break

            # Determine current PlannedStep (inertia after plan completes)
            if plan_cursor < len(plan.steps):
                current_step = plan.steps[plan_cursor]
            else:
                current_step = PlannedStep(
                    action="inertia", target=None, near=None, rule=None,
                )

            primitive = _expand_to_primitive(current_step, sim, self)
            prev_inv = dict(sim.inventory)
            prev_enemies = {e.concept_id for e in sim.dynamic_entities}

            self._apply_tick(sim, primitive, tracker, traj, tick, planned_step=current_step)

            # Snapshot body_series
            for var, value in sim.body.items():
                if var not in traj.body_series:
                    traj.body_series[var] = []
                traj.body_series[var].append(value)

            # Check if current PlannedStep completed
            if plan_cursor < len(plan.steps):
                if _is_plan_step_complete(current_step, sim, prev_inv, prev_enemies):
                    plan_cursor += 1

        if not traj.terminated:
            traj.terminated_reason = (
                "plan_complete" if plan_cursor >= len(plan.steps) else "horizon"
            )

        traj.final_state = sim
        traj.plan_progress = plan_cursor
        return traj

    def _clamp_body(
        self, sim: SimState, tracker: "HomeostaticTracker"
    ) -> None:
        """Clamp body variables to [reference_min, reference_max]."""
        for var in list(sim.body.keys()):
            lo = tracker.reference_min.get(var, 0.0)
            hi = tracker.reference_max.get(var)
            if hi is None:
                continue
            if sim.body[var] > hi:
                sim.body[var] = hi
            elif sim.body[var] < lo:
                sim.body[var] = lo

    def _apply_tick(
        self,
        sim: SimState,
        primitive: str,
        tracker: "HomeostaticTracker",
        traj: Trajectory,
        tick: int,
        planned_step: "PlannedStep | None" = None,
    ) -> None:
        """One tick of simulation. Six phases in fixed order.

        Stage 77a Attempt 2: planned_step is optionally passed so Phase 6 "do"
        can fire a rule by target proximity (manhattan ≤1) instead of relying
        on last_action facing direction. In the sim, navigation walks through
        target tiles, making facing-based rules never fire — proximity check
        works around that.
        """

        # === Phase 1: Dynamic entities move ===
        for entity in sim.dynamic_entities:
            move_rule = self.movement_rule_for(entity.concept_id)
            if move_rule is None or move_rule.confidence < self.CONFIDENCE_THRESHOLD:
                continue
            entity.pos = _apply_movement(
                entity.pos,
                sim.player_pos,
                move_rule.effect.movement_behavior,
                tick,
            )

        # === Phase 2: Player moves ===
        if primitive.startswith("move_"):
            sim.player_pos = _apply_player_move(sim.player_pos, primitive)

        # === Phase 3: Background body rates ===
        # Observation-driven: uses tracker.get_rate(var) which Bayesian-
        # combines the textbook innate rate (rough prior) with the running-
        # mean observed rate. As the agent accumulates experience, sim
        # predictions converge to real env dynamics without reading exact
        # values from the textbook. The textbook only needs rough directional
        # facts ("food depletes slowly") — observation does the refinement.
        for rule in self.body_rate_rules():
            if rule.confidence < self.CONFIDENCE_THRESHOLD:
                continue
            var = rule.effect.body_rate_variable
            # Prefer tracker's learned rate if this var is known to tracker;
            # otherwise fall back to textbook static rate.
            if var in tracker.innate_rates:
                rate = tracker.get_rate(var)
            else:
                rate = rule.effect.body_rate
            sim.body[var] = sim.body.get(var, 0.0) + rate
            traj.events.append(SimEvent(
                step=tick, kind="body_delta", var=var,
                amount=rate, source="_background",
            ))

        # Clamp between Phase 3 and Phase 4 so stateful conditions see
        # clean values (e.g. `food == 0` fires when food has been clamped
        # from -0.04 to 0 by Phase 3's decay hitting the floor).
        self._clamp_body(sim, tracker)

        # === Phase 4: Stateful rules ===
        for rule in self.stateful_rules():
            if rule.confidence < self.CONFIDENCE_THRESHOLD:
                continue
            cond = rule.effect.stateful_condition
            if cond is None or not cond.satisfied(sim):
                continue
            for var, delta in rule.effect.body_delta.items():
                sim.body[var] = sim.body.get(var, 0.0) + delta
                traj.events.append(SimEvent(
                    step=tick, kind="body_delta", var=var,
                    amount=delta, source=f"stateful:{cond.var}",
                ))

        # === Phase 5: Spatial rules ===
        for entity in sim.dynamic_entities:
            for rule in self.spatial_rules_for(entity.concept_id):
                if rule.confidence < self.CONFIDENCE_THRESHOLD:
                    continue
                if _manhattan(entity.pos, sim.player_pos) > rule.effect.spatial_range:
                    continue
                for var, delta in rule.effect.body_delta.items():
                    sim.body[var] = sim.body.get(var, 0.0) + delta
                    traj.events.append(SimEvent(
                        step=tick, kind="body_delta", var=var,
                        amount=delta, source=entity.concept_id,
                    ))

        # === Phase 6: Action-driven effects ===
        if primitive == "do":
            # Attempt 2: use planned_step.target + proximity if available
            # (sim navigation walks through tiles, so facing-based lookup fails).
            fired = False
            if (
                planned_step
                and planned_step.action == "do"
                and planned_step.target
                and planned_step.rule
                and planned_step.rule.confidence >= self.CONFIDENCE_THRESHOLD
            ):
                target = planned_step.target
                # Check adjacency via dynamic_entities (for mobs) or spatial_map
                target_pos = None
                for e in sim.dynamic_entities:
                    if e.concept_id == target:
                        target_pos = e.pos
                        break
                if target_pos is None and sim.spatial_map is not None:
                    target_pos = sim.spatial_map.find_nearest(target, sim.player_pos)
                if target_pos is not None and _manhattan(target_pos, sim.player_pos) <= 1:
                    # Check requires still satisfied in sim inventory
                    if all(
                        sim.inventory.get(r, 0) >= c
                        for r, c in planned_step.rule.requires.items()
                    ):
                        _apply_effect_to_sim(
                            sim, planned_step.rule.effect, traj, tick,
                            f"do:{target}",
                        )
                        fired = True

            # Fallback: original facing-based lookup (for non-planned "do")
            if not fired:
                near = _nearest_concept(sim)
                if near:
                    rule = self._find_do_rule(near, sim.inventory)
                    if rule and rule.confidence >= self.CONFIDENCE_THRESHOLD:
                        _apply_effect_to_sim(sim, rule.effect, traj, tick, near)
        elif primitive.startswith("place_"):
            item = primitive[len("place_"):]
            rule = self._find_place_rule(item, sim.inventory)
            if rule and rule.confidence >= self.CONFIDENCE_THRESHOLD:
                _apply_effect_to_sim(sim, rule.effect, traj, tick, f"place:{item}")
        elif primitive.startswith("make_"):
            item = primitive[len("make_"):]
            rule = self._find_make_rule(item, sim.inventory)
            if rule and rule.confidence >= self.CONFIDENCE_THRESHOLD:
                _apply_effect_to_sim(sim, rule.effect, traj, tick, f"make:{item}")
        elif primitive == "sleep":
            rule = self._find_sleep_rule()
            if rule and rule.confidence >= self.CONFIDENCE_THRESHOLD:
                _apply_effect_to_sim(sim, rule.effect, traj, tick, "sleep")

        # Clamp all body variables to their reference bounds
        self._clamp_body(sim, tracker)

        sim.last_action = primitive
        sim.step = tick + 1

    def _find_do_rule(
        self, target_concept: str, inventory: dict[str, int]
    ) -> CausalLink | None:
        concept = self.concepts.get(target_concept)
        if concept is None:
            return None
        # Prefer the most specific rule (most requires that are satisfied)
        candidates: list[tuple[int, CausalLink]] = []
        for link in concept.causal_links:
            if link.action != "do":
                continue
            if not all(inventory.get(r, 0) >= c for r, c in link.requires.items()):
                continue
            candidates.append((len(link.requires), link))
        if not candidates:
            return None
        candidates.sort(key=lambda c: -c[0])  # most specific first
        return candidates[0][1]

    def _find_make_rule(
        self, result_item: str, inventory: dict[str, int]
    ) -> CausalLink | None:
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if link.action != "make" or not link.effect:
                    continue
                if link.effect.inventory_delta.get(result_item, 0) <= 0:
                    continue
                if not all(inventory.get(r, 0) >= c for r, c in link.requires.items()):
                    continue
                return link
        return None

    def _find_place_rule(
        self, item: str, inventory: dict[str, int]
    ) -> CausalLink | None:
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if link.action != "place" or not link.effect:
                    continue
                if not link.effect.world_place or link.effect.world_place[0] != item:
                    continue
                if not all(inventory.get(r, 0) >= c for r, c in link.requires.items()):
                    continue
                return link
        return None

    def _find_sleep_rule(self) -> CausalLink | None:
        self_concept = self.concepts.get("_self")
        if self_concept is None:
            return None
        for link in self_concept.causal_links:
            if link.action == "sleep":
                return link
        return None


# =========================================================================
# Stage 77a: Module-level helpers for forward simulation
# =========================================================================


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _apply_movement(
    entity_pos: tuple[int, int],
    player_pos: tuple[int, int],
    behavior: str | None,
    tick: int,
) -> tuple[int, int]:
    """Move an entity by 1 tile according to its behavior.

    chase_player: step toward player along greater axis (Manhattan greedy).
    flee_player:  step away.
    random_walk:  deterministic pseudo-random based on tick (so sim is repeatable).
    """
    if behavior is None:
        return entity_pos

    if behavior == "chase_player":
        return _step_toward_pos(entity_pos, player_pos)

    if behavior == "flee_player":
        dx = entity_pos[0] - player_pos[0]
        dy = entity_pos[1] - player_pos[1]
        if abs(dx) >= abs(dy) and dx != 0:
            return (entity_pos[0] + (1 if dx > 0 else -1), entity_pos[1])
        if dy != 0:
            return (entity_pos[0], entity_pos[1] + (1 if dy > 0 else -1))
        return entity_pos

    if behavior == "random_walk":
        # Deterministic pseudo-random based on entity_pos + tick
        seed = (entity_pos[0] * 31 + entity_pos[1] * 17 + tick) & 3
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = deltas[seed]
        return (entity_pos[0] + dx, entity_pos[1] + dy)

    return entity_pos


def _step_toward_pos(
    current: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int]:
    """One Manhattan-greedy step from current toward target."""
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return (current[0] + (1 if dx > 0 else -1), current[1])
    if dy != 0:
        return (current[0], current[1] + (1 if dy > 0 else -1))
    return current


def _apply_player_move(
    pos: tuple[int, int], primitive: str
) -> tuple[int, int]:
    """Apply a move primitive to player position.

    Crafter convention (see Stage 75 report):
      pos[0] = horizontal X, pos[1] = vertical Y
      move_left → x-1, move_right → x+1
      move_up → y-1, move_down → y+1
    """
    if primitive == "move_left":
        return (pos[0] - 1, pos[1])
    if primitive == "move_right":
        return (pos[0] + 1, pos[1])
    if primitive == "move_up":
        return (pos[0], pos[1] - 1)
    if primitive == "move_down":
        return (pos[0], pos[1] + 1)
    return pos


def _nearest_concept(sim: SimState) -> str | None:
    """Concept in the tile immediately in front of the player.

    Direction is determined by last_action (the previous move). If last_action
    is not a move, defaults to facing "down". Reads from sim.spatial_map._map
    by (x, y) key matching CrafterSpatialMap.update's storage convention.
    """
    dx, dy = 0, 1  # default: facing down
    if sim.last_action == "move_left":
        dx, dy = -1, 0
    elif sim.last_action == "move_right":
        dx, dy = 1, 0
    elif sim.last_action == "move_up":
        dx, dy = 0, -1
    elif sim.last_action == "move_down":
        dx, dy = 0, 1

    front = (sim.player_pos[0] + dx, sim.player_pos[1] + dy)

    # Also check if a dynamic entity is at `front` — that takes precedence
    for entity in sim.dynamic_entities:
        if entity.pos == front:
            return entity.concept_id

    # Fall back to spatial map
    if sim.spatial_map is not None and hasattr(sim.spatial_map, "_map"):
        return sim.spatial_map._map.get(front, "empty")
    return None


def _explore_direction(sim: SimState) -> str:
    """Pick a move primitive for exploration.

    Prefers the least-visited neighbor whose immediate step is NOT a known
    blocked tile. `unvisited_neighbors` already filters blocked targets, but
    the path TOWARD an unblocked target may still cross a blocked tile (e.g.
    target at (X-2, Y) requires stepping into (X-1, Y) first). Without this
    check the agent loops issuing the same move into a wall it already
    learned about. Fall-back cycling also skips blocked directions.
    """
    has_map = sim.spatial_map is not None and hasattr(sim.spatial_map, "is_blocked")

    def step_is_blocked(primitive: str) -> bool:
        if not has_map:
            return False
        return sim.spatial_map.is_blocked(_apply_player_move(sim.player_pos, primitive))

    if sim.spatial_map is not None and hasattr(sim.spatial_map, "unvisited_neighbors"):
        try:
            unvisited = sim.spatial_map.unvisited_neighbors(sim.player_pos, radius=3)
        except Exception:
            unvisited = []
        # Sort by Manhattan distance — try closest first, fall back to next
        # if the immediate step is blocked.
        unvisited.sort(
            key=lambda p: abs(p[0] - sim.player_pos[0]) + abs(p[1] - sim.player_pos[1])
        )
        for target in unvisited:
            next_step = _step_toward_pos(sim.player_pos, target)
            if next_step == sim.player_pos:
                continue  # already there, picks no direction
            primitive = _direction_primitive(sim.player_pos, next_step)
            if step_is_blocked(primitive):
                continue
            return primitive

    # Fall back: cycle through 4 cardinal directions so the agent doesn't
    # walk in one direction forever. Skip directions known to be blocked.
    dirs = ["move_up", "move_right", "move_down", "move_left"]
    for offset in range(4):
        primitive = dirs[(sim.step + offset) % 4]
        if not step_is_blocked(primitive):
            return primitive
    # Surrounded — return arbitrary cycle direction (the env will reject it
    # but at least the agent isn't infinite-looping in this function).
    return dirs[sim.step % 4]


def _step_toward_target(sim: SimState, target_pos: tuple[int, int]) -> str:
    """Pick a move primitive toward `target_pos` that is not known-blocked.

    Tries the Manhattan-greedy direction first; if that immediate step is
    a known blocked tile, tries the orthogonal direction; if that's also
    blocked, falls through to _explore_direction.
    """
    has_map = sim.spatial_map is not None and hasattr(sim.spatial_map, "is_blocked")

    def step_blocked(primitive: str) -> bool:
        if not has_map:
            return False
        return sim.spatial_map.is_blocked(_apply_player_move(sim.player_pos, primitive))

    dx = target_pos[0] - sim.player_pos[0]
    dy = target_pos[1] - sim.player_pos[1]

    # Preferred and orthogonal candidates, in order of preference
    candidates: list[str] = []
    if abs(dx) >= abs(dy) and dx != 0:
        candidates.append("move_right" if dx > 0 else "move_left")
        if dy != 0:
            candidates.append("move_down" if dy > 0 else "move_up")
    elif dy != 0:
        candidates.append("move_down" if dy > 0 else "move_up")
        if dx != 0:
            candidates.append("move_right" if dx > 0 else "move_left")

    for primitive in candidates:
        if not step_blocked(primitive):
            return primitive

    # Both Manhattan-greedy directions blocked — fall back to exploration
    return _explore_direction(sim)


def _expand_to_primitive(
    step: PlannedStep, sim: SimState, store: "ConceptStore"
) -> str:
    """Convert a symbolic PlannedStep into a primitive env action for this tick.

    When a plan target is unknown (not in spatial_map), falls through to
    exploration instead of a fixed move_right default. This lets the agent
    actually find trees/cows/water by walking to unvisited places rather
    than stuck moving east forever.
    """
    if step.action == "inertia":
        # Inertia baseline = explore. We don't repeat last_action because
        # that locks the agent into walking in one direction forever —
        # the cycling in _explore_direction is what actually moves the
        # agent through the world to find resources.
        return _explore_direction(sim)

    target = step.target

    if step.action == "do":
        # Attempt 2: proximity-based "do". If the target is in a dynamic
        # entity or spatial_map within manhattan 1, emit "do". Don't rely
        # on _nearest_concept facing direction.
        target_pos = None
        # Dynamic entities (mobs) take precedence
        for e in sim.dynamic_entities:
            if e.concept_id == target:
                target_pos = e.pos
                break
        # Fall back to spatial_map
        if target_pos is None and target and sim.spatial_map is not None:
            target_pos = sim.spatial_map.find_nearest(target, sim.player_pos)

        if target_pos is not None:
            dist = abs(target_pos[0] - sim.player_pos[0]) + abs(target_pos[1] - sim.player_pos[1])
            if dist <= 1:
                return "do"
            return _step_toward_target(sim, target_pos)

        # Unknown target location — explore to find it
        return _explore_direction(sim)

    if step.action == "place":
        effect = step.rule.effect if step.rule else None
        item = None
        if effect and effect.world_place:
            item = effect.world_place[0]
        if item is None:
            return _explore_direction(sim)
        near = _nearest_concept(sim)
        if near == "empty":
            return f"place_{item}"
        return _explore_direction(sim)

    if step.action == "make":
        near_target = step.near or (step.rule.concept if step.rule else None)
        effect = step.rule.effect if step.rule else None
        result_item = None
        if effect:
            positives = [k for k, v in effect.inventory_delta.items() if v > 0]
            if positives:
                result_item = positives[0]
        if result_item is None:
            return _explore_direction(sim)
        if _nearest_concept(sim) == near_target:
            return f"make_{result_item}"
        if near_target and sim.spatial_map is not None:
            target_pos = sim.spatial_map.find_nearest(near_target, sim.player_pos)
            if target_pos is not None:
                return _step_toward_target(sim, target_pos)
        return _explore_direction(sim)

    if step.action == "sleep":
        return "sleep"

    return _explore_direction(sim)


def _direction_primitive(
    current: tuple[int, int], target: tuple[int, int]
) -> str:
    """Convert a (dx, dy) step into a move_* primitive."""
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if dx > 0:
        return "move_right"
    if dx < 0:
        return "move_left"
    if dy > 0:
        return "move_down"
    if dy < 0:
        return "move_up"
    return "move_right"


def _is_plan_step_complete(
    step: PlannedStep,
    sim: SimState,
    prev_inv: dict[str, int],
    prev_enemies: set[str],
) -> bool:
    """Did the current PlannedStep's rule fire this tick?

    Checked by observing the delta on relevant state since the tick began.
    """
    if step.action == "inertia":
        return False  # never completes

    if step.rule is None or step.rule.effect is None:
        return False

    effect = step.rule.effect

    if effect.kind in ("gather", "craft", "consume", "self"):
        # Inventory/body delta from this rule fired?
        for item, delta in effect.inventory_delta.items():
            if delta > 0 and sim.inventory.get(item, 0) > prev_inv.get(item, 0):
                return True
        for var, delta in effect.body_delta.items():
            if delta > 0:
                # We can't easily know prev_body here — approximate by checking
                # the inventory dict (body vars that came from body_delta consume)
                if sim.inventory.get(var, 0) > prev_inv.get(var, 0):
                    return True
        return False

    if effect.kind == "remove":
        # Target entity was in prev_enemies but not now
        current_enemies = {e.concept_id for e in sim.dynamic_entities}
        return (
            effect.scene_remove in prev_enemies
            and effect.scene_remove not in current_enemies
        )

    if effect.kind == "place":
        # Placed item should now appear in inventory_delta (-2 wood tracked)
        # More reliable: check that a dynamic state change happened
        if effect.world_place:
            item = effect.world_place[0]
            # After place, inventory.wood decreases — check that delta
            for inv_var, inv_delta in effect.inventory_delta.items():
                if inv_delta < 0 and sim.inventory.get(inv_var, 0) < prev_inv.get(inv_var, 0):
                    return True
        return False

    return False


def _apply_effect_to_sim(
    sim: SimState,
    effect: RuleEffect,
    traj: Trajectory,
    tick: int,
    source: str,
) -> None:
    """Mutate sim state according to rule effect. Log events."""
    for item, delta in effect.inventory_delta.items():
        sim.inventory[item] = sim.inventory.get(item, 0) + delta
        traj.events.append(SimEvent(
            step=tick, kind="inv_gain", var=item, amount=delta, source=source,
        ))
    for var, delta in effect.body_delta.items():
        sim.body[var] = sim.body.get(var, 0.0) + delta
        traj.events.append(SimEvent(
            step=tick, kind="body_delta", var=var, amount=delta, source=source,
        ))
    if effect.scene_remove:
        # Remove first matching entity
        for i, entity in enumerate(sim.dynamic_entities):
            if entity.concept_id == effect.scene_remove:
                sim.dynamic_entities.pop(i)
                traj.events.append(SimEvent(
                    step=tick, kind="entity_removed",
                    var=None, amount=0.0, source=source,
                ))
                break


def _rule_prevents(rule: CausalLink, failure: Failure) -> bool:
    """Does applying this rule counteract the observed failure?"""
    if rule.effect is None:
        return False
    effect = rule.effect

    if failure.kind == "var_depleted" and failure.var:
        # Rule raises the depleted variable (directly or via stateful regen)
        if effect.body_delta.get(failure.var, 0) > 0:
            return True
        return False

    if failure.kind == "attributed_to" and failure.cause:
        # Rule removes the damaging entity from scene
        if effect.scene_remove == failure.cause:
            return True
        return False

    return False
