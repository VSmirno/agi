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
    result: str = ""  # DEPRECATED: string outcome id, kept for backward compat (removed in Commit 8)
    requires: dict[str, int] = field(default_factory=dict)  # {wood_pickaxe: 1}
    condition: str | None = None  # "nearby", None
    confidence: float = 0.5  # starts at 0.5 (from textbook), grows with verification

    # Stage 77a additions — structured dispatch
    kind: str = "action_triggered"  # discriminator for new effect-based dispatch
    effect: "RuleEffect | None" = None  # structured effect (replaces `result`)
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

    def plan(
        self,
        goal_id: str,
        inventory: dict[str, int] | None = None,
    ) -> list[PlannedStep]:
        """Backward chaining: find sequence of steps to achieve goal.

        Returns forward-ordered list of PlannedStep (ready to execute).

        Args:
            goal_id: target concept id to produce.
            inventory: current inventory. If provided, prerequisites already
                       present are skipped — avoids redundant steps when the
                       agent already holds required items (e.g. don't re-craft
                       sword if already wielding one).
        """
        steps: list[PlannedStep] = []
        visited: set[str] = set()
        inv = dict(inventory) if inventory else {}
        self._plan_recursive(goal_id, steps, visited, inv)
        return steps

    def _plan_recursive(
        self,
        goal_id: str,
        steps: list[PlannedStep],
        visited: set[str],
        inventory: dict[str, int] | None = None,
    ) -> None:
        """Backward chaining helper. Builds plan in forward order.

        Skips subgoals whose result is already in inventory (quantity ≥ 1).
        """
        if goal_id in visited:
            return
        visited.add(goal_id)

        # Skip if the goal itself is already satisfied in inventory
        # (e.g., don't plan 'make wood_sword' if sword already held)
        if inventory is not None and inventory.get(goal_id, 0) >= 1:
            return

        # Find which concept produces this goal
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if link.result != goal_id:
                    continue

                # For make/place: ensure the near-target concept exists
                # e.g. "make wood_pickaxe near table" → need table first
                if link.action in ("make", "place"):
                    self._plan_recursive(concept.id, steps, visited, inventory)

                # Recurse into prerequisites (inventory items)
                for req_item in link.requires:
                    self._plan_recursive(req_item, steps, visited, inventory)

                steps.append(
                    PlannedStep(
                        action=link.action,
                        target=concept.id,
                        near=concept.id if link.action in ("make", "place") else None,
                        rule=link,
                        expected_gain=link.result,
                        requires=dict(link.requires),
                    )
                )
                return  # found a way to produce goal_id

    # --- Verification ---

    def verify(
        self, concept_id: str, action: str, actual_outcome: str | None
    ) -> None:
        """Update confidence based on prediction vs actual outcome."""
        concept = self.query_text(concept_id)
        if concept is None:
            return

        for link in concept.causal_links:
            if link.action != action:
                continue
            if actual_outcome == link.result:
                link.confidence = min(1.0, link.confidence + CONFIRM_DELTA)
            else:
                link.confidence = max(0.0, link.confidence - REFUTE_DELTA)
            return

    def record_surprise(self, outcome: str, action: str, near: str | None = None) -> None:
        """Log an unexpected event (no matching rule)."""
        self.surprises.append(SurpriseEvent(outcome=outcome, action=action, near=near))

    # --- Prediction helper ---

    def predict_before_action(
        self, near: str, action: str, inventory: dict[str, int]
    ) -> Prediction | None:
        """Create a prediction for upcoming action."""
        concept = self.query_text(near)
        if concept is None:
            return None
        link = concept.find_causal(action=action, check_requires=inventory)
        if link is None:
            return None
        return Prediction(
            concept_id=near,
            action=action,
            expected=link.result,
            confidence=link.confidence,
            link=link,
        )

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
        """Save concept store to directory (JSON metadata + tensors)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {}
        tensors = {}
        for cid, concept in self.concepts.items():
            meta[cid] = {
                "attributes": concept.attributes,
                "confidence": concept.confidence,
                "causal_links": [
                    {
                        "action": l.action,
                        "result": l.result,
                        "requires": l.requires,
                        "condition": l.condition,
                        "confidence": l.confidence,
                    }
                    for l in concept.causal_links
                ],
            }
            if concept.visual is not None:
                tensors[f"{cid}_visual"] = concept.visual
            if concept.text_sdr is not None:
                tensors[f"{cid}_text_sdr"] = concept.text_sdr
            for i, obs in enumerate(concept.observations):
                tensors[f"{cid}_obs_{i}"] = obs

        (p / "concepts.json").write_text(json.dumps(meta, indent=2))
        if tensors:
            torch.save(tensors, p / "tensors.pt")

    def load(self, path: str) -> None:
        """Load concept store from directory."""
        p = Path(path)
        meta = json.loads((p / "concepts.json").read_text())
        tensors_path = p / "tensors.pt"
        tensors = torch.load(tensors_path, weights_only=True) if tensors_path.exists() else {}

        for cid, info in meta.items():
            concept = self.register(cid, info["attributes"])
            concept.confidence = info["confidence"]
            for ldata in info["causal_links"]:
                concept.causal_links.append(
                    CausalLink(
                        action=ldata["action"],
                        result=ldata["result"],
                        requires=ldata.get("requires", {}),
                        condition=ldata.get("condition"),
                        confidence=ldata.get("confidence", 0.5),
                    )
                )
            vis_key = f"{cid}_visual"
            if vis_key in tensors:
                concept.visual = tensors[vis_key]
            sdr_key = f"{cid}_text_sdr"
            if sdr_key in tensors:
                concept.text_sdr = tensors[sdr_key]
            # Load observations
            for i in range(MAX_OBSERVATIONS):
                obs_key = f"{cid}_obs_{i}"
                if obs_key in tensors:
                    concept.observations.append(tensors[obs_key])
                else:
                    break

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

        Assumes `inventory` is a working copy we can mutate to reflect
        prerequisites already satisfied by earlier plan steps.
        """
        # Identity key for this rule: (concept, action)
        key = (rule.concept or "", rule.action or "_passive")
        if key in visited:
            return
        visited.add(key)

        # Resolve item prerequisites
        for item, count in rule.requires.items():
            if inventory.get(item, 0) >= count:
                continue
            producer = self._find_rule_producing_item(item)
            if producer is not None:
                self._plan_for_rule(producer, steps, visited, inventory)
                # Assume the producer adds to inventory (for further chaining)
                if producer.effect:
                    for k, v in producer.effect.inventory_delta.items():
                        inventory[k] = inventory.get(k, 0) + v

        # Resolve spatial preconditions for make/place (must be near concept)
        if rule.action == "make" and rule.concept:
            # Need to be adjacent to `rule.concept` (the crafting station).
            # If it's a crafted item (e.g. "table"), we need a rule that places it.
            producer = self._find_rule_producing_world_place(rule.concept)
            if producer is not None:
                self._plan_for_rule(producer, steps, visited, inventory)

        # Add this rule's execution as the final step
        steps.append(PlannedStep(
            action=rule.action or "do",
            target=rule.concept,
            near=rule.concept if rule.action in ("make", "place") else None,
            rule=rule,
            # Legacy fields for backward compat (removed in Commit 8)
            expected_gain=rule.result,
            requires=dict(rule.requires),
        ))

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

            self._apply_tick(sim, primitive, tracker, traj, tick)

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
    ) -> None:
        """One tick of simulation. Six phases in fixed order."""

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
        for rule in self.body_rate_rules():
            if rule.confidence < self.CONFIDENCE_THRESHOLD:
                continue
            var = rule.effect.body_rate_variable
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

    Prefers the least-visited neighbor. If no spatial_map info available,
    cycles through cardinal directions based on step counter so the agent
    doesn't stall on one axis.
    """
    if sim.spatial_map is not None and hasattr(sim.spatial_map, "unvisited_neighbors"):
        try:
            unvisited = sim.spatial_map.unvisited_neighbors(sim.player_pos, radius=3)
        except Exception:
            unvisited = []
        if unvisited:
            # Pick the closest unvisited neighbor (Manhattan distance)
            closest = min(
                unvisited,
                key=lambda p: abs(p[0] - sim.player_pos[0]) + abs(p[1] - sim.player_pos[1]),
            )
            return _direction_primitive(
                sim.player_pos, _step_toward_pos(sim.player_pos, closest)
            )

    # Fall back: cycle through 4 cardinal directions so the agent doesn't
    # walk in one direction forever. Uses sim.step to pick direction.
    dirs = ["move_up", "move_right", "move_down", "move_left"]
    return dirs[sim.step % 4]


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
        if _nearest_concept(sim) == target:
            return "do"
        target_pos = None
        if target and sim.spatial_map is not None:
            target_pos = sim.spatial_map.find_nearest(target, sim.player_pos)
        if target_pos is not None:
            primitive_pos = _step_toward_pos(sim.player_pos, target_pos)
            return _direction_primitive(sim.player_pos, primitive_pos)
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
                return _direction_primitive(
                    sim.player_pos, _step_toward_pos(sim.player_pos, target_pos)
                )
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
