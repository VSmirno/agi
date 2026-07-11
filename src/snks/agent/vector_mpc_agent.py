"""Stage 83: Vector MPC Agent — MPC planning through VectorWorldModel.

Replaces the ConceptStore-based MPC loop with vector-based forward
imagination. Same structure: perceive → generate candidates → simulate
→ score → execute first primitive → learn from surprise.

Key differences from mpc_agent.py:
- generate_candidate_plans uses forward imagination (predict per concept×action)
- simulate_forward works through VectorWorldModel.predict + decode
- score_trajectory uses total_gain (cumulative) not binary has_gain
- Surprise-driven learning: every step updates model from observation
- Entity-correlated damage discovery without textbook declaration
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.capability_state import extract_capability_state
from snks.agent.textbook_promoter import TextbookPromoter
from snks.agent.perception import (
    HomeostaticTracker,
    VisualField,
    perceive_semantic_field,
    perceive_tile_field,
)
from snks.agent.vector_world_model import VectorWorldModel, bind, hamming_similarity
from snks.agent.stimuli import (
    HomeostasisStimulus,
    StimuliLayer,
    SurvivalAversion,
    VitalDeltaStimulus,
)
from snks.agent.vector_sim import (
    DynamicEntityState,
    VectorState,
    VectorPlan,
    VectorPlanStep,
    VectorTrajectory,
    simulate_forward,
    score_trajectory,
)
from snks.agent.post_mortem import DamageEvent, PostMortemAnalyzer, dominant_cause
from snks.agent.stage90_diagnostics import (
    DEFAULT_DEATH_TRACE_HORIZON,
    build_death_trace_bundle,
    infer_error_label,
    summarize_dynamic_entities,
    summarize_scored_candidates,
)
from snks.agent.stage90r_local_model import (
    build_local_advisory_entry,
    rank_local_actor_candidates,
    rank_local_action_candidates,
)
from snks.agent.stage90r_emergency_controller import (
    EmergencySafetyController,
    EmergencyWorldFacts,
)
from snks.agent.stage90r_local_policy import (
    BeliefStateEncoder,
    build_local_observation_package,
    build_local_trace_entry,
    infer_local_regime,
    nearest_hostile_distance,
)


@dataclass(frozen=True)
class StrategyOption:
    """Trace-visible strategy abstraction over existing planner mechanisms."""

    kind: str
    target: str | None = None

    @property
    def option_id(self) -> str:
        if self.target:
            return f"{self.kind}:{self.target}"
        return self.kind

    def to_trace(self) -> dict[str, str | None]:
        return {
            "id": self.option_id,
            "kind": self.kind,
            "target": self.target,
        }


@dataclass(frozen=True)
class OptionContext:
    """Compact conflict context for learned option-outcome memory."""

    health_bucket: str
    food_bucket: str
    drink_bucket: str
    energy_bucket: str
    threat_pressure: str
    local_restore: str
    capability_state: str
    intent_state: str
    progress_state: str
    goal_family: str

    def to_trace(self) -> dict[str, str]:
        return {
            "health_bucket": self.health_bucket,
            "food_bucket": self.food_bucket,
            "drink_bucket": self.drink_bucket,
            "energy_bucket": self.energy_bucket,
            "threat_pressure": self.threat_pressure,
            "local_restore": self.local_restore,
            "capability_state": self.capability_state,
            "intent_state": self.intent_state,
            "progress_state": self.progress_state,
            "goal_family": self.goal_family,
        }


# ---------------------------------------------------------------------------
# Dynamic entity tracker (lightweight, from mpc_agent)
# ---------------------------------------------------------------------------


def _promoted_nodes_path(world_model_path: str | Path | None) -> Path | None:
    """Return the schema-v1 promoted YAML sibling for a world-model snapshot."""
    if world_model_path is None:
        return None
    path = Path(world_model_path)
    return path.with_name(f"{path.stem}_promoted.yaml")


def _load_promoted_entities_into_spatial_map(
    promoter: TextbookPromoter,
    promoted_path: Path | None,
    spatial_map: CrafterSpatialMap,
) -> list[dict[str, Any]]:
    """Load persisted entity observations without asserting them in the live map.

    Promoted entity nodes are cross-episode evidence. Crafter resets placed
    objects between episodes, so writing these positions into the current
    spatial map makes the planner navigate to stale tables/furnaces and
    suppresses legitimate `place_*` plans.
    """
    if promoted_path is None:
        return []
    _ = spatial_map
    return promoter.load_nodes(promoted_path)


def _next_promoted_episode_index(prior_nodes: list[dict[str, Any]]) -> int:
    """Infer the next per-seed episode index from persisted promoted nodes."""
    last_seen = [
        int(node.get("provenance", {}).get("last_seen_episode", -1))
        for node in prior_nodes
        if isinstance(node, dict)
    ]
    return max(last_seen, default=-1) + 1


class DynamicEntityTracker:
    """Track positions of moving entities between ticks.

    Stage 89 preparation:
    - retain previous positions across updates
    - infer simple per-entity velocity from consecutive observations
    - keep entities alive for one missed frame to tolerate segmenter flicker
    """

    def __init__(self) -> None:
        self._dynamic_concepts: set[str] = set()
        self._positions: dict[str, list[tuple[int, int]]] = {}
        self._prev_positions: dict[str, list[tuple[int, int]]] = {}
        self._states: dict[str, list[DynamicEntityState]] = {}
        self._step = 0

    def register_dynamic_concept(self, concept_id: str) -> None:
        self._dynamic_concepts.add(concept_id)

    def update(self, vf: VisualField, player_pos: tuple[int, int]) -> None:
        self._step += 1
        self._prev_positions = {
            cid: list(positions) for cid, positions in self._positions.items()
        }
        prev_states = {
            cid: list(states) for cid, states in self._states.items()
        }
        self._positions = {}
        self._states = {}
        px, py = int(player_pos[0]), int(player_pos[1])
        center_row, center_col = 3, 4  # 7x9 viewport center
        for cid, _conf, gy, gx in vf.detections:
            if cid in self._dynamic_concepts:
                wx = px + (gx - center_col)
                wy = py + (gy - center_row)
                self._positions.setdefault(cid, []).append((wx, wy))
        for cid in self._dynamic_concepts:
            current_positions = list(self._positions.get(cid, []))
            previous = list(self._prev_positions.get(cid, []))
            prev_state_list = list(prev_states.get(cid, []))

            used_prev: set[int] = set()
            states: list[DynamicEntityState] = []
            for pos in current_positions:
                match_idx = self._nearest_prev_index(pos, previous, used_prev)
                if match_idx is None:
                    states.append(DynamicEntityState(
                        concept_id=cid,
                        position=pos,
                        velocity=None,
                        age=0,
                        last_seen_step=self._step,
                    ))
                    continue
                used_prev.add(match_idx)
                prev_pos = previous[match_idx]
                old_age = 0
                if match_idx < len(prev_state_list):
                    old_age = prev_state_list[match_idx].age
                states.append(DynamicEntityState(
                    concept_id=cid,
                    position=pos,
                    velocity=(pos[0] - prev_pos[0], pos[1] - prev_pos[1]),
                    age=old_age + 1,
                    last_seen_step=self._step,
                ))

            # One-frame persistence for missed detections (especially arrow flicker).
            for idx, prev_state in enumerate(prev_state_list):
                if idx in used_prev:
                    continue
                if self._step - prev_state.last_seen_step <= 1:
                    states.append(prev_state)

            if states:
                self._states[cid] = states
                self._positions[cid] = [s.position for s in states]

    def visible_entities(self) -> list[tuple[str, tuple[int, int]]]:
        result = []
        for cid, states in self._states.items():
            for state in states:
                result.append((cid, state.position))
        return result

    def current(self) -> list[DynamicEntityState]:
        result = []
        for states in self._states.values():
            result.extend(states)
        return result

    def current_for(self, concept_id: str) -> list[DynamicEntityState]:
        return list(self._states.get(concept_id, []))

    def min_distance(self, concept_id: str, player_pos: tuple[int, int]) -> int | None:
        positions = [state.position for state in self._states.get(concept_id, [])]
        if not positions:
            return None
        px, py = player_pos
        return min(abs(px - ex) + abs(py - ey) for ex, ey in positions)

    @staticmethod
    def _nearest_prev_index(
        pos: tuple[int, int],
        previous: list[tuple[int, int]],
        used_prev: set[int],
    ) -> int | None:
        best_idx: int | None = None
        best_dist: int | None = None
        for idx, prev in enumerate(previous):
            if idx in used_prev:
                continue
            dist = abs(pos[0] - prev[0]) + abs(pos[1] - prev[1])
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx


# ---------------------------------------------------------------------------
# Forward imagination: generate candidate plans
# ---------------------------------------------------------------------------

PredictionCache = dict[tuple[str, str], tuple[torch.Tensor, float]]


def build_prediction_cache(
    model: VectorWorldModel,
    known_concepts: set[str],
    target_actions: list[str],
) -> PredictionCache:
    """Precompute predictions for all (concept, action) pairs in one GPU op."""
    pairs = [(c, a) for c in known_concepts for a in target_actions]
    return model.batch_predict(pairs)


def _cached_predict(
    cache: PredictionCache,
    model: VectorWorldModel,
    concept_id: str,
    action: str,
) -> tuple[torch.Tensor, float]:
    """Lookup in cache; fall back to individual predict on miss."""
    key = (concept_id, action)
    if key in cache:
        return cache[key]
    return model.predict(concept_id, action)


def _adjacent_concepts(
    spatial_map: CrafterSpatialMap,
    player_pos: tuple[int, int] | None,
) -> set[str]:
    """Concepts present in the four cardinal-adjacent tiles of player_pos."""
    if player_pos is None or not hasattr(spatial_map, "concept_at"):
        return set()
    px, py = player_pos
    out: set[str] = set()
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        concept = spatial_map.concept_at((px + dx, py + dy))
        if concept is not None:
            out.add(concept)
    return out


def _remove_entity_target_from_textbook(
    textbook: Any | None,
    *,
    action: str,
    target: str | None,
) -> str | None:
    """Return expected removed entity for an action/target fact, if declared."""
    if textbook is None or target is None:
        return None
    for rule in getattr(textbook, "rules", []):
        if rule.get("action") != action or rule.get("target") != target:
            continue
        effect = rule.get("effect", {}) or {}
        remove_entity = effect.get("remove_entity")
        if remove_entity:
            return str(remove_entity)
    return None


def _positive_body_effect_from_textbook(
    textbook: Any | None,
    *,
    action: str,
    target: str | None,
) -> dict[str, float]:
    """Return textbook body gains for an action/target fact."""
    if textbook is None or target is None:
        return {}
    effects: dict[str, float] = {}
    for rule in getattr(textbook, "rules", []):
        if rule.get("action") != action or rule.get("target") != target:
            continue
        effect = rule.get("effect", {}) or {}
        body = effect.get("body", {}) or {}
        for key, value in body.items():
            gain = float(value)
            if gain > 0:
                effects[str(key)] = max(gain, effects.get(str(key), 0.0))
    return effects


def _has_immediate_emergency_threat(
    *,
    nearest_threat_distances: dict[str, int | None],
    emergency_facts: EmergencyWorldFacts,
) -> bool:
    for concept_id, distance in nearest_threat_distances.items():
        if distance is None:
            continue
        if int(distance) <= emergency_facts.emergency_range(str(concept_id)):
            return True
    return False


def _opportunistic_survival_plan(
    *,
    textbook: Any | None,
    model: VectorWorldModel,
    inventory: dict[str, int],
    body: dict[str, float],
    near_concept: str | None,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    nearest_threat_distances: dict[str, int | None],
    emergency_facts: EmergencyWorldFacts,
) -> VectorPlan | None:
    """Take adjacent survival resources before vitals become critical."""
    if _has_immediate_emergency_threat(
        nearest_threat_distances=nearest_threat_distances,
        emergency_facts=emergency_facts,
    ):
        return None

    px, py = int(player_pos[0]), int(player_pos[1])
    local_targets: set[str] = set()
    if near_concept:
        local_targets.add(str(near_concept))
    if hasattr(spatial_map, "concept_at"):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            concept = spatial_map.concept_at((px + dx, py + dy))
            if concept:
                local_targets.add(str(concept))

    candidates: list[tuple[float, str]] = []
    for target in sorted(local_targets):
        if target in {"empty", "unknown", "None"}:
            continue
        if not model.requirements_met(target, "do", inventory):
            continue
        gains = _positive_body_effect_from_textbook(
            textbook,
            action="do",
            target=target,
        )
        depleted_vitals = [
            float(body.get(vital, 9.0))
            for vital, gain in gains.items()
            if gain > 0 and float(body.get(vital, 9.0)) < 9.0
        ]
        if depleted_vitals:
            candidates.append((min(depleted_vitals), target))

    if not candidates:
        return None
    _vital_level, target = min(candidates)
    return VectorPlan(
        steps=[VectorPlanStep(action="do", target=target)],
        origin=f"opportunistic:{target}:do_survival_buffer",
    )


def _is_interaction_target_present(
    *,
    target: str,
    near_concept: str | None,
    dynamic_entities: list[DynamicEntityState],
) -> bool:
    if near_concept == target:
        return True
    return any(entity.concept_id == target for entity in dynamic_entities)


def _should_continue_interaction(
    *,
    interaction_intent: dict[str, Any] | None,
    current_goal: Goal | None,
    near_concept: str | None,
    inventory: dict[str, int],
    model: VectorWorldModel,
) -> str | None:
    """Return target to keep interacting with until its declared outcome lands."""
    if interaction_intent is None:
        return None
    target = str(interaction_intent.get("target"))
    action = str(interaction_intent.get("action"))
    if (
        current_goal is not None
        and current_goal.id == f"fight_{target}"
        and action == "do"
        and model.requirements_met(target, "do", inventory)
    ):
        return target
    return None


_INTERACTION_COMPLETION_MAX_ATTEMPTS = 3


def _action_geometry(action: str) -> dict[str, Any]:
    """Small generic action-geometry bridge for embodied interactions.

    Crafter declares this in the textbook (`do.dispatch=facing_tile`). The
    vector planner does not yet carry the full textbook schema, so keep the
    bridge action-scoped rather than target-scoped: `do` requires the target
    on the facing tile, regardless of whether the target is tree, water, cow,
    or an enemy.
    """
    if action == "do":
        return {
            "operates_on": "facing_tile",
            "requires_relation": "facing",
            "range": 1,
        }
    return {
        "operates_on": "self",
        "requires_relation": "none",
        "range": 0,
    }


def _positive_expected_effect(
    *,
    model: VectorWorldModel,
    state: VectorState,
    target: str,
    action: str,
) -> dict[str, int] | None:
    if not model.requirements_met(target, action, state.inventory):
        return None
    effect_vec, confidence = model.predict(target, action)
    if confidence < 0.2:
        return None
    decoded = model.decode_effect(effect_vec)
    expected: dict[str, int] = {}
    for var, delta in decoded.items():
        if int(delta) <= 0:
            continue
        if var in state.body and float(state.body.get(var, 0.0)) >= 9.0:
            continue
        expected[var] = int(delta)
    return expected or None


def _expected_remove_entity_outcome(
    *,
    textbook: Any | None,
    model: VectorWorldModel,
    inventory: dict[str, int],
    target: str,
    action: str,
) -> dict[str, str] | None:
    """Return a declared remove-entity outcome when the action is feasible."""
    expected_remove = _remove_entity_target_from_textbook(
        textbook,
        action=action,
        target=target,
    )
    if expected_remove is None:
        return None
    if not model.requirements_met(target, action, inventory):
        return None
    return {"remove_entity": expected_remove}


def _interaction_relation_state(
    *,
    action: str,
    player_pos: tuple[int, int],
    target_pos: tuple[int, int] | None,
    last_move: str | None,
) -> dict[str, Any]:
    geometry = _action_geometry(action)
    relation = str(geometry["requires_relation"])
    if target_pos is None:
        return {
            "relation": relation,
            "distance": None,
            "is_adjacent": False,
            "is_facing_target": False,
        }
    px, py = int(player_pos[0]), int(player_pos[1])
    tx, ty = int(target_pos[0]), int(target_pos[1])
    dx, dy = tx - px, ty - py
    distance = abs(dx) + abs(dy)
    is_adjacent = distance <= int(geometry["range"])
    if relation == "facing" and distance == 1:
        facing = _facing_delta(last_move)
        if facing == (0, 0):
            facing = (0, 1)
        is_facing = (dx, dy) == facing
    else:
        is_facing = relation in ("none", "")
    return {
        "relation": relation,
        "distance": int(distance),
        "is_adjacent": bool(is_adjacent),
        "is_facing_target": bool(is_facing),
    }


def _interaction_intent_from_goal_target(
    *,
    model: VectorWorldModel,
    state: VectorState,
    active_goal: Goal | None,
    spatial_map: CrafterSpatialMap,
    dynamic_entities: list[DynamicEntityState],
    player_pos: tuple[int, int],
    existing_intent: dict[str, Any] | None,
    step: int,
    textbook: Any | None = None,
) -> dict[str, Any] | None:
    """Create or refresh a generic completion intent for the active goal target."""
    if active_goal is None or not active_goal.target_concept:
        return existing_intent
    target = str(active_goal.target_concept)
    action = "do"
    expected_outcome = _expected_remove_entity_outcome(
        textbook=textbook,
        model=model,
        inventory=state.inventory,
        target=target,
        action=action,
    )
    expected_effect = None
    if expected_outcome is None:
        expected_effect = _positive_expected_effect(
            model=model,
            state=state,
            target=target,
            action=action,
        )
    if expected_effect is None and expected_outcome is None:
        return existing_intent
    target_pos = _nearest_known_target_position(
        target,
        player_pos,
        spatial_map,
        dynamic_entities,
    )
    if target_pos is None:
        return existing_intent

    if (
        existing_intent is not None
        and existing_intent.get("action") == action
        and existing_intent.get("target") == target
        and existing_intent.get("expected_effect") == (expected_effect or {})
        and existing_intent.get("expected_outcome") == (expected_outcome or {})
    ):
        refreshed = dict(existing_intent)
        refreshed["target_pos"] = [int(target_pos[0]), int(target_pos[1])]
        return refreshed

    return {
        "action": action,
        "target": target,
        "expected_effect": expected_effect or {},
        "expected_outcome": expected_outcome or {},
        "target_pos": [int(target_pos[0]), int(target_pos[1])],
        "started_step": int(step),
        "attempts": 0,
        "status": "active",
    }


def _select_interaction_completion_plan(
    *,
    interaction_intent: dict[str, Any] | None,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    dynamic_entities: list[DynamicEntityState],
    last_move: str | None,
) -> tuple[VectorPlan | None, dict[str, Any] | None]:
    """Select the next approach/align/act plan for an active interaction intent."""
    if interaction_intent is None:
        return None, None

    action = str(interaction_intent.get("action"))
    target = str(interaction_intent.get("target"))
    expected_effect = dict(interaction_intent.get("expected_effect") or {})
    expected_outcome = dict(interaction_intent.get("expected_outcome") or {})
    attempts = int(interaction_intent.get("attempts", 0))
    target_pos = _nearest_known_target_position(
        target,
        player_pos,
        spatial_map,
        dynamic_entities,
    )
    relation_state = _interaction_relation_state(
        action=action,
        player_pos=player_pos,
        target_pos=target_pos,
        last_move=last_move,
    )
    trace = {
        "status": str(interaction_intent.get("status", "active")),
        "action": action,
        "target_concept": target,
        "target_pos": (
            [int(target_pos[0]), int(target_pos[1])]
            if target_pos is not None
            else None
        ),
        "expected_effect": expected_effect,
        "expected_outcome": expected_outcome,
        "relation": relation_state["relation"],
        "is_adjacent": bool(relation_state["is_adjacent"]),
        "is_facing_target": bool(relation_state["is_facing_target"]),
        "attempts": attempts,
        "selected_phase": None,
        "reason": None,
        "expected_effect_achieved": None,
    }

    if attempts >= _INTERACTION_COMPLETION_MAX_ATTEMPTS:
        trace.update({
            "status": "failed",
            "selected_phase": "failed",
            "reason": "max_attempts_exceeded",
        })
        return None, trace
    if target_pos is None:
        trace.update({
            "status": "failed",
            "selected_phase": "failed",
            "reason": "target_lost",
        })
        return None, trace

    distance = relation_state["distance"]
    if distance is not None and int(distance) > int(_action_geometry(action)["range"]):
        trace.update({
            "status": "approaching",
            "selected_phase": "approach",
            "reason": "target_not_reached",
        })
        return (
            VectorPlan(
                steps=[VectorPlanStep(action="navigate_known", target=target)],
                origin=f"navigate_known:{target}",
            ),
            trace,
        )

    if not bool(relation_state["is_facing_target"]):
        trace.update({
            "status": "aligning",
            "selected_phase": "align",
            "reason": "relation_not_satisfied",
        })
        return (
            VectorPlan(
                steps=[VectorPlanStep(action=action, target=target)],
                origin=f"align_interaction:{target}:{action}",
            ),
            trace,
        )

    trace.update({
        "status": "acting",
        "selected_phase": "act",
        "reason": "relation_satisfied",
    })
    return (
        VectorPlan(
            steps=[VectorPlanStep(action=action, target=target)],
            origin=f"complete_interaction:{target}:{action}",
        ),
        trace,
    )


def _expected_effect_achieved(
    expected_effect: dict[str, int],
    *,
    inventory_delta: dict[str, int],
    body_delta: dict[str, float],
) -> bool:
    if not expected_effect:
        return False
    for key, expected_delta in expected_effect.items():
        if int(expected_delta) <= 0:
            continue
        actual = float(body_delta.get(key, inventory_delta.get(key, 0)))
        # Body variables are capped by the environment, so a declared +5
        # drink effect may realise as +2 when the body is near max. The
        # completion question is whether the expected positive effect
        # occurred, not whether the uncapped textbook magnitude landed.
        if actual <= 0:
            return False
    return True


def _update_interaction_completion_after_step(
    *,
    interaction_intent: dict[str, Any] | None,
    interaction_trace: dict[str, Any] | None,
    primitive: str,
    control_origin: str,
    rescue_trigger: str | None,
    inventory_delta: dict[str, int],
    body_delta: dict[str, float],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Verify or fail an interaction completion attempt after env feedback."""
    if interaction_intent is None or interaction_trace is None:
        return interaction_intent, interaction_trace, False

    updated_intent = dict(interaction_intent)
    updated_trace = dict(interaction_trace)
    updated_trace["actual_primitive"] = primitive

    if (
        control_origin == "emergency_safety"
        and not bool(updated_trace.get("emergency_alignment_preserved", False))
    ):
        reason = f"emergency_override:{rescue_trigger or 'emergency_safety'}"
        updated_intent["status"] = "interrupted"
        updated_trace.update({
            "status": "interrupted",
            "selected_phase": "interrupted",
            "reason": reason,
            "expected_effect_achieved": False,
        })
        return updated_intent, updated_trace, True

    if (
        control_origin == "emergency_safety"
        and bool(updated_trace.get("emergency_alignment_preserved", False))
    ):
        updated_intent["status"] = "aligning"
        updated_trace.update({
            "status": "aligning",
            "selected_phase": "align",
            "reason": "emergency_alignment_required",
            "expected_effect_achieved": False,
        })
        return updated_intent, updated_trace, False

    achieved = _expected_effect_achieved(
        dict(updated_intent.get("expected_effect") or {}),
        inventory_delta=inventory_delta,
        body_delta=body_delta,
    )
    updated_trace["expected_effect_achieved"] = bool(achieved)

    selected_phase = str(updated_trace.get("selected_phase"))
    if selected_phase == "failed":
        updated_intent["status"] = "failed"
        return updated_intent, updated_trace, True

    if achieved:
        updated_intent["status"] = "verified"
        updated_trace.update({
            "status": "verified",
            "selected_phase": "verify",
            "reason": "expected_effect_achieved",
        })
        return updated_intent, updated_trace, True

    if selected_phase == "act" and primitive == str(updated_intent.get("action")):
        attempts = int(updated_intent.get("attempts", 0)) + 1
        updated_intent["attempts"] = attempts
        updated_trace["attempts"] = attempts
        if attempts >= _INTERACTION_COMPLETION_MAX_ATTEMPTS:
            updated_intent["status"] = "failed"
            updated_trace.update({
                "status": "failed",
                "selected_phase": "failed",
                "reason": "max_attempts_exceeded",
            })
            return updated_intent, updated_trace, True
        updated_intent["status"] = "acting"
        updated_trace.update({
            "status": "acting",
            "reason": "expected_effect_not_observed",
        })
        return updated_intent, updated_trace, False

    updated_intent["status"] = str(updated_trace.get("status", "active"))
    return updated_intent, updated_trace, False


def _interaction_intent_from_plan(
    *,
    textbook: Any | None,
    plan: VectorPlan,
    existing_intent: dict[str, Any] | None,
    step: int,
) -> dict[str, Any] | None:
    if not plan.steps:
        return None
    first_step = plan.steps[0]
    expected_remove = _remove_entity_target_from_textbook(
        textbook,
        action=first_step.action,
        target=first_step.target,
    )
    if expected_remove is None:
        return None
    return {
        "action": first_step.action,
        "target": first_step.target,
        "expected_effect": {},
        "expected_outcome": {"remove_entity": expected_remove},
        "started_step": (
            existing_intent.get("started_step")
            if existing_intent is not None
            else step
        ),
        "status": "continuing",
    }


def _combat_alignment_for_emergency_do(
    *,
    textbook: Any | None,
    model: VectorWorldModel,
    inventory: dict[str, int],
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    dynamic_entities: list[DynamicEntityState],
    last_move: str | None,
    near_concept: str | None,
    rng: np.random.RandomState,
    target_hint: str | None = None,
) -> tuple[str, VectorPlan, dict[str, Any]] | None:
    """Convert an emergency `do` into required hostile alignment if needed."""

    def _candidate_targets() -> list[str]:
        ordered: list[str] = []
        if target_hint:
            ordered.append(str(target_hint))
        px, py = int(player_pos[0]), int(player_pos[1])
        dynamic_adjacent = sorted(
            (
                int(abs(int(entity.position[0]) - px) + abs(int(entity.position[1]) - py)),
                str(entity.concept_id),
            )
            for entity in dynamic_entities
        )
        ordered.extend(cid for dist, cid in dynamic_adjacent if dist <= 1)
        if hasattr(spatial_map, "concept_at"):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                concept = spatial_map.concept_at((px + dx, py + dy))
                if concept:
                    ordered.append(str(concept))
        seen: set[str] = set()
        out: list[str] = []
        for target in ordered:
            if target in seen:
                continue
            seen.add(target)
            out.append(target)
        return out

    for target in _candidate_targets():
        expected_outcome = _expected_remove_entity_outcome(
            textbook=textbook,
            model=model,
            inventory=inventory,
            target=target,
            action="do",
        )
        if expected_outcome is None:
            continue
        intent = {
            "action": "do",
            "target": target,
            "expected_effect": {},
            "expected_outcome": expected_outcome,
            "started_step": 0,
            "attempts": 0,
            "status": "active",
        }
        plan, trace = _select_interaction_completion_plan(
            interaction_intent=intent,
            player_pos=player_pos,
            spatial_map=spatial_map,
            dynamic_entities=dynamic_entities,
            last_move=last_move,
        )
        if plan is None or trace is None:
            continue
        if trace.get("selected_phase") != "align":
            continue
        primitive = expand_to_primitive(
            plan.steps[0],
            player_pos,
            spatial_map,
            model,
            rng,
            last_action=last_move,
            near_concept=near_concept,
            dynamic_entities=dynamic_entities,
        )
        if primitive == "do":
            continue
        trace.update({
            "status": "aligning",
            "selected_phase": "align",
            "reason": "emergency_alignment_required",
            "actual_primitive": primitive,
            "emergency_alignment_preserved": True,
        })
        return primitive, plan, trace
    return None


def _derive_strategy_option(plan: VectorPlan) -> StrategyOption:
    """Classify an existing plan into a stable strategy option id.

    This is trace-only in Phase 1. It does not change plan ranking or action
    selection; it gives the later learned option-outcome role a stable key.
    """
    if not plan.steps:
        return StrategyOption("baseline_motion")

    origin = str(plan.origin)
    first = plan.steps[0]
    if origin.startswith(("align_interaction:", "complete_interaction:")):
        return StrategyOption("complete_interaction", f"{first.target}:{first.action}")
    if origin.startswith("continue:"):
        return StrategyOption("continue_interaction", first.target)
    if origin.startswith("opportunistic:"):
        return StrategyOption("take_local_survival", first.target)
    if first.action == "navigate_known":
        return StrategyOption("seek_known", first.target)
    if first.action == "frontier_seek":
        return StrategyOption("seek_frontier", first.target)
    if first.action in {"make", "place"}:
        return StrategyOption("craft_capability", first.target)
    if first.action == "sleep":
        return StrategyOption("recover_self", "sleep")
    if first.action == "do" and first.target not in {None, "self"}:
        return StrategyOption("engage_target", first.target)
    if first.target not in {None, "self"}:
        return StrategyOption("seek_known", first.target)
    return StrategyOption("baseline_motion")


def _vital_bucket(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v <= 2.0:
        return "critical"
    if v <= 4.0:
        return "low"
    return "ok"


def _goal_family(goal: Goal | None) -> str:
    if goal is None:
        return "none"
    goal_id = str(goal.id)
    if "_" not in goal_id:
        return goal_id
    return goal_id.split("_", 1)[0]


def _threat_pressure(
    nearest_threat_distances: dict[str, int | None],
    emergency_facts: EmergencyWorldFacts,
) -> str:
    active = 0
    contact = False
    near = False
    for concept_id, distance in nearest_threat_distances.items():
        if distance is None:
            continue
        d = int(distance)
        if d <= emergency_facts.emergency_range(str(concept_id)):
            active += 1
            near = True
        if d <= 1:
            contact = True
    if active >= 2:
        return "multi"
    if contact:
        return "contact"
    if near:
        return "near"
    return "none"


def _local_restore_context(
    *,
    textbook: Any | None,
    model: VectorWorldModel,
    inventory: dict[str, int],
    body: dict[str, float],
    near_concept: str | None,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
) -> str:
    px, py = int(player_pos[0]), int(player_pos[1])
    local_targets: set[str] = set()
    if near_concept:
        local_targets.add(str(near_concept))
    if hasattr(spatial_map, "concept_at"):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            concept = spatial_map.concept_at((px + dx, py + dy))
            if concept:
                local_targets.add(str(concept))

    vitals: set[str] = set()
    for target in sorted(local_targets):
        if target in {"empty", "unknown", "None"}:
            continue
        if not model.requirements_met(target, "do", inventory):
            continue
        for vital, gain in _positive_body_effect_from_textbook(
            textbook,
            action="do",
            target=target,
        ).items():
            if gain > 0 and float(body.get(vital, 9.0)) < 9.0:
                vitals.add(str(vital))
    if not vitals:
        return "none"
    if len(vitals) > 1:
        return "multi"
    return next(iter(vitals))


def _build_option_context(
    *,
    body: dict[str, float],
    inventory: dict[str, int],
    capability_state: Any,
    current_goal: Goal | None,
    interaction_intent: dict[str, Any] | None,
    best_plan: VectorPlan,
    nearest_threat_distances: dict[str, int | None],
    emergency_facts: EmergencyWorldFacts,
    textbook: Any | None,
    model: VectorWorldModel,
    near_concept: str | None,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
) -> OptionContext:
    """Build compact trace context for later option-outcome learning."""
    if interaction_intent is not None:
        intent_state = "continuing_interaction"
    elif best_plan.steps and best_plan.steps[0].action in {
        "frontier_seek",
        "navigate_known",
    }:
        intent_state = "seeking_resource"
    elif best_plan.steps and best_plan.steps[0].action in {"make", "place"}:
        intent_state = "crafting_capability"
    else:
        intent_state = "none"

    return OptionContext(
        health_bucket=_vital_bucket(body.get("health")),
        food_bucket=_vital_bucket(body.get("food")),
        drink_bucket=_vital_bucket(body.get("drink")),
        energy_bucket=_vital_bucket(body.get("energy")),
        threat_pressure=_threat_pressure(nearest_threat_distances, emergency_facts),
        local_restore=_local_restore_context(
            textbook=textbook,
            model=model,
            inventory=inventory,
            body=body,
            near_concept=near_concept,
            player_pos=player_pos,
            spatial_map=spatial_map,
        ),
        capability_state=(
            "armed_melee"
            if bool(getattr(capability_state, "armed_melee", False))
            else "unarmed"
        ),
        intent_state=intent_state,
        progress_state="normal",
        goal_family=_goal_family(current_goal),
    )


_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up": (0, -1),
    "move_down": (0, 1),
}


def _nearest_known_target_position(
    target: str | None,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    dynamic_entities: list[DynamicEntityState] | None = None,
) -> tuple[int, int] | None:
    """Resolve the closest mapped or tracked instance of a target concept."""
    if target in (None, "self"):
        return None
    candidates: list[tuple[int, int]] = []
    spatial_pos = spatial_map.find_nearest(str(target), player_pos)
    if spatial_pos is not None:
        candidates.append(spatial_pos)
    if dynamic_entities:
        candidates.extend(
            tuple(entity.position)
            for entity in dynamic_entities
            if entity.concept_id == target
        )
    if not candidates:
        return None
    px, py = int(player_pos[0]), int(player_pos[1])
    return min(candidates, key=lambda p: abs(int(p[0]) - px) + abs(int(p[1]) - py))


def _known_move_blocked(
    next_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    dynamic_entities: list[DynamicEntityState] | None,
) -> bool:
    """Return True when local experience says the move target is blocked."""
    if hasattr(spatial_map, "is_blocked") and spatial_map.is_blocked(next_pos):
        return True
    if dynamic_entities:
        return any(tuple(entity.position) == tuple(next_pos) for entity in dynamic_entities)
    return False


def _known_target_move_candidates(
    *,
    target: str | None,
    player_pos: tuple[int, int],
    target_pos: tuple[int, int] | None,
    spatial_map: CrafterSpatialMap,
    model: VectorWorldModel,
    dynamic_entities: list[DynamicEntityState] | None,
) -> list[dict[str, Any]]:
    """Rank one-step movement primitives by target-distance improvement."""
    if target_pos is None:
        return []
    px, py = int(player_pos[0]), int(player_pos[1])
    tx, ty = int(target_pos[0]), int(target_pos[1])
    dist_before = abs(tx - px) + abs(ty - py)
    preferred_order = ("move_right", "move_left", "move_down", "move_up")
    move_actions = [action for action in preferred_order if action in model.actions]
    move_actions.extend(
        sorted(
            action
            for action in model.actions
            if action.startswith("move_") and action not in move_actions
        )
    )

    candidates: list[dict[str, Any]] = []
    for action in move_actions:
        delta = _MOVE_DELTAS.get(action)
        if delta is None:
            continue
        nx, ny = px + delta[0], py + delta[1]
        dist_after = abs(tx - nx) + abs(ty - ny)
        blocked = _known_move_blocked(
            (nx, ny),
            spatial_map=spatial_map,
            dynamic_entities=dynamic_entities,
        )
        candidates.append({
            "action": action,
            "target": target,
            "next_pos": [int(nx), int(ny)],
            "blocked": bool(blocked),
            "dist_before": int(dist_before),
            "dist_after": int(dist_after),
            "reduces_distance": bool(dist_after < dist_before),
        })

    candidates.sort(
        key=lambda c: (
            bool(c["blocked"]),
            not bool(c["reduces_distance"]),
            int(c["dist_after"]),
            str(c["action"]),
        )
    )
    return candidates


def _select_known_target_move(
    *,
    target: str | None,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    model: VectorWorldModel,
    dynamic_entities: list[DynamicEntityState] | None,
) -> tuple[str | None, tuple[int, int] | None, list[dict[str, Any]]]:
    target_pos = _nearest_known_target_position(
        target,
        player_pos,
        spatial_map,
        dynamic_entities,
    )
    candidates = _known_target_move_candidates(
        target=target,
        player_pos=player_pos,
        target_pos=target_pos,
        spatial_map=spatial_map,
        model=model,
        dynamic_entities=dynamic_entities,
    )
    if not candidates:
        return None, target_pos, candidates
    return str(candidates[0]["action"]), target_pos, candidates


def generate_candidate_plans(
    model: VectorWorldModel,
    state: VectorState,
    spatial_map: CrafterSpatialMap,
    visible_concepts: set[str],
    beam_width: int = 5,
    max_depth: int = 3,
    cache: PredictionCache | None = None,
    enable_motion_plans: bool = True,
    enable_motion_chains: bool = True,
    player_pos: tuple[int, int] | None = None,
    active_goal: "Goal | None" = None,
) -> list[VectorPlan]:
    """Generate plans via forward imagination.

    For each visible/known concept × action with positive predicted effect,
    build a plan. Then recursively extend promising plans (generate_chains).

    Phase-1 addition: if `active_goal.target_concept` names a textbook
    concept that is not yet on the cognitive map, emit a single
    `frontier:<concept>` plan so the planner can score directed exploration
    against the baseline plan instead of falling through to uniform RNG.

    Stage9X addition: if that target is already known but not adjacent,
    emit `navigate_known:<concept>` so goal progress can reward a concrete
    distance-reducing approach step instead of relying on baseline motion.
    """
    candidates: list[VectorPlan] = []

    # Gather all known concepts. Three sources:
    #   visible_concepts → what perception sees this tick
    #   spatial_map.known_objects → resource tiles remembered from prior ticks
    #   state.dynamic_entities → cow/zombie/skeleton/arrow currently tracked
    #     by DynamicEntityTracker (may sit outside the viewport but the
    #     tracker still knows their position). Without this third source the
    #     planner generates no `single:cow:do` plan once cow walks off
    #     screen, and the agent ignores it entirely (seed 17 ep 0 finding).
    known = (
        set(visible_concepts)
        | set(spatial_map.known_objects.keys())
        | {entity.concept_id for entity in state.dynamic_entities}
    )

    # Single-step plans: try each concept × target-action
    action_ids = list(model.actions.keys())
    target_actions = [a for a in action_ids if a in ("do", "make")]
    self_actions = [a for a in action_ids if a in ("sleep",)]
    move_actions = [a for a in action_ids if a.startswith("move_")]

    if cache is None:
        cache = build_prediction_cache(model, known, target_actions)

    # Concepts that are never valid resource/crafting plan targets.
    # "empty" — background tile, no resource to gather.
    # "self"  — handled separately via self_actions (sleep).
    # zombie/skeleton are intentionally absent: textbook declares
    # `do <entity> requires {wood_sword}` so the requirements-only loop
    # below emits `single:<entity>:do` plans when the agent is armed.
    # Before the Phase-2 removal of those names from non_targetable, the
    # first loop's positive-effect gate filtered them out and the planner
    # generated no combat plan at all even with `wood_sword=2` in inventory.
    non_targetable = {"empty", "self"}

    for concept_id in known:
        if concept_id in non_targetable:
            continue
        for action in target_actions:
            # For "make": only allow concepts that have a textbook requirement entry.
            # Blocks spurious SDM associations (diamond:make, coal:make, etc.)
            # that were never declared in the textbook.
            if action == "make" and not _is_declared_crafting_rule(model, concept_id, action):
                continue
            # Requirement check — facts from textbook (category 1)
            if not model.requirements_met(concept_id, action, state.inventory):
                continue
            effect_vec, confidence = _cached_predict(cache, model, concept_id, action)
            if confidence < 0.2:
                continue
            decoded = model.decode_effect(effect_vec)
            if _has_positive_effect(decoded, state):
                candidates.append(VectorPlan(
                    steps=[VectorPlanStep(action=action, target=concept_id)],
                    origin=f"single:{concept_id}:{action}",
                ))

    # Requirement-only `do` rules, such as combat rules with
    # `effect.remove_entity`, do not produce inventory/body deltas and
    # therefore do not pass the positive-effect gate above. They are still
    # legitimate plans when the textbook declares requirements and the target
    # is visible/known; goal progress decides whether they matter.
    for (target, action), _reqs in model.action_requirements.items():
        if action != "do" or target not in known:
            continue
        if not model.requirements_met(target, action, state.inventory):
            continue
        candidates.append(VectorPlan(
            steps=[VectorPlanStep(action=action, target=target)],
            origin=f"single:{target}:{action}",
        ))

    # Self-actions as standalone plans (no target concept).
    # No confidence/effect gate — scoring handles everything:
    # sleep wins only when min_vital improves in simulation (vitals low);
    # baseline (dist=0, -steps=0) beats idle sleep when vitals full.
    for action in self_actions:
        # Suppress `sleep` plan while energy is comfortable. Crafter is
        # split into a safe early phase (no zombies / skeletons) and a
        # hostile night phase; turns spent sleeping during the safe phase
        # are lost gathering/crafting opportunity. The energy<3 threshold
        # matches the textbook-derived `sleep` goal trigger in
        # GoalSelector, so the planner only proposes sleep when the goal
        # layer would have promoted it anyway. Vitals are capped at 9.0 by
        # apply_effect, so without this gate sleep ties with motion plans
        # in scoring and the RNG fallback fires it ~1/N of the time.
        if action == "sleep":
            if state.body.get("energy", 0.0) >= 3.0:
                continue
        candidates.append(VectorPlan(
            steps=[VectorPlanStep(action=action, target="self")],
            origin=f"self:{action}",
        ))

    # Crafting plans (make/place). The targets here are inventory items or
    # placed objects (wood_pickaxe, table, …), never tile concepts in
    # `known`. Iterate `model.action_requirements` (seeded from textbook
    # `make`/`place` rules) so these plans get a chance to be picked.
    # Without this, the agent could carry 9 wood and never try to craft.
    #
    # Adjacency requirements (e.g. `make_wood_pickaxe` requires `table`
    # adjacent): handled via `model.near_requirements`. When the required
    # concept isn't adjacent, prepend `place_X` if feasible. "empty" near is
    # treated as always satisfied (the env finds an open tile).
    adj_concepts = _adjacent_concepts(spatial_map, player_pos)
    for (target, action), _reqs in model.action_requirements.items():
        if action not in ("make", "place"):
            continue
        if not _is_declared_crafting_rule(model, target, action):
            continue
        if not model.requirements_met(target, action, state.inventory):
            continue
        effect_vec, confidence = _cached_predict(cache, model, target, action)
        if confidence < 0.2:
            continue
        decoded = model.decode_effect(effect_vec)
        if not _has_positive_effect(decoded, state):
            continue

        near_req = model.near_requirements.get((target, action))
        if near_req is None or near_req == "empty" or near_req in adj_concepts:
            candidates.append(VectorPlan(
                steps=[VectorPlanStep(action=action, target=target)],
                origin=f"single:{target}:{action}",
            ))
            continue

        # Adjacency requirement not met. If an instance of the required tile
        # already exists somewhere on the map, emit the single make plan —
        # `expand_to_primitive` will navigate to it. Only when no instance
        # exists do we fall through to a `[place_near_req, make_target]`
        # chain plan that prepends a fresh placement. Without this guard the
        # planner kept generating chain plans every step the agent strayed
        # from its prior table, and the top-band RNG fired `place_table`
        # over and over, leaving rows of stacked tables on the map.
        existing_near_pos = spatial_map.find_nearest(near_req, player_pos) if player_pos is not None else None
        if existing_near_pos is not None:
            candidates.append(VectorPlan(
                steps=[VectorPlanStep(action=action, target=target)],
                origin=f"single:{target}:{action}",
            ))
            continue

        # No instance exists — prepend a place plan when feasible against
        # the post-place inventory.
        place_key = (near_req, "place")
        place_reqs = model.action_requirements.get(place_key)
        if place_reqs is None:
            continue
        if not model.requirements_met(near_req, "place", state.inventory):
            continue
        post_inv = dict(state.inventory)
        for item, cost in place_reqs.items():
            post_inv[item] = post_inv.get(item, 0) - int(cost)
        if not model.requirements_met(target, action, post_inv):
            continue
        candidates.append(VectorPlan(
            steps=[
                VectorPlanStep(action="place", target=near_req),
                VectorPlanStep(action=action, target=target),
            ],
            origin=f"chain:place_{near_req}+{action}_{target}",
        ))

    # Motion-only plans. Needed for dynamic-threat avoidance:
    # projected danger may make a move valuable even when it has no
    # immediate inventory/body gain. Scoring decides whether these lose
    # to baseline or win because they avoid future damage.
    if enable_motion_plans:
        for action in move_actions:
            candidates.append(VectorPlan(
                steps=[VectorPlanStep(action=action, target="self")],
                origin=f"self:{action}",
            ))

    if enable_motion_chains and state.dynamic_entities:
        candidates.extend(_generate_motion_chains(move_actions, max_depth=max_depth))

    # Multi-step chains via beam search (target actions only)
    chains = _generate_chains(model, state, known, target_actions,
                              beam_width=beam_width, max_depth=max_depth,
                              cache=cache)
    candidates.extend(chains)

    # Phase 1: goal-conditioned frontier exploration. When the active goal
    # names a concept (e.g. find_water → water) that the agent has never
    # observed on the cognitive map, emit a single frontier_seek plan so
    # the planner can score directed exploration against baseline instead
    # of falling through to uniform RNG. Without this, the agent burns
    # tens of steps under goal=find_water with plan_origin=baseline while
    # vitals decay (see Phase 0b audit, seed 17 ep 0).
    if active_goal is not None and active_goal.target_concept:
        target = active_goal.target_concept
        if player_pos is not None:
            known_target_pos = _nearest_known_target_position(
                target,
                player_pos,
                spatial_map,
                state.dynamic_entities,
            )
            if known_target_pos is not None:
                known_dist = (
                    abs(int(known_target_pos[0]) - int(player_pos[0]))
                    + abs(int(known_target_pos[1]) - int(player_pos[1]))
                )
                if known_dist > 1:
                    candidates.append(VectorPlan(
                        steps=[VectorPlanStep(action="navigate_known", target=target)],
                        origin=f"navigate_known:{target}",
                    ))
                target_is_known = True
            else:
                target_is_known = target in known
        else:
            target_is_known = target in known
        if target not in known and player_pos is not None:
            # Only emit when target is genuinely unknown — once the concept
            # appears on the map, a real `single:<target>:do` plan will be
            # generated by the loop above and the frontier plan is no
            # longer needed.
            existing = spatial_map.find_nearest(target, player_pos)
            if existing is None and not target_is_known:
                candidates.append(VectorPlan(
                    steps=[VectorPlanStep(action="frontier_seek", target=target)],
                    origin=f"frontier:{target}",
                ))

    # Always include a "do nothing" plan as baseline
    candidates.append(VectorPlan(steps=[], origin="baseline"))

    return candidates


def _generate_motion_chains(
    move_actions: list[str],
    max_depth: int = 3,
) -> list[VectorPlan]:
    """Generate short generic motion chains for threat-driven repositioning.

    Stage 89b: one-step motion plans are often too myopic for dynamic threats.
    These chains are still generic and threat-agnostic: they simply expand the
    planner's movement horizon without introducing enemy-specific reflex logic.
    """
    if max_depth < 2:
        return []

    opposite = {
        "move_up": "move_down",
        "move_down": "move_up",
        "move_left": "move_right",
        "move_right": "move_left",
    }
    chains: list[VectorPlan] = []

    def orthogonal(first: str) -> list[str]:
        return [
            action for action in move_actions
            if action != first and action != opposite.get(first)
        ]

    seen: set[tuple[str, ...]] = set()
    patterns: list[tuple[str, ...]] = []
    for first in move_actions:
        patterns.append((first, first))
        for second in orthogonal(first):
            patterns.append((first, second))
        if max_depth >= 3:
            patterns.append((first, first, first))
            for second in orthogonal(first):
                patterns.append((first, second, second))

    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        chains.append(VectorPlan(
            steps=[VectorPlanStep(action=action, target="self") for action in pattern],
            origin=f"self:motion_chain:{'+'.join(pattern)}",
        ))

    return chains


def _has_positive_effect(decoded: dict[str, int], state: VectorState) -> bool:
    """Check if an effect can improve inventory or currently depleted body state."""
    for var, val in decoded.items():
        if var not in state.body and val > 0:
            return True
        if var in state.body and val > 0 and float(state.body.get(var, 0.0)) < 9.0:
            return True
    return False


def _is_declared_crafting_rule(model: VectorWorldModel, target: str, action: str) -> bool:
    """Return True for real textbook make/place rules, not helper requirement keys."""
    if action not in ("make", "place"):
        return False
    return (target, action) in model.near_requirements


def _generate_chains(
    model: VectorWorldModel,
    state: VectorState,
    known_concepts: set[str],
    plan_actions: list[str],
    beam_width: int = 5,
    max_depth: int = 3,
    cache: PredictionCache | None = None,
) -> list[VectorPlan]:
    """Recursive forward search: 'if I do X, what can I do next?'

    Beam search: keep top beam_width plans at each depth.
    """
    if cache is None:
        cache = build_prediction_cache(model, known_concepts, plan_actions)

    # Start with all single-step plans that have positive effect
    beam: list[tuple[float, VectorPlan, VectorState]] = []

    non_targetable = {"empty", "self", "zombie", "skeleton"}
    for concept_id in known_concepts:
        if concept_id in non_targetable:
            continue
        for action in plan_actions:
            if action == "make" and not _is_declared_crafting_rule(model, concept_id, action):
                continue
            # First step: check requirements against actual inventory.
            if not model.requirements_met(concept_id, action, state.inventory):
                continue
            # Note: chain requirements check uses hypothetical state after
            # previous steps' predicted effects — may differ from real inv
            effect_vec, conf = _cached_predict(cache, model, concept_id, action)
            if conf < 0.2:
                continue
            decoded = model.decode_effect(effect_vec)
            if not decoded:
                continue
            new_state = state.apply_effect(decoded)
            gain = sum(v for v in decoded.values() if v > 0
                       and v not in state.body)
            if gain > 0:
                plan = VectorPlan(
                    steps=[VectorPlanStep(action=action, target=concept_id)],
                    origin=f"chain:{concept_id}:{action}",
                )
                beam.append((gain, plan, new_state))

    beam.sort(key=lambda x: -x[0])
    beam = beam[:beam_width]
    result: list[VectorPlan] = [b[1] for b in beam]

    for _depth in range(1, max_depth):
        next_beam: list[tuple[float, VectorPlan, VectorState]] = []
        for prev_gain, prev_plan, prev_state in beam:
            for concept_id in known_concepts:
                if concept_id in non_targetable:
                    continue
                for action in plan_actions:
                    if action == "make" and not _is_declared_crafting_rule(model, concept_id, action):
                        continue
                    # Check requirements against hypothetical state after prior steps.
                    if not model.requirements_met(concept_id, action, prev_state.inventory):
                        continue
                    effect_vec, conf = _cached_predict(cache, model, concept_id, action)
                    if conf < 0.2:
                        continue
                    decoded = model.decode_effect(effect_vec)
                    if not decoded:
                        continue
                    new_state = prev_state.apply_effect(decoded)
                    step_gain = sum(v for v in decoded.values() if v > 0
                                    and v not in prev_state.body)
                    total_gain = prev_gain + step_gain
                    if step_gain > 0:
                        new_plan = VectorPlan(
                            steps=prev_plan.steps + [
                                VectorPlanStep(action=action, target=concept_id),
                            ],
                            origin=prev_plan.origin + f"+{concept_id}:{action}",
                        )
                        next_beam.append((total_gain, new_plan, new_state))

        next_beam.sort(key=lambda x: -x[0])
        next_beam = next_beam[:beam_width]
        result.extend([b[1] for b in next_beam])
        beam = next_beam
        if not beam:
            break

    return result


# ---------------------------------------------------------------------------
# Primitive expansion (simplified)
# ---------------------------------------------------------------------------

def _step_toward(
    player_pos: tuple[int, int],
    target_pos: tuple[int, int],
    model: VectorWorldModel,
    rng: np.random.RandomState,
) -> str:
    """Pick a move primitive toward target using textbook primitives."""
    px, py = player_pos
    tx, ty = target_pos
    dx, dy = tx - px, ty - py

    moves = []
    if dx > 0:
        moves.append("move_right")
    elif dx < 0:
        moves.append("move_left")
    if dy > 0:
        moves.append("move_down")
    elif dy < 0:
        moves.append("move_up")

    if not moves:
        move_actions = [a for a in model.actions if a.startswith("move_")]
        return str(rng.choice(move_actions)) if move_actions else "move_right"
    return str(rng.choice(moves))


def expand_to_primitive(
    plan_step: VectorPlanStep,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    model: VectorWorldModel,
    rng: np.random.RandomState,
    last_action: str | None = None,
    near_concept: str | None = None,
    dynamic_entities: "list[DynamicEntityState] | None" = None,
    navigation_debug: dict[str, Any] | None = None,
) -> str:
    """Expand a plan step to a single env primitive.

    Maps abstract actions to Crafter env primitives:
    - "do" → "do"
    - "sleep" → "sleep"
    - "make" + target "wood_sword" → "make_wood_sword"
    - "place" + target "table" → "place_table"

    If target is not adjacent, navigate toward it first.
    """
    # Phase-1 frontier exploration. The planner emits this pseudo-action
    # when the goal target is unknown to the cognitive map; expand resolves
    # it to a real move toward the nearest unvisited cell.
    if plan_step.action == "frontier_seek":
        unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
        if unvisited:
            # Deterministic under PYTHONHASHSEED=0: pick the closest by
            # manhattan; ties broken by sorted iteration order (the list
            # produced by unvisited_neighbors is already in nested-loop
            # order over (dy, dx)).
            closest = min(
                unvisited,
                key=lambda c: abs(c[0] - player_pos[0]) + abs(c[1] - player_pos[1]),
            )
            return _step_toward(player_pos, closest, model, rng)
        move_actions = [a for a in model.actions if a.startswith("move_")]
        return str(rng.choice(move_actions)) if move_actions else "move_right"

    if plan_step.action == "navigate_known":
        chosen_move, target_pos, move_candidates = _select_known_target_move(
            target=plan_step.target,
            player_pos=player_pos,
            spatial_map=spatial_map,
            model=model,
            dynamic_entities=dynamic_entities,
        )
        if navigation_debug is not None:
            px, py = int(player_pos[0]), int(player_pos[1])
            navigation_debug.update({
                "target_concept": plan_step.target,
                "target_pos": (
                    [int(target_pos[0]), int(target_pos[1])]
                    if target_pos is not None
                    else None
                ),
                "dist_before": (
                    int(abs(int(target_pos[0]) - px) + abs(int(target_pos[1]) - py))
                    if target_pos is not None
                    else None
                ),
                "chosen_move": chosen_move,
                "candidate_moves": move_candidates,
            })
        if chosen_move is not None:
            return chosen_move
        move_actions = [a for a in model.actions if a.startswith("move_")]
        fallback = str(rng.choice(move_actions)) if move_actions else "move_right"
        if navigation_debug is not None:
            navigation_debug["chosen_move"] = fallback
            navigation_debug["fallback"] = "random_move_no_known_target"
        return fallback

    target_pos = spatial_map.find_nearest(plan_step.target, player_pos)

    # Phase-2: fall back to dynamic-entity tracker for cow/zombie/skeleton/arrow
    # that are tracked but not in the static spatial_map. Without this, even
    # though `_plan_distance` now allows the plan to win, `expand_to_primitive`
    # would still drop into random-move because find_nearest returned None.
    if target_pos is None and dynamic_entities:
        tracked = [
            e.position for e in dynamic_entities
            if e.concept_id == plan_step.target
        ]
        if tracked:
            target_pos = min(
                tracked,
                key=lambda p: abs(p[0] - player_pos[0]) + abs(p[1] - player_pos[1]),
            )

    if (
        plan_step.action == "do"
        and near_concept == plan_step.target
        and target_pos is not None
        and abs(target_pos[0] - player_pos[0]) + abs(target_pos[1] - player_pos[1]) <= 1
        and (plan_step.target, "do") in getattr(model, "action_requirements", {})
    ):
        dx = int(target_pos[0] - player_pos[0])
        dy = int(target_pos[1] - player_pos[1])
        facing = _facing_delta(last_action)
        if facing == (0, 0):
            facing = (0, 1)
        if (dx, dy) == facing:
            return "do"
        return _step_toward(player_pos, target_pos, model, rng)

    if target_pos is None and plan_step.action not in ("sleep",):
        # near_concept == target: resource is at the player's center tile —
        # find_nearest skipped it (Bug 5 guard for stale perception entries).
        # The player is physically adjacent; execute the action directly.
        if near_concept is not None and near_concept == plan_step.target and plan_step.action == "do":
            return "do"
        # make/place act on inventory or facing tile — no spatial target needed
        # in the spatial map (the named target lives in inventory). But the
        # textbook may declare an adjacency requirement (e.g. `make_*`
        # requires `near: table`). Honor it here so the primitive expansion
        # navigates to an existing required-tile if any exist on the map and
        # only places a new one when none does. Without this guard the agent
        # ends up placing a fresh table every time it strays from the prior
        # one — visible as 3 tables stacked next to each other.
        if plan_step.action in ("make", "place"):
            from snks.agent.crafter_pixel_env import ACTION_TO_IDX
            near_req = getattr(model, "near_requirements", {}).get(
                (plan_step.target, plan_step.action)
            )
            if near_req and near_req != "empty":
                px, py = player_pos
                adj: set[str] = set()
                if hasattr(spatial_map, "concept_at"):
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        concept = spatial_map.concept_at((px + dx, py + dy))
                        if concept is not None:
                            adj.add(concept)
                if near_req not in adj:
                    near_pos = spatial_map.find_nearest(near_req, player_pos)
                    if near_pos is not None:
                        return _step_toward(player_pos, near_pos, model, rng)
                    place_compound = f"place_{near_req}"
                    if place_compound in ACTION_TO_IDX:
                        return place_compound
            compound = f"{plan_step.action}_{plan_step.target}"
            if compound in ACTION_TO_IDX:
                return compound
        # Target not in spatial map — explore
        move_actions = [a for a in model.actions if a.startswith("move_")]
        return str(rng.choice(move_actions)) if move_actions else "move_right"

    # Sleep doesn't need a target position
    if plan_step.action == "sleep":
        return "sleep"

    if target_pos is not None:
        px, py = player_pos
        tx, ty = target_pos
        dx, dy = tx - px, ty - py
        dist = abs(dx) + abs(dy)

        if dist > 1:
            # Navigate toward target
            return _step_toward(player_pos, target_pos, model, rng)

        # Adjacent — check if we're facing the target.
        # Crafter "do" acts on the FACING tile only.
        # Facing direction from last_action: move_right→(+1,0), etc.
        if plan_step.action == "do" and dist == 1:
            facing_map = {
                "move_left": (-1, 0), "move_right": (1, 0),
                "move_up": (0, -1), "move_down": (0, 1),
            }
            facing = facing_map.get(last_action, (0, 1))  # default: down
            if (dx, dy) != facing:
                # Target not on facing tile — turn to face it by moving toward it
                return _step_toward(player_pos, target_pos, model, rng)

    # Adjacent or no position needed — map to env primitive
    action = plan_step.action
    target = plan_step.target

    if action == "do":
        return "do"
    elif action == "sleep":
        return "sleep"
    elif action in ("make", "place"):
        compound = f"{action}_{target}"
        # Validate against known env actions
        from snks.agent.crafter_pixel_env import ACTION_TO_IDX
        if compound in ACTION_TO_IDX:
            return compound
        # Invalid compound — explore instead
        move_actions = [a for a in model.actions if a.startswith("move_")]
        return str(rng.choice(move_actions)) if move_actions else "move_right"
    else:
        return action


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

class _OutcomeRecorder:
    """Per-episode pending-ring buffer for outcome-role writes.

    On `push(step, plan_steps, near, health_now)` we resolve the
    `(concept, action)` pair for the chosen plan and remember the
    decision-time health. After `horizon` env steps `flush_due(...)`
    computes `damage_h = health_at_decision - health_now` and writes
    `(survived=True, damage, died_to=None)` via `model.learn_outcome`.
    On death `flush_on_death(...)` writes the remaining pending snapshots
    with `survived=False, died_to=<cause>`.

    Identical lifecycle to the previous EpisodicEpisodeRecorder, but
    every write goes into the same `VectorWorldModel.memory` SDM at the
    outcome-role address — no separate substrate object.
    """

    def __init__(self, model, horizon: int, resolver) -> None:
        self.model = model
        self.horizon = max(1, int(horizon))
        self._resolve = resolver
        self._pending: list[dict] = []

    def push(self, step: int, plan_steps: list, near: "str | None",
             health_now: float) -> None:
        pair = self._resolve(plan_steps, near)
        if pair is None:
            return
        self._pending.append({
            "due": step + self.horizon,
            "concept": pair[0],
            "action": pair[1],
            "health_start": float(health_now),
        })

    def flush_due(self, current_step: int, health_now: float) -> int:
        kept: list[dict] = []
        flushed = 0
        for snap in self._pending:
            if snap["due"] <= current_step:
                damage = max(0, int(round(snap["health_start"] - health_now)))
                self.model.learn_outcome(snap["concept"], snap["action"], {
                    "survived_h": True,
                    "damage_h": damage,
                    "died_to": None,
                })
                flushed += 1
            else:
                kept.append(snap)
        self._pending = kept
        return flushed

    def flush_on_death(self, health_now: float, died_to: "str | None") -> int:
        flushed = 0
        for snap in self._pending:
            damage = max(0, int(round(snap["health_start"] - health_now)))
            self.model.learn_outcome(snap["concept"], snap["action"], {
                "survived_h": False,
                "damage_h": damage,
                "died_to": died_to,
            })
            flushed += 1
        self._pending = []
        return flushed


class _OptionOutcomeRecorder:
    """Pending-ring buffer for selected strategy-option outcome writes."""

    _FAILURE_REASON_LABELS = frozenset({
        "health_critical",
        "food_critical",
        "drink_critical",
        "energy_critical",
        "interrupted",
        "failed",
        "target_lost",
        "max_attempts_exceeded",
    })

    def __init__(self, model: VectorWorldModel, horizon: int) -> None:
        self.model = model
        self.horizon = max(1, int(horizon))
        self._pending: list[dict[str, Any]] = []
        self._recent: list[dict[str, Any]] = []
        self._recent_limit = max(8, self.horizon + 6)

    @staticmethod
    def _critical_vital_reason(body_now: dict[str, float] | None) -> str | None:
        if not body_now:
            return None
        for key in ("health", "food", "drink", "energy"):
            try:
                value = float(body_now.get(key, 9.0))
            except (TypeError, ValueError):
                continue
            if value <= 1.0:
                return f"{key}_critical"
        return None

    @classmethod
    def _normalise_failure_reason(cls, reason: str | None) -> str:
        label = str(reason or "").strip()
        if label.startswith("emergency_override"):
            return "interrupted"
        if label in cls._FAILURE_REASON_LABELS:
            return label
        return "failed"

    def push(
        self,
        *,
        step: int,
        context: OptionContext,
        option: StrategyOption,
        health_now: float,
        body_now: dict[str, float] | None = None,
    ) -> None:
        snap = {
            "due": int(step) + self.horizon,
            "context": context.to_trace(),
            "option_id": option.option_id,
            "health_start": float(health_now),
            "body_start": dict(body_now or {}),
        }
        self._pending.append(snap)
        self._recent.append(snap)
        del self._recent[:-self._recent_limit]

    def flush_due(
        self,
        *,
        current_step: int,
        health_now: float,
        body_now: dict[str, float] | None = None,
    ) -> int:
        kept: list[dict[str, Any]] = []
        flushed = 0
        for snap in self._pending:
            if int(snap["due"]) <= int(current_step):
                self._write_snapshot(
                    snap,
                    health_now=health_now,
                    died_to=self._critical_vital_reason(body_now),
                )
                flushed += 1
            else:
                kept.append(snap)
        self._pending = kept
        return flushed

    def mark_latest_failed(
        self,
        *,
        health_now: float,
        reason: str,
        body_now: dict[str, float] | None = None,
    ) -> int:
        if not self._pending:
            return 0
        snap = self._pending.pop()
        self._write_snapshot(
            snap,
            health_now=health_now,
            died_to=self._normalise_failure_reason(
                reason or self._critical_vital_reason(body_now)
            ),
        )
        return 1

    def flush_on_death(self, *, health_now: float, died_to: str | None) -> int:
        self._credit_precursors(health_now=health_now, died_to=died_to)
        flushed = 0
        for snap in self._pending:
            self._write_snapshot(snap, health_now=health_now, died_to=died_to)
            flushed += 1
        self._pending = []
        return flushed

    def _credit_precursors(self, *, health_now: float, died_to: str | None) -> None:
        cause = self.model.normalize_option_failure_cause({"died_to": died_to, "damage_h": 1})
        if cause != "hostile_damage":
            return
        # A small deterministic window retains pre-terminal hostile-risk
        # choices after their normal horizon write has already flushed.
        for snap in self._recent[-self._recent_limit:]:
            context = snap["context"]
            if (context.get("goal_family") == "fight" or
                    context.get("threat_pressure") not in {None, "none"}):
                damage = max(0, int(round(float(snap["health_start"]) - health_now)))
                self.model.learn_option_failure_credit(context, str(snap["option_id"]), {
                    "survived_h": False, "damage_h": damage, "died_to": died_to,
                }, credit_type="precursor")

    def _write_snapshot(
        self,
        snap: dict[str, Any],
        *,
        health_now: float,
        died_to: str | None,
    ) -> None:
        damage = max(0, int(round(float(snap["health_start"]) - health_now)))
        self.model.learn_option_outcome(
            snap["context"],
            str(snap["option_id"]),
            {
                "survived_h": died_to is None,
                "damage_h": damage,
                "died_to": died_to,
            },
        )


def run_vector_mpc_episode(
    env: Any,
    segmenter: Any,
    model: VectorWorldModel,
    tracker: HomeostaticTracker,
    rng: np.random.RandomState | None = None,
    max_steps: int = 1000,
    horizon: int = 10,
    beam_width: int = 5,
    max_depth: int = 3,
    vital_vars: list[str] | None = None,
    stimuli: StimuliLayer | None = None,
    textbook: "Any | None" = None,
    verbose: bool = False,
    enable_dynamic_threat_model: bool = True,
    enable_dynamic_threat_goals: bool = True,
    enable_motion_plans: bool = True,
    enable_motion_chains: bool = True,
    enable_post_plan_passive_rollout: bool = True,
    record_stage89c_trace: bool = False,
    record_step_trace: bool = False,
    record_death_bundle: bool = False,
    record_local_trace: bool = False,
    record_local_counterfactuals: bool | str = True,
    local_counterfactual_horizon: int = 3,
    local_action_advisor: Any | None = None,
    local_actor_policy: Any | None = None,
    local_advisory_allowed_actions: list[str] | None = None,
    record_local_advisory_trace: bool = False,
    local_advisory_top_k: int = 3,
    local_advisory_device: torch.device | str = "cpu",
    mixed_control_actor_share: float = 0.0,
    enable_planner_rescue: bool = False,
    rescue_low_vitals_threshold: float = 4.0,
    rescue_hostile_distance_threshold: int = 1,
    rescue_stall_streak_threshold: int = 2,
    death_capture_steps: int = DEFAULT_DEATH_TRACE_HORIZON,
    perception_mode: str = "pixel",
    enable_outcome_learning: bool = False,
    world_model_path: "str | Path | None" = None,
    outcome_horizon: int = 5,
    outcome_stimulus_weight: float = 1.0,
    enable_option_outcome_learning: bool = False,
    option_outcome_horizon: int = 5,
    enable_option_outcome_stimulus: bool = False,
    option_outcome_stimulus_weight: float = 1.0,
    option_outcome_confidence_floor: float = 0.25,
) -> dict:
    """Run one episode with vector MPC planning.

    Each step:
      1. Perceive → update spatial_map, entity_tracker, tracker
      2. Generate candidate plans via forward imagination
      3. Simulate each, score, pick best
      4. Execute first primitive
      5. Surprise-driven learning from observation
    """
    if rng is None:
        rng = np.random.RandomState()
    vitals = vital_vars or ["health", "food", "drink", "energy"]
    if stimuli is None:
        stimuli = StimuliLayer([
            SurvivalAversion(),
            VitalDeltaStimulus(["health"]),
            HomeostasisStimulus(vitals),
        ])

    # Outcome-role lifecycle: load persistent world-model state if requested,
    # wire `OutcomeStimulus` into the planner's stimuli list, and prepare an
    # outcome recorder that writes realised trajectory outcomes back into the
    # SAME `model.memory` at the outcome-role address. Persistence happens
    # via the existing VectorWorldModel.save/load (one .pt per seed contains
    # physics-role + requirement dicts + outcome-role writes together).
    outcome_recorder = None
    option_outcome_recorder = (
        _OptionOutcomeRecorder(model=model, horizon=int(option_outcome_horizon))
        if enable_option_outcome_learning
        else None
    )
    _outcome_near_holder: dict[str, "str | None"] = {"value": None}
    _option_eval_holder: dict[str, Any] = {}
    promoter = TextbookPromoter()
    promoted_path = _promoted_nodes_path(world_model_path)
    promoted_nodes_prior: list[dict[str, Any]] = []
    should_use_persistent_world_model = (
        enable_outcome_learning
        or enable_option_outcome_learning
        or enable_option_outcome_stimulus
    )
    if should_use_persistent_world_model and world_model_path is not None:
        model.load(Path(world_model_path))
    if enable_outcome_learning:
        from snks.agent.stimuli import OutcomeStimulus, resolve_outcome_pair
        stimuli.stimuli.append(OutcomeStimulus(
            model=model,
            weight=outcome_stimulus_weight,
            near_concept_provider=lambda: _outcome_near_holder["value"],
        ))
        outcome_recorder = _OutcomeRecorder(
            model=model, horizon=int(outcome_horizon),
            resolver=resolve_outcome_pair,
        )

    from snks.agent.goal_selector import Goal, GoalSelector
    goal_selector = (
        GoalSelector(
            textbook,
            allow_dynamic_entity_goals=enable_dynamic_threat_goals,
        )
        if textbook is not None
        else None
    )
    emergency_facts = EmergencyWorldFacts.from_textbook(textbook)
    emergency_controller = EmergencySafetyController(
        facts=emergency_facts,
        low_vitals_threshold=float(rescue_low_vitals_threshold),
        hostile_distance_threshold=int(rescue_hostile_distance_threshold),
        stall_streak_threshold=int(rescue_stall_streak_threshold),
    )
    if enable_option_outcome_stimulus:
        from snks.agent.stimuli import OptionOutcomeStimulus

        def _option_context_for_trajectory(
            trajectory: VectorTrajectory,
        ) -> dict[str, str] | None:
            if not _option_eval_holder:
                return None
            return _build_option_context(
                body=_option_eval_holder["body"],
                inventory=_option_eval_holder["inventory"],
                capability_state=_option_eval_holder["capability_state"],
                current_goal=_option_eval_holder["current_goal"],
                interaction_intent=_option_eval_holder["interaction_intent"],
                best_plan=trajectory.plan,
                nearest_threat_distances=_option_eval_holder["nearest_threat_distances"],
                emergency_facts=emergency_facts,
                textbook=textbook,
                model=model,
                near_concept=_option_eval_holder["near_concept"],
                player_pos=_option_eval_holder["player_pos"],
                spatial_map=spatial_map,
            ).to_trace()

        def _option_id_for_trajectory(trajectory: VectorTrajectory) -> str:
            return _derive_strategy_option(trajectory.plan).option_id

        stimuli.stimuli.append(OptionOutcomeStimulus(
            model=model,
            weight=float(option_outcome_stimulus_weight),
            confidence_floor=float(option_outcome_confidence_floor),
            context_provider=_option_context_for_trajectory,
            option_id_provider=_option_id_for_trajectory,
        ))

    entity_tracker = DynamicEntityTracker()
    # Register known dynamic concepts
    for cid in model.concepts:
        if cid in ("zombie", "skeleton", "cow", "arrow"):
            entity_tracker.register_dynamic_concept(cid)

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    promoted_nodes_prior = _load_promoted_entities_into_spatial_map(
        promoter=promoter,
        promoted_path=promoted_path,
        spatial_map=spatial_map,
    )
    prev_inv: dict[str, int] | None = None
    prev_body: dict[str, float] | None = None
    prev_action: str | None = None
    prev_move: str | None = None  # last move primitive — determines facing
    prev_plan_target: str | None = None  # target concept of last executed plan step
    prev_player_pos: tuple[int, int] | None = None
    action_counts: Counter = Counter()
    steps_taken = 0
    cause_of_death = "alive"
    total_surprise = 0.0
    n_surprise_events = 0
    damage_log: list = []
    arrow_threat_steps = 0
    defensive_action_steps = 0
    danger_prediction_errors: list[float] = []
    pending_prediction_diag: dict[str, float | bool] | None = None
    arrow_visible_steps = 0
    arrow_velocity_known_steps = 0
    arrow_velocity_unknown_steps = 0
    first_arrow_threat_step: int | None = None
    first_defensive_action_step: int | None = None
    defensive_events: list[dict[str, Any]] = []
    defensive_sequences = 0
    defensive_window_targets = (10, 20)
    step_trace: list[dict[str, Any]] = []
    death_trace_steps: list[dict[str, Any]] = []
    local_trace: list[dict[str, Any]] = []
    local_advisory_trace: list[dict[str, Any]] = []
    rescue_trace: list[dict[str, Any]] = []
    local_advisory_allowed_actions = (
        list(local_advisory_allowed_actions)
        if local_advisory_allowed_actions is not None
        else ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]
    )
    local_belief_tracker = BeliefStateEncoder()
    actor_non_progress_streak = 0
    interaction_intent: dict[str, Any] | None = None

    for step in range(max_steps):
        steps_taken = step + 1
        raw_inv = dict(info.get("inventory", {}))
        _vital_set = {"health", "food", "drink", "energy"}
        body = {v: float(raw_inv.get(v, 9.0)) for v in vitals}
        inv = {k: v for k, v in raw_inv.items() if k not in _vital_set}
        player_pos = tuple(info.get("player_pos", (32, 32)))

        if pending_prediction_diag is not None:
            actual_loss = max(
                0.0,
                float(pending_prediction_diag["health_before"]) - body.get("health", 0.0),
            )
            if bool(pending_prediction_diag["arrow_threat"]):
                danger_prediction_errors.append(
                    abs(float(pending_prediction_diag["predicted_loss"]) - actual_loss)
                )
            pending_prediction_diag = None

        # --- Blocked movement detection ---
        if (
            prev_action
            and prev_action.startswith("move_")
            and prev_player_pos is not None
            and prev_player_pos == player_pos
        ):
            dx, dy = 0, 0
            if prev_action == "move_right":
                dx = 1
            elif prev_action == "move_left":
                dx = -1
            elif prev_action == "move_down":
                dy = 1
            elif prev_action == "move_up":
                dy = -1
            blocked_tile = (player_pos[0] + dx, player_pos[1] + dy)
            spatial_map.mark_blocked(blocked_tile)

        # --- Perception ---
        if perception_mode == "pixel":
            vf = perceive_tile_field(pixels, segmenter)
        elif perception_mode == "symbolic":
            vf = perceive_semantic_field(info)
        else:
            raise ValueError(f"Unknown perception_mode: {perception_mode}")
        station_diag_before_perception = _station_spatial_debug(
            spatial_map,
            player_pos,
            concepts=_SPATIAL_DEBUG_CONCEPTS,
        )
        _update_spatial_map(spatial_map, vf, player_pos, prev_move=prev_move)
        _update_spatial_map_hazards(spatial_map, info, player_pos)
        station_diag_after_perception = _station_spatial_debug(
            spatial_map,
            player_pos,
            concepts=_SPATIAL_DEBUG_CONCEPTS,
        )
        entity_tracker.update(vf, player_pos)

        # --- Homeostatic tracker ---
        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # --- Surprise-driven learning ---
        if prev_inv is not None and prev_body is not None and prev_action is not None:
            # Compute actual deltas
            inv_deltas = {}
            for k in set(inv.keys()) | set(prev_inv.keys()):
                d = inv.get(k, 0) - prev_inv.get(k, 0)
                if d != 0:
                    inv_deltas[k] = d
            body_deltas = {}
            for k in vitals:
                d = body.get(k, 0) - prev_body.get(k, 0)
                if abs(d) > 0.01:
                    body_deltas[k] = int(round(d))

            all_deltas = {**inv_deltas, **body_deltas}
            if all_deltas:
                # Target concept = what the previous plan step was aimed at.
                # `near_concept` is the tile under the player (usually grass);
                # Crafter's `do` acts on the FACING tile, so we must use the
                # plan's declared target instead. Example: plan=single:tree:do
                # → prev_plan_target='tree', delta={wood:+1} → learn(tree,do,...).
                target_concept = (
                    prev_plan_target
                    if prev_action in ("do", "place", "make")
                    else None
                )
                if target_concept and target_concept not in ("empty", "self"):
                    surprise = model.learn(target_concept, prev_action, all_deltas)
                    total_surprise += surprise
                    n_surprise_events += 1

                # Entity-correlated surprise for unexpected damage
                health_delta = body_deltas.get("health", 0)
                if health_delta < 0:
                    nearby_cids = []
                    for entity_cid, entity_pos in entity_tracker.visible_entities():
                        ex, ey = entity_pos
                        dist = abs(ex - player_pos[0]) + abs(ey - player_pos[1])
                        if dist <= 6:
                            model.learn(entity_cid, "proximity",
                                        {"health": health_delta})
                        nearby_cids.append((entity_cid, dist))
                    # Accumulate damage event for post-mortem analysis
                    damage_log.append(DamageEvent(
                        step=step,
                        health_delta=float(health_delta),
                        vitals={k: prev_body.get(k, 9.0)
                                for k in ("food", "drink", "energy")},
                        nearby_cids=nearby_cids,
                    ))

        # --- Build VectorState ---
        observed_dynamic_entities = entity_tracker.current()
        state = VectorState(
            inventory=inv,
            body=body,
            player_pos=player_pos,
            step=step,
            last_action=prev_action,
            spatial_map=spatial_map,
            dynamic_entities=(
                observed_dynamic_entities if enable_dynamic_threat_model else []
            ),
        )
        arrow_states = [e for e in observed_dynamic_entities if e.concept_id == "arrow"]
        if arrow_states:
            arrow_visible_steps += 1
            if any(e.velocity is not None for e in arrow_states):
                arrow_velocity_known_steps += 1
            else:
                arrow_velocity_unknown_steps += 1

        if record_stage89c_trace:
            _update_defensive_event_windows(
                defensive_events=defensive_events,
                current_step=step,
                body=body,
                alive=True,
                window_targets=defensive_window_targets,
            )

        # --- Build per-step prediction cache (one batched GPU op) ---
        # Include dynamic-entity concept_ids so prediction cache covers
        # `do <entity>` lookups when the entity is currently tracked but
        # not in the static spatial map.
        known_step = (
            set(vf.visible_concepts())
            | set(spatial_map.known_objects.keys())
            | {entity.concept_id for entity in observed_dynamic_entities}
        )
        target_acts = [a for a in model.actions if a in ("do", "make", "place")]
        if enable_dynamic_threat_model and observed_dynamic_entities and "proximity" in model.actions:
            target_acts.append("proximity")
        step_cache = build_prediction_cache(model, known_step, target_acts)

        # Goal is a pure function of state; compute it first so plan
        # generation can emit goal-conditioned plans (e.g. frontier
        # exploration toward an unseen goal target).
        current_goal = goal_selector.select(state) if goal_selector else Goal("explore")

        if interaction_intent is not None:
            target = str(interaction_intent.get("target"))
            goal_matches = (
                current_goal is not None
                and (
                    current_goal.id == f"fight_{target}"
                    or current_goal.target_concept == target
                )
            )
            target_pos = _nearest_known_target_position(
                target,
                player_pos,
                spatial_map,
                observed_dynamic_entities,
            )
            target_present = target_pos is not None or _is_interaction_target_present(
                target=target,
                near_concept=vf.near_concept,
                dynamic_entities=observed_dynamic_entities,
            )
            if not goal_matches or not target_present:
                interaction_intent = None

        interaction_intent = _interaction_intent_from_goal_target(
            textbook=textbook,
            model=model,
            state=state,
            active_goal=current_goal,
            spatial_map=spatial_map,
            dynamic_entities=observed_dynamic_entities,
            player_pos=player_pos,
            existing_intent=interaction_intent,
            step=step,
        )

        # --- Generate + simulate + score ---
        candidates = generate_candidate_plans(
            model, state, spatial_map, vf.visible_concepts(),
            beam_width=beam_width,
            max_depth=max_depth,
            cache=step_cache,
            enable_motion_plans=enable_motion_plans,
            enable_motion_chains=enable_motion_chains,
            player_pos=player_pos,
            active_goal=current_goal,
        )

        # Sort candidates by proximity to first target — closer first.
        # Stable sort below keeps proximity order within equal scores.
        def _plan_distance(plan: VectorPlan) -> int:
            if not plan.steps:
                # Baseline plan = exploration (random move). Always "reachable"
                # so known=1. -steps=0 beats sleep (-steps=-1) when both have
                # total_gain=0 — agent explores rather than sleeps uselessly.
                return 0
            # Phase-1 frontier plan: distance is the manhattan to the closest
            # unvisited cell within radius. Returning a real distance (not
            # 9999) keeps `known=1` in the score tuple so the plan participates
            # in normal lex sorting; goal_progress earns the
            # FRONTIER_PROGRESS_EPSILON tiebreaker via Goal.progress.
            if plan.steps[0].action == "frontier_seek":
                unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
                if not unvisited:
                    return 9999
                return min(
                    abs(c[0] - player_pos[0]) + abs(c[1] - player_pos[1])
                    for c in unvisited
                )
            # All steps must have known targets — if any step's target is not in
            # spatial_map, the whole plan is unreachable (known=0 in scoring).
            max_dist = 0
            for step in plan.steps:
                if step.target == "self":
                    continue
                # make/place act on inventory or the facing/adjacent tile and
                # do not depend on the named target existing in the spatial map
                # (`wood_pickaxe` lives in inventory; `table` will be placed
                # adjacent). Adjacency is enforced separately during plan
                # generation via `near_requirements`.
                if step.action in ("make", "place"):
                    continue
                # near_concept == target means the resource is immediately adjacent
                # to the player. find_nearest skips player_pos (Bug 5 guard), so
                # it would return None for center-tile resources — treat as dist=0.
                if step.target == vf.near_concept:
                    continue
                pos = spatial_map.find_nearest(step.target, player_pos)
                if pos is None:
                    # Fall back to dynamic-entity tracker: cow/zombie/skeleton
                    # can be tracked outside the spatial-map view. Use the
                    # closest tracked instance of `step.target`. Without this
                    # fallback the plan's known=0 and baseline always wins,
                    # even when DynamicEntityTracker has a live position.
                    tracked = [
                        e.position for e in state.dynamic_entities
                        if e.concept_id == step.target
                    ]
                    if not tracked:
                        return 9999
                    pos = min(
                        tracked,
                        key=lambda p: abs(p[0] - player_pos[0]) + abs(p[1] - player_pos[1]),
                    )
                max_dist = max(max_dist, abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]))
            return max_dist

        candidates.sort(key=_plan_distance)

        capability_state = extract_capability_state(inv, textbook)
        nearest_threats_now = _nearest_emergency_threat_distances(
            emergency_facts,
            player_pos,
            spatial_map,
            observed_dynamic_entities,
        )

        # Refresh outcome-stimulus near-concept holder so motion/baseline
        # plans query the substrate conditioned on the current facing tile.
        if enable_outcome_learning:
            _outcome_near_holder["value"] = str(vf.near_concept) if vf.near_concept else None
        if enable_option_outcome_stimulus:
            _option_eval_holder.clear()
            _option_eval_holder.update({
                "body": body,
                "inventory": inv,
                "capability_state": capability_state,
                "current_goal": current_goal,
                "interaction_intent": interaction_intent,
                "nearest_threat_distances": nearest_threats_now,
                "near_concept": str(vf.near_concept) if vf.near_concept else None,
                "player_pos": player_pos,
            })

        scored: list[tuple[tuple, VectorPlan, VectorTrajectory]] = []
        for plan in candidates:
            traj = simulate_forward(
                model,
                plan,
                state,
                horizon,
                vitals,
                cache=step_cache,
                enable_post_plan_passive_rollout=enable_post_plan_passive_rollout,
            )
            sim_score = score_trajectory(traj, stimuli=stimuli, goal=current_goal)
            dist = _plan_distance(plan)
            # known=1 if target exists in spatial_map, 0 otherwise.
            # Inserted after goal_prog so any reachable plan beats a speculative one.
            known = 1 if dist < 9999 else 0
            # sim_score = (base_score, goal_prog, -steps) — 3-tuple
            # goal_prog is self-normalizing: sleep with goal=sleep → vital_delta>0,
            # sleep with goal=fight_zombie → vital_delta≈0. No suppression needed.
            score = (sim_score[0], sim_score[1], known, sim_score[2])
            scored.append((score, plan, traj))

        scored.sort(key=lambda x: x[0], reverse=True)
        option_candidate_score_debug: list[dict[str, Any]] = []
        if enable_option_outcome_stimulus and record_local_trace:
            for candidate_score, candidate_plan, _candidate_traj in scored[:8]:
                candidate_option = _derive_strategy_option(candidate_plan)
                candidate_context = _build_option_context(
                    body=body,
                    inventory=inv,
                    capability_state=capability_state,
                    current_goal=current_goal,
                    interaction_intent=interaction_intent,
                    best_plan=candidate_plan,
                    nearest_threat_distances=nearest_threats_now,
                    emergency_facts=emergency_facts,
                    textbook=textbook,
                    model=model,
                    near_concept=str(vf.near_concept) if vf.near_concept else None,
                    player_pos=player_pos,
                    spatial_map=spatial_map,
                )
                candidate_decoded, candidate_confidence = model.predict_option_outcome(
                    candidate_context.to_trace(),
                    candidate_option.option_id,
                )
                option_candidate_score_debug.append({
                    "score": [float(x) for x in candidate_score],
                    "plan_origin": str(candidate_plan.origin),
                    "first_step": (
                        {
                            "action": str(candidate_plan.steps[0].action),
                            "target": str(candidate_plan.steps[0].target),
                        }
                        if candidate_plan.steps
                        else None
                    ),
                    "strategy_option": candidate_option.to_trace(),
                    "option_context": candidate_context.to_trace(),
                    "option_outcome_recall": {
                        "confidence": float(candidate_confidence),
                        "decoded": candidate_decoded,
                        "used_for_scoring": bool(
                            candidate_decoded is not None
                            and float(candidate_confidence) >= float(option_outcome_confidence_floor)
                            and not bool(candidate_decoded.get("survived_h", True))
                        ),
                    },
                })
        candidate_summaries = (
            summarize_scored_candidates(scored, body)
            if record_death_bundle
            else []
        )
        baseline_traj = next(
            (traj for _score, plan, traj in scored if plan.origin == "baseline"),
            None,
        )
        best_score, best_plan, best_traj = scored[0]
        interaction_completion_trace: dict[str, Any] | None = None
        interaction_completion_plan, interaction_completion_trace = (
            _select_interaction_completion_plan(
                interaction_intent=interaction_intent,
                player_pos=player_pos,
                spatial_map=spatial_map,
                dynamic_entities=observed_dynamic_entities,
                last_move=prev_move,
            )
        )
        if interaction_completion_plan is not None:
            completion_traj = simulate_forward(
                model,
                interaction_completion_plan,
                state,
                horizon,
                vitals,
                cache=step_cache,
                enable_post_plan_passive_rollout=enable_post_plan_passive_rollout,
            )
            sim_score = score_trajectory(completion_traj, stimuli=stimuli, goal=current_goal)
            dist = _plan_distance(interaction_completion_plan)
            completion_score = (sim_score[0], sim_score[1], 1 if dist < 9999 else 0, sim_score[2])
            if completion_score >= best_score:
                best_plan = interaction_completion_plan
                best_traj = completion_traj
                best_score = completion_score
            elif interaction_completion_trace is not None:
                interaction_completion_trace.update({
                    "status": "suppressed",
                    "selected_phase": "suppressed",
                    "reason": "ranked_candidate_score_higher",
                    "completion_score": [float(x) for x in completion_score],
                    "ranked_best_score": [float(x) for x in best_score],
                    "ranked_best_origin": str(best_plan.origin),
                })
        selected_target = best_plan.steps[0].target if best_plan.steps else None
        selected_action = best_plan.steps[0].action if best_plan.steps else None
        target_pos_before = (
            _nearest_known_target_position(
                selected_target,
                player_pos,
                spatial_map,
                observed_dynamic_entities,
            )
            if selected_target not in (None, "self")
            else None
        )
        target_dist_before = (
            abs(target_pos_before[0] - player_pos[0]) + abs(target_pos_before[1] - player_pos[1])
            if target_pos_before is not None
            else None
        )
        facing_vec_before = _facing_delta(prev_move)
        facing_tile_before = (
            (player_pos[0] + facing_vec_before[0], player_pos[1] + facing_vec_before[1])
            if facing_vec_before != (0, 0)
            else None
        )
        facing_label_before = (
            _spatial_label_at(spatial_map, facing_tile_before)
            if facing_tile_before is not None
            else None
        )
        env_facing_before = _env_tile_truth(env, facing_tile_before)

        health_now = body.get("health", 0.0)
        predicted_best_health = (
            best_traj.final_state.body.get("health", health_now)
            if best_traj.final_state is not None
            else health_now
        )
        predicted_best_loss = max(0.0, health_now - predicted_best_health)
        predicted_baseline_health = (
            baseline_traj.final_state.body.get("health", health_now)
            if baseline_traj is not None and baseline_traj.final_state is not None
            else health_now
        )
        predicted_baseline_loss = max(0.0, health_now - predicted_baseline_health)

        # --- Choose planner / actor primitive, then let emergency controller arbitrate ---
        navigation_debug: dict[str, Any] | None = None
        if best_plan.steps:
            if best_plan.steps[0].action == "navigate_known":
                navigation_debug = {}
            planner_primitive = expand_to_primitive(
                best_plan.steps[0], player_pos, spatial_map, model, rng,
                last_action=prev_move,
                near_concept=vf.near_concept,
                dynamic_entities=observed_dynamic_entities,
                navigation_debug=navigation_debug,
            )
        else:
            # Baseline plan (empty) won the ranking. RNG fallback now
            # restricts to plans that are
            #   (a) alive-predicted in score (score[0] >= 0; not lethal)
            #   (b) NOT a move whose sim rollout produced zero
            #       displacement (i.e. not a blocked move pretending to
            #       be a no-op). `sleep`/`do` plans naturally have zero
            #       displacement and are kept — they can still be
            #       productive (recover energy, gather resource).
            # Without (b), in a water-peninsula the planner draws RNG
            # over {move_up, move_right, move_down (all blocked, no-op),
            # move_left (real move), sleep, do} and only escapes 1/6th
            # of the time, leaving the agent stuck for many ticks.
            def _is_meaningful(plan, traj):
                if not plan.steps:
                    return False
                first = plan.steps[0]
                if first.action.startswith("move_"):
                    if not traj.states or len(traj.states) < 2:
                        return False
                    if tuple(traj.states[0].player_pos) == tuple(traj.states[1].player_pos):
                        return False
                return True

            alive_concrete = [
                (s, p, t) for s, p, t in scored
                if s[0] >= 0 and _is_meaningful(p, t)
            ]
            if alive_concrete:
                # `scored` is already sorted desc by full score tuple. Pick
                # plans in the top score band and RNG within only that band.
                # Without this, uniform RNG across all alive plans dilutes
                # rare-but-valuable plans (e.g. `make_wood_pickaxe`) among
                # the many tied-zero motion plans — agent gathers wood and
                # stands next to a table but almost never executes the make.
                top_score = alive_concrete[0][0]
                top_band = [p for s, p, t in alive_concrete if s == top_score]
                chosen_plan = top_band[rng.randint(0, len(top_band))]
                planner_primitive = expand_to_primitive(
                    chosen_plan.steps[0], player_pos, spatial_map, model, rng,
                    last_action=prev_move,
                    near_concept=vf.near_concept,
                    dynamic_entities=observed_dynamic_entities,
                )
            else:
                move_actions = [a for a in model.actions if a.startswith("move_")]
                planner_primitive = str(rng.choice(move_actions)) if move_actions else "move_right"

        continued_target = _should_continue_interaction(
            interaction_intent=interaction_intent,
            current_goal=current_goal,
            near_concept=vf.near_concept,
            inventory=inv,
            model=model,
        )
        if continued_target is not None and interaction_completion_plan is None:
            target = continued_target
            best_plan = VectorPlan(
                steps=[VectorPlanStep(action="do", target=target)],
                origin=f"continue:{target}:do_until_remove_entity",
            )
            selected_target = target
            selected_action = "do"
            target_pos_before = spatial_map.find_nearest(target, player_pos)
            target_dist_before = (
                abs(target_pos_before[0] - player_pos[0]) + abs(target_pos_before[1] - player_pos[1])
                if target_pos_before is not None
                else None
            )
            planner_primitive = expand_to_primitive(
                best_plan.steps[0],
                player_pos,
                spatial_map,
                model,
                rng,
                last_action=prev_move,
                near_concept=vf.near_concept,
                dynamic_entities=observed_dynamic_entities,
            )
            navigation_debug = None

        if continued_target is None:
            opportunistic_plan = _opportunistic_survival_plan(
                textbook=textbook,
                model=model,
                inventory=inv,
                body=body,
                near_concept=vf.near_concept,
                player_pos=player_pos,
                spatial_map=spatial_map,
                nearest_threat_distances=nearest_threats_now,
                emergency_facts=emergency_facts,
            )
            if opportunistic_plan is not None:
                best_plan = opportunistic_plan
                selected_target = best_plan.steps[0].target
                selected_action = best_plan.steps[0].action
                target_pos_before = spatial_map.find_nearest(selected_target, player_pos)
                target_dist_before = (
                    abs(target_pos_before[0] - player_pos[0]) + abs(target_pos_before[1] - player_pos[1])
                    if target_pos_before is not None
                    else None
                )
                planner_primitive = expand_to_primitive(
                    best_plan.steps[0],
                    player_pos,
                    spatial_map,
                    model,
                    rng,
                    last_action=prev_move,
                    near_concept=vf.near_concept,
                    dynamic_entities=observed_dynamic_entities,
                )
                navigation_debug = None

        primitive = planner_primitive
        control_origin = "planner_bootstrap"
        learner_action: str | None = None
        learner_action_index: int | None = None
        rescue_trigger: str | None = None
        rescue_pending: dict[str, Any] | None = None
        actor_observation = build_local_observation_package(
            vf,
            body,
            inv,
            belief_context=local_belief_tracker.build_context(
                near_concept=str(vf.near_concept)
            ),
        )
        advisory_ranked: list[dict[str, Any]] = []
        emergency_candidate_outcomes: list[dict[str, Any]] = []
        if local_actor_policy is not None and (
            float(mixed_control_actor_share) > 0.0 or bool(enable_planner_rescue)
        ):
            from snks.agent.crafter_pixel_env import ACTION_NAMES, ACTION_TO_IDX

            actor_ranked = rank_local_actor_candidates(
                evaluator=local_actor_policy,
                observation=actor_observation,
                allowed_actions=local_advisory_allowed_actions,
                action_to_idx=ACTION_TO_IDX,
                action_names=list(ACTION_NAMES),
                device=local_advisory_device,
            )
            if actor_ranked and float(rng.rand()) < float(mixed_control_actor_share):
                learner_action = str(actor_ranked[0]["action"])
                learner_action_index = int(actor_ranked[0]["action_index"])
                primitive = learner_action
                control_origin = "learner_actor"
            if bool(enable_planner_rescue):
                advisory_ranked = rank_local_action_candidates(
                    evaluator=local_actor_policy,
                    observation=actor_observation,
                    allowed_actions=local_advisory_allowed_actions,
                    action_to_idx=ACTION_TO_IDX,
                    device=local_advisory_device,
                )

        if bool(enable_planner_rescue):
            emergency_candidate_outcomes = _build_local_counterfactual_outcomes(
                model=model,
                state=state,
                vf=vf,
                cache=step_cache,
                vitals=vitals,
                horizon=min(local_counterfactual_horizon, horizon),
                enable_post_plan_passive_rollout=enable_post_plan_passive_rollout,
            )
            emergency_features = emergency_controller.evaluate(
                body=body,
                nearest_threat_distances=nearest_threats_now,
                actor_non_progress_streak=actor_non_progress_streak,
                planner_action=planner_primitive,
                current_action=primitive,
                learner_action=learner_action,
                belief_state_signature=dict(actor_observation.get("belief_state_signature", {})),
                candidate_outcomes=emergency_candidate_outcomes,
                predicted_baseline_loss=float(predicted_baseline_loss),
                predicted_selected_loss=float(predicted_best_loss),
            )
            if emergency_features.activated:
                pre_emergency_action = primitive
                emergency_selection = emergency_controller.select_action(
                    current_action=primitive,
                    planner_action=planner_primitive,
                    learner_action=learner_action,
                    candidate_outcomes=emergency_candidate_outcomes,
                    advisory_ranked=advisory_ranked,
                    allowed_actions=local_advisory_allowed_actions,
                )
                primitive = emergency_selection.action
                control_origin = "emergency_safety"
                rescue_trigger = emergency_features.primary_reason or "emergency_score"
                if primitive == "do":
                    alignment_override = _combat_alignment_for_emergency_do(
                        textbook=textbook,
                        model=model,
                        inventory=inv,
                        player_pos=player_pos,
                        spatial_map=spatial_map,
                        dynamic_entities=observed_dynamic_entities,
                        last_move=prev_move,
                        near_concept=vf.near_concept,
                        rng=rng,
                        target_hint=selected_target,
                    )
                    if alignment_override is not None:
                        primitive, alignment_plan, alignment_trace = alignment_override
                        best_plan = alignment_plan
                        selected_target = alignment_plan.steps[0].target
                        selected_action = alignment_plan.steps[0].action
                        target_pos_before = _nearest_known_target_position(
                            selected_target,
                            player_pos,
                            spatial_map,
                            observed_dynamic_entities,
                        )
                        target_dist_before = (
                            abs(target_pos_before[0] - player_pos[0])
                            + abs(target_pos_before[1] - player_pos[1])
                            if target_pos_before is not None
                            else None
                        )
                        planner_primitive = primitive
                        interaction_completion_trace = alignment_trace
                _regime_labels, primary_regime = infer_local_regime(actor_observation, nearest_threats_now)
                rescue_pending = {
                    "step": int(step),
                    "trigger": rescue_trigger,
                    "activation_reason": rescue_trigger,
                    "activation_reasons": list(emergency_features.reasons),
                    "activation_score": emergency_features.score,
                    "activation_features": dict(emergency_features.values),
                    "planner_action": planner_primitive,
                    "learner_action": learner_action,
                    "pre_emergency_action": pre_emergency_action,
                    "rescue_action": emergency_selection.action,
                    "emergency_action": emergency_selection.action,
                    "executed_action": primitive,
                    "rescue_policy": emergency_selection.override_source,
                    "override_source": emergency_selection.override_source,
                    "action_selection_reason": emergency_selection.reason,
                    "utility_components": dict(emergency_selection.utility_components),
                    "ranked_emergency_actions": list(emergency_selection.ranked_actions),
                    "rescue_applied": True,
                    "rescue_improved_outcome": None,
                    "pre_rescue_state": {
                        "primary_regime": primary_regime,
                        "body": {
                            key: round(float(value), 3)
                            for key, value in body.items()
                        },
                        "nearest_threat_distances": dict(nearest_threats_now),
                        "belief_state_signature": dict(actor_observation.get("belief_state_signature", {})),
                    },
                    "candidate_outcome_excerpt": emergency_candidate_outcomes[:4],
                    "post_rescue_outcome": {},
                }
                if (
                    interaction_completion_trace is not None
                    and interaction_completion_trace.get("reason") == "emergency_alignment_required"
                ):
                    rescue_pending["combat_alignment_override"] = {
                        "target": interaction_completion_trace.get("target_concept"),
                        "original_action": emergency_selection.action,
                        "executed_action": primitive,
                        "reason": "hostile_do_requires_facing",
                        "is_adjacent": interaction_completion_trace.get("is_adjacent"),
                        "is_facing_target": interaction_completion_trace.get("is_facing_target"),
                    }

        if (
            interaction_completion_trace is not None
            and control_origin == "emergency_safety"
            and not bool(interaction_completion_trace.get("emergency_alignment_preserved", False))
        ):
            interaction_completion_trace.update({
                "status": "interrupted",
                "selected_phase": "interrupted",
                "reason": f"emergency_override:{rescue_trigger or 'emergency_safety'}",
                "expected_effect_achieved": False,
                "actual_primitive": primitive,
            })

        arrow_threat_now = any(entity.concept_id == "arrow" for entity in observed_dynamic_entities)
        # Stage 89 telemetry fix:
        # visible projectile != imminent threat. Many steps contain an arrow in
        # view, but the passive baseline still predicts zero health loss within
        # the planning horizon. Counting every visible arrow as a "threat step"
        # made defensive_action_rate look artificially close to zero even when
        # the planner correctly dodged every actually imminent hit. Treat threat
        # as "baseline would take damage within horizon", and keep visibility as
        # a separate metric.
        imminent_arrow_threat_now = arrow_threat_now and predicted_baseline_loss > 0.0
        if imminent_arrow_threat_now:
            arrow_threat_steps += 1
            if first_arrow_threat_step is None:
                first_arrow_threat_step = step
            if primitive.startswith("move_") and predicted_best_loss < predicted_baseline_loss:
                defensive_action_steps += 1
                if first_defensive_action_step is None:
                    first_defensive_action_step = step
                if not defensive_events or defensive_events[-1]["step"] != step:
                    defensive_sequences += 1
                if record_stage89c_trace:
                    defensive_events.append({
                        "step": step,
                        "primitive": primitive,
                        "plan_origin": best_plan.origin,
                        "predicted_best_loss": round(float(predicted_best_loss), 3),
                        "predicted_baseline_loss": round(float(predicted_baseline_loss), 3),
                        "health": round(float(body.get("health", 0.0)), 3),
                        "food": round(float(body.get("food", 0.0)), 3),
                        "drink": round(float(body.get("drink", 0.0)), 3),
                        "energy": round(float(body.get("energy", 0.0)), 3),
                        "nearest_zombie_dist": _nearest_hostile_distance(
                            "zombie", player_pos, spatial_map, observed_dynamic_entities
                        ),
                        "nearest_skeleton_dist": _nearest_hostile_distance(
                            "skeleton", player_pos, spatial_map, observed_dynamic_entities
                        ),
                        "nearest_arrow_dist": _nearest_dynamic_distance(
                            "arrow", player_pos, observed_dynamic_entities
                        ),
                        "nearest_tree_dist": _nearest_spatial_distance(
                            "tree", player_pos, spatial_map
                        ),
                        "nearest_water_dist": _nearest_spatial_distance(
                            "water", player_pos, spatial_map
                        ),
                        "nearest_cow_dist": _nearest_spatial_distance(
                            "cow", player_pos, spatial_map
                        ),
                        "post_defense_vitals": {
                            "health": round(float(body.get("health", 0.0)), 3),
                            "food": round(float(body.get("food", 0.0)), 3),
                            "drink": round(float(body.get("drink", 0.0)), 3),
                            "energy": round(float(body.get("energy", 0.0)), 3),
                        },
                        "resource_access_loss": {
                            "tree": _nearest_spatial_distance("tree", player_pos, spatial_map),
                            "water": _nearest_spatial_distance("water", player_pos, spatial_map),
                            "cow": _nearest_spatial_distance("cow", player_pos, spatial_map),
                        },
                        "threat_distance_after_defense": {
                            "zombie": _nearest_hostile_distance(
                                "zombie", player_pos, spatial_map, observed_dynamic_entities
                            ),
                            "skeleton": _nearest_hostile_distance(
                                "skeleton", player_pos, spatial_map, observed_dynamic_entities
                            ),
                            "arrow": _nearest_dynamic_distance(
                                "arrow", player_pos, observed_dynamic_entities
                            ),
                        },
                        "survived_10": None,
                        "survived_20": None,
                    })
        pending_prediction_diag = {
            "health_before": float(health_now),
            "predicted_loss": float(predicted_best_loss),
            "arrow_threat": imminent_arrow_threat_now,
        }

        action_counts[primitive] += 1

        if verbose and step % 20 == 0:
            print(
                f"s{step:3d} H{body.get('health', 0):.0f} "
                f"F{body.get('food', 0):.0f} D{body.get('drink', 0):.0f} "
                f"near={vf.near_concept:9s} → {primitive:12s} "
                f"plan={best_plan.origin[:30]}"
            )

        # --- Step environment ---
        prev_inv = dict(inv)
        prev_body = dict(body)
        prev_action = primitive
        # Track last MOVE separately: Crafter's facing is set by moves only,
        # do/place/make/sleep don't change facing.
        if primitive.startswith("move_"):
            prev_move = primitive
        # else: prev_move keeps previous value (facing unchanged)
        # Track plan target so surprise-driven learn uses the *intended* target,
        # not near_concept (which is the player's own tile).
        if best_plan.steps:
            prev_plan_target = best_plan.steps[0].target
        else:
            prev_plan_target = None
        next_interaction_intent = _interaction_intent_from_plan(
            textbook=textbook,
            plan=best_plan,
            existing_intent=interaction_intent,
            step=step,
        )
        if next_interaction_intent is not None:
            interaction_intent = next_interaction_intent
        prev_player_pos = player_pos
        strategy_option = _derive_strategy_option(best_plan)
        option_context = _build_option_context(
            body=body,
            inventory=inv,
            capability_state=capability_state,
            current_goal=current_goal,
            interaction_intent=interaction_intent,
            best_plan=best_plan,
            nearest_threat_distances=nearest_threats_now,
            emergency_facts=emergency_facts,
            textbook=textbook,
            model=model,
            near_concept=str(vf.near_concept),
            player_pos=player_pos,
            spatial_map=spatial_map,
        )
        option_outcome_recall = None
        if enable_option_outcome_stimulus:
            decoded, confidence = model.predict_option_outcome(
                option_context.to_trace(),
                strategy_option.option_id,
            )
            option_outcome_recall = {
                "option_id": strategy_option.option_id,
                "confidence": float(confidence),
                "decoded": decoded,
                "used_for_scoring": bool(
                    decoded is not None
                    and float(confidence) >= float(option_outcome_confidence_floor)
                    and not bool(decoded.get("survived_h", True))
                ),
            }

        # Outcome-recorder: snapshot the chosen plan's (concept, action) at
        # decision time so that after `outcome_horizon` env steps we can
        # write the realised outcome back into the world model.
        if enable_outcome_learning and outcome_recorder is not None:
            outcome_recorder.push(
                step=step,
                plan_steps=best_plan.steps,
                near=str(vf.near_concept) if vf.near_concept else None,
                health_now=float(body.get("health", 9.0)),
            )
        if option_outcome_recorder is not None:
            option_outcome_recorder.push(
                step=step,
                context=option_context,
                option=strategy_option,
                health_now=float(body.get("health", 9.0)),
                body_now={key: float(body.get(key, 9.0)) for key in vitals},
            )

        pixels, _reward, done, info = env.step(primitive)
        player_pos_after = tuple(info.get("player_pos", player_pos))
        if navigation_debug is not None:
            target_pos_for_debug = navigation_debug.get("target_pos")
            if target_pos_for_debug is not None:
                tx, ty = int(target_pos_for_debug[0]), int(target_pos_for_debug[1])
                navigation_debug["dist_after"] = int(
                    abs(tx - int(player_pos_after[0]))
                    + abs(ty - int(player_pos_after[1]))
                )
            else:
                navigation_debug["dist_after"] = None
            navigation_debug["actual_primitive"] = primitive
        raw_inv_after = dict(info.get("inventory", {}))
        body_after = {v: float(raw_inv_after.get(v, body.get(v, 0.0))) for v in vitals}
        inv_after = {
            key: value
            for key, value in raw_inv_after.items()
            if key not in _vital_set
        }
        facing_vec_after = _facing_delta(prev_move if not primitive.startswith("move_") else primitive)
        facing_tile_after = (
            (player_pos_after[0] + facing_vec_after[0], player_pos_after[1] + facing_vec_after[1])
            if facing_vec_after != (0, 0)
            else None
        )
        env_facing_after = _env_tile_truth(env, facing_tile_after)
        item_delta_after = {
            key: raw_inv_after.get(key, 0) - raw_inv.get(key, 0)
            for key in set(raw_inv_after.keys()) | set(raw_inv.keys())
            if key not in _vital_set and raw_inv_after.get(key, 0) - raw_inv.get(key, 0) != 0
        }
        body_delta_after = {
            key: float(body_after.get(key, 0.0)) - float(body.get(key, 0.0))
            for key in vitals
            if abs(float(body_after.get(key, 0.0)) - float(body.get(key, 0.0))) > 0.01
        }
        close_interaction_after_trace = False
        if interaction_completion_trace is not None:
            (
                interaction_intent,
                interaction_completion_trace,
                close_interaction_after_trace,
            ) = _update_interaction_completion_after_step(
                interaction_intent=interaction_intent,
                interaction_trace=interaction_completion_trace,
                primitive=primitive,
                control_origin=control_origin,
                rescue_trigger=rescue_trigger,
                inventory_delta=item_delta_after,
                body_delta=body_delta_after,
            )
            if (
                option_outcome_recorder is not None
                and interaction_completion_trace is not None
                and str(interaction_completion_trace.get("status")) in ("interrupted", "failed")
            ):
                option_outcome_recorder.mark_latest_failed(
                    health_now=float(body_after.get("health", 9.0)),
                    body_now={key: float(body_after.get(key, 9.0)) for key in vitals},
                    reason=str(interaction_completion_trace.get("reason") or "failed"),
                )
        if record_death_bundle:
            chosen_predicted_loss = (
                float(candidate_summaries[0].get("predicted_loss", predicted_best_loss))
                if candidate_summaries
                else float(predicted_best_loss)
            )
            other_candidates = candidate_summaries[1:] if candidate_summaries else []
            better_safe_candidate_exists = any(
                float(candidate.get("predicted_loss", chosen_predicted_loss)) + 0.01
                < chosen_predicted_loss
                for candidate in other_candidates
            )
            better_move_candidate_exists = any(
                str(candidate.get("plan", {}).get("first_action") or "").startswith("move_")
                and float(candidate.get("predicted_loss", chosen_predicted_loss)) + 0.01
                < chosen_predicted_loss
                for candidate in other_candidates
            )
            move_candidates_present = any(
                str(candidate.get("plan", {}).get("first_action") or "").startswith("move_")
                for candidate in candidate_summaries
            )
            hostile_entities = [
                entity
                for entity in observed_dynamic_entities
                if entity.concept_id in ("zombie", "skeleton", "arrow")
            ]
            snapshot = {
                "step": step,
                "goal": current_goal.id if current_goal is not None else None,
                "goal_trace": current_goal.to_trace() if current_goal is not None else None,
                "capability_state": capability_state.to_trace(),
                "player_pos_before": list(player_pos),
                "player_pos_after": list(player_pos_after),
                "body_before": {
                    key: round(float(value), 3)
                    for key, value in body.items()
                },
                "body_after": {
                    key: round(float(value), 3)
                    for key, value in body_after.items()
                },
                "primitive": primitive,
                "blocked_move": bool(
                    primitive.startswith("move_") and player_pos_after == player_pos
                ),
                "plan_origin": best_plan.origin,
                "plan_target": selected_target,
                "plan_action": selected_action,
                "chosen_predicted_loss": round(float(chosen_predicted_loss), 3),
                "baseline_predicted_loss": round(float(predicted_baseline_loss), 3),
                "actual_health_delta": round(
                    float(body_after.get("health", 0.0) - body.get("health", 0.0)),
                    3,
                ),
                "actual_damage": round(
                    max(0.0, float(body.get("health", 0.0) - body_after.get("health", 0.0))),
                    3,
                ),
                "hostile_present": bool(hostile_entities),
                "nearest_zombie_dist": _nearest_hostile_distance(
                    "zombie", player_pos, spatial_map, observed_dynamic_entities
                ),
                "nearest_skeleton_dist": _nearest_hostile_distance(
                    "skeleton", player_pos, spatial_map, observed_dynamic_entities
                ),
                "nearest_arrow_dist": _nearest_dynamic_distance(
                    "arrow", player_pos, observed_dynamic_entities
                ),
                "dynamic_entities": summarize_dynamic_entities(hostile_entities),
                "move_candidates_present": move_candidates_present,
                "better_safe_candidate_exists": better_safe_candidate_exists,
                "better_move_candidate_exists": better_move_candidate_exists,
                "top_candidates": candidate_summaries,
                "done_after_step": bool(done),
            }
            snapshot["error_label"] = infer_error_label(snapshot)
            death_trace_steps.append(snapshot)
            if len(death_trace_steps) > death_capture_steps:
                death_trace_steps = death_trace_steps[-death_capture_steps:]
        if record_step_trace:
            step_trace.append({
                "step": step,
                "player_pos_before": list(player_pos),
                "player_pos_after": list(player_pos_after),
                "facing_before": prev_move,
                "facing_tile_before": list(facing_tile_before) if facing_tile_before is not None else None,
                "facing_label_before": facing_label_before,
                "env_material_before": env_facing_before.get("material"),
                "env_object_before": env_facing_before.get("object"),
                "near_concept": vf.near_concept,
                "primitive": primitive,
                "plan_origin": best_plan.origin,
                "plan_target": selected_target,
                "plan_action": selected_action,
                "target_pos_before": list(target_pos_before) if target_pos_before is not None else None,
                "target_dist_before": target_dist_before,
                "target_matches_near": selected_target is not None and selected_target == vf.near_concept,
                "predicted_best_loss": round(float(predicted_best_loss), 3),
                "predicted_baseline_loss": round(float(predicted_baseline_loss), 3),
                "inventory_delta": item_delta_after,
                "wood_gain": int(item_delta_after.get("wood", 0)),
                "did_gain": bool(item_delta_after),
                "facing_tile_after": list(facing_tile_after) if facing_tile_after is not None else None,
                "env_material_after": env_facing_after.get("material"),
                "env_object_after": env_facing_after.get("object"),
                "done_after_step": bool(done),
            })
        if record_local_trace:
            capability_state_after = extract_capability_state(inv_after, textbook)
            counterfactual_outcomes = (
                _build_local_counterfactual_outcomes(
                    model=model,
                    state=state,
                    vf=vf,
                    cache=step_cache,
                    vitals=vitals,
                    horizon=min(local_counterfactual_horizon, horizon),
                    enable_post_plan_passive_rollout=enable_post_plan_passive_rollout,
                )
                if _should_record_local_counterfactuals(
                    record_local_counterfactuals,
                    near_concept=str(vf.near_concept),
                    body=body,
                    player_pos=player_pos,
                    observed_dynamic_entities=observed_dynamic_entities,
                )
                else []
            )
            local_entry = build_local_trace_entry(
                step=step,
                vf=vf,
                body=body,
                inventory=inv,
                primitive=primitive,
                plan_origin=best_plan.origin,
                controller=control_origin,
                planner_action=planner_primitive,
                learner_action=learner_action,
                learner_action_index=learner_action_index,
                rescue_applied=bool(rescue_pending is not None),
                rescue_trigger=rescue_trigger,
                nearest_threat_distances=nearest_threats_now,
                near_concept=str(vf.near_concept),
                player_pos_before=player_pos,
                player_pos_after=player_pos_after,
                body_after=body_after,
                inventory_after=inv_after,
                counterfactual_outcomes=counterfactual_outcomes,
                done_after_step=bool(done),
            )
            local_entry["station_spatial_debug"] = {
                "before_perception_update": station_diag_before_perception,
                "after_perception_update": station_diag_after_perception,
                "selected_target": str(selected_target) if selected_target is not None else None,
                "selected_action": str(selected_action) if selected_action is not None else None,
                "target_pos_before": (
                    [int(target_pos_before[0]), int(target_pos_before[1])]
                    if target_pos_before is not None
                    else None
                ),
                "target_dist_before": (
                    int(target_dist_before)
                    if target_dist_before is not None
                    else None
                ),
            }
            local_entry["goal"] = current_goal.to_trace() if current_goal is not None else None
            local_entry["interaction_intent"] = dict(interaction_intent) if interaction_intent is not None else None
            if interaction_completion_trace is not None:
                local_entry["interaction_completion"] = dict(interaction_completion_trace)
            local_entry["strategy_option"] = strategy_option.to_trace()
            local_entry["option_context"] = option_context.to_trace()
            if navigation_debug is not None:
                local_entry["navigation_debug"] = navigation_debug
            if option_outcome_recall is not None:
                local_entry["option_outcome_recall"] = option_outcome_recall
            if option_candidate_score_debug:
                local_entry["option_candidate_score_debug"] = option_candidate_score_debug
            local_entry["capability_state"] = capability_state.to_trace()
            local_entry["capability_state_after"] = capability_state_after.to_trace()
            local_entry["capability_delta"] = {
                key: capability_state_after.to_trace()[key]
                for key in capability_state_after.to_trace()
                if capability_state_after.to_trace()[key] != capability_state.to_trace()[key]
            }
            local_trace.append(local_entry)
        if close_interaction_after_trace:
            interaction_intent = None
        if record_local_advisory_trace and local_action_advisor is not None:
            from snks.agent.crafter_pixel_env import ACTION_TO_IDX

            advisory_observation = build_local_observation_package(
                vf,
                body,
                inv,
                belief_context=local_belief_tracker.build_context(
                    near_concept=str(vf.near_concept)
                ),
            )
            advisory_threats = {
                "zombie": _nearest_hostile_distance(
                    "zombie", player_pos, spatial_map, observed_dynamic_entities
                ),
                "skeleton": _nearest_hostile_distance(
                    "skeleton", player_pos, spatial_map, observed_dynamic_entities
                ),
                "arrow": _nearest_dynamic_distance(
                    "arrow", player_pos, observed_dynamic_entities
                ),
            }
            regime_labels, primary_regime = infer_local_regime(advisory_observation, advisory_threats)
            ranked_candidates = rank_local_action_candidates(
                evaluator=local_action_advisor,
                observation=advisory_observation,
                allowed_actions=local_advisory_allowed_actions,
                action_to_idx=ACTION_TO_IDX,
                device=local_advisory_device,
            )
            advisory_entry = build_local_advisory_entry(
                planner_action=primitive,
                planner_plan_origin=best_plan.origin,
                ranked_candidates=ranked_candidates,
                top_k=local_advisory_top_k,
            )
            advisory_entry.update(
                {
                    "step": int(step),
                    "near_concept": str(vf.near_concept),
                    "body": {
                        key: round(float(value), 3)
                        for key, value in body.items()
                    },
                    "regime_labels": regime_labels,
                    "primary_regime": primary_regime,
                    "belief_state_signature": dict(advisory_observation.get("belief_state_signature", {})),
                }
            )
            local_advisory_trace.append(advisory_entry)
        local_belief_tracker.observe_transition(
            near_concept=str(vf.near_concept),
            player_pos_before=player_pos,
            player_pos_after=player_pos_after,
            body_before=body,
            body_after=body_after,
            inventory_before=inv,
            inventory_after=inv_after,
            nearest_threat_distance_before=nearest_hostile_distance(
                {
                    "zombie": _nearest_hostile_distance(
                        "zombie", player_pos, spatial_map, observed_dynamic_entities
                    ),
                    "skeleton": _nearest_hostile_distance(
                        "skeleton", player_pos, spatial_map, observed_dynamic_entities
                    ),
                    "arrow": _nearest_dynamic_distance(
                        "arrow", player_pos, observed_dynamic_entities
                    ),
                }
            ),
        )
        step_displacement = (
            abs(int(player_pos_after[0]) - int(player_pos[0]))
            + abs(int(player_pos_after[1]) - int(player_pos[1]))
        )
        step_inventory_gain = sum(max(0, int(value)) for value in item_delta_after.values())
        if control_origin == "learner_actor" and step_displacement == 0 and step_inventory_gain == 0:
            actor_non_progress_streak += 1
        elif control_origin == "learner_actor":
            actor_non_progress_streak = 0
        else:
            actor_non_progress_streak = 0

        if rescue_pending is not None:
            post_nearest_threats = {
                "zombie": _nearest_hostile_distance(
                    "zombie", player_pos_after, spatial_map, observed_dynamic_entities
                ),
                "skeleton": _nearest_hostile_distance(
                    "skeleton", player_pos_after, spatial_map, observed_dynamic_entities
                ),
                "arrow": _nearest_dynamic_distance(
                    "arrow", player_pos_after, observed_dynamic_entities
                ),
            }
            post_nearest = nearest_hostile_distance(post_nearest_threats)
            pre_nearest = nearest_hostile_distance(nearest_threats_now)
            rescue_pending["post_rescue_outcome"] = {
                "damage_step": round(
                    max(0.0, float(body.get("health", 0.0) - body_after.get("health", 0.0))),
                    3,
                ),
                "health_delta_step": round(
                    float(body_after.get("health", 0.0) - body.get("health", 0.0)),
                    3,
                ),
                "displacement_step": int(step_displacement),
                "inventory_gain_step": int(step_inventory_gain),
                "nearest_hostile_after": post_nearest,
            }
            rescue_pending["immediate_outcome_delta"] = dict(rescue_pending["post_rescue_outcome"])
            if rescue_trigger == "repeated_non_progress":
                rescue_pending["rescue_improved_outcome"] = bool(step_displacement > 0 or step_inventory_gain > 0)
            else:
                rescue_pending["rescue_improved_outcome"] = bool(
                    float(body_after.get("health", 0.0)) >= float(body.get("health", 0.0))
                    and (
                        post_nearest is None
                        or pre_nearest is None
                        or int(post_nearest) >= int(pre_nearest)
                    )
                )
            rescue_trace.append(rescue_pending)

        # --- Bug 6: clear chopped tile ---
        new_inv = dict(info.get("inventory", {}))
        inv_changed = False
        for item_key in model.roles:
            if item_key.startswith("__"):
                continue
            old_count = inv.get(item_key, 0)
            new_count = new_inv.get(item_key, 0)
            if new_count > old_count and primitive in ("do",):
                # Gathered something — clear facing tile.
                # Facing direction = last MOVE primitive (not prev_action,
                # which was just set to "do" a few lines above).
                dx, dy = _facing_delta(prev_move)
                facing_tile = (player_pos[0] + dx, player_pos[1] + dy)
                spatial_map.update(facing_tile, "empty")
                inv_changed = True
                break

        # --- Bug 6b: frustrated do — clear stale resource entries ---
        # If `do` produced no inventory delta, the facing tile is probably
        # empty (e.g., tree was already harvested but segmenter still labels
        # the sapling as "tree"). Force-clear it with conf=1.0 so that
        # subsequent segmenter re-observations at lower conf can't restore
        # the stale label, breaking the "do on empty tile forever" loop.
        if primitive == "do" and not inv_changed:
            dx, dy = _facing_delta(prev_move)
            facing_tile = (player_pos[0] + dx, player_pos[1] + dy)
            if facing_tile != player_pos:
                spatial_map.update(facing_tile, "empty", 1.0)

        if done:
            raw_inv_end = dict(info.get("inventory", {}))
            body_at_end = {v: float(raw_inv_end.get(v, 0.0)) for v in vitals}
            if record_stage89c_trace:
                _update_defensive_event_windows(
                    defensive_events=defensive_events,
                    current_step=step + 1,
                    body=body_at_end,
                    alive=False,
                    window_targets=defensive_window_targets,
                )
            if pending_prediction_diag is not None:
                actual_loss = max(
                    0.0,
                    float(pending_prediction_diag["health_before"])
                    - body_at_end.get("health", 0.0),
                )
                if bool(pending_prediction_diag["arrow_threat"]):
                    danger_prediction_errors.append(
                        abs(float(pending_prediction_diag["predicted_loss"]) - actual_loss)
                    )
                pending_prediction_diag = None
            if any(body_at_end.get(v, 0.0) <= 0 for v in vitals):
                cause_of_death = "health"
                # Record killing blow: last env.step() reduced health but the
                # loop exits before the next iteration can compute the delta.
                final_health_delta = body_at_end.get("health", 0.0) - body.get("health", 9.0)
                if final_health_delta < 0:
                    nearby_cids = []
                    for entity_cid, entity_pos in entity_tracker.visible_entities():
                        ex, ey = entity_pos
                        dist = abs(ex - player_pos[0]) + abs(ey - player_pos[1])
                        nearby_cids.append((entity_cid, dist))
                    damage_log.append(DamageEvent(
                        step=step,
                        health_delta=float(final_health_delta),
                        vitals={k: body.get(k, 9.0) for k in ("food", "drink", "energy")},
                        nearby_cids=nearby_cids,
                    ))
            else:
                cause_of_death = "done"
            # Flush any pending outcome-recorder snapshots with the
            # episode's death cause before exiting the loop.
            if enable_outcome_learning and outcome_recorder is not None:
                outcome_recorder.flush_on_death(
                    health_now=float(body_at_end.get("health", 0.0)),
                    died_to=cause_of_death,
                )
            if option_outcome_recorder is not None:
                option_outcome_recorder.flush_on_death(
                    health_now=float(body_at_end.get("health", 0.0)),
                    died_to=cause_of_death,
                )
            break

        # Non-done branch: flush any outcome snapshots whose horizon ended
        # at this step. body_after is defined right after env.step above.
        if enable_outcome_learning and outcome_recorder is not None:
            outcome_recorder.flush_due(
                current_step=step,
                health_now=float(body_after.get("health", 9.0)),
            )
        if option_outcome_recorder is not None:
            option_outcome_recorder.flush_due(
                current_step=step,
                health_now=float(body_after.get("health", 9.0)),
                body_now={key: float(body_after.get(key, 9.0)) for key in vitals},
            )

    # --- Metrics ---
    total_actions = sum(action_counts.values())
    entropy = 0.0
    if total_actions > 0:
        for count in action_counts.values():
            p = count / total_actions
            if p > 0:
                entropy -= p * np.log2(p)

    attribution = PostMortemAnalyzer().attribute(damage_log, steps_taken)
    death_cause = dominant_cause(attribution)
    final_raw_inv = dict(info.get("inventory", {}))
    final_body = {v: float(final_raw_inv.get(v, 0.0)) for v in vitals}
    # Persist the world model (concepts/actions/roles/SDM/outcomes/
    # requirements) so the next episode of this seed starts with all
    # accumulated experience. Save BEFORE building the death-trace bundle
    # so a crash during death-trace doesn't lose state.
    if should_use_persistent_world_model and world_model_path is not None:
        model.save(Path(world_model_path))
    if promoted_path is not None:
        promoted_nodes_new = promoter.collect_entity_observations(
            spatial_map,
            episode_index=_next_promoted_episode_index(promoted_nodes_prior),
        )
        promoter.save_nodes(
            promoter.merge_nodes(promoted_nodes_prior, promoted_nodes_new),
            promoted_path,
        )

    death_trace_bundle = None
    if record_death_bundle and death_cause != "alive":
        death_trace_bundle = build_death_trace_bundle(
            episode_steps=steps_taken,
            death_cause=death_cause,
            env_cause=cause_of_death,
            final_body=final_body,
            final_inventory=final_raw_inv,
            capture_horizon=death_capture_steps,
            recent_steps=death_trace_steps,
        )

    return {
        "avg_len": steps_taken,        # legacy name kept for backward compat
        "episode_steps": steps_taken,
        "cause": cause_of_death,
        "death_cause": death_cause,
        "damage_log": damage_log,
        "final_inv": final_raw_inv,
        "final_body": final_body,
        "action_counts": dict(action_counts),
        "action_entropy": round(entropy, 3),
        "total_surprise": round(total_surprise, 3),
        "n_surprise_events": n_surprise_events,
        "mean_surprise": round(total_surprise / max(n_surprise_events, 1), 3),
        "arrow_threat_steps": arrow_threat_steps,
        "defensive_action_steps": defensive_action_steps,
        "defensive_action_rate": round(
            defensive_action_steps / max(arrow_threat_steps, 1), 3
        ),
        "danger_prediction_error": round(
            float(np.mean(danger_prediction_errors)) if danger_prediction_errors else 0.0,
            3,
        ),
        "arrow_visible_steps": arrow_visible_steps,
        "arrow_velocity_known_steps": arrow_velocity_known_steps,
        "arrow_velocity_unknown_steps": arrow_velocity_unknown_steps,
        "arrow_velocity_known_rate": round(
            arrow_velocity_known_steps / max(arrow_visible_steps, 1), 3
        ),
        "first_arrow_threat_step": first_arrow_threat_step,
        "first_defensive_action_step": first_defensive_action_step,
        "n_defensive_sequences": defensive_sequences,
        "defensive_events": defensive_events if record_stage89c_trace else [],
        "step_trace": step_trace if record_step_trace else [],
        "death_trace_bundle": death_trace_bundle,
        "local_trace": local_trace if record_local_trace else [],
        "local_advisory_trace": local_advisory_trace if record_local_advisory_trace else [],
        "rescue_trace": rescue_trace,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_record_local_counterfactuals(
    record_local_counterfactuals: bool | str,
    *,
    near_concept: str | None,
    body: dict[str, float],
    player_pos: tuple[int, int],
    observed_dynamic_entities: list[DynamicEntityState],
) -> bool:
    if isinstance(record_local_counterfactuals, bool):
        return record_local_counterfactuals
    mode = str(record_local_counterfactuals).strip().lower()
    if mode in {"", "false", "none", "off"}:
        return False
    if mode not in {"salient_only", "salient"}:
        return True
    if str(near_concept or "empty") in {"tree", "water", "stone", "coal", "iron", "diamond", "cow"}:
        return True
    if body and min(float(body.get(key, 9.0)) for key in ("health", "food", "drink", "energy")) <= 4.0:
        return True
    px, py = int(player_pos[0]), int(player_pos[1])
    for entity in observed_dynamic_entities:
        if entity.concept_id not in {"zombie", "skeleton", "arrow"}:
            continue
        dist = abs(int(entity.position[0]) - px) + abs(int(entity.position[1]) - py)
        if dist <= 3:
            return True
    return False


def _nearest_emergency_threat_distances(
    facts: EmergencyWorldFacts,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    observed_dynamic_entities: list[DynamicEntityState],
) -> dict[str, int | None]:
    distances: dict[str, int | None] = {}
    for concept_id in facts.hostile_concepts:
        dynamic_dist = _nearest_dynamic_distance(
            concept_id,
            player_pos,
            observed_dynamic_entities,
        )
        if concept_id == "arrow":
            distances[concept_id] = dynamic_dist
        else:
            distances[concept_id] = _nearest_hostile_distance(
                concept_id,
                player_pos,
                spatial_map,
                observed_dynamic_entities,
            )
    return distances


def _mixed_control_rescue_trigger(
    *,
    body: dict[str, float],
    nearest_threat_distances: dict[str, int | None],
    actor_non_progress_streak: int,
    low_vitals_threshold: float,
    hostile_distance_threshold: int,
    stall_streak_threshold: int,
) -> str | None:
    if body and min(float(body.get(key, 9.0)) for key in ("health", "food", "drink", "energy")) <= low_vitals_threshold:
        return "low_vitals"
    nearest_hostile = nearest_hostile_distance(nearest_threat_distances)
    if nearest_hostile is not None and nearest_hostile <= hostile_distance_threshold:
        return "hostile_contact"
    if actor_non_progress_streak >= stall_streak_threshold:
        return "repeated_non_progress"
    return None


def _select_mixed_control_rescue_action(
    *,
    actor_action: str,
    planner_action: str,
    rescue_trigger: str | None,
    advisory_ranked: list[dict[str, Any]] | None = None,
) -> tuple[str, str] | None:
    if rescue_trigger is None:
        return None
    if actor_action != planner_action:
        return planner_action, "planner_override"
    if advisory_ranked:
        advisory_best = str(advisory_ranked[0].get("action", actor_action))
        if advisory_best != actor_action:
            return advisory_best, "advisory_override"
    return None

def _build_local_counterfactual_outcomes(
    *,
    model: VectorWorldModel,
    state: VectorState,
    vf: VisualField,
    cache: dict | None,
    vitals: list[str],
    horizon: int,
    enable_post_plan_passive_rollout: bool,
) -> list[dict[str, Any]]:
    from snks.agent.crafter_pixel_env import ACTION_TO_IDX

    def _counterfactual_do_target() -> str | None:
        near_concept = str(vf.near_concept)
        if near_concept not in {"None", "empty", "unknown"}:
            return near_concept
        px, py = int(state.player_pos[0]), int(state.player_pos[1])
        candidates: list[tuple[int, str]] = []
        for (target, action), _reqs in getattr(model, "action_requirements", {}).items():
            if action != "do" or not model.requirements_met(target, action, state.inventory):
                continue
            target_pos = _nearest_known_target_position(
                target,
                state.player_pos,
                state.spatial_map,
                state.dynamic_entities,
            )
            if target_pos is None:
                continue
            dist = abs(int(target_pos[0]) - px) + abs(int(target_pos[1]) - py)
            if dist <= 1:
                candidates.append((dist, str(target)))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    allowed_actions = ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]
    start_threats = {
        "zombie": _nearest_hostile_distance("zombie", state.player_pos, state.spatial_map, state.dynamic_entities),
        "skeleton": _nearest_hostile_distance("skeleton", state.player_pos, state.spatial_map, state.dynamic_entities),
        "arrow": _nearest_dynamic_distance("arrow", state.player_pos, state.dynamic_entities),
    }
    start_hostile = min(
        [distance for distance in start_threats.values() if distance is not None],
        default=None,
    )
    outcomes: list[dict[str, Any]] = []
    for primitive in allowed_actions:
        target = "self"
        if primitive == "do":
            do_target = _counterfactual_do_target()
            if do_target is None:
                continue
            target = do_target
        plan = VectorPlan(
            steps=[VectorPlanStep(action=primitive, target=target)],
            origin=f"stage90r_counterfactual:{primitive}",
        )
        trajectory = simulate_forward(
            model=model,
            plan=plan,
            initial_state=state,
            horizon=max(1, horizon),
            vital_vars=vitals,
            cache=cache,
            enable_post_plan_passive_rollout=enable_post_plan_passive_rollout,
        )
        final_state = trajectory.final_state or state
        displacement_h = (
            abs(int(final_state.player_pos[0]) - int(state.player_pos[0]))
            + abs(int(final_state.player_pos[1]) - int(state.player_pos[1]))
        )
        end_threats = {
            "zombie": _nearest_hostile_distance("zombie", final_state.player_pos, final_state.spatial_map, final_state.dynamic_entities),
            "skeleton": _nearest_hostile_distance("skeleton", final_state.player_pos, final_state.spatial_map, final_state.dynamic_entities),
            "arrow": _nearest_dynamic_distance("arrow", final_state.player_pos, final_state.dynamic_entities),
        }
        end_hostile = min(
            [distance for distance in end_threats.values() if distance is not None],
            default=None,
        )
        inventory_delta: dict[str, int] = {}
        resource_gain = 0
        for key in set(state.inventory.keys()) | set(final_state.inventory.keys()):
            delta = int(final_state.inventory.get(key, 0)) - int(state.inventory.get(key, 0))
            if delta != 0:
                inventory_delta[key] = delta
            if delta > 0:
                resource_gain += delta
        if start_hostile is None or end_hostile is None:
            escape_delta = None
        else:
            escape_delta = int(end_hostile - start_hostile)
        outcomes.append(
            {
                "action": primitive,
                "action_index": int(ACTION_TO_IDX.get(primitive, 0)),
                "source": "explicit_vector_rollout",
                "target": target,
                "mean_confidence": round(
                    sum(trajectory.confidences) / max(len(trajectory.confidences), 1),
                    4,
                ),
                "terminated": bool(trajectory.terminated),
                "terminated_reason": trajectory.terminated_reason,
                "label": {
                    "health_delta_h": round(
                        float(final_state.body.get("health", 0.0)) - float(state.body.get("health", 0.0)),
                        3,
                    ),
                    "damage_h": round(
                        max(0.0, float(state.body.get("health", 0.0)) - float(final_state.body.get("health", 0.0))),
                        3,
                    ),
                    "resource_gain_h": int(resource_gain),
                    "inventory_delta_h": inventory_delta,
                    "survived_h": not final_state.is_dead(vitals),
                    "escape_delta_h": escape_delta,
                    "nearest_hostile_now": start_hostile,
                    "nearest_hostile_h": end_hostile,
                    "effective_displacement_h": int(displacement_h),
                    "blocked_h": bool(
                        primitive.startswith("move_") and displacement_h == 0
                    ),
                    "adjacent_hostile_after_h": bool(
                        end_hostile is not None and int(end_hostile) <= 1
                    ),
                },
            }
        )
    return outcomes

def _facing_delta(last_move: str | None) -> tuple[int, int]:
    if last_move == "move_right":
        return (1, 0)
    if last_move == "move_left":
        return (-1, 0)
    if last_move == "move_down":
        return (0, 1)
    if last_move == "move_up":
        return (0, -1)
    return (0, 0)


def _spatial_label_at(
    spatial_map: CrafterSpatialMap,
    pos: tuple[int, int] | None,
) -> str | None:
    if pos is None:
        return None
    entry = spatial_map._map.get((int(pos[0]), int(pos[1])))
    if entry is None:
        return None
    return str(entry[0])


# Concepts surfaced in `local_trace[*].station_spatial_debug.nearest` for
# Phase-1 frontier-exploration validation. Stations stay first to preserve the
# field's original ordering; resources are appended so a recorder can answer
# "does the agent know where water/cow/coal/iron/tree/stone is right now?".
_SPATIAL_DEBUG_CONCEPTS: tuple[str, ...] = (
    "table",
    "furnace",
    "water",
    "cow",
    "coal",
    "iron",
    "tree",
    "stone",
    "diamond",
    "plant",
)


def _station_spatial_debug(
    spatial_map: CrafterSpatialMap,
    player_pos: tuple[int, int],
    *,
    concepts: tuple[str, ...],
    entry_limit: int = 12,
) -> dict[str, Any]:
    """Small JSON-safe snapshot of known crafting-station positions."""
    py, px = int(player_pos[0]), int(player_pos[1])
    entries: list[dict[str, Any]] = []
    for (y, x), (concept, confidence, count) in getattr(spatial_map, "_map", {}).items():
        if concept not in concepts:
            continue
        dist = abs(int(y) - py) + abs(int(x) - px)
        entries.append({
            "concept": str(concept),
            "pos": [int(y), int(x)],
            "dist": int(dist),
            "confidence": round(float(confidence), 4),
            "count": int(count),
        })
    entries.sort(key=lambda item: (item["dist"], item["concept"], item["pos"]))

    nearest: dict[str, Any] = {}
    for concept in concepts:
        pos = spatial_map.find_nearest(concept, player_pos)
        nearest[concept] = {
            "pos": [int(pos[0]), int(pos[1])] if pos is not None else None,
            "dist": (
                int(abs(pos[0] - py) + abs(pos[1] - px))
                if pos is not None
                else None
            ),
        }

    return {
        "player_pos": [py, px],
        "nearest": nearest,
        "entries": entries[:entry_limit],
        "n_entries": len(entries),
    }


def _env_tile_truth(env: Any, pos: tuple[int, int] | None) -> dict[str, str | None]:
    if pos is None:
        return {"material": None, "object": None}
    inner = getattr(env, "_env", None)
    world = getattr(inner, "_world", None)
    if world is None:
        return {"material": None, "object": None}
    try:
        material, obj = world[pos]
    except Exception:
        return {"material": None, "object": None}
    object_name = None
    if obj is not None:
        object_name = getattr(obj, "texture", None) or obj.__class__.__name__.lower()
    return {
        "material": str(material) if material is not None else None,
        "object": str(object_name) if object_name is not None else None,
    }

_HAZARD_CONCEPTS = {"lava"}


def _update_spatial_map_hazards(
    spatial_map: "CrafterSpatialMap",
    info: dict,
    player_pos: tuple[int, int],
    radius: int = 5,
) -> None:
    """Scan info["semantic"] around the player for static hazard tiles and
    write them into spatial_map.

    Bypasses NEAR_TO_IDX / NEAR_CLASSES (which the trained local actor
    `.pt` depends on for its output dimension). Adding hazards there
    triggers a CUDA index-out-of-bounds in the actor's logits because
    the model was trained with the old class set.

    Direct semantic scan keeps the actor untouched while still letting
    the planner see lava tiles via spatial_map.concept_at(...).
    """
    import numpy as np
    from snks.agent.crafter_pixel_env import SEMANTIC_NAMES
    semantic = info.get("semantic")
    if semantic is None:
        return
    semantic = np.asarray(semantic)
    py, px = int(player_pos[0]), int(player_pos[1])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            wy, wx = py + dy, px + dx
            if not (0 <= wy < semantic.shape[0] and 0 <= wx < semantic.shape[1]):
                continue
            name = SEMANTIC_NAMES.get(int(semantic[wy, wx]), "unknown")
            if name in _HAZARD_CONCEPTS:
                # Hazards are immutable — write directly into _map to
                # bypass spatial_map.update's "empty overrides stale
                # resource labels" rule that would otherwise erase
                # this entry on the next perception step.
                key = (int(wy), int(wx))
                spatial_map._map[key] = (name, 1.0, 1)
                spatial_map._visited.add(key)


def _update_spatial_map(
    spatial_map: CrafterSpatialMap,
    vf: VisualField,
    player_pos: tuple[int, int],
    prev_move: str | None = None,
) -> None:
    """Write viewport detections into spatial_map with confidence."""
    # Only naturally-occurring world objects go into spatial_map.
    # Placed/crafted items (table) are never spawned by the world and
    # should not appear from segmenter false-positives.
    _NATURAL_CONCEPTS = {
        "tree", "stone", "coal", "iron", "diamond",
        "water", "cow", "zombie", "skeleton", "empty",
        # Placed objects (`table`, `furnace`) are persistent within an episode
        # and even more important to remember than consumable resources:
        # without them in the map, `find_nearest("table", ...)` returns None,
        # which made the planner regenerate `place_<near_req>` plans every
        # step the agent strayed from its prior table, leaving rows of
        # stacked tables on the map and never executing the make plan
        # they were chained to.
        "table", "furnace",
    }

    px, py = int(player_pos[0]), int(player_pos[1])
    center_row, center_col = 3, 4  # 7×9 viewport

    if vf.near_concept in _NATURAL_CONCEPTS:
        # near_concept describes the FACING tile, so record the label at the
        # facing tile coordinate — not the player's own tile, which is
        # actually empty (player can't stand on a resource/table). Writing at
        # player_pos produced lying entries like `(28,34) → "table"` even
        # though the table sits at (28,33), and `find_nearest("table")` then
        # returned the player's prior position instead of the placed table.
        facing_delta = _facing_delta(prev_move)
        if facing_delta == (0, 0):
            facing_delta = (0, 1)  # default facing: down
        fx, fy = px + facing_delta[0], py + facing_delta[1]
        spatial_map.update((fx, fy), vf.near_concept, vf.near_similarity)

    for cid, conf, gy, gx in vf.detections:
        if cid not in _NATURAL_CONCEPTS:
            continue
        wx = px + (gx - center_col)
        wy = py + (gy - center_row)
        spatial_map.update((wx, wy), cid, conf)


def _nearest_spatial_distance(
    concept_id: str,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
) -> int | None:
    pos = spatial_map.find_nearest(concept_id, player_pos)
    if pos is None:
        return None
    return abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1])


def _nearest_dynamic_distance(
    concept_id: str,
    player_pos: tuple[int, int],
    dynamic_entities: list[DynamicEntityState],
) -> int | None:
    distances = [
        abs(entity.position[0] - player_pos[0]) + abs(entity.position[1] - player_pos[1])
        for entity in dynamic_entities
        if entity.concept_id == concept_id
    ]
    return min(distances) if distances else None


def _nearest_hostile_distance(
    concept_id: str,
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    dynamic_entities: list[DynamicEntityState],
) -> int | None:
    distances = []
    spatial_dist = _nearest_spatial_distance(concept_id, player_pos, spatial_map)
    dynamic_dist = _nearest_dynamic_distance(concept_id, player_pos, dynamic_entities)
    if spatial_dist is not None:
        distances.append(spatial_dist)
    if dynamic_dist is not None:
        distances.append(dynamic_dist)
    return min(distances) if distances else None


def _update_defensive_event_windows(
    defensive_events: list[dict[str, Any]],
    current_step: int,
    body: dict[str, float],
    alive: bool,
    window_targets: tuple[int, ...],
) -> None:
    for event in defensive_events:
        elapsed = current_step - int(event["step"])
        for window in window_targets:
            health_key = f"health_delta_{window}"
            food_key = f"food_delta_{window}"
            drink_key = f"drink_delta_{window}"
            survive_key = f"survived_{window}"
            threat_delta_key = f"threat_distance_delta_{window}"
            if health_key in event:
                continue
            if elapsed < window and alive:
                continue
            event[health_key] = round(float(body.get("health", 0.0)) - float(event["health"]), 3)
            event[food_key] = round(float(body.get("food", 0.0)) - float(event["food"]), 3)
            event[drink_key] = round(float(body.get("drink", 0.0)) - float(event["drink"]), 3)
            event[survive_key] = bool(alive and elapsed >= window)
            event[threat_delta_key] = None
