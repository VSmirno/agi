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
    rank_local_action_candidates,
)
from snks.agent.stage90r_local_policy import (
    TemporalBeliefTracker,
    build_local_observation_package,
    build_local_trace_entry,
    infer_local_regime,
)


# ---------------------------------------------------------------------------
# Dynamic entity tracker (lightweight, from mpc_agent)
# ---------------------------------------------------------------------------

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
) -> list[VectorPlan]:
    """Generate plans via forward imagination.

    For each visible/known concept × action with positive predicted effect,
    build a plan. Then recursively extend promising plans (generate_chains).
    """
    candidates: list[VectorPlan] = []

    # Gather all known concepts (visible + in spatial map)
    known = set(visible_concepts) | set(spatial_map.known_objects.keys())

    # Single-step plans: try each concept × target-action
    action_ids = list(model.actions.keys())
    target_actions = [a for a in action_ids if a in ("do", "make")]
    self_actions = [a for a in action_ids if a in ("sleep",)]
    move_actions = [a for a in action_ids if a.startswith("move_")]

    if cache is None:
        cache = build_prediction_cache(model, known, target_actions)

    # Concepts that are never valid plan targets.
    # "empty" — background tile, no resource to gather.
    # "self"  — handled separately via self_actions (sleep).
    # enemies — attacking doesn't yield inventory resources.
    non_targetable = {"empty", "self", "zombie", "skeleton"}

    for concept_id in known:
        if concept_id in non_targetable:
            continue
        for action in target_actions:
            # For "make": only allow concepts that have a textbook requirement entry.
            # Blocks spurious SDM associations (diamond:make, coal:make, etc.)
            # that were never declared in the textbook.
            if action == "make" and (concept_id, action) not in model.action_requirements:
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

    # Self-actions as standalone plans (no target concept).
    # No confidence/effect gate — scoring handles everything:
    # sleep wins only when min_vital improves in simulation (vitals low);
    # baseline (dist=0, -steps=0) beats idle sleep when vitals full.
    for action in self_actions:
        candidates.append(VectorPlan(
            steps=[VectorPlanStep(action=action, target="self")],
            origin=f"self:{action}",
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
    """Check if effect has any positive inventory delta."""
    for var, val in decoded.items():
        if var not in state.body and val > 0:
            return True
    return False


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
            if action == "make" and (concept_id, action) not in model.action_requirements:
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
                    if action == "make" and (concept_id, action) not in model.action_requirements:
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
) -> str:
    """Expand a plan step to a single env primitive.

    Maps abstract actions to Crafter env primitives:
    - "do" → "do"
    - "sleep" → "sleep"
    - "make" + target "wood_sword" → "make_wood_sword"
    - "place" + target "table" → "place_table"

    If target is not adjacent, navigate toward it first.
    """
    target_pos = spatial_map.find_nearest(plan_step.target, player_pos)

    if target_pos is None and plan_step.action not in ("sleep",):
        # near_concept == target: resource is at the player's center tile —
        # find_nearest skipped it (Bug 5 guard for stale perception entries).
        # The player is physically adjacent; execute the action directly.
        if near_concept is not None and near_concept == plan_step.target and plan_step.action == "do":
            return "do"
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
    record_local_counterfactuals: bool = True,
    local_counterfactual_horizon: int = 3,
    local_action_advisor: Any | None = None,
    local_advisory_allowed_actions: list[str] | None = None,
    record_local_advisory_trace: bool = False,
    local_advisory_top_k: int = 3,
    local_advisory_device: torch.device | str = "cpu",
    death_capture_steps: int = DEFAULT_DEATH_TRACE_HORIZON,
    perception_mode: str = "pixel",
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

    from snks.agent.goal_selector import Goal, GoalSelector
    goal_selector = (
        GoalSelector(
            textbook,
            allow_dynamic_entity_goals=enable_dynamic_threat_goals,
        )
        if textbook is not None
        else None
    )

    entity_tracker = DynamicEntityTracker()
    # Register known dynamic concepts
    for cid in model.concepts:
        if cid in ("zombie", "skeleton", "cow", "arrow"):
            entity_tracker.register_dynamic_concept(cid)

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
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
    local_advisory_allowed_actions = (
        list(local_advisory_allowed_actions)
        if local_advisory_allowed_actions is not None
        else ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]
    )
    local_belief_tracker = TemporalBeliefTracker()

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
        _update_spatial_map(spatial_map, vf, player_pos)
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
        known_step = set(vf.visible_concepts()) | set(spatial_map.known_objects.keys())
        target_acts = [a for a in model.actions if a in ("do", "make", "place")]
        if enable_dynamic_threat_model and observed_dynamic_entities and "proximity" in model.actions:
            target_acts.append("proximity")
        step_cache = build_prediction_cache(model, known_step, target_acts)

        # --- Generate + simulate + score ---
        candidates = generate_candidate_plans(
            model, state, spatial_map, vf.visible_concepts(),
            beam_width=beam_width,
            max_depth=max_depth,
            cache=step_cache,
            enable_motion_plans=enable_motion_plans,
            enable_motion_chains=enable_motion_chains,
        )

        # Sort candidates by proximity to first target — closer first.
        # Stable sort below keeps proximity order within equal scores.
        def _plan_distance(plan: VectorPlan) -> int:
            if not plan.steps:
                # Baseline plan = exploration (random move). Always "reachable"
                # so known=1. -steps=0 beats sleep (-steps=-1) when both have
                # total_gain=0 — agent explores rather than sleeps uselessly.
                return 0
            # All steps must have known targets — if any step's target is not in
            # spatial_map, the whole plan is unreachable (known=0 in scoring).
            max_dist = 0
            for step in plan.steps:
                if step.target == "self":
                    continue
                # near_concept == target means the resource is immediately adjacent
                # to the player. find_nearest skips player_pos (Bug 5 guard), so
                # it would return None for center-tile resources — treat as dist=0.
                if step.target == vf.near_concept:
                    continue
                pos = spatial_map.find_nearest(step.target, player_pos)
                if pos is None:
                    return 9999
                max_dist = max(max_dist, abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]))
            return max_dist

        candidates.sort(key=_plan_distance)

        # Current goal for this step (pure function of current state)
        current_goal = goal_selector.select(state) if goal_selector else Goal("explore")

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
        selected_target = best_plan.steps[0].target if best_plan.steps else None
        selected_action = best_plan.steps[0].action if best_plan.steps else None
        target_pos_before = (
            spatial_map.find_nearest(selected_target, player_pos)
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

        # --- Execute first primitive ---
        if best_plan.steps:
            primitive = expand_to_primitive(
                best_plan.steps[0], player_pos, spatial_map, model, rng,
                last_action=prev_move,  # facing = last move, not last action
                near_concept=vf.near_concept,
            )
        else:
            move_actions = [a for a in model.actions if a.startswith("move_")]
            primitive = str(rng.choice(move_actions)) if move_actions else "move_right"

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
        prev_player_pos = player_pos

        pixels, _reward, done, info = env.step(primitive)
        player_pos_after = tuple(info.get("player_pos", player_pos))
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
                if record_local_counterfactuals
                else []
            )
            local_trace.append(
                build_local_trace_entry(
                    step=step,
                    vf=vf,
                    body=body,
                    inventory=inv,
                    primitive=primitive,
                    plan_origin=best_plan.origin,
                    nearest_threat_distances={
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
                    near_concept=str(vf.near_concept),
                    player_pos_before=player_pos,
                    player_pos_after=player_pos_after,
                    body_after=body_after,
                    inventory_after=inv_after,
                    counterfactual_outcomes=counterfactual_outcomes,
                    done_after_step=bool(done),
                )
            )
        if record_local_advisory_trace and local_action_advisor is not None:
            from snks.agent.crafter_pixel_env import ACTION_TO_IDX

            advisory_observation = build_local_observation_package(
                vf,
                body,
                inv,
                temporal_context=local_belief_tracker.build_context(
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
                    "temporal_signature": dict(advisory_observation.get("temporal_signature", {})),
                }
            )
            local_advisory_trace.append(advisory_entry)
        local_belief_tracker.observe_transition(
            action=primitive,
            near_concept=str(vf.near_concept),
            player_pos_before=player_pos,
            player_pos_after=player_pos_after,
            body_before=body,
            body_after=body_after,
            inventory_before=inv,
            inventory_after=inv_after,
        )

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
            break

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
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            near_concept = str(vf.near_concept)
            if near_concept in {"None", "empty", "unknown"}:
                continue
            target = near_concept
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

def _update_spatial_map(
    spatial_map: CrafterSpatialMap,
    vf: VisualField,
    player_pos: tuple[int, int],
) -> None:
    """Write viewport detections into spatial_map with confidence."""
    # Only naturally-occurring world objects go into spatial_map.
    # Placed/crafted items (table) are never spawned by the world and
    # should not appear from segmenter false-positives.
    _NATURAL_CONCEPTS = {
        "tree", "stone", "coal", "iron", "diamond",
        "water", "cow", "zombie", "skeleton", "empty",
    }

    px, py = int(player_pos[0]), int(player_pos[1])
    center_row, center_col = 3, 4  # 7×9 viewport

    if vf.near_concept in _NATURAL_CONCEPTS:
        spatial_map.update((px, py), vf.near_concept, vf.near_similarity)

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
