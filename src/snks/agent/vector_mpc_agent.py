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
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.perception import (
    HomeostaticTracker,
    VisualField,
    perceive_tile_field,
)
from snks.agent.vector_world_model import VectorWorldModel, bind, hamming_similarity
from snks.agent.vector_sim import (
    VectorState,
    VectorPlan,
    VectorPlanStep,
    VectorTrajectory,
    simulate_forward,
    score_trajectory,
)


# ---------------------------------------------------------------------------
# Dynamic entity tracker (lightweight, from mpc_agent)
# ---------------------------------------------------------------------------

class DynamicEntityTracker:
    """Track positions of moving entities between ticks."""

    def __init__(self) -> None:
        self._dynamic_concepts: set[str] = set()
        self._positions: dict[str, list[tuple[int, int]]] = {}

    def register_dynamic_concept(self, concept_id: str) -> None:
        self._dynamic_concepts.add(concept_id)

    def update(self, vf: VisualField, player_pos: tuple[int, int]) -> None:
        self._positions.clear()
        px, py = int(player_pos[0]), int(player_pos[1])
        center_row, center_col = 3, 4  # 7x9 viewport center
        for cid, _conf, gy, gx in vf.detections:
            if cid in self._dynamic_concepts:
                wx = px + (gx - center_col)
                wy = py + (gy - (center_row - 1))
                self._positions.setdefault(cid, []).append((wx, wy))

    def visible_entities(self) -> list[tuple[str, tuple[int, int]]]:
        result = []
        for cid, positions in self._positions.items():
            for pos in positions:
                result.append((cid, pos))
        return result

    def min_distance(self, concept_id: str, player_pos: tuple[int, int]) -> int | None:
        positions = self._positions.get(concept_id, [])
        if not positions:
            return None
        px, py = player_pos
        return min(abs(px - ex) + abs(py - ey) for ex, ey in positions)


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

    if cache is None:
        cache = build_prediction_cache(model, known, target_actions)

    # Concepts that are never valid plan targets.
    # "empty" — background tile, no resource to gather.
    # "self"  — handled separately via self_actions (sleep).
    non_targetable = {"empty", "self"}

    for concept_id in known:
        if concept_id in non_targetable:
            continue
        for action in target_actions:
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

    # Self-actions as standalone plans (no target concept)
    for action in self_actions:
        candidates.append(VectorPlan(
            steps=[VectorPlanStep(action=action, target="self")],
            origin=f"self:{action}",
        ))

    # Multi-step chains via beam search (target actions only)
    chains = _generate_chains(model, state, known, target_actions,
                              beam_width=beam_width, max_depth=max_depth,
                              cache=cache)
    candidates.extend(chains)

    # Always include a "do nothing" plan as baseline
    candidates.append(VectorPlan(steps=[], origin="baseline"))

    return candidates


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

    non_targetable = {"empty", "self"}
    for concept_id in known_concepts:
        if concept_id in non_targetable:
            continue
        for action in plan_actions:
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
    verbose: bool = False,
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

    entity_tracker = DynamicEntityTracker()
    # Register known dynamic concepts
    for cid in model.concepts:
        if cid in ("zombie", "skeleton", "cow"):
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

    for step in range(max_steps):
        steps_taken = step + 1
        inv = dict(info.get("inventory", {}))
        body = {v: float(info.get(v, 9.0)) for v in vitals}
        player_pos = tuple(info.get("player_pos", (32, 32)))

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
        vf = perceive_tile_field(pixels, segmenter)
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
                    for entity_cid, entity_pos in entity_tracker.visible_entities():
                        ex, ey = entity_pos
                        dist = abs(ex - player_pos[0]) + abs(ey - player_pos[1])
                        if dist <= 6:
                            model.learn(entity_cid, "proximity",
                                        {"health": health_delta})

        # --- Build VectorState ---
        state = VectorState(
            inventory=inv,
            body=body,
            player_pos=player_pos,
            step=step,
            last_action=prev_action,
            spatial_map=spatial_map,
        )

        # --- Build per-step prediction cache (one batched GPU op) ---
        known_step = set(vf.visible_concepts()) | set(spatial_map.known_objects.keys())
        target_acts = [a for a in model.actions if a in ("do", "make", "place")]
        step_cache = build_prediction_cache(model, known_step, target_acts)

        # --- Generate + simulate + score ---
        candidates = generate_candidate_plans(
            model, state, spatial_map, vf.visible_concepts(),
            beam_width=beam_width, max_depth=max_depth, cache=step_cache,
        )

        # Sort candidates by proximity to first target — closer first.
        # Stable sort below keeps proximity order within equal scores.
        def _plan_distance(plan: VectorPlan) -> int:
            if not plan.steps:
                return 9999  # baseline last
            first = plan.steps[0]
            if first.target == "self":
                return 0
            pos = spatial_map.find_nearest(first.target, player_pos)
            if pos is None:
                return 9999
            return abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1])

        candidates.sort(key=_plan_distance)

        scored: list[tuple[tuple, VectorPlan, VectorTrajectory]] = []
        for plan in candidates:
            traj = simulate_forward(model, plan, state, horizon, vitals, cache=step_cache)
            score = score_trajectory(traj, vitals)
            scored.append((score, plan, traj))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_plan, best_traj = scored[0]

        # --- Execute first primitive ---
        if best_plan.steps:
            primitive = expand_to_primitive(
                best_plan.steps[0], player_pos, spatial_map, model, rng,
                last_action=prev_move,  # facing = last move, not last action
            )
        else:
            move_actions = [a for a in model.actions if a.startswith("move_")]
            primitive = str(rng.choice(move_actions)) if move_actions else "move_right"

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
                dx, dy = 0, 0
                if prev_move == "move_right":
                    dx = 1
                elif prev_move == "move_left":
                    dx = -1
                elif prev_move == "move_down":
                    dy = 1
                elif prev_move == "move_up":
                    dy = -1
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
            dx, dy = 0, 0
            if prev_move == "move_right":
                dx = 1
            elif prev_move == "move_left":
                dx = -1
            elif prev_move == "move_down":
                dy = 1
            elif prev_move == "move_up":
                dy = -1
            facing_tile = (player_pos[0] + dx, player_pos[1] + dy)
            if facing_tile != player_pos:
                spatial_map.update(facing_tile, "empty", 1.0)

        if done:
            body_at_end = {v: float(info.get(v, 0)) for v in vitals}
            if any(body_at_end.get(v, 0) <= 0 for v in vitals):
                cause_of_death = "health"
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

    return {
        "avg_len": steps_taken,
        "cause": cause_of_death,
        "final_inv": dict(info.get("inventory", {})),
        "action_counts": dict(action_counts),
        "action_entropy": round(entropy, 3),
        "total_surprise": round(total_surprise, 3),
        "n_surprise_events": n_surprise_events,
        "mean_surprise": round(total_surprise / max(n_surprise_events, 1), 3),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_spatial_map(
    spatial_map: CrafterSpatialMap,
    vf: VisualField,
    player_pos: tuple[int, int],
) -> None:
    """Write viewport detections into spatial_map with confidence."""
    px, py = int(player_pos[0]), int(player_pos[1])
    center_row, center_col = 3, 4  # 7×9 viewport

    spatial_map.update((px, py), vf.near_concept, vf.near_similarity)

    for cid, conf, gy, gx in vf.detections:
        wx = px + (gx - center_col)
        wy = py + (gy - (center_row - 1))
        spatial_map.update((wx, wy), cid, conf)
