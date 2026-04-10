"""Stage 77a: MPC agent loop and supporting functions.

This module assembles the Stage 77a forward-simulation architecture into
a working episode loop. Each step:

  1. Perceive current frame → VisualField + SimState snapshot
  2. Generate candidate plans via baseline rollout + find_remedies
  3. Simulate each candidate with ConceptStore.simulate_forward
  4. Score trajectories lexicographically: (alive, min_body, ticks, final)
  5. Execute the first primitive action of the winning plan
  6. Re-plan from scratch on the next step (pure MPC — no commitment)

Design: docs/superpowers/specs/2026-04-10-stage77a-conceptstore-forward-sim-design.md
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    """JSON encoder hook for numpy scalars/arrays that sneak into trace dicts."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")

from snks.agent.concept_store import (
    ConceptStore,
    _apply_player_move,
    _expand_to_primitive as expand_to_primitive,
)
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.forward_sim_types import (
    DynamicEntity,
    Failure,
    Plan,
    PlannedStep,
    SimState,
    Trajectory,
)
from snks.agent.perception import (
    HomeostaticTracker,
    VisualField,
    perceive_tile_field,
    verify_outcome,
)


# ---------------------------------------------------------------------------
# DynamicEntityTracker — track moving entities across episode steps
# ---------------------------------------------------------------------------


@dataclass
class DynamicEntityTracker:
    """Track positions of dynamic entities (enemies, mobs) across steps.

    Distinct from CrafterSpatialMap which is for static objects. Entities
    are added when first seen, updated when re-seen at a different position,
    and removed when no longer visible. This is a lightweight simplification
    — a real tracker would use association heuristics across frames.
    """

    # Known entities by a synthetic id: (concept_id, index).
    entities: list[DynamicEntity] = field(default_factory=list)

    # Concepts we consider "dynamic" — populated from textbook passive_movement rules.
    # If empty, any non-empty detection in the viewport can be a dynamic entity.
    dynamic_concepts: set[str] = field(default_factory=set)

    def register_dynamic_concept(self, concept_id: str) -> None:
        """Mark a concept as dynamic — any detection of it becomes an entity."""
        self.dynamic_concepts.add(concept_id)

    def update(
        self,
        visual_field: VisualField,
        player_pos: tuple[int, int],
        viewport_center: tuple[int, int] = (3, 4),  # 7x9 viewport center (row, col)
    ) -> None:
        """Update entity positions from the current visual field.

        Converts viewport coordinates to world coordinates using player_pos
        as the anchor. Re-initializes entities list each update — entities
        not visible this frame are dropped.
        """
        new_entities: list[DynamicEntity] = []
        center_row, center_col = viewport_center
        px, py = int(player_pos[0]), int(player_pos[1])

        for cid, _conf, gy, gx in visual_field.detections:
            if self.dynamic_concepts and cid not in self.dynamic_concepts:
                continue
            # Convert viewport (gy, gx) → world (wx, wy)
            # Follows Stage 75 coordinate mapping: sprite offset +1 row
            wx = px + (gx - center_col)
            wy = py + (gy - (center_row - 1))
            new_entities.append(DynamicEntity(concept_id=cid, pos=(wx, wy)))

        self.entities = new_entities

    def current(self) -> list[DynamicEntity]:
        """Snapshot of currently tracked entities (returned as a copy)."""
        return [
            DynamicEntity(concept_id=e.concept_id, pos=tuple(e.pos))
            for e in self.entities
        ]


# ---------------------------------------------------------------------------
# Candidate plan generation — baseline → failures → remedies
# ---------------------------------------------------------------------------


def extract_failures(traj: Trajectory) -> list[Failure]:
    """Scan a trajectory for catastrophic events, attributed to sources.

    Two failure kinds:
      - var_depleted: a body var hit zero (or its reference_min)
      - attributed_to: a non-background source caused negative body_delta

    Returns failures sorted by step ascending (earliest = highest priority).
    """
    failures: list[Failure] = []

    # Type 1: body variable depletion
    for var, series in traj.body_series.items():
        for i, value in enumerate(series):
            if value <= 0:
                failures.append(Failure(
                    kind="var_depleted",
                    var=var,
                    cause=None,
                    step=i,
                    severity=1.0,
                ))
                break  # first depletion only

    # Type 2: attributed damage
    damage_sources: dict[str, int] = {}
    for event in traj.events:
        if (
            event.kind == "body_delta"
            and event.amount < 0
            and event.source not in ("_background", None)
        ):
            # Filter out stateful:* sources (those aren't external attribution)
            if event.source.startswith("stateful:"):
                continue
            if event.source not in damage_sources:
                damage_sources[event.source] = event.step

    for source, first_step in damage_sources.items():
        failures.append(Failure(
            kind="attributed_to",
            var=None,
            cause=source,
            step=first_step,
            severity=1.0,
        ))

    failures.sort(key=lambda f: f.step)
    return failures


def generate_candidate_plans(
    state: SimState,
    store: ConceptStore,
    tracker: HomeostaticTracker,
    horizon: int = 20,
) -> list[Plan]:
    """Produce candidate plans via baseline rollout + remedy search.

    1. Run an inertia baseline rollout for `horizon` ticks.
    2. Extract failures (what went wrong, when) from baseline.
    3. Inject PROACTIVE failures for things the world model knows are
       dangerous even if baseline doesn't currently predict them:
       - body vars with negative innate rate (will eventually deplete)
       - entity concepts with known spatial damage on vital vars
    4. For each failure, query the world model (find_remedies) for rules
       that counteract it.
    5. For each remedy rule, backward-chain a plan (plan_toward_rule).
    6. Return baseline + all generated plans as candidates.

    Drives emerge from predicted failures + proactive world-model threats —
    no hardcoded drive categories, no Strategy 1/2/Preparation branches.
    The "proactive" injection isn't a strategy, it's an observation: the
    world model says "this concept damages vital vars" and the agent plans
    accordingly, regardless of whether the threat is currently visible.
    """
    # 1. Baseline rollout
    baseline = Plan(
        steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)],
        origin="baseline",
    )
    baseline_traj = store.simulate_forward(baseline, state, tracker, horizon=horizon)

    candidates: list[Plan] = [baseline]

    # 2. Extract failures from baseline trajectory
    failures = extract_failures(baseline_traj)

    # 3. Proactive failure injection: things the world model knows are
    # dangerous even if baseline doesn't see them within `horizon` ticks.
    seen_causes = {f.cause for f in failures if f.cause}
    seen_var_depleted = {f.var for f in failures if f.kind == "var_depleted"}

    # 3a. Body vars with negative innate rate → future depletion
    for var, rate in tracker.innate_rates.items():
        if rate < 0 and var not in seen_var_depleted:
            failures.append(Failure(
                kind="var_depleted",
                var=var,
                cause=None,
                step=horizon * 10,  # far future, lower priority than baseline-observed
                severity=0.5,
            ))

    # 3b. Entity concepts with known spatial damage on vital body vars
    for rule in store.passive_rules:
        if rule.kind != "passive_spatial" or not rule.concept or not rule.effect:
            continue
        # Does this rule damage any vital var?
        for var, delta in rule.effect.body_delta.items():
            if delta < 0 and var in tracker.vital_mins:
                if rule.concept not in seen_causes:
                    failures.append(Failure(
                        kind="attributed_to",
                        var=None,
                        cause=rule.concept,
                        step=horizon * 10,  # proactive, not imminent
                        severity=0.5,
                    ))
                    seen_causes.add(rule.concept)
                break

    # 4-5. For each failure, find remedies and plan toward each
    seen_rules: set[int] = set()
    for failure in failures:
        remedies = store.find_remedies(failure)
        for rule in remedies:
            if id(rule) in seen_rules:
                continue
            seen_rules.add(id(rule))
            plan_steps = store.plan_toward_rule(rule, state.inventory)
            if plan_steps:
                candidates.append(Plan(steps=plan_steps, origin="remedy"))

    return candidates


# ---------------------------------------------------------------------------
# Trajectory scoring — lexicographic tuple (alive, min_body, ticks, final)
# ---------------------------------------------------------------------------


def score_trajectory(
    traj: Trajectory,
    tracker: HomeostaticTracker,
) -> tuple:
    """Return a lexicographic sort key — higher tuple = better plan.

    The tuple shape differs between alive and dead trajectories because the
    dominant concern is different:

    ALIVE bucket — primary concern is safety:
        (1, min_body, n_ticks, final_body)
      - All alive trajectories run to horizon, so n_ticks is typically equal.
      - min_body (worst moment) rewards safer plans.
      - final_body is a tiebreaker.

    DEAD bucket — primary concern is how close we came to surviving:
        (0, n_ticks, min_body, final_body)
      - Longer survival means the plan held on more, gives MPC more room
        for the next-step re-plan to find a better option.
      - min_body is a tiebreaker (among equally-long dead, safer is better).

    Python compares tuples lexicographically; the leading 1/0 puts alive
    above any dead regardless of the remaining elements. Within a bucket,
    the order is consistent and meaningful.

    Normalization: each var divided by its reference_max (from textbook).
    If reference_max unknown, uses observed_max as fallback, else 1.
    """
    n_ticks = traj.tick_count()
    catastrophic = traj.terminated and traj.terminated_reason == "body_dead"
    alive = not catastrophic

    def normalized_body_sum(step_idx: int) -> float:
        total = 0.0
        for var, series in traj.body_series.items():
            ref_max = tracker.reference_max.get(var, 0.0)
            if ref_max <= 0:
                ref_max = float(tracker.observed_max.get(var, 1))
            ref_max = max(ref_max, 1.0)
            total += series[step_idx] / ref_max
        return total

    if n_ticks > 0:
        min_body = min(normalized_body_sum(i) for i in range(n_ticks))
        final_body = normalized_body_sum(n_ticks - 1)
    else:
        min_body = 0.0
        final_body = 0.0

    if alive:
        return (1, min_body, n_ticks, final_body)
    return (0, n_ticks, min_body, final_body)


# ---------------------------------------------------------------------------
# Helpers: spatial map update, SimState construction, outcome verify
# ---------------------------------------------------------------------------


def update_spatial_map_from_viewport(
    spatial_map: CrafterSpatialMap,
    visual_field: VisualField,
    player_pos: tuple[int, int],
    viewport_rows: int = 7,
    viewport_cols: int = 9,
) -> None:
    """Write all viewport detections into the spatial_map at world coordinates.

    Follows the Stage 75 coordinate mapping: sprite offset +1 row.
    """
    center_row = viewport_rows // 2
    center_col = viewport_cols // 2
    px, py = int(player_pos[0]), int(player_pos[1])

    # Player's own tile — use near_concept
    spatial_map.update((px, py), visual_field.near_concept)

    for cid, _conf, gy, gx in visual_field.detections:
        wx = px + (gx - center_col)
        wy = py + (gy - (center_row - 1))
        spatial_map.update((wx, wy), cid)


def build_sim_state(
    inventory: dict[str, int],
    player_pos: tuple[int, int],
    spatial_map: CrafterSpatialMap,
    entity_tracker: DynamicEntityTracker,
    tracker: HomeostaticTracker,
    last_action: str | None,
    step: int,
) -> SimState:
    """Construct a SimState snapshot from the current real-world state.

    Body vars come from inventory (Crafter stores health/food/drink/energy
    inside the inventory dict alongside tool counts). The set of body vars
    is derived from tracker (innate_rates + vital_mins + observed_max keys).
    """
    body_var_names = (
        set(tracker.innate_rates.keys())
        | set(tracker.vital_mins.keys())
        | {k for k in tracker.observed_max.keys() if tracker.reference_max.get(k, 0) > 0}
    )
    body = {var: float(inventory.get(var, 0)) for var in body_var_names}

    return SimState(
        inventory=dict(inventory),
        body=body,
        player_pos=tuple(player_pos),
        dynamic_entities=entity_tracker.current(),
        spatial_map=spatial_map,
        last_action=last_action,
        step=step,
    )


def outcome_to_verify(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
) -> str | None:
    """Compute a verify label for the outcome of an action, for confidence update.

    Mirrors the legacy perception.outcome_to_verify but inlined here so
    mpc_agent has no dependency on perception's dead functions.
    """
    gains, losses = {}, {}
    for k in set(inv_before) | set(inv_after):
        d = inv_after.get(k, 0) - inv_before.get(k, 0)
        if d > 0:
            gains[k] = d
        elif d < 0:
            losses[k] = -d

    body_vars = {"health", "food", "drink", "energy"}

    if action == "do":
        for k in gains:
            if k not in body_vars:
                return k
        for stat in ("food", "drink"):
            if gains.get(stat, 0) > 0:
                return f"restore_{stat}"
        return None
    if action.startswith("place_"):
        return action.replace("place_", "") if losses else None
    if action.startswith("make_"):
        crafted = action.replace("make_", "")
        return crafted if gains.get(crafted, 0) > 0 else None
    return None


# ---------------------------------------------------------------------------
# Main MPC episode loop
# ---------------------------------------------------------------------------


def run_mpc_episode(
    env: Any,
    segmenter: Any,
    store: ConceptStore,
    tracker: HomeostaticTracker,
    rng: np.random.RandomState,
    max_steps: int = 500,
    horizon: int = 20,
    perceive_fn: Any = None,
    verbose: bool = False,
    trace_path: str | Path | None = None,
) -> dict:
    """Run one episode with MPC + forward-sim planning.

    Each step:
      1. Perceive → update spatial_map, entity_tracker, homeostatic tracker
      2. Build SimState snapshot
      3. generate_candidate_plans (baseline → failures → remedies)
      4. simulate_forward each candidate, score with score_trajectory
      5. Execute the first primitive action of the winning plan
      6. verify_outcome → confidence update for rule that fired

    Args:
        env: CrafterPixelEnv-compatible environment.
        segmenter: tile segmenter (Stage 75 checkpoint).
        store: ConceptStore with textbook rules loaded.
        tracker: HomeostaticTracker initialized from textbook.
        rng: random state (used only for fallback random choices).
        max_steps: episode length cap.
        horizon: rollout horizon for each candidate plan.
        perceive_fn: override perception function for testing (default: imports
            from snks.agent.continuous_agent).
        verbose: print per-step diagnostics.

    Returns:
        Dict with episode metrics: length, final_inv, cause_of_death,
        action_counts, bootstrap_ratio (always 1.0 — no bootstrap in Stage 77a),
        action_entropy.
    """
    if perceive_fn is None:
        perceive_fn = perceive_tile_field

    trace_fh = None
    if trace_path is not None:
        trace_fh = open(str(trace_path), "a")

    # Pre-populate entity_tracker with known dynamic concepts from store
    entity_tracker = DynamicEntityTracker()
    for rule in store.passive_rules:
        if rule.kind == "passive_movement" and rule.concept:
            entity_tracker.register_dynamic_concept(rule.concept)

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    prev_inv: dict[str, int] | None = None
    prev_action: str | None = None
    prev_player_pos: tuple[int, int] | None = None
    action_counts: Counter = Counter()
    steps_taken = 0
    cause_of_death = "alive"
    inv_after: dict[str, int] = dict(info.get("inventory", {}))
    # Trace state — predicted body_series for tick 0 of the best plan chosen
    # on the previous step, so we can measure actual-vs-predicted surprise.
    prev_predicted_next_body: dict[str, float] | None = None
    prev_chosen_origin: str | None = None
    prev_primitive: str | None = None

    for step in range(max_steps):
        steps_taken = step + 1
        inv = dict(info.get("inventory", {}))
        player_pos = tuple(info.get("player_pos", (32, 32)))

        # --- Observation: did the last move action succeed? ---
        # If the previous action was a move_* but player_pos didn't change,
        # the target tile is impassable. Record this in spatial_map as a
        # blocked tile so exploration stops trying it. This is observation-
        # based world modeling (factual update), not a hardcoded
        # stuck-avoidance rule in policy code.
        if (
            prev_action
            and prev_action.startswith("move_")
            and prev_player_pos is not None
            and prev_player_pos == player_pos
        ):
            blocked_tile = _apply_player_move(prev_player_pos, prev_action)
            spatial_map.mark_blocked(blocked_tile)

        # --- Perception ---
        vf = perceive_fn(pixels, segmenter)
        update_spatial_map_from_viewport(spatial_map, vf, player_pos)
        entity_tracker.update(vf, player_pos)

        # --- Update homeostatic tracker ---
        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # --- Build SimState snapshot ---
        state = build_sim_state(
            inventory=inv,
            player_pos=player_pos,
            spatial_map=spatial_map,
            entity_tracker=entity_tracker,
            tracker=tracker,
            last_action=prev_action,
            step=step,
        )

        # --- Generate candidate plans via baseline rollout ---
        candidates = generate_candidate_plans(state, store, tracker, horizon=horizon)

        # --- Simulate and score each candidate ---
        scored: list[tuple[tuple, Plan, Trajectory]] = []
        for plan in candidates:
            traj = store.simulate_forward(plan, state, tracker, horizon=horizon)
            score = score_trajectory(traj, tracker)
            scored.append((score, plan, traj))

        # --- Pick best, execute first primitive ---
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_plan, best_traj = scored[0]

        if best_plan.steps:
            primitive = expand_to_primitive(best_plan.steps[0], state, store)
        else:
            primitive = str(rng.choice(["move_left", "move_right", "move_up", "move_down"]))

        action_counts[primitive] += 1

        if verbose:
            print(
                f"s{step:3d} H{inv.get('health', 0)} F{inv.get('food', 0)} "
                f"D{inv.get('drink', 0)} E{inv.get('energy', 0)} "
                f"near={vf.near_concept:9s} → {primitive} "
                f"(plan {best_plan.origin}, {len(best_plan.steps)} steps)"
            )

        # --- Trace: measure surprise from PREVIOUS tick's prediction ---
        surprise: dict[str, float] = {}
        if trace_fh is not None and prev_predicted_next_body is not None:
            for var, predicted in prev_predicted_next_body.items():
                actual = float(inv.get(var, 0))
                surprise[var] = actual - predicted

        # --- Execute in real env ---
        inv_before_action = inv
        pixels, _reward, done, info = env.step(primitive)
        inv_after = dict(info.get("inventory", {}))

        # --- Verify outcome (updates confidence) ---
        outcome = outcome_to_verify(primitive, inv_before_action, inv_after)
        verified = False
        if outcome:
            verify_outcome(vf.near_concept, primitive, outcome, store)
            verified = True

        # --- Trace: write one JSONL line per tick ---
        if trace_fh is not None:
            predicted_next_body: dict[str, float] = {}
            for var, series in best_traj.body_series.items():
                if series:
                    predicted_next_body[var] = float(series[0])
            origins = Counter(p.origin for _, p, _ in scored)
            # Damage sources in the chosen trajectory (first 5 body_delta events
            # with a real source — filter _background and stateful:*).
            damage_preview = []
            for ev in best_traj.events:
                if ev.kind != "body_delta" or ev.amount >= 0:
                    continue
                if ev.source in (None, "_background") or ev.source.startswith("stateful:"):
                    continue
                damage_preview.append({
                    "step": ev.step, "var": ev.var, "amount": ev.amount,
                    "source": ev.source,
                })
                if len(damage_preview) >= 5:
                    break
            trace_entry = {
                "step": step,
                "pos": list(player_pos),
                "body": {k: float(inv.get(k, 0)) for k in ("health", "food", "drink", "energy")},
                "near": vf.near_concept,
                "visible": sorted(vf.visible_concepts()),
                "n_entities": len(entity_tracker.current()),
                "n_candidates": len(scored),
                "plan_origins": dict(origins),
                "chosen": {
                    "origin": best_plan.origin,
                    "n_steps": len(best_plan.steps),
                    "first_action": best_plan.steps[0].action if best_plan.steps else None,
                    "first_target": best_plan.steps[0].target if best_plan.steps else None,
                    "first_near": best_plan.steps[0].near if best_plan.steps else None,
                    "score": [float(x) for x in best_score],
                    "traj_terminated": best_traj.terminated,
                    "traj_reason": best_traj.terminated_reason,
                    "traj_ticks": best_traj.tick_count(),
                    "predicted_final_body": {
                        k: float(v[-1]) if v else None
                        for k, v in best_traj.body_series.items()
                    },
                    "damage_preview": damage_preview,
                },
                "primitive": primitive,
                "surprise": surprise,
                "verified": verified,
                "outcome": outcome,
                "prev_primitive_landed": (
                    prev_action is not None
                    and prev_action.startswith("move_")
                    and prev_player_pos is not None
                    and prev_player_pos != player_pos
                ),
            }
            trace_fh.write(json.dumps(trace_entry, default=_json_default) + "\n")
            trace_fh.flush()
            prev_predicted_next_body = predicted_next_body
            prev_chosen_origin = best_plan.origin
            prev_primitive = primitive

        prev_inv = inv
        prev_action = primitive
        prev_player_pos = player_pos

        if done:
            # Diagnose cause of death — which vital var hit min?
            zeroed = [
                var for var, min_val in tracker.vital_mins.items()
                if inv_after.get(var, 1) <= min_val
            ]
            cause_of_death = ",".join(zeroed) if zeroed else "other"
            break

    total = sum(action_counts.values())
    if total > 0:
        probs = np.array([c / total for c in action_counts.values()])
        action_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        action_entropy = 0.0

    if trace_fh is not None:
        # End-of-episode marker line — easier to aggregate across episodes.
        trace_fh.write(json.dumps({
            "episode_end": True,
            "length": steps_taken,
            "cause_of_death": cause_of_death,
            "final_inv": {str(k): int(v) for k, v in inv_after.items()},
        }, default=_json_default) + "\n")
        trace_fh.close()

    return {
        "length": steps_taken,
        "final_inv": inv_after,
        "action_counts": dict(action_counts),
        "action_entropy": action_entropy,
        "cause_of_death": cause_of_death,
    }
