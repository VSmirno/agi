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
    RESIDUAL_BODY_ORDER,
    ConceptStore,
    _apply_player_move,
    _expand_to_primitive as expand_to_primitive,
    primitive_to_action_idx,
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

    # 6. Stage 80 (Bug 3 fix): proactive gather plans.
    #
    # The original Stage 77a generator only emits plans as REMEDIES for
    # failures (body vars that will deplete or entities that will damage
    # vital vars). Wood / stone / coal / iron are not body vars and
    # never appear as failures, so the agent never plans to gather them
    # unless a tool-requiring need bubbles up via plan_toward_rule.
    # Combined with the score_trajectory bias toward "high min_body",
    # this produced agents that spent ~70% of their time sleeping (it
    # locally maximises min_body) and never gathered ANYTHING.
    #
    # Fix: for every action-triggered rule whose effect is `inventory:
    # { X: +N }` (i.e. it produces an item), emit a candidate plan toward
    # that rule. This adds wood/stone/coal/iron gathering and table /
    # tool crafting plans to the candidate set on every step. Whether
    # they're chosen depends on score_trajectory (the lex tuple now
    # prefers any-gain over no-gain among alive plans).
    for concept in store.concepts.values():
        for link in concept.causal_links:
            if link.kind != "action_triggered":
                continue
            if id(link) in seen_rules:
                continue
            if not link.effect:
                continue
            # Must produce at least one positive inventory delta —
            # filters out combat rules (scene_remove only) and any
            # rules that don't add to inventory.
            if not any(d > 0 for d in link.effect.inventory_delta.values()):
                continue
            seen_rules.add(id(link))
            plan_steps = store.plan_toward_rule(link, state.inventory)
            if plan_steps:
                candidates.append(Plan(steps=plan_steps, origin="gather"))

    return candidates


# ---------------------------------------------------------------------------
# Trajectory scoring — lexicographic tuple (alive, min_body, ticks, final)
# ---------------------------------------------------------------------------


def score_trajectory(
    traj: Trajectory,
    tracker: HomeostaticTracker,
) -> tuple:
    """Return a lexicographic sort key — higher tuple = better plan.

    Stage 80 Bug 3 fix: the alive tuple now includes `has_inv_gain` as
    a top-priority component (just below `alive`). The original
    formulation `(1, min_body, n_ticks, final_body)` produced
    sleep-dominance: sleep plans always slightly raise min_body via
    energy clamping + stateful health regen, so they outscored every
    alternative — including gathering plans that ACTUALLY make
    progress. The Stage 80 diagnostic showed 69.5% of agent actions
    were `sleep` and 0/5 episodes gathered any wood/stone/coal/iron.

    The new tuple shape:

    ALIVE: (1, has_gain, min_body, n_ticks, final_body)
      - has_gain = 1 if the rollout produced ANY positive inventory_delta
        event (wood +1, stone_item +1, etc), else 0
      - Among alive plans, gathering plans beat non-gathering plans
      - Within "any-gain" or "no-gain" subsets, min_body is the next
        priority (safer wins)
      - n_ticks is the next tiebreaker (longer rollout = more room)
      - final_body is the last tiebreaker

    DEAD: (0, n_ticks, min_body, final_body)
      - Unchanged. Once you're dead, gathering doesn't matter.

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
        # Stage 80: has_gain dominates min_body. Any rollout that
        # produces an inventory positive (do tree, do stone with
        # pickaxe, place table, make pickaxe) outranks any rollout
        # that doesn't (sleep, navigate-aimlessly).
        has_gain = 0
        for ev in traj.events:
            if ev.kind == "inv_gain" and ev.amount > 0:
                has_gain = 1
                break
        return (1, has_gain, min_body, n_ticks, final_body)
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
    residual_predictor: Any = None,
    residual_optimizer: Any = None,
    residual_train: bool = False,
    surprise_accumulator: Any = None,
    rule_nursery: Any = None,
    nursery_tick_every: int = 1,
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
        residual_predictor: Stage 78c — optional ResidualBodyPredictor; when
            provided it is passed to simulate_forward for MPC inference and
            can also be trained online via SGD when residual_train=True.
        residual_optimizer: torch.optim.Optimizer for the residual, required
            when residual_train=True.
        residual_train: if True, after each env.step the residual is
            trained with a single SGD step on the observed body gap.
        surprise_accumulator: Stage 79 — optional SurpriseAccumulator. When
            provided, after each env.step the (predicted_delta, actual_delta)
            pair is fed to the accumulator for runtime rule induction.
            Predicted delta is computed via the same 1-tick rules-only
            replay (with planned_step propagation) as Stage 78c training.
        rule_nursery: Stage 79 — optional RuleNursery. When provided,
            `nursery.tick(accumulator, store, current_tick=step)` is called
            every `nursery_tick_every` env steps. Promoted rules go into
            `store.learned_rules` and become available to subsequent
            `simulate_forward` calls in the same episode (and beyond).
        nursery_tick_every: how often to run the nursery emit/verify cycle.
            Default 1 = every env step. Increase to amortise cost in long
            episodes if profiling shows it.

    Returns:
        Dict with episode metrics: length, final_inv, cause_of_death,
        action_counts, action_entropy, plus residual_loss_* (if training)
        and nursery_stats / accumulator_stats (if Stage 79 enabled).
    """
    if perceive_fn is None:
        perceive_fn = perceive_tile_field
    if residual_train and (residual_predictor is None or residual_optimizer is None):
        raise ValueError(
            "residual_train=True requires both residual_predictor and residual_optimizer"
        )
    if rule_nursery is not None and surprise_accumulator is None:
        raise ValueError(
            "rule_nursery requires surprise_accumulator (the nursery reads from the accumulator)"
        )

    # Lazy import to avoid making mpc_agent depend on snks.learning
    # at module load time when the nursery isn't being used.
    if surprise_accumulator is not None:
        from snks.learning.surprise_accumulator import (
            BODY_ORDER as _NURSERY_BODY_ORDER,
            ContextKey as _NurseryContextKey,
        )

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
    # Stage 78c — residual SGD loss history (populated per training step).
    residual_losses: list[float] = []

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
        # Stage 78c: residual is held constant across rollouts for one env
        # step (same perception snapshot). All candidates see the same
        # corrected dynamics, so the scoring stays consistent.
        visible_concepts = vf.visible_concepts()
        scored: list[tuple[tuple, Plan, Trajectory]] = []
        for plan in candidates:
            traj = store.simulate_forward(
                plan,
                state,
                tracker,
                horizon=horizon,
                residual_predictor=residual_predictor,
                visible_concepts=visible_concepts,
            )
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

        # --- Stage 80 Bug 6 fix: clear chopped tile from spatial_map ---
        # When env.step("do") successfully gathered a resource, the
        # facing tile in env was emptied (tree chopped, stone broken,
        # cow eaten, water drained). Stage 75 segmenter sometimes
        # mis-classifies the now-empty tile as still being the
        # resource (player sprite confusion / latent pixels), so the
        # spatial_map keeps reporting the resource at the old position.
        # find_nearest then loops the planner. Patch: explicitly mark
        # the facing tile as 'empty' after a successful do interaction.
        if primitive == "do" and outcome and any(
            inv_after.get(k, 0) > inv_before_action.get(k, 0)
            for k in ("wood", "stone", "coal", "iron", "diamond", "sapling")
        ):
            facing_dx, facing_dy = 0, 1
            if prev_action == "move_left":
                facing_dx, facing_dy = -1, 0
            elif prev_action == "move_right":
                facing_dx, facing_dy = 1, 0
            elif prev_action == "move_up":
                facing_dx, facing_dy = 0, -1
            elif prev_action == "move_down":
                facing_dx, facing_dy = 0, 1
            facing_tile = (
                int(player_pos[0]) + facing_dx,
                int(player_pos[1]) + facing_dy,
            )
            spatial_map.update(facing_tile, "empty")

        # --- Stage 78c: online residual SGD step -----------------------------
        # After env.step we know the actual body at t+1. Train residual on
        # (prev_state_t, prev_primitive_t) → (actual_delta_t - rules_delta_t).
        # rules_delta is computed by a 1-tick rules-only replay on a copy of
        # the prev sim-state (residual is NOT passed), isolating the symbolic
        # prediction from the residual-corrected one used for planning.
        #
        # IMPORTANT (Bug 1 fix): the rules-only replay must propagate the
        # SAME planned_step that the planner's simulate_forward used for the
        # chosen plan's first tick. Without it, Phase 6 'do' falls into the
        # facing-based fallback (`_nearest_concept`) which the Stage 77a
        # comment in concept_store.py:646 explicitly notes as broken in sim
        # contexts ("navigation walks through target tiles"). The result was
        # systematically zero rules_delta for do-water and (intermittently)
        # do-cow, training the residual to over-predict the body gap and
        # mislead future planning iterations.
        if (
            residual_train
            and residual_predictor is not None
            and residual_optimizer is not None
        ):
            import torch

            # Use the CURRENT step as the training example: we planned from
            # `state` at step t with `primitive`, then observed inv_after.
            rules_sim = state.copy()
            rules_traj = Trajectory(
                plan=Plan(steps=[], origin="train_probe"),
                body_series={var: [] for var in rules_sim.body},
                events=[],
                final_state=rules_sim,
                terminated=False,
                terminated_reason="horizon",
                plan_progress=0,
            )
            chosen_planned_step = best_plan.steps[0] if best_plan.steps else None
            store._apply_tick(
                rules_sim,
                primitive,
                tracker,
                rules_traj,
                tick=0,
                planned_step=chosen_planned_step,
            )
            prev_body = state.body
            actual_delta = [
                float(inv_after.get(var, 0)) - float(prev_body.get(var, 0.0))
                for var in RESIDUAL_BODY_ORDER
            ]
            rules_delta = [
                float(rules_sim.body.get(var, 0.0)) - float(prev_body.get(var, 0.0))
                for var in RESIDUAL_BODY_ORDER
            ]
            device = next(residual_predictor.parameters()).device
            fp = residual_predictor.encode(
                visible=set(visible_concepts),
                body=dict(prev_body),
                action_idx=primitive_to_action_idx(primitive),
                device=device,
            )
            rules_tensor = torch.tensor(rules_delta, dtype=torch.float32, device=device)
            target_tensor = torch.tensor(actual_delta, dtype=torch.float32, device=device)
            loss = residual_predictor.residual_loss(fp, rules_tensor, target_tensor)
            residual_optimizer.zero_grad()
            loss.backward()
            residual_optimizer.step()
            residual_losses.append(float(loss.item()))

        # --- Stage 79: surprise accumulator + rule nursery -------------------
        # Reuses the same 1-tick rules-only replay (with planned_step
        # propagation, the Bug 1 fix from Stage 78c) to compute the
        # planner's actual prediction. Surprise is fed to the accumulator;
        # the nursery emits/verifies/promotes candidate rules from
        # saturated buckets. Promoted rules go into store.learned_rules
        # and become available to subsequent simulate_forward calls.
        if surprise_accumulator is not None:
            # 1-tick rules-only replay (a SECOND copy if we already did
            # one for residual training; the cost is microseconds and
            # keeps the two consumers independent so future stages can
            # disable one without breaking the other).
            nursery_rules_sim = state.copy()
            nursery_rules_traj = Trajectory(
                plan=Plan(steps=[], origin="nursery_probe"),
                body_series={var: [] for var in nursery_rules_sim.body},
                events=[],
                final_state=nursery_rules_sim,
                terminated=False,
                terminated_reason="horizon",
                plan_progress=0,
            )
            nursery_chosen_step = best_plan.steps[0] if best_plan.steps else None
            store._apply_tick(
                nursery_rules_sim,
                primitive,
                tracker,
                nursery_rules_traj,
                tick=0,
                planned_step=nursery_chosen_step,
                visible_concepts=visible_concepts,
            )

            nursery_predicted_delta = {
                var: float(nursery_rules_sim.body.get(var, 0.0))
                     - float(state.body.get(var, 0.0))
                for var in _NURSERY_BODY_ORDER
            }
            nursery_actual_delta = {
                var: float(inv_after.get(var, 0))
                     - float(state.body.get(var, 0.0))
                for var in _NURSERY_BODY_ORDER
            }
            nursery_context = _NurseryContextKey.from_state(
                visible=set(visible_concepts),
                body=dict(state.body),
                action=primitive,
            )
            surprise_accumulator.observe(
                context=nursery_context,
                predicted=nursery_predicted_delta,
                actual=nursery_actual_delta,
                tick_id=step,
            )

            if rule_nursery is not None and (step % nursery_tick_every == 0):
                rule_nursery.tick(
                    surprise_accumulator,
                    store,
                    current_tick=step,
                )

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
                "next_step_blocked": (
                    primitive.startswith("move_")
                    and spatial_map.is_blocked(_apply_player_move(player_pos, primitive))
                ),
                "n_blocked_known": len(spatial_map._blocked),
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

    result: dict[str, Any] = {
        "length": steps_taken,
        "final_inv": inv_after,
        "action_counts": dict(action_counts),
        "action_entropy": action_entropy,
        "cause_of_death": cause_of_death,
    }
    if residual_losses:
        result["residual_loss_mean"] = float(np.mean(residual_losses))
        result["residual_loss_final"] = float(residual_losses[-1])
        result["residual_loss_early"] = float(np.mean(residual_losses[: max(1, len(residual_losses) // 5)]))
        result["residual_steps"] = len(residual_losses)
    if surprise_accumulator is not None:
        result["accumulator_stats"] = surprise_accumulator.stats()
    if rule_nursery is not None:
        result["nursery_stats"] = rule_nursery.stats()
        result["learned_rule_count"] = len(store.learned_rules)
    return result
