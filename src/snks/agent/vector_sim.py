"""Stage 83: VectorState + simulate_forward — simulation in vector space.

VectorState stores structured data (inventory, body) for exact accounting.
simulate_forward uses VectorWorldModel.predict() for predictions, then
decodes effects to apply on the structured state.

Vectors are the knowledge layer; dicts are the accounting layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from snks.agent.crafter_spatial_map import CrafterSpatialMap
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.agent.stimuli import StimuliLayer
    from snks.agent.goal_selector import Goal


# ---------------------------------------------------------------------------
# VectorState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicEntityState:
    """Tracked dynamic entity in world coordinates for short-horizon sim."""

    concept_id: str
    position: tuple[int, int]
    velocity: tuple[int, int] | None = None
    age: int = 0
    last_seen_step: int = 0

@dataclass
class VectorState:
    """World snapshot for vector-based simulation.

    Structured dict (inventory + body) is the source of truth for
    simulation. to_vector() encodes for SDM addressing when needed.
    """

    inventory: dict[str, int] = field(default_factory=dict)
    body: dict[str, float] = field(default_factory=dict)
    player_pos: tuple[int, int] = (0, 0)
    step: int = 0
    last_action: str | None = None
    spatial_map: "CrafterSpatialMap | None" = None
    dynamic_entities: list[DynamicEntityState] = field(default_factory=list)

    def apply_effect(self, decoded_effect: dict[str, int]) -> "VectorState":
        """Apply decoded effect dict. Returns new VectorState (immutable)."""
        new_inv = dict(self.inventory)
        new_body = dict(self.body)

        for var, delta in decoded_effect.items():
            if var in new_body:
                new_body[var] = max(0.0, min(9.0, new_body[var] + delta))
            else:
                new_inv[var] = max(0, new_inv.get(var, 0) + delta)

        return VectorState(
            inventory=new_inv,
            body=new_body,
            player_pos=self.player_pos,
            step=self.step + 1,
            last_action=self.last_action,
            spatial_map=self.spatial_map,
            dynamic_entities=list(self.dynamic_entities),
        )

    def move_player(self, action: str) -> "VectorState":
        """Apply a move primitive to player position only, without ticking time."""
        px, py = self.player_pos
        dx, dy = 0, 0
        if action == "move_right":
            dx = 1
        elif action == "move_left":
            dx = -1
        elif action == "move_down":
            dy = 1
        elif action == "move_up":
            dy = -1
        target = (px + dx, py + dy)
        if self._move_target_blocked(target):
            target = (px, py)

        return VectorState(
            inventory=dict(self.inventory),
            body=dict(self.body),
            player_pos=target,
            step=self.step,
            last_action=action,
            spatial_map=self.spatial_map,
            dynamic_entities=list(self.dynamic_entities),
        )

    def _move_target_blocked(self, target: tuple[int, int]) -> bool:
        spatial_map = self.spatial_map
        if spatial_map is not None and hasattr(spatial_map, "is_blocked"):
            if bool(spatial_map.is_blocked(target)):
                return True
        return any(tuple(entity.position) == tuple(target) for entity in self.dynamic_entities)

    def is_dead(self, vital_vars: list[str] | None = None) -> bool:
        """Check if any vital body variable <= 0."""
        vitals = vital_vars or ["health"]
        for v in vitals:
            if self.body.get(v, 1.0) <= 0:
                return True
        return False

    def to_vector(self, model: "VectorWorldModel") -> torch.Tensor:
        """Encode state as binary vector for similarity / SDM addressing."""
        from snks.agent.vector_world_model import bind, bundle, encode_scalar

        parts = []
        for var, val in self.body.items():
            role = model._ensure_role(var)
            scalar = encode_scalar(int(val), model.dim, model.max_scalar).to(model.device)
            parts.append(bind(role, scalar))
        for item, count in self.inventory.items():
            role = model._ensure_role(item)
            scalar = encode_scalar(int(count), model.dim, model.max_scalar).to(model.device)
            parts.append(bind(role, scalar))

        if not parts:
            from snks.agent.vector_world_model import random_bitvector
            return random_bitvector(model.dim, model.device)

        return bundle(parts)

    def copy(self) -> "VectorState":
        return VectorState(
            inventory=dict(self.inventory),
            body=dict(self.body),
            player_pos=self.player_pos,
            step=self.step,
            last_action=self.last_action,
            spatial_map=self.spatial_map.copy() if self.spatial_map else None,
            dynamic_entities=list(self.dynamic_entities),
        )


# ---------------------------------------------------------------------------
# PlanStep + Plan (minimal, replaces forward_sim_types for vector path)
# ---------------------------------------------------------------------------

@dataclass
class VectorPlanStep:
    """One step in a plan: do action on target concept."""
    action: str
    target: str


@dataclass
class VectorPlan:
    """Sequence of steps to execute."""
    steps: list[VectorPlanStep]
    origin: str = ""  # diagnostic: where this plan came from


# ---------------------------------------------------------------------------
# VectorTrajectory
# ---------------------------------------------------------------------------

@dataclass
class VectorTrajectory:
    """Result of simulate_forward."""
    plan: VectorPlan
    states: list[VectorState]
    terminated: bool = False
    terminated_reason: str = ""
    confidences: list[float] = field(default_factory=list)
    # Per-step prediction confidence from model.predict().
    # Populated by simulate_forward. Range [0,1]: 1.0=certain, 0.0=surprised.

    @property
    def final_state(self) -> VectorState | None:
        return self.states[-1] if self.states else None

    def total_inventory_gain(self) -> int:
        """Sum of all positive inventory deltas across the trajectory."""
        if not self.states:
            return 0
        first = self.states[0]
        last = self.states[-1]
        gain = 0
        all_items = set(first.inventory.keys()) | set(last.inventory.keys())
        for item in all_items:
            delta = last.inventory.get(item, 0) - first.inventory.get(item, 0)
            if delta > 0:
                gain += delta
        return gain

    def vital_delta(self, var: str) -> float:
        """Change in body variable from first to last state."""
        if len(self.states) < 2:
            return 0.0
        return self.states[-1].body.get(var, 0.0) - self.states[0].body.get(var, 0.0)

    def inventory_delta(self, item: str) -> float:
        """Change in inventory item count from first to last state."""
        if len(self.states) < 2:
            return 0.0
        return float(
            self.states[-1].inventory.get(item, 0)
            - self.states[0].inventory.get(item, 0)
        )

    def item_gained(self, item: str) -> bool:
        """True if item count went from 0 to >0 during trajectory."""
        if len(self.states) < 2:
            return False
        return (
            self.states[0].inventory.get(item, 0) == 0
            and self.states[-1].inventory.get(item, 0) > 0
        )

    def avg_surprise(self) -> float:
        """Average prediction surprise (1 - avg_confidence). 0 if no confidences."""
        if not self.confidences:
            return 0.0
        return 1.0 - sum(self.confidences) / len(self.confidences)


# ---------------------------------------------------------------------------
# simulate_forward
# ---------------------------------------------------------------------------

def simulate_forward(
    model: "VectorWorldModel",
    plan: VectorPlan,
    initial_state: VectorState,
    horizon: int = 20,
    vital_vars: list[str] | None = None,
    cache: dict | None = None,
    enable_post_plan_passive_rollout: bool = True,
) -> VectorTrajectory:
    """Run forward simulation through VectorWorldModel predictions.

    If a prediction cache is provided, uses it to skip redundant SDM reads.
    """
    states = [initial_state.copy()]
    state = initial_state.copy()
    confidences: list[float] = []

    if not plan.steps:
        return _passive_rollout(
            model=model,
            state=state,
            states=states,
            horizon=horizon,
            vital_vars=vital_vars,
            cache=cache,
            plan=plan,
            confidences=confidences,
            enabled=enable_post_plan_passive_rollout,
        )

    for step in plan.steps[:horizon]:
        primitive_action, allow_effect = _materialize_plan_step(step, state)
        state = _advance_dynamic_entities(model, state, primitive_action, cache)
        if not allow_effect:
            state = _tick_without_effect(state, primitive_action)
            states.append(state)
            continue
        key = (step.target, step.action)
        if cache is not None and key in cache:
            effect_vec, confidence = cache[key]
        else:
            effect_vec, confidence = model.predict(step.target, step.action)

        confidences.append(confidence)  # capture before early-continue

        if confidence < 0.2:
            # No knowledge about this action — skip step
            state = VectorState(
                inventory=dict(state.inventory),
                body=dict(state.body),
                player_pos=state.player_pos,
                step=state.step,
                last_action=step.action,
                spatial_map=state.spatial_map,
                dynamic_entities=list(state.dynamic_entities),
            )
            states.append(state)
            continue

        decoded = model.decode_effect(effect_vec)
        state = state.apply_effect(decoded)
        state.last_action = step.action
        states.append(state)

        if state.is_dead(vital_vars):
            return VectorTrajectory(
                plan=plan, states=states,
                terminated=True, terminated_reason="dead",
                confidences=confidences,
            )

    if (
        enable_post_plan_passive_rollout
        and len(plan.steps) < horizon
        and _has_relevant_dynamic_threat(state)
    ):
        return _passive_rollout(
            model=model,
            state=state,
            states=states,
            horizon=horizon - len(plan.steps),
            vital_vars=vital_vars,
            cache=cache,
            plan=plan,
            confidences=confidences,
            enabled=enable_post_plan_passive_rollout,
        )

    return VectorTrajectory(plan=plan, states=states, confidences=confidences)


def _materialize_plan_step(
    step: VectorPlanStep,
    state: VectorState,
) -> tuple[str, bool]:
    """Convert a high-level plan step into the immediate simulated primitive.

    Vector MPC plans are abstract (`do water`, `do tree`). Execution expands
    those steps into one primitive at a time and navigates toward the target
    before the `do` fires. The simulator must mirror that geometry, otherwise
    distant resource plans look immediately rewarding in imagination while the
    real agent only takes a movement step.
    """
    if step.action != "do" or step.target == "self" or state.spatial_map is None:
        return step.action, True

    target_pos = state.spatial_map.find_nearest(step.target, state.player_pos)
    if target_pos is None:
        return "wait", False

    px, py = state.player_pos
    tx, ty = target_pos
    dx, dy = tx - px, ty - py
    dist = abs(dx) + abs(dy)

    if dist > 1:
        return _step_toward_sim(player_pos=state.player_pos, target_pos=target_pos), False

    if dist == 1 and not _is_facing_target(state.last_action, (dx, dy)):
        return _step_toward_sim(player_pos=state.player_pos, target_pos=target_pos), False

    return step.action, True


def _step_toward_sim(
    player_pos: tuple[int, int],
    target_pos: tuple[int, int],
) -> str:
    """Deterministic one-step movement toward a target for imagination."""
    px, py = player_pos
    tx, ty = target_pos
    dx, dy = tx - px, ty - py

    if abs(dx) >= abs(dy) and dx != 0:
        return "move_right" if dx > 0 else "move_left"
    if dy != 0:
        return "move_down" if dy > 0 else "move_up"
    if dx != 0:
        return "move_right" if dx > 0 else "move_left"
    return "wait"


def _is_facing_target(last_action: str | None, delta: tuple[int, int]) -> bool:
    facing_map = {
        "move_left": (-1, 0),
        "move_right": (1, 0),
        "move_up": (0, -1),
        "move_down": (0, 1),
    }
    facing = facing_map.get(last_action, (0, 1))
    return facing == delta


def _passive_rollout(
    model: "VectorWorldModel",
    state: VectorState,
    states: list[VectorState],
    horizon: int,
    vital_vars: list[str] | None,
    cache: dict | None,
    plan: VectorPlan,
    confidences: list[float],
    enabled: bool,
) -> VectorTrajectory:
    """Continue short-horizon world dynamics after explicit plan steps end."""
    if not enabled:
        return VectorTrajectory(plan=plan, states=states, confidences=confidences)
    if not _has_relevant_dynamic_threat(state):
        return VectorTrajectory(plan=plan, states=states, confidences=confidences)
    for _ in range(max(0, horizon)):
        if not state.dynamic_entities:
            break
        state = _advance_dynamic_entities(model, state, action="wait", cache=cache)
        states.append(state)
        if state.is_dead(vital_vars):
            return VectorTrajectory(
                plan=plan,
                states=states,
                terminated=True,
                terminated_reason="dead",
                confidences=confidences,
            )
    return VectorTrajectory(plan=plan, states=states, confidences=confidences)


def _has_relevant_dynamic_threat(state: VectorState) -> bool:
    """Return True only for dynamic entities that can change short-horizon safety.

    Stage 89 debugging showed that benign dynamic entities such as `cow`
    were enough to trigger a full passive rollout. That made baseline,
    sleep, and movement plans tie at episode start, after which stable
    candidate ordering let `sleep` win repeatedly. Restrict passive threat
    rollout to hostile/projectile concepts only.
    """
    return any(
        entity.concept_id in {"arrow", "zombie", "skeleton"}
        for entity in state.dynamic_entities
    )


def _advance_dynamic_entities(
    model: "VectorWorldModel",
    state: VectorState,
    action: str,
    cache: dict | None = None,
) -> VectorState:
    """Advance player and dynamic entities one short-horizon tick.

    Stage 89:
    - player movement is applied for explicit move primitives
    - arrows continue along inferred velocity
    - passive spatial damage is applied generically from textbook-seeded
      `concept -> proximity` facts using the concept's configured range
    """
    next_state = state.move_player(action) if action.startswith("move_") else state.copy()
    next_state.last_action = action

    updated_entities: list[DynamicEntityState] = []
    for entity in next_state.dynamic_entities:
        pos = entity.position
        if entity.velocity is not None:
            pos = (pos[0] + entity.velocity[0], pos[1] + entity.velocity[1])
        else:
            pos = _apply_movement_behavior(
                entity_pos=entity.position,
                player_pos=next_state.player_pos,
                behavior=model.movement_behaviors.get(entity.concept_id),
                tick=next_state.step,
            )
        moved = DynamicEntityState(
            concept_id=entity.concept_id,
            position=pos,
            velocity=entity.velocity,
            age=entity.age + 1,
            last_seen_step=next_state.step,
        )
        updated_entities.append(moved)

        hit_key = (moved.concept_id, "proximity")
        proximity_range = int(model.proximity_ranges.get(moved.concept_id, 0))
        distance = (
            abs(moved.position[0] - next_state.player_pos[0])
            + abs(moved.position[1] - next_state.player_pos[1])
        )
        should_apply = False
        if moved.concept_id == "arrow":
            should_apply = moved.position == next_state.player_pos
        elif proximity_range > 0 and distance <= proximity_range:
            should_apply = True

        if should_apply:
            if cache is not None and hit_key in cache:
                effect_vec, confidence = cache[hit_key]
            else:
                effect_vec, confidence = model.predict(*hit_key)
            if confidence >= 0.2:
                decoded = model.decode_effect(effect_vec)
                next_state = _apply_effect_same_tick(next_state, decoded)
                next_state.last_action = action

    next_state.dynamic_entities = updated_entities
    return next_state


def _apply_movement_behavior(
    *,
    entity_pos: tuple[int, int],
    player_pos: tuple[int, int],
    behavior: str | None,
    tick: int,
) -> tuple[int, int]:
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
        seed = (entity_pos[0] * 31 + entity_pos[1] * 17 + tick) & 3
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = deltas[seed]
        return (entity_pos[0] + dx, entity_pos[1] + dy)
    return entity_pos


def _step_toward_pos(
    current: tuple[int, int],
    target: tuple[int, int],
) -> tuple[int, int]:
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return (current[0] + (1 if dx > 0 else -1), current[1])
    if dy != 0:
        return (current[0], current[1] + (1 if dy > 0 else -1))
    return current


def _apply_effect_same_tick(state: VectorState, decoded_effect: dict[str, int]) -> VectorState:
    """Apply effect without advancing simulated time."""
    new_inv = dict(state.inventory)
    new_body = dict(state.body)

    for var, delta in decoded_effect.items():
        if var in new_body:
            new_body[var] = max(0.0, min(9.0, new_body[var] + delta))
        else:
            new_inv[var] = max(0, new_inv.get(var, 0) + delta)

    return VectorState(
        inventory=new_inv,
        body=new_body,
        player_pos=state.player_pos,
        step=state.step,
        last_action=state.last_action,
        spatial_map=state.spatial_map,
        dynamic_entities=list(state.dynamic_entities),
    )


def _tick_without_effect(state: VectorState, action: str) -> VectorState:
    """Advance simulated time by one step without inventory/body changes."""
    return VectorState(
        inventory=dict(state.inventory),
        body=dict(state.body),
        player_pos=state.player_pos,
        step=state.step + 1,
        last_action=action,
        spatial_map=state.spatial_map,
        dynamic_entities=list(state.dynamic_entities),
    )


# ---------------------------------------------------------------------------
# score_trajectory
# ---------------------------------------------------------------------------

def score_trajectory(
    trajectory: VectorTrajectory,
    stimuli: "StimuliLayer | None" = None,
    goal: "Goal | None" = None,
) -> tuple:
    """Score trajectory: 3-tuple (base_score, goal_prog, -steps).

    base_score: StimuliLayer.evaluate() if provided, else survived (0/1).
    goal_prog:  Goal.progress(trajectory) if goal provided, else 0.

    Stage 85 note: the Crafter-specific cumulative inventory delta is removed.
    Goal.progress() carries the equivalent signal but only when the goal
    is active — e.g. goal=gather_wood → inventory_delta("wood"),
    goal=fight_zombie → vital_delta("health").
    """
    goal_prog = goal.progress(trajectory) if goal is not None else 0.0
    # Tie-break on explicit planner commitment, not on passive rollout length.
    # Stage 89 debugging found that baseline, sleep, and movement plans could
    # all inherit the same `-steps` once a passive rollout extended them to the
    # full horizon. That turned many clearly different choices into exact ties
    # and let stable candidate ordering select `sleep` at episode start. The
    # rollout is part of safety estimation, not of action complexity; use the
    # number of planned steps as the tie-break instead.
    steps = len(trajectory.plan.steps)

    if stimuli is not None:
        base = stimuli.evaluate(trajectory)
    else:
        base = 0 if trajectory.terminated else 1

    return (base, goal_prog, -steps)
