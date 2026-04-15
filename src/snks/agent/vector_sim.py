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
        )

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
) -> VectorTrajectory:
    """Run forward simulation through VectorWorldModel predictions.

    If a prediction cache is provided, uses it to skip redundant SDM reads.
    """
    states = [initial_state.copy()]
    state = initial_state.copy()
    confidences: list[float] = []

    for step in plan.steps[:horizon]:
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
                step=state.step + 1,
                last_action=step.action,
                spatial_map=state.spatial_map,
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

    return VectorTrajectory(plan=plan, states=states, confidences=confidences)


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

    total_gain (Crafter-specific cumulative inventory delta) is removed.
    Goal.progress() carries the equivalent signal but only when the goal
    is active — e.g. goal=gather_wood → inventory_delta("wood"),
    goal=fight_zombie → vital_delta("health").
    """
    goal_prog = goal.progress(trajectory) if goal is not None else 0.0
    steps = len(trajectory.states) - 1  # exclude initial state

    if stimuli is not None:
        base = stimuli.evaluate(trajectory)
    else:
        base = 0 if trajectory.terminated else 1

    return (base, goal_prog, -steps)
