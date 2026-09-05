"""Bounded discrete MPC; batches candidates without hiding their compute cost."""

from dataclasses import dataclass

import torch

from snks.agent.core_cost import GoalCost
from snks.agent.core_world_model import CoreWorldModel, LatentState


@dataclass
class PlanResult:
    actions: tuple[int, ...]
    cost: float
    model_calls: int
    trace: list[dict]


@dataclass
class _Node:
    state: LatentState
    actions: tuple[int, ...] = ()
    cost: float = 0.0


def _stack(states: list[LatentState]) -> LatentState:
    return LatentState(*(torch.cat([getattr(s, name) for s in states], dim=0)
                         for name in ("z", "sensors", "sensor_mask", "hidden")),
                       states[0].schema)


def _slice(state: LatentState, index: int) -> LatentState:
    return LatentState(*(getattr(state, name)[index:index + 1]
                         for name in ("z", "sensors", "sensor_mask", "hidden")),
                       state.schema)


@torch.no_grad()
def beam_plan(model: CoreWorldModel, root: LatentState, cost: GoalCost,
              n_actions: int, horizon: int, beam_width: int,
              max_calls: int) -> PlanResult:
    """Search cumulative goal cost; each candidate transition consumes one call.

    The short-horizon pilot uses fixed-depth costs (no survival discount that
    would reward predicted death). Termination is an optional explicit cost.
    """
    if min(n_actions, horizon, beam_width, max_calls) < 1 or len(root.z) != 1:
        raise ValueError("positive search budgets and one root are required")
    beam = [_Node(root)]
    calls, trace = 0, []
    for depth in range(horizon):
        pairs = [(node, action) for node in beam for action in range(n_actions)]
        pairs = pairs[:max_calls - calls]
        if not pairs:
            break
        actions = torch.tensor([action for _, action in pairs], device=root.z.device)
        prediction = model.step(_stack([node.state for node, _ in pairs]), actions)
        scores = cost(prediction)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite planning cost")
        calls += len(pairs)
        scores_cpu = scores.cpu().tolist()
        uncertainty = prediction.uncertainty.cpu().tolist()
        expanded = []
        for i, (node, action) in enumerate(pairs):
            candidate = _Node(_slice(prediction.next_state, i),
                              node.actions + (action,), node.cost + scores_cpu[i])
            expanded.append(candidate)
            trace.append({"actions": list(candidate.actions), "cost": candidate.cost,
                          "step_cost": scores_cpu[i], "uncertainty": uncertainty[i],
                          "depth": depth + 1})
        beam = sorted(expanded, key=lambda node: (node.cost, node.actions))[:beam_width]
    winner = beam[0]
    if not winner.actions:
        raise RuntimeError("no candidate within planning budget")
    return PlanResult(winner.actions, winner.cost, calls, trace)
