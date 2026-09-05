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
    absorbing_cost: float | None = None


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
    """Search depth-local goal cost; each candidate transition consumes one call.

    Latent distance is a state score, not a calibrated path cost: summing it
    would reject valid detours. Predicted terminal states stay absorbing, so an
    untrained post-terminal rollout cannot change their fixed-depth ranking.
    """
    if min(n_actions, horizon, beam_width, max_calls) < 1 or len(root.z) != 1:
        raise ValueError("positive search budgets and one root are required")
    beam = [_Node(root)]
    calls, trace = 0, []
    for depth in range(horizon):
        expanded = []
        for node in beam:
            if node.absorbing_cost is None:
                continue
            expanded.append(
                _Node(
                    node.state,
                    node.actions,
                    node.cost,
                    node.absorbing_cost,
                )
            )
            trace.append({
                "actions": list(node.actions),
                "cost": node.cost,
                "step_cost": node.absorbing_cost,
                "uncertainty": 0.0,
                "depth": depth + 1,
                "absorbing": True,
            })
        pairs = [(node, action) for node in beam if node.absorbing_cost is None
                 for action in range(n_actions)]
        pairs = pairs[:max_calls - calls]
        if pairs:
            actions = torch.tensor([action for _, action in pairs], device=root.z.device)
            prediction = model.step(_stack([node.state for node, _ in pairs]), actions)
            scores = cost(prediction)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("non-finite planning cost")
            calls += len(pairs)
            scores_cpu = scores.cpu().tolist()
            uncertainty = prediction.uncertainty.cpu().tolist()
            terminated = prediction.terminated_prob.cpu().tolist()
            for i, (node, action) in enumerate(pairs):
                absorbing_cost = scores_cpu[i] if terminated[i] >= 0.5 else None
                candidate = _Node(
                    _slice(prediction.next_state, i),
                    node.actions + (action,),
                    scores_cpu[i],
                    absorbing_cost,
                )
                expanded.append(candidate)
                trace.append({"actions": list(candidate.actions), "cost": candidate.cost,
                              "step_cost": scores_cpu[i], "uncertainty": uncertainty[i],
                              "depth": depth + 1})
        if not expanded:
            break
        beam = sorted(expanded, key=lambda node: (node.cost, node.actions))[:beam_width]
    winner = beam[0]
    if not winner.actions:
        raise RuntimeError("no candidate within planning budget")
    return PlanResult(winner.actions, winner.cost, calls, trace)
