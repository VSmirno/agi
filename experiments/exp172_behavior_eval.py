"""Closed-loop behavior and canonical rollout evaluation for exp172."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from experiments.exp145_physics_transfer import (
    SOURCE_LAYOUTS,
    TARGET_LAYOUTS,
    _adapter,
)
from experiments.exp148_source_target_one_step import _validate_protocol
from snks.agent.core_world_model import LatentState
from snks.env.core_grid import GRID_ACTIONS
from snks.pipeline import core_experiment as core


EVAL_SEED = 20000
MAX_REAL_STEPS = 16
ACTION_COUNT = 5
SEARCH_HORIZON = 3
SEARCH_WIDTH = 5
SEARCH_MAX_CALLS = 55


@dataclass(frozen=True)
class _SearchNode:
    state: Any
    actions: tuple[int, ...] = ()
    cost: float = 0.0
    prefix_costs: tuple[float, ...] = ()


@dataclass(frozen=True)
class _ActualState:
    candidate_actions: tuple[int, ...]
    latent: LatentState
    terminated: bool = False
    truncated: bool = False


def _depth_local_search(
    root,
    expand,
    score,
    *,
    action_count: int,
    horizon: int,
    width: int,
    max_calls: int,
    early_progress_tie_break: bool = False,
):
    """Run one shared fixed-width search with a state-local endpoint score."""

    if min(action_count, horizon, width, max_calls) < 1:
        raise ValueError("search budgets must be positive")
    beam = [_SearchNode(root)]
    calls = 0
    trace = []
    depths = []
    for depth in range(1, horizon + 1):
        pairs = [
            (node.state, action)
            for node in beam
            for action in range(action_count)
        ][: max_calls - calls]
        if not pairs:
            break
        states = list(expand(pairs))
        if len(states) != len(pairs):
            raise ValueError("expand must return one state per candidate")
        costs = [float(value) for value in score(states)]
        if len(costs) != len(states):
            raise ValueError("score must return one cost per candidate")
        if any(not math.isfinite(value) for value in costs):
            raise FloatingPointError("non-finite planning cost")
        expanded = []
        for (node, action), state, cost in zip(
            ((node, action) for node in beam for action in range(action_count)),
            states,
            costs,
        ):
            actions = node.actions + (action,)
            prefix_costs = node.prefix_costs + (cost,)
            expanded.append(_SearchNode(state, actions, cost, prefix_costs))
            row = {
                "actions": list(actions),
                "cost": cost,
                "depth": depth,
                "uncertainty": 0.0,
            }
            if early_progress_tie_break:
                row["prefix_costs"] = list(prefix_costs)
            trace.append(row)
        calls += len(pairs)
        depths.append({"depth": depth, "candidate_count": len(pairs)})
        if early_progress_tie_break:
            key = lambda node: (node.cost, node.prefix_costs, node.actions)
        else:
            key = lambda node: (node.cost, node.actions)
        beam = sorted(expanded, key=key)[:width]
    if not beam or not beam[0].actions:
        raise RuntimeError("no candidate within planning budget")
    winner = beam[0]
    result = {
        "actions": list(winner.actions),
        "cost": winner.cost,
        "candidate_calls": calls,
        "depths": depths,
        "trace": trace,
    }
    if early_progress_tie_break:
        result["prefix_costs"] = list(winner.prefix_costs)
    return result


def _stack(states: list[LatentState]) -> LatentState:
    first = states[0]
    return LatentState(
        z=torch.cat([state.z for state in states]),
        sensors=torch.cat([state.sensors for state in states]),
        sensor_mask=torch.cat([state.sensor_mask for state in states]),
        hidden=torch.cat([state.hidden for state in states]),
        schema=first.schema,
    )


def _slice(state: LatentState, index: int) -> LatentState:
    return LatentState(
        state.z[index : index + 1],
        state.sensors[index : index + 1],
        state.sensor_mask[index : index + 1],
        state.hidden[index : index + 1],
        state.schema,
    )


def _model_expand(model):
    def expand(pairs):
        states = [state for state, _action in pairs]
        actions = torch.tensor(
            [action for _state, action in pairs],
            device=states[0].z.device,
            dtype=torch.long,
        )
        prediction = model.step(_stack(states), actions)
        # Termination and ensemble variance are deliberately absent from search.
        return [_slice(prediction.next_state, index) for index in range(len(pairs))]

    return expand


def _ordered_score(ordered, goal_z, latent=lambda state: state.z):
    def score(states):
        z = torch.cat([latent(state) for state in states])
        normalized_horizon = torch.ones(len(z), device=z.device, dtype=z.dtype)
        costs = -ordered(z, goal_z.expand(len(z), -1), normalized_horizon)
        return costs.detach().cpu().tolist()

    return score


def _teacher_forced_observation(model, state, action: int, observation):
    action_tensor = torch.tensor([action], device=state.z.device, dtype=torch.long)
    predicted = model.step(state, action_tensor)
    actual = model.initial(observation)
    return LatentState(
        actual.z,
        actual.sensors,
        actual.sensor_mask,
        predicted.next_state.hidden.detach(),
        actual.schema,
    )


def _layout_specs():
    source = [
        ("source", name, layout, tuple(actions))
        for name, (layout, actions) in SOURCE_LAYOUTS.items()
    ]
    unseen = [
        ("unseen", name, layout, tuple(actions))
        for name, (layout, actions, _push_two) in TARGET_LAYOUTS.items()
    ]
    return source + unseen


def _actual_expand(baseline, layout, real_history, counters):
    """Build oracle candidates from fresh resets without exporting outcomes."""

    def expand(pairs):
        expanded = []
        for state, action in pairs:
            candidate = state.candidate_actions + (action,)
            adapter = _adapter(layout, 1, EVAL_SEED, MAX_REAL_STEPS)
            replayed = 0
            terminated = truncated = False
            try:
                observation = adapter.reset(EVAL_SEED)
                for replay_action in (*real_history, *candidate):
                    transition = adapter.step(replay_action)
                    replayed += 1
                    observation = transition.after
                    terminated = bool(transition.terminated)
                    truncated = bool(transition.truncated)
                    if terminated or truncated:
                        break
            finally:
                adapter.close()
            counters["candidate_calls"] += 1
            counters["resets"] += 1
            counters["replayed_actions"] += replayed
            counters["terminal_candidates"] += int(terminated)
            expanded.append(
                _ActualState(
                    candidate,
                    baseline.initial(observation),
                    terminated,
                    truncated,
                )
            )
        return expanded

    return expand


def _episode(
    model,
    baseline,
    ordered,
    split,
    layout_name,
    layout,
    arm,
    early_progress_tie_break=False,
):
    adapter = _adapter(layout, 1, EVAL_SEED, MAX_REAL_STEPS)
    actions = []
    selected_costs = []
    selected_prefix_costs = []
    selected_plans = []
    decision_traces = []
    planner_calls = 0
    history_update_calls = 0
    oracle = {
        "candidate_calls": 0,
        "resets": 0,
        "replayed_actions": 0,
        "terminal_candidates": 0,
    }
    actual_terminated = actual_truncated = success = False
    try:
        observation = adapter.reset(EVAL_SEED)
        goal_observation = adapter.goal_observation()
        predictor = baseline if arm == "original" else model
        live_state = predictor.initial(observation) if arm != "actual" else None
        goal_z = baseline.initial(goal_observation).z
        for _step in range(MAX_REAL_STEPS):
            if arm == "actual":
                root = _ActualState((), baseline.initial(observation))
                expand = _actual_expand(baseline, layout, tuple(actions), oracle)
                score = _ordered_score(ordered, goal_z, lambda state: state.latent.z)
            else:
                root = live_state
                expand = _model_expand(predictor)
                score = _ordered_score(ordered, goal_z)
            plan = _depth_local_search(
                root,
                expand,
                score,
                action_count=ACTION_COUNT,
                horizon=SEARCH_HORIZON,
                width=SEARCH_WIDTH,
                max_calls=SEARCH_MAX_CALLS,
                early_progress_tie_break=early_progress_tie_break,
            )
            if plan["candidate_calls"] != SEARCH_MAX_CALLS:
                raise AssertionError("fixed H3 search did not consume 55 candidates")
            action = int(plan["actions"][0])
            transition = adapter.step(action)
            observation = transition.after
            actions.append(action)
            selected_costs.append(plan["cost"])
            if early_progress_tie_break:
                selected_prefix_costs.append(plan["prefix_costs"])
            selected_plans.append(plan["actions"])
            decision_traces.append(plan["trace"])
            planner_calls += plan["candidate_calls"]
            actual_terminated = bool(transition.terminated)
            actual_truncated = bool(transition.truncated)
            success = bool(adapter.diagnostic_snapshot().get("success"))
            if arm != "actual":
                live_state = _teacher_forced_observation(
                    predictor, live_state, action, observation
                )
                history_update_calls += 1
            if success or actual_terminated or actual_truncated:
                break
        final_diagnostic = adapter.diagnostic_snapshot()
    finally:
        adapter.close()
    result = {
        "arm": arm,
        "split": split,
        "layout": layout_name,
        "seed": EVAL_SEED,
        "success": success,
        "steps": len(actions),
        "actual_terminated": actual_terminated,
        "actual_truncated": actual_truncated,
        "actions": actions,
        "action_names": [GRID_ACTIONS[action] for action in actions],
        "selected_costs": selected_costs,
        "selected_plans": selected_plans,
        "planner_candidate_calls": planner_calls,
        "history_update_model_calls": history_update_calls,
        "oracle_evaluator_cost": oracle,
        "final_diagnostic": final_diagnostic,
        "decision_traces": decision_traces,
    }
    if early_progress_tie_break:
        result["selected_prefix_costs"] = selected_prefix_costs
    return result


def _canonical_diagnostic(model, role, split, layout_name, layout, actions):
    prefix, continuation = _validate_protocol(
        split, layout_name, layout, actions, EVAL_SEED
    )
    adapter = _adapter(layout, 1, EVAL_SEED, 32)
    try:
        observation = adapter.reset(EVAL_SEED)
        root = model.initial(observation)
        for action in prefix:
            transition = adapter.step(action)
            if transition.terminated or transition.truncated:
                raise RuntimeError("canonical real prefix unexpectedly ended")
            root = _teacher_forced_observation(
                model, root, action, transition.after
            )
        predicted = root
        rows = []
        for depth, action in enumerate(continuation, start=1):
            prediction = model.step(
                predicted,
                torch.tensor([action], device=root.z.device, dtype=torch.long),
            )
            predicted = prediction.next_state
            transition = adapter.step(action)
            actual = model.initial(transition.after)
            if depth in (1, 3):
                prediction_mse = float(
                    (predicted.z - actual.z).square().mean().item()
                )
                persistence_mse = float(
                    (root.z - actual.z).square().mean().item()
                )
                rows.append(
                    {
                        "horizon": depth,
                        "actions": list(continuation[:depth]),
                        "predicted_vs_actual_latent_mse": prediction_mse,
                        "persistence_vs_actual_latent_mse": persistence_mse,
                        "prediction_to_persistence_ratio": (
                            prediction_mse / persistence_mse
                            if persistence_mse > 0.0
                            else None
                        ),
                        "actual_success": bool(
                            adapter.diagnostic_snapshot().get("success")
                        ),
                    }
                )
        return {
            "model": role,
            "split": split,
            "layout": layout_name,
            "seed": EVAL_SEED,
            "real_prefix": list(prefix),
            "canonical_continuation": list(continuation),
            "autoregressive": rows,
        }
    finally:
        adapter.close()


def _summarize(rows):
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "planner_candidate_calls": sum(
            row["planner_candidate_calls"] for row in rows
        ),
        "history_update_model_calls": sum(
            row["history_update_model_calls"] for row in rows
        ),
        "oracle_evaluator_cost": {
            key: sum(row["oracle_evaluator_cost"][key] for row in rows)
            for key in (
                "candidate_calls",
                "resets",
                "replayed_actions",
                "terminal_candidates",
            )
        },
    }


@torch.inference_mode()
def evaluate_behavior(
    model,
    baseline,
    ordered,
    config,
    journal,
    out,
    *,
    early_progress_tie_break=False,
):
    """Evaluate learned, original, and actual-transition development arms."""

    del config  # Search and episode budgets are fixed by the exp172 protocol.
    if model.schemas.get("grid-v1") != (ACTION_COUNT, 1):
        raise ValueError("learned model must expose the established grid-v1 schema")
    if baseline.schemas.get("grid-v1") != (ACTION_COUNT, 1):
        raise ValueError("baseline must expose the established grid-v1 schema")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    behavior_path = out / "behavior_rows.jsonl"
    diagnostic_path = out / "canonical_rollout_rows.jsonl"
    layouts = _layout_specs()

    episodes = []
    behavior_writer = core.TraceWriter(behavior_path)
    try:
        total = 3 * len(layouts)
        journal.update("behavior_eval", 0, total)
        for arm in ("original", "learned", "actual"):
            for split, layout_name, layout, _actions in layouts:
                row = _episode(
                    model,
                    baseline,
                    ordered,
                    split,
                    layout_name,
                    layout,
                    arm,
                    early_progress_tie_break,
                )
                episodes.append(row)
                behavior_writer.write(row)
                journal.update(
                    "behavior_eval",
                    len(episodes),
                    total,
                    arm=arm,
                    split=split,
                    layout=layout_name,
                    success=row["success"],
                    steps=row["steps"],
                )
    finally:
        behavior_writer.close()

    diagnostics = []
    diagnostic_writer = core.TraceWriter(diagnostic_path)
    try:
        total = 2 * len(layouts)
        journal.update("canonical_rollout", 0, total)
        for role, predictor in (("original", baseline), ("learned", model)):
            for split, layout_name, layout, actions in layouts:
                row = _canonical_diagnostic(
                    predictor,
                    role,
                    split,
                    layout_name,
                    layout,
                    actions,
                )
                diagnostics.append(row)
                diagnostic_writer.write(row)
                journal.update(
                    "canonical_rollout",
                    len(diagnostics),
                    total,
                    model=role,
                    split=split,
                    layout=layout_name,
                )
    finally:
        diagnostic_writer.close()

    by_arm = {
        arm: {
            "summary": _summarize([row for row in episodes if row["arm"] == arm]),
            "layouts": [row for row in episodes if row["arm"] == arm],
        }
        for arm in ("original", "learned", "actual")
    }
    result = {
        "status": "completed",
        "protocol": {
            "development_layouts": 8,
            "deterministic_seed": EVAL_SEED,
            "push_distance": 1,
            "max_real_steps": MAX_REAL_STEPS,
            "search": {
                "horizon": SEARCH_HORIZON,
                "width": SEARCH_WIDTH,
                "max_candidate_calls_per_decision": SEARCH_MAX_CALLS,
                "score": "fixed ordered TemporalProbe endpoint logit",
                "probe_normalized_horizon": 1.0,
                "depth_local_not_path_sum": True,
                "uncertainty_penalty": 0.0,
                "model_termination_used": False,
            },
            "closed_loop": "execute first action then replan from real observation",
            "physical_stop": "actual success, termination, truncation, or 16 actions",
            "actual_oracle": (
                "evaluator-only fresh reset and real-history-plus-candidate replay; "
                "outcomes are never features or inputs to learned/original"
            ),
            "actual_terminal_candidates": (
                "replay stops at the first terminal/truncated transition and reuses "
                "that endpoint for longer candidate suffixes; the candidate still "
                "consumes search budget and termination is not scored"
            ),
            "canonical_diagnostic": (
                "eight real late-prefix roots per model; H1/H3 continuation is "
                "autoregressive with no future teacher forcing"
            ),
        },
        "behavior": by_arm,
        "canonical_rollout": diagnostics,
        "artifacts": {
            "behavior_rows": behavior_path.name,
            "canonical_rollout_rows": diagnostic_path.name,
        },
    }
    if early_progress_tie_break:
        result["protocol"]["search"]["tie_break"] = (
            "exact endpoint ties only: lexicographic prefix costs, then actions"
        )
    return result
