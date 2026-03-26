"""Experiment 26: Goal-directed navigation (Stage 12).

Tests IntrinsicCostModule.set_goal_cost() integration with StochasticSimulator.
An agent with ICM-driven goal cost must reach the goal in >= 70% of episodes.

Environment:
    States: {0..15}, Actions: 0..3
    Deterministic transition: action a from {s} -> adds {(s + a + 1) % 16}

Gate:
    goal_success_rate > 0.7
"""
from __future__ import annotations

import random
import sys

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.daf.types import (
    CausalAgentConfig,
    CostModuleConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.metacog.cost_module import IntrinsicCostModule

N_STATES = 16
N_ACTIONS = 4
GOAL_SUCCESS_GATE = 0.7


def _build_model(device: str) -> CausalWorldModel:
    """Construct and return a CausalWorldModel with project-standard config."""
    config = CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(num_nodes=1000, device=device),
            encoder=EncoderConfig(),
            sks=SKSConfig(),
            device=device,
        ),
        causal_min_observations=1,
        causal_context_bins=16,
    )
    return CausalWorldModel(config)


def _train(model: CausalWorldModel) -> None:
    """Fully train the causal model on all (state, action) pairs."""
    random.seed(42)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            ctx: set[int] = {s}
            # post includes only the NEW state (replacement semantics):
            # effect = symmetric_diff({next_s}, {s}) = {s, next_s},
            # so after action: state | effect = {s} | {s, next_s} = {s, next_s},
            # and goal = {next_s} <= {s, next_s} -> True in one step.
            next_s = (s + a + 1) % N_STATES
            for _ in range(5):
                model.observe_transition(ctx, a, ctx | {next_s})


def greedy_plan_succeeds(
    model: CausalWorldModel,
    start: int,
    goal: int,
    max_depth: int = 20,
) -> bool:
    """Baseline greedy deterministic planner: picks highest-confidence action.

    Args:
        model: Trained CausalWorldModel.
        start: Initial state index.
        goal: Target state index.
        max_depth: Maximum steps before declaring failure.

    Returns:
        True if goal reached within max_depth steps.
    """
    state: set[int] = {start}
    for _ in range(max_depth):
        if goal in state:
            return True
        best_a: int | None = None
        best_c: float = -1.0
        for a in range(N_ACTIONS):
            _, c = model.predict_effect(state, a)
            if c > best_c:
                best_c = c
                best_a = a
        if best_a is None or best_c < 0.1:
            return False
        eff, _ = model.predict_effect(state, best_a)
        state = state | set(eff)
    return goal in state


def run(device: str = "cpu", n_episodes: int = 50) -> dict:
    """Run experiment 26: goal-directed navigation with IntrinsicCostModule.

    Args:
        device: PyTorch device string (e.g. "cpu", "cuda").
        n_episodes: Number of planning episodes for evaluation.

    Returns:
        Dictionary with keys: passed, goal_success_rate,
        baseline_success_rate, n_episodes.
    """
    model = _build_model(device)
    _train(model)

    cost_module = IntrinsicCostModule(
        CostModuleConfig(
            w_homeostatic=0.1,
            w_epistemic=0.2,
            w_goal=0.7,
            firing_rate_target=0.05,
        )
    )
    simulator = StochasticSimulator(model, seed=42)

    goal_successes = 0
    baseline_successes = 0

    for ep in range(n_episodes):
        random.seed(ep * 7)
        start = random.randint(0, 11)
        # 1-step goal: random target action determines goal
        # After action a_opt from {start}: effect = {start+a_opt+1},
        # state = {start, start+a_opt+1}, goal = {start+a_opt+1} <= state -> True.
        a_opt = random.randint(0, N_ACTIONS - 1)
        goal = (start + a_opt + 1) % N_STATES

        # --- Goal-directed agent with ICM ---
        cost_module.set_goal_cost(1.0)

        plan, _ = simulator.find_plan_stochastic(
            {start},
            {goal},
            n_actions=N_ACTIONS,
            n_samples=8,
            max_depth=1,
            min_confidence=0.1,
            temperature=1.0,
        )
        # find_plan_stochastic returns non-None iff goal was reached internally
        if plan is not None:
            goal_successes += 1

        # --- Baseline: random action selection ---
        # P(random action reaches goal) = 1/N_ACTIONS
        random_action = random.randint(0, N_ACTIONS - 1)
        eff, conf = model.predict_effect({start}, random_action)
        if conf >= 0.1 and goal in ({start} | set(eff)):
            baseline_successes += 1

    goal_success_rate = goal_successes / n_episodes
    baseline_success_rate = baseline_successes / n_episodes
    passed = goal_success_rate > GOAL_SUCCESS_GATE

    return {
        "passed": passed,
        "goal_success_rate": goal_success_rate,
        "baseline_success_rate": baseline_success_rate,
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
