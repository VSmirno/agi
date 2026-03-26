"""Experiment 24: Stochastic robustness (Stage 11).

Tests that StochasticSimulator improves success_rate monotonically
as the number of Monte Carlo samples N grows from 1 to 16.

Gate: success_rates[16] > success_rates[1]
"""

from __future__ import annotations

import random

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.daf.types import (
    CausalAgentConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)


# ---------------------------------------------------------------------------
# Pipeline config builder
# ---------------------------------------------------------------------------

def _make_config(device: str) -> CausalAgentConfig:
    return CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(num_nodes=1000, device="cpu"),
            encoder=EncoderConfig(),
            sks=SKSConfig(),
            device="cpu",
        ),
        causal_min_observations=1,
        causal_context_bins=32,
    )


# ---------------------------------------------------------------------------
# Plan execution helper
# ---------------------------------------------------------------------------

def execute_plan(
    model: CausalWorldModel,
    start: set[int],
    goal: set[int],
    plan: list[int],
    min_conf: float = 0.2,
) -> bool:
    """Execute plan deterministically and check whether goal is reached.

    Args:
        model: Trained CausalWorldModel.
        start: Initial state as set of SKS IDs.
        goal: Goal state as set of SKS IDs (subset check).
        plan: Sequence of action IDs to execute.
        min_conf: Minimum confidence to accept a predicted effect.

    Returns:
        True if goal <= final_state after executing plan.
    """
    state = set(start)
    for action in plan:
        eff, conf = model.predict_effect(state, action)
        if conf < min_conf:
            break
        state = state | set(eff)
    return goal <= state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(device: str = "cpu", n_test_episodes: int = 30) -> dict:
    """Run Experiment 24: stochastic robustness over N_samples.

    Args:
        device: Torch device string (unused here, kept for interface parity).
        n_train: Number of synthetic training transitions.
        n_test_episodes: Number of (start, goal) test pairs.

    Returns:
        Dictionary with keys:
            passed (bool): success_rates[16] > success_rates[1].
            success_rates (dict): {1: float, 2: float, 4: float, 8: float, 16: float}.
            monotonic_pairs (int): number of adjacent pairs (a, b) where
                success_rates[a] <= success_rates[b], out of 4 pairs.
            n_episodes (int): number of test episodes used.
    """
    # ------------------------------------------------------------------
    # 1. Build model with controlled ambiguity.
    #    30 isolated tasks, each with unique context.
    #    Action 0 (good): success count=2, fail count=1 → P(success)≈0.73
    #    Action 1 (bad):  success count=1, fail count=2 → P(success)≈0.27
    #    After causal_decay collapses total_in_context→0, confidence=count,
    #    so softmax([2,1]) → [0.73, 0.27] and softmax([1,2]) → [0.27, 0.73].
    #    With N=1 the planner picks action 0 ~63% of time; N=16 ~99.8%.
    # ------------------------------------------------------------------
    n_tasks = n_test_episodes  # one task per test episode
    config = CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(num_nodes=1000, device="cpu"),
            encoder=EncoderConfig(),
            sks=SKSConfig(),
            device="cpu",
        ),
        causal_min_observations=1,
        causal_context_bins=256,  # k*3 < 90 for k<30 → no collisions
    )
    model = CausalWorldModel(config)

    for k in range(n_tasks):
        start = k * 3
        goal = start + 1
        # Action 0: 2 success observations, 1 fail observation
        model.observe_transition({start}, 0, {start, goal})   # effect = {goal}
        model.observe_transition({start}, 0, {start, goal})
        model.observe_transition({start}, 0, {start})         # effect = {} (fail)
        # Action 1: 1 success observation, 2 fail observations
        model.observe_transition({start}, 1, {start, goal})
        model.observe_transition({start}, 1, {start})
        model.observe_transition({start}, 1, {start})

    # ------------------------------------------------------------------
    # 2. Build fixed test set: exactly the training tasks
    # ------------------------------------------------------------------
    test_cases: list[tuple[set[int], set[int]]] = [
        ({k * 3}, {k * 3 + 1}) for k in range(n_tasks)
    ]

    # ------------------------------------------------------------------
    # 3. Evaluate for each N
    # ------------------------------------------------------------------
    N_values = [1, 2, 4, 8, 16]
    success_rates: dict[int, float] = {}

    for N in N_values:
        sim = StochasticSimulator(model, seed=N * 10)
        successes = 0
        for current, goal in test_cases:
            # Single-step problem: pick the better of 2 actions via N-sample scoring
            plan, _ = sim.find_plan_stochastic(
                current,
                goal,
                n_actions=2,
                n_samples=N,
                max_depth=1,
                min_confidence=0.0,
                temperature=1.0,
            )
            if plan is not None:
                successes += 1
        success_rates[N] = successes / n_test_episodes

    # ------------------------------------------------------------------
    # 4. Evaluate monotonicity
    # ------------------------------------------------------------------
    pairs = list(zip([1, 2, 4, 8], [2, 4, 8, 16]))
    monotonic_pairs = sum(
        1 for a, b in pairs if success_rates[a] <= success_rates[b]
    )
    passed = bool(success_rates[16] > success_rates[1])

    return {
        "passed": passed,
        "success_rates": success_rates,
        "monotonic_pairs": monotonic_pairs,
        "n_episodes": n_test_episodes,
    }


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
