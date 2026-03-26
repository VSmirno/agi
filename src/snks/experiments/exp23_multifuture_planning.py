"""Experiment 23: Multi-trajectory planning (Stage 11).

Stochastic planner vs deterministic planner in a noisy transition environment.

Environment:
    States: {0..19}, Actions: 0..4
    Deterministic transition: action a from {s} -> adds {(s + a + 1) % 20}
    Noisy transition (p=0.2): random state instead of deterministic

Gate:
    stochastic_success_rate > deterministic_success_rate * 1.2 (20% improvement)
    Fallback: if both < 0.3, PASS if stochastic >= deterministic + 0.05
"""
from __future__ import annotations

import random
import sys

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.daf.types import (
    CausalAgentConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)

NOISE_PROB = 0.2
N_STATES = 20
N_ACTIONS = 5
IMPROVEMENT_RATIO = 1.2
FALLBACK_MARGIN = 0.05
FALLBACK_THRESHOLD = 0.3


def _build_model(device: str) -> CausalWorldModel:
    config = CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(num_nodes=1000, device=device),
            encoder=EncoderConfig(),
            sks=SKSConfig(),
            device=device,
        ),
        causal_min_observations=1,
        causal_context_bins=32,
    )
    return CausalWorldModel(config)


def _train(model: CausalWorldModel, n_train: int) -> None:
    """Populate the causal model with deterministic + noisy transitions."""
    random.seed(42)
    for _ in range(n_train):
        s = random.randint(0, N_STATES - 1)
        a = random.randint(0, N_ACTIONS - 1)
        ctx: set[int] = {s}
        det_eff = ctx | {(s + a + 1) % N_STATES}
        model.observe_transition(ctx, a, det_eff)
        if random.random() < NOISE_PROB:
            noise_s = random.randint(0, N_STATES - 1)
            noise_eff = ctx | {noise_s}
            model.observe_transition(ctx, a, noise_eff)


def _det_plan(
    model: CausalWorldModel,
    start: set[int],
    goal: set[int],
    n_actions: int = N_ACTIONS,
    max_depth: int = 10,
    min_conf: float = 0.3,
) -> bool:
    """Greedy deterministic planner: always picks highest-confidence action."""
    state = set(start)
    for _ in range(max_depth):
        if goal <= state:
            return True
        best_a: int | None = None
        best_conf = -1.0
        for a in range(n_actions):
            _, conf = model.predict_effect(state, a)
            if conf > best_conf:
                best_conf = conf
                best_a = a
        if best_a is None or best_conf < min_conf:
            return False
        eff, _ = model.predict_effect(state, best_a)
        state = state | set(eff)
    return goal <= state


def run(device: str = "cpu", n_train: int = 500, n_episodes: int = 100) -> dict:
    """Run experiment 23: stochastic vs deterministic planning in noisy env.

    Args:
        device: PyTorch device string (e.g. "cpu", "cuda").
        n_train: Number of training transitions for the causal model.
        n_episodes: Number of planning episodes for evaluation.

    Returns:
        Dictionary with keys: passed, stochastic_success_rate,
        deterministic_success_rate, ratio, n_episodes.
    """
    model = _build_model(device)
    _train(model, n_train)

    random.seed(123)
    stoch_sim = StochasticSimulator(model, seed=42)

    stoch_successes = 0
    det_successes = 0

    for _ep in range(n_episodes):
        s = random.randint(0, 14)
        g = (s + 5) % N_STATES
        current: set[int] = {s}
        goal: set[int] = {g}

        # Stochastic planner
        plan, _ = stoch_sim.find_plan_stochastic(
            current,
            goal,
            n_actions=N_ACTIONS,
            n_samples=8,
            max_depth=10,
            min_confidence=0.2,
        )
        if plan is not None:
            state = set(current)
            for action in plan:
                eff, conf = model.predict_effect(state, action)
                if conf < 0.2:
                    break
                state = state | set(eff)
            if goal <= state:
                stoch_successes += 1

        # Deterministic planner
        if _det_plan(model, current, goal):
            det_successes += 1

    stoch_rate = stoch_successes / n_episodes
    det_rate = det_successes / n_episodes
    ratio = stoch_rate / det_rate if det_rate > 0.0 else float("inf")

    primary_pass = stoch_rate > det_rate * IMPROVEMENT_RATIO
    fallback_pass = (
        stoch_rate < FALLBACK_THRESHOLD
        and det_rate < FALLBACK_THRESHOLD
        and stoch_rate >= det_rate + FALLBACK_MARGIN
    )
    passed = primary_pass or fallback_pass

    return {
        "passed": passed,
        "stochastic_success_rate": stoch_rate,
        "deterministic_success_rate": det_rate,
        "ratio": ratio,
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
