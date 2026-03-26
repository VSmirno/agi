"""Experiment 25: Cost-driven exploration (Stage 12).

ICM-guided exploration vs random baseline in CausalGridWorld MultiRoom.

Gate:
    coverage_ratio = icm_coverage / random_coverage >= 1.0
Desired:
    coverage_ratio >= 1.258 (Exp9 baseline)
"""
from __future__ import annotations

import random as pyrandom

from snks.agent.agent import CausalAgent
from snks.daf.types import (
    CausalAgentConfig,
    CostModuleConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import CausalGridWorld
from snks.metacog.cost_module import IntrinsicCostModule

COVERAGE_GATE = 1.0
GRID_SIZE = 12
LEVEL = "MultiRoom"


def _make_config(device: str, num_nodes: int) -> CausalAgentConfig:
    return CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(
                num_nodes=num_nodes,
                avg_degree=20,
                oscillator_model="fhn",
                coupling_strength=0.05,
                dt=0.01,
                noise_sigma=0.005,
                fhn_I_base=0.0,
                device=device,
            ),
            encoder=EncoderConfig(sdr_size=4096, sdr_sparsity=0.04),
            sks=SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5),
            steps_per_cycle=100,
            device=device,
        ),
        motor_sdr_size=256,
        causal_min_observations=2,
        curiosity_epsilon=0.5,
    )


def _get_coverage(env: CausalGridWorld) -> float:
    """Return coverage fraction, falling back to _visited_cells if needed."""
    try:
        cov = env.coverage
        if isinstance(cov, float):
            return cov
    except AttributeError:
        pass
    # Manual fallback: fraction of interior cells visited
    total = (env.width - 2) * (env.height - 2)
    return len(env._visited_cells) / max(total, 1)


def _run_icm_agent(n_steps: int, device: str, num_nodes: int) -> float:
    """Run ICM-guided agent and return coverage fraction.

    Args:
        n_steps: Total environment steps to execute.
        device: PyTorch device string.
        num_nodes: Number of DAF oscillator nodes.

    Returns:
        Coverage fraction achieved by the ICM agent.
    """
    config = _make_config(device, num_nodes)
    agent = CausalAgent(config)
    cost_module = IntrinsicCostModule(
        CostModuleConfig(
            w_homeostatic=0.3,
            w_epistemic=0.4,
            w_goal=0.3,
            firing_rate_target=0.05,
        )
    )
    env = CausalGridWorld(level=LEVEL, size=GRID_SIZE, max_steps=n_steps + 100)
    obs, _ = env.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for _ in range(n_steps):
        img_tensor = agent.obs_adapter.convert(img)
        cycle = agent.pipeline.perception_cycle(img_tensor)
        current_sks = set(cycle.sks_clusters.keys())

        # Adaptive epsilon: high curiosity → more exploration
        epsilon = 0.5
        if cycle.metacog is not None:
            cost_state = cost_module.compute(cycle.metacog, cycle.mean_firing_rate)
            epsilon = 0.9 if cost_state.epistemic_value > 0.4 else 0.3

        if pyrandom.random() < epsilon:
            action = pyrandom.randint(0, 4)
        else:
            best_a, best_c = 0, -1.0
            for a in range(5):
                _, conf = agent.causal_model.predict_effect(current_sks, a)
                if conf > best_c:
                    best_c, best_a = conf, a
            action = best_a

        obs, _, terminated, truncated, _ = env.step(action)
        img_next = obs["image"] if isinstance(obs, dict) else obs

        img_next_tensor = agent.obs_adapter.convert(img_next)
        next_cycle = agent.pipeline.perception_cycle(img_next_tensor)
        next_sks = set(next_cycle.sks_clusters.keys())
        agent.causal_model.observe_transition(current_sks, action, next_sks)

        img = img_next
        if terminated or truncated:
            obs, _ = env.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    return _get_coverage(env)


def _run_random(n_steps: int) -> float:
    """Run random baseline agent and return coverage fraction.

    Args:
        n_steps: Total environment steps to execute.

    Returns:
        Coverage fraction achieved by the random agent.
    """
    env = CausalGridWorld(level=LEVEL, size=GRID_SIZE, max_steps=n_steps + 100)
    env.reset()
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(pyrandom.randint(0, 4))
        if term or trunc:
            env.reset()
    return _get_coverage(env)


def run(device: str = "cpu", n_steps: int = 2000, num_nodes: int = 5000) -> dict:
    """Run experiment 25: ICM-guided exploration vs random baseline.

    Args:
        device: PyTorch device string (e.g. "cpu", "cuda", "hip").
        n_steps: Number of environment steps per agent.
        num_nodes: Number of DAF oscillator nodes.

    Returns:
        Dictionary with keys:
            passed (bool): True if coverage_ratio >= 1.0.
            icm_coverage (float): Coverage by ICM agent.
            random_coverage (float): Coverage by random agent.
            coverage_ratio (float): icm_coverage / random_coverage.
    """
    pyrandom.seed(42)
    icm_coverage = _run_icm_agent(n_steps=n_steps, device=device, num_nodes=num_nodes)

    pyrandom.seed(42)
    random_coverage = _run_random(n_steps=n_steps)

    if random_coverage > 0.0:
        coverage_ratio = icm_coverage / random_coverage
    else:
        coverage_ratio = float("inf") if icm_coverage > 0.0 else 1.0

    passed = coverage_ratio >= COVERAGE_GATE

    return {
        "passed": passed,
        "icm_coverage": icm_coverage,
        "random_coverage": random_coverage,
        "coverage_ratio": coverage_ratio,
    }


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
