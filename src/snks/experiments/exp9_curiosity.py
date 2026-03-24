"""Experiment 9: Curiosity-driven Exploration.

Environment: L6 (MultiRoom) — 12×12, 2 rooms, door, boxes.
Compare CausalAgent (with IntrinsicMotivation) vs random agent.

Metrics:
- Coverage ratio: curious_coverage / random_coverage
- Discovery speed: steps to discover all causal link types

Gate: Coverage ratio > 1.5
"""

from __future__ import annotations

import random as pyrandom

import numpy as np

from snks.agent.agent import CausalAgent
from snks.daf.types import (
    CausalAgentConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import Action, CausalGridWorld, make_level


def make_config(device: str = "cpu", num_nodes: int = 5000) -> CausalAgentConfig:
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
            encoder=EncoderConfig(
                sdr_size=4096,
                sdr_sparsity=0.04,
            ),
            sks=SKSConfig(
                coherence_mode="rate",
                min_cluster_size=5,
                dbscan_min_samples=5,
            ),
            steps_per_cycle=100,
            device=device,
        ),
        motor_sdr_size=256,
        causal_min_observations=2,
        curiosity_epsilon=0.30,        # C1: start high for chaotic exploration
        curiosity_epsilon_min=0.05,     # C1: decay to this floor
        curiosity_epsilon_horizon=2000, # C1: decay horizon in steps
    )


def _run_random_agent(n_steps: int) -> tuple[float, int]:
    """Run random agent, return (coverage, visited_cells)."""
    env = CausalGridWorld(level="MultiRoom", size=12, max_steps=n_steps + 100)
    env.reset()

    for _ in range(n_steps):
        action = pyrandom.randint(0, 4)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    coverage = env.coverage
    visited = len(env._visited_cells)
    env.close()
    return coverage, visited


def _run_curious_agent(
    config: CausalAgentConfig,
    n_steps: int,
) -> tuple[float, int, int]:
    """Run curious agent, return (coverage, visited_cells, n_causal_links)."""
    agent = CausalAgent(config)
    env = make_level("MultiRoom", max_steps=n_steps + 100)
    obs, info = env.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for step in range(n_steps):
        action = agent.step(img)
        obs, reward, terminated, truncated, info = env.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent.observe_result(img)

        # Track coverage in unwrapped env
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_visited_cells"):
            pass  # coverage tracked in unwrapped

        if terminated or truncated:
            obs, info = env.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    # Get coverage from unwrapped env
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    coverage = unwrapped.coverage if hasattr(unwrapped, "coverage") else 0.0
    visited = len(unwrapped._visited_cells) if hasattr(unwrapped, "_visited_cells") else 0
    n_links = agent.causal_model.n_links

    env.close()
    return coverage, visited, n_links


def run(
    device: str = "cpu",
    num_nodes: int = 5000,
    n_steps: int = 500,
    n_trials: int = 3,
) -> dict:
    """Run Experiment 9: Curiosity-driven Exploration.

    Returns dict with: coverage_ratio, curious_coverage, random_coverage.
    """
    config = make_config(device=device, num_nodes=num_nodes)

    # Run multiple trials and average
    curious_coverages = []
    random_coverages = []
    curious_links = []

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}...")

        # Random agent
        rand_cov, rand_visited = _run_random_agent(n_steps)
        random_coverages.append(rand_cov)

        # Curious agent
        cur_cov, cur_visited, n_links = _run_curious_agent(config, n_steps)
        curious_coverages.append(cur_cov)
        curious_links.append(n_links)

    avg_curious = np.mean(curious_coverages)
    avg_random = np.mean(random_coverages)
    coverage_ratio = avg_curious / max(avg_random, 0.001)

    results = {
        "coverage_ratio": float(coverage_ratio),
        "curious_coverage": float(avg_curious),
        "random_coverage": float(avg_random),
        "avg_causal_links": float(np.mean(curious_links)),
        "n_steps": n_steps,
        "n_trials": n_trials,
    }

    print(f"Exp 9 Results:")
    print(f"  Coverage ratio:   {coverage_ratio:.3f} (gate > 1.5)")
    print(f"  Curious coverage: {avg_curious:.3f}")
    print(f"  Random coverage:  {avg_random:.3f}")
    print(f"  Avg causal links: {np.mean(curious_links):.0f}")

    return results


if __name__ == "__main__":
    run()
