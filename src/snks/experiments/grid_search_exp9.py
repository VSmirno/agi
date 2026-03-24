"""Grid search over curiosity hyperparameters for Exp 9.

Tests combinations of:
- state_novelty_denominator: controls how aggressive state exploration is
- state_weight / action_weight: balance between state vs action novelty

Results saved to results.csv for analysis.
"""

from __future__ import annotations

import csv
import random as pyrandom
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

from snks.agent.agent import CausalAgent
from snks.daf.types import (
    CausalAgentConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import CausalGridWorld, make_level


class GridSearchResult(NamedTuple):
    denominator: float
    state_weight: float
    action_weight: float
    epsilon: float
    coverage_ratio: float
    curious_coverage: float
    random_coverage: float
    causal_links: int


def make_config(
    device: str = "cpu",
    num_nodes: int = 5000,
    epsilon: float = 0.15,
    denominator: float = 10.0,
    state_weight: float = 0.8,
    action_weight: float = 0.2,
) -> CausalAgentConfig:
    """Create config with tunable curiosity parameters.

    Note: denominator and weights are NOT yet in config, will inject via monkey-patch.
    """
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
        curiosity_epsilon=epsilon,
    )


def patch_motivation(agent: CausalAgent, denominator: float, state_weight: float, action_weight: float) -> None:
    """Monkey-patch IntrinsicMotivation to use custom hyperparameters."""
    original_select = agent.motivation.select_action

    def patched_select_action(current_sks: set[int], causal_model, n_actions: int) -> int:
        if pyrandom.random() < agent.motivation.epsilon:
            return pyrandom.randint(0, n_actions - 1)

        from snks.agent.motivation import _stable_context
        full_ctx = _stable_context(current_sks)
        best_action = 0
        best_interest = -1.0

        for a in range(n_actions):
            key = (full_ctx, a)
            visit_count = agent.motivation._visit_counts[key]
            action_novelty = 1.0 / (1.0 + visit_count)

            # Use custom denominator
            state_novelty = 1.0 - (visit_count / (visit_count + denominator))

            # Use custom weights
            interest = state_weight * state_novelty + action_weight * action_novelty

            if interest > best_interest:
                best_interest = interest
                best_action = a

        return best_action

    agent.motivation.select_action = patched_select_action


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
    denominator: float,
    state_weight: float,
    action_weight: float,
) -> tuple[float, int, int]:
    """Run curious agent with tuned hyperparameters."""
    agent = CausalAgent(config)
    patch_motivation(agent, denominator, state_weight, action_weight)

    env = make_level("MultiRoom", max_steps=n_steps + 100)
    obs, info = env.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for step in range(n_steps):
        action = agent.step(img)
        obs, reward, terminated, truncated, info = env.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent.observe_result(img)

        if terminated or truncated:
            obs, info = env.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    coverage = env.unwrapped.coverage if hasattr(env, "unwrapped") else 0.0
    visited = len(env.unwrapped._visited_cells) if hasattr(env, "unwrapped") else 0
    n_causal_links = agent.causal_model.n_links
    env.close()

    return coverage, visited, n_causal_links


def grid_search(
    device: str = "cpu",
    n_trials: int = 2,
    n_steps: int = 500,
    fast_mode: bool = False,
) -> list[GridSearchResult]:
    """Run grid search over hyperparameter combinations.

    Args:
        fast_mode: If True, test only promising subset (3-4 hours vs 16+ hours)
    """

    # Grid parameters
    if fast_mode:
        # Promising subset: focus on denominator 3-10, epsilon 0.15-0.2, weights 0.85+
        denominators = [3.0, 5.0, 7.0, 10.0]
        epsilons = [0.15, 0.2]
        weight_pairs = [
            (0.85, 0.15),
            (0.9, 0.1),
            (0.95, 0.05),
        ]
    else:
        # Full grid
        denominators = [3.0, 5.0, 7.0, 10.0, 15.0]
        epsilons = [0.1, 0.15, 0.2]
        weight_pairs = [
            (0.8, 0.2),
            (0.85, 0.15),
            (0.9, 0.1),
            (0.95, 0.05),
        ]

    results: list[GridSearchResult] = []
    total_runs = len(denominators) * len(epsilons) * len(weight_pairs) * n_trials
    run_idx = 0

    print(f"Grid Search: {len(denominators)} × {len(epsilons)} × {len(weight_pairs)} × {n_trials} trials = {total_runs} runs")
    print()

    for denominator in denominators:
        for epsilon in epsilons:
            for state_w, action_w in weight_pairs:
                config = make_config(device=device, epsilon=epsilon)

                curious_coverages = []
                causal_links_list = []

                for trial in range(n_trials):
                    run_idx += 1
                    print(f"  [{run_idx}/{total_runs}] denom={denominator} ε={epsilon} w=({state_w:.2f},{action_w:.2f}) trial={trial+1}")

                    curious_cov, _, n_links = _run_curious_agent(
                        config, n_steps, denominator, state_w, action_w
                    )
                    curious_coverages.append(curious_cov)
                    causal_links_list.append(n_links)

                # Get random baseline (only once per epsilon, shared)
                # For speed, sample once per epsilon
                random_cov, _ = _run_random_agent(n_steps)

                mean_curious = np.mean(curious_coverages)
                mean_causal = np.mean(causal_links_list)
                ratio = mean_curious / random_cov if random_cov > 0 else 0.0

                result = GridSearchResult(
                    denominator=denominator,
                    state_weight=state_w,
                    action_weight=action_w,
                    epsilon=epsilon,
                    coverage_ratio=ratio,
                    curious_coverage=mean_curious,
                    random_coverage=random_cov,
                    causal_links=int(mean_causal),
                )
                results.append(result)

                print(f"      → ratio={ratio:.3f} curious={mean_curious:.3f} random={random_cov:.3f} links={int(mean_causal)}")

    return results


def main(device: str = "cpu", fast_mode: bool = False) -> None:
    """Run grid search and save results.

    Args:
        device: "cuda" or "cpu"
        fast_mode: If True, test only promising subset (faster)
    """
    print("=" * 80)
    mode_str = "FAST MODE (3-4 hours)" if fast_mode else "FULL MODE (16+ hours)"
    print(f"Exp 9 Grid Search: Curiosity Hyperparameter Tuning — {mode_str}")
    print("=" * 80)
    print()

    results = grid_search(device=device, n_trials=2, n_steps=500, fast_mode=fast_mode)

    # Sort by coverage_ratio
    results.sort(key=lambda r: r.coverage_ratio, reverse=True)

    print()
    print("=" * 80)
    print("Top 10 Results (by coverage_ratio):")
    print("=" * 80)
    for i, r in enumerate(results[:10], 1):
        print(
            f"{i}. denom={r.denominator:4.1f} ε={r.epsilon:.2f} "
            f"w=({r.state_weight:.2f},{r.action_weight:.2f}) "
            f"→ ratio={r.coverage_ratio:.3f} (curious={r.curious_coverage:.3f} "
            f"random={r.random_coverage:.3f} links={r.causal_links})"
        )

    # Save to CSV
    results_file = Path(__file__).parent / "grid_search_results.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "denominator", "state_weight", "action_weight", "epsilon",
            "coverage_ratio", "curious_coverage", "random_coverage", "causal_links"
        ])
        for r in results:
            writer.writerow([
                r.denominator, r.state_weight, r.action_weight, r.epsilon,
                f"{r.coverage_ratio:.4f}", f"{r.curious_coverage:.4f}",
                f"{r.random_coverage:.4f}", r.causal_links
            ])

    print()
    print(f"Results saved to: {results_file}")

    # Best result
    best = results[0]
    print()
    print("=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    print(f"  state_novelty_denominator: {best.denominator}")
    print(f"  state_weight:              {best.state_weight}")
    print(f"  action_weight:             {best.action_weight}")
    print(f"  epsilon:                   {best.epsilon}")
    print(f"  Coverage ratio:            {best.coverage_ratio:.3f}")
    print(f"  Causal links discovered:   {best.causal_links}")
    print()
    if best.coverage_ratio > 1.5:
        print("✅ GATE PASSED (> 1.5)")
    else:
        print(f"❌ GATE FAILED (target 1.5, got {best.coverage_ratio:.3f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fast mode (3-4 hours instead of 16+)")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    main(device=device, fast_mode=args.fast)
