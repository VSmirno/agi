"""Final grid search for Exp 9 — refined scope based on quick_tune results.

Best config found: denom=5.0, w=(0.95,0.05), ratio=1.448
Target: ratio > 1.5 (need +2% improvement)

Search strategy:
- Focus on denom 4-6 (5.0 was best)
- Focus on weights 0.90-0.98 (0.95/0.05 was best)
- Test epsilon 0.15, 0.20
- 2 trials per config
"""

from __future__ import annotations

import csv
import random as pyrandom
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
from snks.device import get_device
from snks.env.causal_grid import CausalGridWorld, make_level


class Result(NamedTuple):
    denominator: float
    state_weight: float
    action_weight: float
    epsilon: float
    coverage_ratio: float
    curious_coverage: float
    random_coverage: float


def make_config(device: str = "cpu", epsilon: float = 0.15) -> CausalAgentConfig:
    return CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(
                num_nodes=5000,
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
    """Monkey-patch IntrinsicMotivation with custom hyperparameters."""
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
            state_novelty = 1.0 - (visit_count / (visit_count + denominator))
            interest = state_weight * state_novelty + action_weight * action_novelty

            if interest > best_interest:
                best_interest = interest
                best_action = a

        return best_action

    agent.motivation.select_action = patched_select_action


def run_curious_agent(config: CausalAgentConfig, n_steps: int, denom: float, sw: float, aw: float) -> float:
    """Run curious agent and return coverage."""
    agent = CausalAgent(config)
    patch_motivation(agent, denom, sw, aw)

    env = make_level("MultiRoom", max_steps=n_steps + 100)
    obs, info = env.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for _ in range(n_steps):
        action = agent.step(img)
        obs, _, terminated, truncated, _ = env.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent.observe_result(img)
        if terminated or truncated:
            obs, info = env.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    coverage = env.unwrapped.coverage if hasattr(env, "unwrapped") else 0.0
    env.close()
    return coverage


def run_random_agent(n_steps: int) -> float:
    """Run random agent and return coverage."""
    env = CausalGridWorld(level="MultiRoom", size=12, max_steps=n_steps + 100)
    env.reset()
    for _ in range(n_steps):
        env.step(pyrandom.randint(0, 4))
    coverage = env.coverage
    env.close()
    return coverage


def main(device: str = "cpu") -> None:
    """Run focused grid search."""
    print("=" * 100)
    print("EXP 9 FINAL GRID SEARCH — Refined Scope")
    print("=" * 100)
    print()
    print("Quick tune best: denom=5.0, w=(0.95,0.05), ratio=1.448")
    print("Target: ratio > 1.5 (need +2%)")
    print()

    # Refined grid: focus on 5.0 ± 1.0, weights 0.90-0.98
    denominators = [4.0, 4.5, 5.0, 5.5, 6.0]
    epsilons = [0.15, 0.20]
    weight_pairs = [
        (0.90, 0.10),
        (0.92, 0.08),
        (0.95, 0.05),
        (0.98, 0.02),
    ]

    total = len(denominators) * len(epsilons) * len(weight_pairs) * 2
    print(f"Grid: {len(denominators)} denom × {len(epsilons)} eps × {len(weight_pairs)} weights × 2 trials = {total} runs")
    print()

    base_config = make_config(device=device)
    random_cov = run_random_agent(500)
    print(f"Random baseline: {random_cov:.4f}")
    print()

    results: list[Result] = []
    run_idx = 0

    for denom in denominators:
        for eps in epsilons:
            config = make_config(device=device, epsilon=eps)

            curious_covs = []
            for trial in range(2):
                run_idx += 1
                print(f"[{run_idx:3d}/{total}] denom={denom:.1f} ε={eps:.2f}", end=" ")

                for sw, aw in weight_pairs:
                    curious_cov = run_curious_agent(config, 500, denom, sw, aw)
                    curious_covs.append((sw, aw, curious_cov))
                    ratio = curious_cov / random_cov if random_cov > 0 else 0.0
                    status = "✅" if ratio > 1.5 else ""
                    print(f"\n            w=({sw:.2f},{aw:.2f}) → ratio={ratio:.4f} {status}")

            # Average per weight pair
            for sw, aw in weight_pairs:
                covs = [c for s, a, c in curious_covs if s == sw and a == aw]
                if covs:
                    avg_cov = np.mean(covs)
                    ratio = avg_cov / random_cov if random_cov > 0 else 0.0
                    result = Result(denom, sw, aw, eps, ratio, avg_cov, random_cov)
                    results.append(result)

    # Sort by ratio
    results.sort(key=lambda r: r.coverage_ratio, reverse=True)

    print()
    print("=" * 100)
    print("TOP 15 RESULTS:")
    print("=" * 100)
    for i, r in enumerate(results[:15], 1):
        status = "✅ GATE PASS" if r.coverage_ratio > 1.5 else "❌"
        print(
            f"{i:2d}. ratio={r.coverage_ratio:.4f} denom={r.denominator:.1f} "
            f"ε={r.epsilon:.2f} w=({r.state_weight:.2f},{r.action_weight:.2f}) {status}"
        )

    # Save CSV
    csv_path = Path(__file__).parent / "grid_search_final_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "denominator", "state_weight", "action_weight", "epsilon",
            "coverage_ratio", "curious_coverage", "random_coverage"
        ])
        for r in results:
            writer.writerow([
                r.denominator, r.state_weight, r.action_weight, r.epsilon,
                f"{r.coverage_ratio:.6f}", f"{r.curious_coverage:.6f}", f"{r.random_coverage:.6f}"
            ])

    print()
    print(f"Results saved to: {csv_path}")

    # Summary
    best = results[0]
    print()
    print("=" * 100)
    print("BEST CONFIGURATION:")
    print("=" * 100)
    print(f"  denominator:   {best.denominator}")
    print(f"  state_weight:  {best.state_weight}")
    print(f"  action_weight: {best.action_weight}")
    print(f"  epsilon:       {best.epsilon}")
    print(f"  coverage_ratio: {best.coverage_ratio:.6f}")
    print()
    if best.coverage_ratio > 1.5:
        print("✅ GATE PASSED!")
    else:
        gap = 1.5 - best.coverage_ratio
        pct = gap / 1.5 * 100
        print(f"❌ Gap: {gap:.4f} ({pct:.2f}% below target)")


if __name__ == "__main__":
    # Auto-detect device (NVIDIA CUDA, AMD ROCm, or CPU)
    device_obj = get_device(prefer="auto")
    device_str = str(device_obj)
    main(device=device_str)
