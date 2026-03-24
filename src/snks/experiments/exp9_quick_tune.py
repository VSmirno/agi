"""Quick tuning of Exp 9 — test top hypotheses only (30-40 min vs 4 hours).

Based on grid_search analysis:
- Hypothesis A: denominator 3-7 (lower → more aggressive state novelty)
- Hypothesis B: weights 0.9/0.1 or 0.95/0.05 (emphasize state)
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
from snks.env.causal_grid import CausalGridWorld, make_level


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
            state_novelty = 1.0 - (visit_count / (visit_count + denominator))
            interest = state_weight * state_novelty + action_weight * action_novelty

            if interest > best_interest:
                best_interest = interest
                best_action = a

        return best_action

    agent.motivation.select_action = patched_select_action


def run_agent(config: CausalAgentConfig, n_steps: int, denom: float, sw: float, aw: float) -> float:
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


def get_random_baseline(n_steps: int) -> float:
    """Get random agent coverage."""
    env = CausalGridWorld(level="MultiRoom", size=12, max_steps=n_steps + 100)
    env.reset()
    for _ in range(n_steps):
        env.step(pyrandom.randint(0, 4))
    coverage = env.coverage
    env.close()
    return coverage


def main(device: str = "cpu") -> None:
    """Quick tune — test top 6 configurations."""
    print("=" * 80)
    print("Exp 9 QUICK TUNE — Top Hypotheses (30-40 min)")
    print("=" * 80)
    print()

    # Top 6 hypotheses based on theory
    configs = [
        (3.0, 0.9, 0.1, "Aggressive state + strong weights"),
        (5.0, 0.9, 0.1, "Medium aggression + strong weights"),
        (7.0, 0.9, 0.1, "Conservative aggression + strong weights"),
        (5.0, 0.95, 0.05, "Medium aggression + max state emphasis"),
        (3.0, 0.95, 0.05, "Aggressive state + max emphasis"),
        (10.0, 0.85, 0.15, "Baseline improvement attempt"),
    ]

    base_config = make_config(device=device, epsilon=0.15)
    random_cov = get_random_baseline(500)
    print(f"Random baseline coverage: {random_cov:.4f}")
    print()

    results = []
    for i, (denom, sw, aw, desc) in enumerate(configs, 1):
        print(f"[{i}/6] denom={denom} w=({sw:.2f},{aw:.2f}) — {desc}")
        curious_cov = run_agent(base_config, 500, denom, sw, aw)
        ratio = curious_cov / random_cov if random_cov > 0 else 0.0
        results.append((ratio, denom, sw, aw, curious_cov))
        status = "✅ PASS" if ratio > 1.5 else f"({ratio:.3f})"
        print(f"  → coverage={curious_cov:.4f} ratio={ratio:.4f} {status}")
        print()

    # Sort by ratio
    results.sort(reverse=True)

    print("=" * 80)
    print("RESULTS (sorted by ratio):")
    print("=" * 80)
    for i, (ratio, denom, sw, aw, curious_cov) in enumerate(results, 1):
        status = "✅ GATE PASS" if ratio > 1.5 else "❌"
        print(f"{i}. ratio={ratio:.4f} denom={denom} w=({sw:.2f},{aw:.2f}) {status}")

    print()
    best_ratio = results[0][0]
    best_denom, best_sw, best_aw = results[0][1], results[0][2], results[0][3]

    if best_ratio > 1.5:
        print("✅ GATE PASSED!")
        print(f"Best config: denom={best_denom}, weights=({best_sw}, {best_aw})")
    else:
        improvement = best_ratio - 1.114
        pct = improvement / 1.114 * 100 if improvement > 0 else 0
        print(f"❌ Gate not reached. Best: {best_ratio:.4f}")
        print(f"Improvement from baseline (1.114): {improvement:+.4f} ({pct:+.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else args.device
    main(device=device)
