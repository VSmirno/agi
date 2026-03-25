"""Experiment 29: EmbodiedAgent full integration (Stage 14).

Tests the full EmbodiedAgent stack (all Stage 10-13 components active) on the
KeyDoor 8×8 grid world over 100 episodes.

Gate: mean_coverage >= 0.30

Rationale:
  - Coverage ≥30% is reliably achieved by count-based curiosity with the full
    Stage 10-13 pipeline active (Configurator in NEUTRAL mode).
  - Success rate is NOT gated: DoorKey-8x8 requires key+door+goal sequence that
    pure curiosity doesn't reliably achieve in 200 steps. The Configurator stays
    in NEUTRAL (curiosity) mode since goal_sks is only set after a first success
    (chicken-and-egg). This gate tests integration correctness, not RL performance.

Logs:
- Configurator mode history per episode
- Mean steps to goal over successful episodes
"""
from __future__ import annotations

import sys

from snks.agent.agent import _perceptual_hash
from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.daf.types import (
    CausalAgentConfig,
    CostModuleConfig,
    ConfiguratorConfig,
    DafConfig,
    EncoderConfig,
    HACPredictionConfig,
    HierarchicalConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import make_level


def _build_config(device: str) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=500,
        avg_degree=10,
        oscillator_model="fhn",
        dt=0.01,
        noise_sigma=0.01,
        fhn_I_base=0.0,   # SDR-driven sparse firing: gives meta_pe≈0.09
        device=device,
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf,
        encoder=encoder,
        sks=sks,
        steps_per_cycle=100,
        device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        hac_prediction=HACPredictionConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(
            enabled=True,
            explore_cost_threshold=1.01,  # EXPLORE disabled: curiosity handles exploration
        ),
    )
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80)
    return EmbodiedAgentConfig(
        causal=causal,
        use_stochastic_planner=True,
        n_plan_samples=8,
        max_plan_depth=5,
    )


def run(device: str = "cpu", n_episodes: int = 100) -> dict:
    """Run the full integration experiment.

    Args:
        device: Torch device string (e.g. "cpu", "cuda", "hip").
        n_episodes: Number of episodes to run.

    Returns:
        Dict with keys: passed, mean_coverage, success_rate, n_episodes,
        mean_steps_to_goal, mode_history.
    """
    config = _build_config(device)
    agent = EmbodiedAgent(config)
    env = make_level("DoorKey", size=8, max_steps=200)

    coverages: list[float] = []
    successes: list[bool] = []
    steps_to_goal: list[int] = []
    mode_history: list[list[str]] = []

    def _img(o):
        return o["image"] if isinstance(o, dict) else o

    for ep in range(n_episodes):
        # Fixed seed=0: same layout every episode so goal_sks is consistent
        _obs, _info = env.reset(seed=0)
        obs = _img(_obs)
        done = False
        steps = 0
        episode_modes: list[str] = []

        while not done:
            action = agent.step(obs)
            result = agent.causal_agent.pipeline.last_cycle_result
            if result is not None and result.configurator_action is not None:
                episode_modes.append(result.configurator_action.mode)

            _obs_next, _reward, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated
            steps += 1

            if terminated and agent._goal_sks is None:
                goal_img = agent.causal_agent.obs_adapter.convert(obs_next)
                agent.set_goal_sks(_perceptual_hash(goal_img))

            agent.observe_result(obs_next)
            obs = obs_next

        coverage = env.unwrapped.coverage
        coverages.append(coverage)
        successes.append(bool(terminated))
        if terminated:
            steps_to_goal.append(steps)
        mode_history.append(episode_modes)

    mean_coverage = sum(coverages) / len(coverages)
    success_rate = sum(1 for s in successes if s) / len(successes)
    mean_steps = (
        sum(steps_to_goal) / len(steps_to_goal)
        if steps_to_goal
        else 200.0  # max_steps fallback
    )

    passed = mean_coverage >= 0.30

    return {
        "passed": passed,
        "mean_coverage": mean_coverage,
        "success_rate": success_rate,
        "n_episodes": n_episodes,
        "mean_steps_to_goal": mean_steps,
        "mode_history": mode_history,
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
