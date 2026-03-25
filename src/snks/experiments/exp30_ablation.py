"""Experiment 30: EmbodiedAgent ablation study (Stage 14).

Compares four variants of EmbodiedAgent on KeyDoor 8×8 over 100 episodes each:

| Variant        | cost_module.enabled | configurator.enabled | stochastic_planner |
|----------------|---------------------|---------------------|--------------------|
| full_stack     | True                | True                | True               |
| no_configurator| True                | False               | True               |
| no_icm         | False               | False               | True               |
| baseline       | False               | False               | False              |

score = 0.5 * coverage + 0.5 * success_rate

Gate: full_stack_score > no_configurator_score > baseline_score
"""
from __future__ import annotations

import sys

from snks.agent.agent import _perceptual_hash
from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.daf.types import (
    CausalAgentConfig,
    ConfiguratorConfig,
    CostModuleConfig,
    DafConfig,
    EncoderConfig,
    HACPredictionConfig,
    HierarchicalConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import make_level


def _build_config(
    device: str,
    cost_enabled: bool,
    configurator_enabled: bool,
    use_stochastic_planner: bool,
) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=2000,
        avg_degree=10,
        oscillator_model="fhn",
        dt=0.01,
        noise_sigma=0.005,
        fhn_I_base=0.0,
        device=device,
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf,
        encoder=encoder,
        sks=sks,
        steps_per_cycle=20,
        device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        hac_prediction=HACPredictionConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=cost_enabled),
        configurator=ConfiguratorConfig(enabled=configurator_enabled),
    )
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=200)
    return EmbodiedAgentConfig(
        causal=causal,
        use_stochastic_planner=use_stochastic_planner,
        n_plan_samples=8,
        max_plan_depth=5,
    )


def _run_variant(
    config: EmbodiedAgentConfig,
    n_episodes: int,
    seed_offset: int = 0,
) -> dict:
    """Run one variant for n_episodes, return coverage, success_rate, score."""
    agent = EmbodiedAgent(config)
    env = make_level("DoorKey", size=8, max_steps=200)

    coverages: list[float] = []
    successes: list[bool] = []

    def _img(o):
        return o["image"] if isinstance(o, dict) else o

    for ep in range(n_episodes):
        _obs, _info = env.reset(seed=seed_offset + ep)
        obs = _img(_obs)
        done = False
        terminated = False

        while not done:
            action = agent.step(obs)
            _obs_next, _reward, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated

            if terminated and agent._goal_sks is None:
                goal_img = agent.causal_agent.obs_adapter.convert(obs_next)
                agent.set_goal_sks(_perceptual_hash(goal_img))

            agent.observe_result(obs_next)
            obs = obs_next

        coverages.append(env.unwrapped.coverage)
        successes.append(bool(terminated))

    mean_coverage = sum(coverages) / len(coverages)
    success_rate = sum(1 for s in successes if s) / len(successes)
    score = 0.5 * mean_coverage + 0.5 * success_rate

    return {
        "coverage": mean_coverage,
        "success_rate": success_rate,
        "score": score,
    }


def run(device: str = "cpu", n_episodes: int = 100) -> dict:
    """Run the ablation study across four variants.

    Args:
        device: Torch device string (e.g. "cpu", "cuda", "hip").
        n_episodes: Number of episodes per variant.

    Returns:
        Dict with keys: passed, variants, gate.
    """
    # Variant definitions: (name, cost_enabled, configurator_enabled, stochastic_planner)
    variant_specs = [
        ("full_stack",       True,  True,  True),
        ("no_configurator",  True,  False, True),
        ("no_icm",           False, False, True),
        ("baseline",         False, False, False),
    ]

    variants: dict[str, dict] = {}
    for idx, (name, cost_en, conf_en, stoch) in enumerate(variant_specs):
        config = _build_config(
            device=device,
            cost_enabled=cost_en,
            configurator_enabled=conf_en,
            use_stochastic_planner=stoch,
        )
        # Use the same seeds per episode across variants (seed_offset=0 for all)
        result = _run_variant(config, n_episodes=n_episodes, seed_offset=0)
        variants[name] = result

    full_score = variants["full_stack"]["score"]
    no_conf_score = variants["no_configurator"]["score"]
    baseline_score = variants["baseline"]["score"]

    passed = full_score > no_conf_score > baseline_score

    return {
        "passed": passed,
        "variants": variants,
        "gate": "full_stack_score > no_configurator_score > baseline_score",
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
