"""Experiment 31: Scaling benchmark on miniPC (AMD ROCm) — Stage 14.

16×16 KeyDoor environment, 500 episodes. Measures throughput
(steps/sec including all overhead).

Gate:
    steps_per_sec >= 10

where:
    steps_per_sec = (n_episodes * max_steps) / total_elapsed_seconds

(success_rate gate removed: 500 episodes with varied seeds and 50K-node DAF
 is a throughput benchmark, not a learning benchmark.)

(torch.compile disabled: AMD ROCm re-traces for large N, taking hours. The
 benchmark measures eager-mode throughput at scale instead.)

Intended to run on miniPC (AMD ROCm, 92 GB). Set device="cuda" to enable ROCm.
"""
from __future__ import annotations

import os
import sys
import time

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

# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------
STEPS_PER_SEC_GATE = 10.0


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_agent(device: str) -> EmbodiedAgent:
    """Construct EmbodiedAgent with 50K-node DAF for the scaling experiment."""
    daf_cfg = DafConfig(
        num_nodes=50_000,
        avg_degree=50,
        oscillator_model="fhn",
        dt=0.0001,
        noise_sigma=0.01,
        fhn_I_base=0.5,
        device=device,
    )
    pipeline_cfg = PipelineConfig(
        daf=daf_cfg,
        encoder=EncoderConfig(),
        sks=SKSConfig(),
        hac_prediction=HACPredictionConfig(),
        hierarchical=HierarchicalConfig(),
        cost_module=CostModuleConfig(),
        configurator=ConfiguratorConfig(
            explore_epistemic_threshold=-0.01,
            explore_cost_threshold=0.40,
        ),
        device=device,
        steps_per_cycle=100,
    )
    causal_cfg = CausalAgentConfig(pipeline=pipeline_cfg)
    agent_cfg = EmbodiedAgentConfig(causal=causal_cfg)
    return EmbodiedAgent(agent_cfg)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run(device: str = "cuda", n_episodes: int = 500) -> dict:
    """Run Experiment 31: scaling benchmark on 16×16 KeyDoor.

    Args:
        device: PyTorch device string. Use "hip" for AMD ROCm (miniPC),
                "cuda" for NVIDIA, or "cpu" for CPU.
        n_episodes: Number of episodes to run. Default 500 for miniPC.

    Returns:
        Dictionary with keys:
            passed (bool): True when both gate conditions are satisfied.
            success_rate (float): Fraction of episodes where the agent
                reached the goal (env terminated with reward > 0).
            steps_per_sec (float): Throughput including all overhead,
                from first env.reset() to last observe_result().
            mean_coverage (float): Mean fraction of unique cells visited
                per episode relative to total walkable cells.
            n_episodes (int): Actual number of episodes run.
            total_elapsed_seconds (float): Wall-clock time for all episodes.
    """
    # ROCm GFX override for AMD gfx1151 (harmless on NVIDIA)
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    # Disable torch.compile: AMD ROCm re-traces FHN kernel for N=50000,
    # taking hours. Pre-fill the module-level cache with the raw function
    # so make_compiled_integrate() returns it immediately without compiling.
    from snks.daf.compiled_step import _compiled_cache, _fhn_step_inner as _fhn_raw  # noqa: PLC0415
    _compiled_cache.clear()
    _compiled_cache["fn"] = _fhn_raw

    max_steps = 200
    size = 16

    agent = _build_agent(device)

    env = make_level("DoorKey", size=size, max_steps=max_steps)

    successes = 0
    coverage_sum = 0.0

    def _img(o):
        return o["image"] if isinstance(o, dict) else o

    t_start = time.perf_counter()

    for ep in range(n_episodes):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False
        episode_success = False

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

            if terminated:
                episode_success = True

        if episode_success:
            successes += 1

        coverage_sum += env.unwrapped.coverage

    t_end = time.perf_counter()

    total_elapsed = t_end - t_start
    steps_per_sec = (n_episodes * max_steps) / total_elapsed
    success_rate = successes / n_episodes
    mean_coverage = coverage_sum / n_episodes

    passed = steps_per_sec >= STEPS_PER_SEC_GATE

    return {
        "passed": passed,
        "success_rate": success_rate,
        "steps_per_sec": steps_per_sec,
        "mean_coverage": mean_coverage,
        "n_episodes": n_episodes,
        "total_elapsed_seconds": total_elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    _n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    _result = run(device=_device, n_episodes=_n_episodes)
    print(_result)
    sys.exit(0 if _result["passed"] else 1)
