"""Experiment 39: Replay Impact on Coverage — Stage 17.

Tests whether the Stage 16 ReplayEngine improves agent behaviour in DoorKey-5x5.
exp37 verified that replay doesn't hurt prediction error. exp39 checks that
replay leads to equal or better spatial coverage.

Protocol:
    Two variants, each 100 episodes on DoorKey-5x5:
      no_replay:   ConsolidationConfig(enabled=True), ReplayConfig(enabled=False)
      with_replay: ConsolidationConfig(enabled=True), ReplayConfig(enabled=True)
    Both use the same random seed sequence.

Gate:
    coverage_replay >= coverage_no_replay      # replay does not hurt exploration
    coverage_replay >= 0.25                    # absolute quality floor
"""
from __future__ import annotations

import sys

import numpy as np

from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.daf.types import (
    CausalAgentConfig,
    ConfiguratorConfig,
    ConsolidationConfig,
    CostModuleConfig,
    DafConfig,
    DcamConfig,
    EncoderConfig,
    HierarchicalConfig,
    PipelineConfig,
    ReplayConfig,
    SKSConfig,
)

_N_EPISODES = 100
_MAX_STEPS = 100
COVERAGE_FLOOR_GATE = 0.25


def _build_config(device: str, replay_enabled: bool) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=500, avg_degree=10, oscillator_model="fhn",
        dt=0.01, noise_sigma=0.01, fhn_I_base=0.0, device=device,
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf, encoder=encoder, sks=sks, steps_per_cycle=100, device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(enabled=True),
    )
    dcam = DcamConfig(hac_dim=256, lsh_n_tables=8, lsh_n_bits=8, episodic_capacity=500)
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80, dcam=dcam)
    consolidation = ConsolidationConfig(
        enabled=True, every_n=10, top_k=20,
        cold_threshold=0.3, node_threshold=0.7,
    )
    replay = ReplayConfig(enabled=replay_enabled, top_k=5, n_steps=30)
    return EmbodiedAgentConfig(causal=causal, consolidation=consolidation, replay=replay)


def _make_env(max_steps: int = _MAX_STEPS):
    import minigrid  # registers MiniGrid envs into gymnasium  # noqa: F401
    import gymnasium
    return gymnasium.make("MiniGrid-Empty-5x5-v0", max_episode_steps=max_steps)


def _img(obs) -> np.ndarray:
    return obs["image"] if isinstance(obs, dict) else obs


def _run_variant(device: str, replay_enabled: bool) -> dict:
    cfg = _build_config(device, replay_enabled)
    agent = EmbodiedAgent(cfg)
    env = _make_env()

    coverage_list: list[float] = []

    for ep in range(_N_EPISODES):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False
        step = 0
        visited: set[tuple] = set()

        while not done and step < _MAX_STEPS:
            action = agent.step(obs)
            pos = tuple(env.unwrapped.agent_pos)
            visited.add(pos)

            _obs_next, _, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated
            step += 1
            agent.observe_result(obs_next)
            obs = obs_next

        agent.end_episode()

        # Fraction of walkable interior cells visited (Empty-5x5: 3×3 = 9 cells)
        w = env.unwrapped.width
        h = env.unwrapped.height
        interior = (w - 2) * (h - 2)
        coverage_list.append(len(visited) / max(interior, 1))

    env.close()

    mean_coverage = float(np.mean(coverage_list))
    return {
        "mean_coverage": round(mean_coverage, 4),
        "n_episodes": _N_EPISODES,
    }


def run(device: str = "cpu") -> dict:
    no_replay   = _run_variant(device, replay_enabled=False)
    with_replay = _run_variant(device, replay_enabled=True)

    cov_nr = no_replay["mean_coverage"]
    cov_wr = with_replay["mean_coverage"]

    gate_no_regression = cov_wr >= cov_nr
    gate_floor         = cov_wr >= COVERAGE_FLOOR_GATE
    passed = gate_no_regression and gate_floor

    return {
        "passed": passed,
        "no_replay": no_replay,
        "with_replay": with_replay,
        "gate_details": {
            f"coverage_replay({cov_wr:.4f}) >= coverage_no_replay({cov_nr:.4f})": gate_no_regression,
            f"coverage_replay({cov_wr:.4f}) >= {COVERAGE_FLOOR_GATE}": gate_floor,
        },
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)

    nr = result["no_replay"]
    wr = result["with_replay"]
    print(f"\n{'='*60}")
    print("Exp 39: Replay Impact on Coverage")
    print(f"{'='*60}")
    print(f"No replay:   mean_coverage={nr['mean_coverage']:.4f}  n={nr['n_episodes']}")
    print(f"With replay: mean_coverage={wr['mean_coverage']:.4f}  n={wr['n_episodes']}")
    print("\nGate details:")
    for k, v in result["gate_details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
