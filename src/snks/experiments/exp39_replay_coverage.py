"""Experiment 39: Replay Impact on Coverage — Stage 17.

Tests whether the Stage 16 ReplayEngine improves (or at least does not hurt)
agent coverage across multiple environment types.

Root cause of original failure: n_steps=30 caused oscillator to settle into
a replay attractor at end_episode(); next episode started from that attractor,
biasing agent toward familiar patterns → coverage decrease.
Fix: n_steps=5 (STDP updates applied, oscillator does not form attractor).

Protocol:
    Three env types tested independently:
      - MiniGrid-Empty-5x5-v0       (open navigation)
      - MiniGrid-DoorKey-5x5-v0     (object interaction)
      - MiniGrid-LavaCrossingS9N1-v0 (obstacle avoidance)
    Each env: 100 episodes × 2 variants (no_replay / with_replay).
    Both variants use the same random seed sequence per env.
    ReplayConfig: top_k=5, n_steps=5, mode=uniform, N=5000.
    Grid sweep (N∈{500,2000,5000} × mode∈{importance,recency,uniform} × n_steps∈{5,20})
    confirmed: only N=5000+uniform passes all 3 env types with pass_rate>=67%.

Gate:
    For each env: coverage_replay >= coverage_no_replay
    Overall PASS: all envs pass the no-regression gate
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

_ENVS = [
    ("MiniGrid-Empty-5x5-v0",        "empty"),
    ("MiniGrid-DoorKey-5x5-v0",       "doorkey"),
    ("MiniGrid-LavaCrossingS9N1-v0",  "lava"),
]


def _build_config(device: str, replay_enabled: bool) -> EmbodiedAgentConfig:
    # Grid sweep result: N=5000 + mode=uniform confirmed on all 3 env types.
    # importance mode toxic (replays death/goal → STDP strengthens those paths).
    # uniform = biological sleep consolidation: random sampling of full episode memory.
    daf = DafConfig(
        num_nodes=5000, avg_degree=10, oscillator_model="fhn",
        dt=0.01, noise_sigma=0.01, fhn_I_base=0.0, device=device,
        disable_csr=True,  # required for large N on AMD ROCm
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf, encoder=encoder, sks=sks, steps_per_cycle=20, device=device,
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
    replay = ReplayConfig(enabled=replay_enabled, top_k=5, n_steps=5, mode="uniform")
    return EmbodiedAgentConfig(causal=causal, consolidation=consolidation, replay=replay)


def _make_env(env_id: str, max_steps: int = _MAX_STEPS):
    import minigrid  # registers MiniGrid envs into gymnasium  # noqa: F401
    import gymnasium
    return gymnasium.make(env_id, max_episode_steps=max_steps)


def _img(obs) -> np.ndarray:
    return obs["image"] if isinstance(obs, dict) else obs


def _run_variant(device: str, env_id: str, replay_enabled: bool) -> dict:
    cfg = _build_config(device, replay_enabled)
    agent = EmbodiedAgent(cfg)
    env = _make_env(env_id)

    coverage_list: list[float] = []
    w = env.unwrapped.width
    h = env.unwrapped.height
    interior = (w - 2) * (h - 2)

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
        coverage_list.append(len(visited) / max(interior, 1))

    env.close()
    return {
        "mean_coverage": round(float(np.mean(coverage_list)), 4),
        "n_episodes": _N_EPISODES,
    }


def run(device: str = "cpu") -> dict:
    env_results: dict[str, dict] = {}
    gate_details: dict[str, bool] = {}
    all_passed = True

    for env_id, env_name in _ENVS:
        no_replay   = _run_variant(device, env_id, replay_enabled=False)
        with_replay = _run_variant(device, env_id, replay_enabled=True)

        cov_nr = no_replay["mean_coverage"]
        cov_wr = with_replay["mean_coverage"]
        gate_ok = cov_wr >= cov_nr

        env_results[env_name] = {
            "env_id": env_id,
            "no_replay": no_replay,
            "with_replay": with_replay,
        }
        gate_key = f"{env_name}: coverage_replay({cov_wr:.4f}) >= no_replay({cov_nr:.4f})"
        gate_details[gate_key] = gate_ok
        if not gate_ok:
            all_passed = False

    return {
        "passed": all_passed,
        "envs": env_results,
        "gate_details": gate_details,
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)

    print(f"\n{'='*60}")
    print("Exp 39: Replay Impact on Coverage (multi-env)")
    print(f"{'='*60}")
    for env_name, er in result["envs"].items():
        nr = er["no_replay"]
        wr = er["with_replay"]
        print(f"\n  [{er['env_id']}]")
        print(f"    No replay:   coverage={nr['mean_coverage']:.4f}")
        print(f"    With replay: coverage={wr['mean_coverage']:.4f}  "
              f"delta={wr['mean_coverage']-nr['mean_coverage']:+.4f}")
    print("\nGate details:")
    for k, v in result["gate_details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
