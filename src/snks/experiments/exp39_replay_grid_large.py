"""Exp39 Grid: Large-N sweep (N=2000, N=5000) — concept validation.

Quick sweep (N=500) showed concept unconfirmed at small scale.
This sweep tests whether larger network capacity enables reliable replay benefit.

Configs tested: cross product of
  N        ∈ {2000, 5000}
  mode     ∈ {importance, recency, uniform}
  n_steps  ∈ {5, 20}           # best candidates from N=500 quick sweep
  seeds    = [0, 1, 2]
  envs     = empty, doorkey, lava

CONFIRMED criterion: mean_delta > 0 AND pass_rate >= 2/3 for ALL 3 envs.

Usage:
    python -m snks.experiments.exp39_replay_grid_large cuda
"""
from __future__ import annotations

import json
import random
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

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
_MAX_STEPS  = 100

_ENVS = [
    ("MiniGrid-Empty-5x5-v0",        "empty"),
    ("MiniGrid-DoorKey-5x5-v0",       "doorkey"),
    ("MiniGrid-LavaCrossingS9N1-v0",  "lava"),
]


def _build_config(device: str, n_nodes: int, replay_enabled: bool,
                  replay_mode: str, n_steps: int) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=n_nodes, avg_degree=10, oscillator_model="fhn",
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
    replay = ReplayConfig(enabled=replay_enabled, top_k=5, n_steps=n_steps,
                          mode=replay_mode)
    return EmbodiedAgentConfig(causal=causal, consolidation=consolidation, replay=replay)


def _make_env(env_id: str):
    import minigrid  # noqa: F401
    import gymnasium
    return gymnasium.make(env_id, max_episode_steps=_MAX_STEPS)


def _img(obs) -> np.ndarray:
    return obs["image"] if isinstance(obs, dict) else obs


def _run_variant(device: str, env_id: str, n_nodes: int,
                 replay_enabled: bool, replay_mode: str,
                 n_steps: int, seed: int) -> float:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg = _build_config(device, n_nodes, replay_enabled, replay_mode, n_steps)
    agent = EmbodiedAgent(cfg)
    env = _make_env(env_id)
    w, h = env.unwrapped.width, env.unwrapped.height
    interior = (w - 2) * (h - 2)

    cov_list = []
    for ep in range(_N_EPISODES):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done, step, visited = False, 0, set()
        while not done and step < _MAX_STEPS:
            action = agent.step(obs)
            visited.add(tuple(env.unwrapped.agent_pos))
            _obs_next, _, term, trunc, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = term or trunc
            step += 1
            agent.observe_result(obs_next)
            obs = obs_next
        agent.end_episode()
        cov_list.append(len(visited) / max(interior, 1))

    env.close()
    return float(np.mean(cov_list))


def run(device: str = "cpu") -> dict:
    n_nodes_list  = [2000, 5000]
    replay_modes  = ["importance", "recency", "uniform"]
    n_steps_list  = [5, 20]
    seeds         = [0, 1, 2]

    results = {}
    total = len(n_nodes_list) * len(replay_modes) * len(n_steps_list)
    done  = 0

    for n_nodes, replay_mode, n_steps in product(n_nodes_list, replay_modes, n_steps_list):
        done += 1
        cfg_key = f"N{n_nodes}_mode={replay_mode}_steps={n_steps}"
        print(f"\n[{done}/{total}] {cfg_key}", flush=True)

        env_results = {}
        for env_id, env_name in _ENVS:
            deltas = []
            for seed in seeds:
                cov_nr = _run_variant(device, env_id, n_nodes,
                                      False, replay_mode, n_steps, seed)
                cov_wr = _run_variant(device, env_id, n_nodes,
                                      True,  replay_mode, n_steps, seed)
                delta = cov_wr - cov_nr
                deltas.append(delta)
                print(f"  {env_name} s{seed}: nr={cov_nr:.3f} wr={cov_wr:.3f} "
                      f"Δ={delta:+.3f}", flush=True)

            mean_d = float(np.mean(deltas))
            std_d  = float(np.std(deltas))
            pass_r = sum(1 for d in deltas if d > 0) / len(deltas)
            confirmed = mean_d > 0 and pass_r >= 2 / 3
            env_results[env_name] = {
                "deltas":     [round(d, 4) for d in deltas],
                "mean_delta": round(mean_d, 4),
                "std_delta":  round(std_d, 4),
                "pass_rate":  round(pass_r, 2),
                "confirmed":  confirmed,
            }
            mark = "✓" if confirmed else "✗"
            print(f"  {env_name}: μΔ={mean_d:+.4f} ±{std_d:.4f} "
                  f"p={pass_r:.0%} {mark}", flush=True)

        all_confirmed = all(v["confirmed"] for v in env_results.values())
        results[cfg_key] = {
            "n_nodes": n_nodes, "replay_mode": replay_mode, "n_steps": n_steps,
            "envs": env_results, "all_confirmed": all_confirmed,
        }

        # save incrementally
        out = Path("/opt/agi/results/stage17/exp39_grid_large.json")
        out.write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    results = run(device=device)

    print(f"\n{'='*68}")
    print(f"{'Config':<38} {'empty':>7} {'doorkey':>8} {'lava':>6} {'ALL':>5}")
    print(f"{'='*68}")
    for key, r in results.items():
        e = r["envs"]
        marks = [
            "✓" if e["empty"]["confirmed"]   else "✗",
            "✓" if e["doorkey"]["confirmed"] else "✗",
            "✓" if e["lava"]["confirmed"]    else "✗",
            "✓" if r["all_confirmed"]        else "✗",
        ]
        print(f"{key:<38} {marks[0]:>7} {marks[1]:>8} {marks[2]:>6} {marks[3]:>5}")
