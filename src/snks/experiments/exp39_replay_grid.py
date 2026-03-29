"""Experiment 39 Grid: Replay Parameter Sweep — Stage 17 diagnostic.

Systematic sweep to confirm (or refute) that replay improves coverage.
Tests:
  - N: network capacity (500, 2000, 5000)
  - replay_mode: importance / recency / uniform
  - n_steps: 1, 5, 20
  - 3 seeds per config
  - 3 env types

For each config reports: mean_delta ± std, pass_rate (fraction of seeds where delta > 0).
A config CONFIRMS the concept if: mean_delta > 0 AND pass_rate >= 2/3.

Usage:
    python -m snks.experiments.exp39_replay_grid cuda
    python -m snks.experiments.exp39_replay_grid cuda --quick   # N=500 only, 2 seeds

Output: results/stage17/exp39_grid.json + summary table to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

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
    ("MiniGrid-Empty-5x5-v0",         "empty"),
    ("MiniGrid-DoorKey-5x5-v0",        "doorkey"),
    ("MiniGrid-LavaCrossingS9N1-v0",   "lava"),
]


def _build_config(device: str, n_nodes: int, replay_enabled: bool,
                  replay_mode: str, n_steps: int) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=n_nodes, avg_degree=10, oscillator_model="fhn",
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
                 replay_enabled: bool, replay_mode: str, n_steps: int,
                 seed: int) -> float:
    """Returns mean_coverage for this variant+seed."""
    import random
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg = _build_config(device, n_nodes, replay_enabled, replay_mode, n_steps)
    agent = EmbodiedAgent(cfg)
    env = _make_env(env_id)
    w, h = env.unwrapped.width, env.unwrapped.height
    interior = (w - 2) * (h - 2)

    coverage_list = []
    for ep in range(_N_EPISODES):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False
        step = 0
        visited: set[tuple] = set()
        while not done and step < _MAX_STEPS:
            action = agent.step(obs)
            visited.add(tuple(env.unwrapped.agent_pos))
            _obs_next, _, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated
            step += 1
            agent.observe_result(obs_next)
            obs = obs_next
        agent.end_episode()
        coverage_list.append(len(visited) / max(interior, 1))

    env.close()
    return float(np.mean(coverage_list))


def run(device: str = "cpu", quick: bool = False) -> dict:
    n_nodes_list   = [500] if quick else [500, 2000, 5000]
    replay_modes   = ["importance", "recency", "uniform"]
    n_steps_list   = [1, 5, 20]
    seeds          = [0, 1] if quick else [0, 1, 2]

    results = {}

    for n_nodes, replay_mode, n_steps in product(n_nodes_list, replay_modes, n_steps_list):
        cfg_key = f"N{n_nodes}_mode={replay_mode}_steps={n_steps}"
        print(f"\n[{cfg_key}]", flush=True)

        env_results = {}
        for env_id, env_name in _ENVS:
            deltas = []
            for seed in seeds:
                cov_nr = _run_variant(device, env_id, n_nodes,
                                      replay_enabled=False,
                                      replay_mode=replay_mode,
                                      n_steps=n_steps, seed=seed)
                cov_wr = _run_variant(device, env_id, n_nodes,
                                      replay_enabled=True,
                                      replay_mode=replay_mode,
                                      n_steps=n_steps, seed=seed)
                delta = cov_wr - cov_nr
                deltas.append(delta)
                print(f"  {env_name} seed={seed}: nr={cov_nr:.4f} wr={cov_wr:.4f} "
                      f"delta={delta:+.4f}", flush=True)

            mean_d = float(np.mean(deltas))
            std_d  = float(np.std(deltas))
            pass_r = sum(1 for d in deltas if d > 0) / len(deltas)
            confirmed = mean_d > 0 and pass_r >= 2 / 3
            env_results[env_name] = {
                "deltas": [round(d, 4) for d in deltas],
                "mean_delta": round(mean_d, 4),
                "std_delta":  round(std_d, 4),
                "pass_rate":  round(pass_r, 2),
                "confirmed":  confirmed,
            }
            status = "✓ CONFIRMED" if confirmed else "✗ not confirmed"
            print(f"  {env_name}: mean={mean_d:+.4f} ±{std_d:.4f} "
                  f"pass_rate={pass_r:.0%} → {status}", flush=True)

        results[cfg_key] = {
            "n_nodes": n_nodes,
            "replay_mode": replay_mode,
            "n_steps": n_steps,
            "envs": env_results,
            "all_confirmed": all(v["confirmed"] for v in env_results.values()),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", nargs="?", default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="N=500 only, 2 seeds (fast sanity check)")
    args = parser.parse_args()

    results = run(device=args.device, quick=args.quick)

    out_path = Path("results/stage17/exp39_grid.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    # Summary table
    print(f"\n{'='*72}")
    print(f"{'Config':<40} {'empty':>8} {'doorkey':>8} {'lava':>8} {'all':>6}")
    print(f"{'='*72}")
    for key, r in results.items():
        envs = r["envs"]
        e = "✓" if envs["empty"]["confirmed"]   else "✗"
        d = "✓" if envs["doorkey"]["confirmed"] else "✗"
        l = "✓" if envs["lava"]["confirmed"]    else "✗"
        a = "✓" if r["all_confirmed"]           else "✗"
        print(f"{key:<40} {e:>8} {d:>8} {l:>8} {a:>6}")
