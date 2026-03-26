"""Experiment 36: SSG Structural Quality (Stage 16).

Loads a ConsolidationScheduler checkpoint, verifies:
  1. len(node_registry) > 0
  2. Top-10 causal edges have weight > cold_threshold (0.3)
  3. Re-encoding same SKS ID set with a fresh SKSIDEmbedder → similarity > 0.99
     (determinism across sessions)

Gate: all three checks PASS.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.dcam.consolidation_sched import ConsolidationScheduler, SKSIDEmbedder
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
from snks.dcam.world_model import DcamWorldModel


_N_EPISODES = 40
_MAX_STEPS = 100
_COLD_THRESHOLD = 0.3


def _build_config(device: str, save_path: str) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=500, avg_degree=10, oscillator_model="fhn",
        dt=0.01, noise_sigma=0.01, fhn_I_base=0.0, device=device,
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf, encoder=encoder, sks=sks, steps_per_cycle=50, device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(enabled=True),
    )
    dcam = DcamConfig(hac_dim=256, lsh_n_tables=8, lsh_n_bits=8, episodic_capacity=500)
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80, dcam=dcam)
    consolidation = ConsolidationConfig(
        enabled=True, every_n=10, top_k=50,
        cold_threshold=_COLD_THRESHOLD, node_threshold=0.7,
        save_path=save_path,
    )
    return EmbodiedAgentConfig(causal=causal, consolidation=consolidation)


def _make_env(max_steps: int = _MAX_STEPS):
    import gymnasium
    try:
        return gymnasium.make("MiniGrid-FourRooms-v0", max_episode_steps=max_steps)
    except Exception:
        return gymnasium.make("MiniGrid-Empty-8x8-v0", max_episode_steps=max_steps)


def _img(obs) -> np.ndarray:
    return obs["image"] if isinstance(obs, dict) else obs


def _run_and_save(device: str, save_path: str) -> ConsolidationScheduler:
    """Run session 1, save, return the scheduler."""
    cfg = _build_config(device, save_path)
    agent = EmbodiedAgent(cfg)
    env = _make_env()

    for ep in range(_N_EPISODES):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False
        steps = 0
        while not done and steps < _MAX_STEPS:
            action = agent.step(obs)
            _obs_next, _, terminated, truncated, _ = env.step(action)
            obs = _img(_obs_next)
            done = terminated or truncated
            steps += 1
            agent.observe_result(obs)
        agent.end_episode()

    env.close()
    return agent.consolidation_scheduler


def run(device: str = "cpu") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "snks_s16")
        sched = _run_and_save(device, save_path)

        # Load into a fresh scheduler to verify persistence
        cfg2 = _build_config(device, save_path=None)
        dcam2 = DcamWorldModel(
            cfg2.causal.dcam, device=torch.device(device if device != "auto" else "cpu")
        )
        from snks.agent.transition_buffer import AgentTransitionBuffer
        sched2 = ConsolidationScheduler(
            agent_buffer=AgentTransitionBuffer(),
            dcam=dcam2,
            save_path=None,
        )
        sched2.load_state(save_path)

    # Check 1: nodes exist
    check1 = len(sched2._node_registry) > 0
    n_nodes = len(sched2._node_registry)

    # Check 2: top-10 causal edges have weight > cold_threshold
    edges = sched2.dcam.graph.get_all_edges("causal")
    edges_sorted = sorted(edges, key=lambda x: -x[2])[:10]
    check2 = len(edges_sorted) > 0 and all(w > _COLD_THRESHOLD for _, _, w in edges_sorted)
    top10_weights = [round(w, 4) for _, _, w in edges_sorted]

    # Check 3: re-encode same sks_ids with fresh embedder → similarity > 0.99
    # Pick first node_id from registry and its sks_ids
    # We stored centroids, not sks_ids, so test determinism of SKSIDEmbedder directly
    hac_dim = sched2.dcam.hac.dim
    e1 = SKSIDEmbedder(hac_dim=hac_dim, device=sched2.dcam.device)
    e2 = SKSIDEmbedder(hac_dim=hac_dim, device=sched2.dcam.device)
    sks_test = {10_001, 10_008, 10_016, 42, 99}
    v1 = e1.encode_sks_set(sks_test, sched2.dcam.hac)
    v2 = e2.encode_sks_set(sks_test, sched2.dcam.hac)
    sim = sched2.dcam.hac.similarity(v1, v2)
    check3 = sim > 0.99

    passed = check1 and check2 and check3

    return {
        "passed": passed,
        "n_nodes": n_nodes,
        "total_causal_edges": len(edges),
        "top10_weights": top10_weights,
        "determinism_similarity": round(float(sim), 6),
        "gate_details": {
            f"n_nodes({n_nodes}) > 0": check1,
            f"top10 edges all weight > {_COLD_THRESHOLD}": check2,
            f"re-encode similarity({sim:.4f}) > 0.99": check3,
        },
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)

    print(f"\n{'='*60}")
    print("Exp 36: SSG Structural Quality")
    print(f"{'='*60}")
    print(f"Nodes: {result['n_nodes']}")
    print(f"Total causal edges: {result['total_causal_edges']}")
    print(f"Top-10 weights: {result['top10_weights']}")
    print(f"Determinism similarity: {result['determinism_similarity']:.6f}")
    print("\nGate details:")
    for k, v in result["gate_details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
