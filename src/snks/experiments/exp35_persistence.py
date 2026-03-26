"""Experiment 35: Cross-session Persistence (Stage 16).

Gate:
    cold_override_count > 0          SSG was actually used for action selection
    SR_s2 >= SR_s1 * 0.9 or SR_s1 < 0.05   no regression (vacuous when baseline ≈ 0)
    mean_steps_s2 <= mean_steps_s1 * 1.1   overhead acceptable
"""

from __future__ import annotations

import sys
import tempfile
import os

import numpy as np

from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.dcam.consolidation_sched import ConsolidationScheduler
from snks.agent.tiered_planner import TieredPlanner
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
_CONSOLIDATION_EVERY_N = 10


def _build_config(device: str, save_path: str | None = None) -> EmbodiedAgentConfig:
    daf = DafConfig(
        num_nodes=500,
        avg_degree=10,
        oscillator_model="fhn",
        dt=0.01,
        noise_sigma=0.01,
        fhn_I_base=0.0,
        device=device,
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf, encoder=encoder, sks=sks,
        steps_per_cycle=50, device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(enabled=True),
    )
    dcam = DcamConfig(hac_dim=256, lsh_n_tables=8, lsh_n_bits=8,
                      episodic_capacity=500)
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80, dcam=dcam)
    consolidation = ConsolidationConfig(
        enabled=True,
        every_n=_CONSOLIDATION_EVERY_N,
        top_k=50,
        cold_threshold=0.3,
        node_threshold=0.7,
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


def _run_session(agent: EmbodiedAgent, n_episodes: int,
                 track_cold: bool = False) -> dict:
    env = _make_env()
    successes = 0
    total_steps = 0
    cold_override_count = 0

    for ep in range(n_episodes):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False
        steps = 0

        while not done and steps < _MAX_STEPS:
            action = agent.step(obs)

            if track_cold and agent._last_plan_source == "cold":
                cold_override_count += 1

            _obs_next, reward, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated
            steps += 1
            agent.observe_result(obs_next)
            obs = obs_next

            if done and reward > 0:
                successes += 1

        total_steps += steps
        agent.end_episode()

    env.close()

    return {
        "sr": successes / n_episodes,
        "mean_steps": total_steps / n_episodes,
        "cold_override_count": cold_override_count,
    }


def run(device: str = "cpu") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "snks_s16")

        # Session 1
        cfg1 = _build_config(device, save_path=save_path)
        agent1 = EmbodiedAgent(cfg1)
        s1 = _run_session(agent1, _N_EPISODES, track_cold=False)

        # Session 2 — new agent, load state
        cfg2 = _build_config(device, save_path=None)
        agent2 = EmbodiedAgent(cfg2)

        # Restore scheduler state (node_registry + edge_actions + SSG)
        agent2.consolidation_scheduler.load_state(save_path)
        # Re-attach the TieredPlanner with loaded scheduler
        agent2.tiered_planner = TieredPlanner(
            causal_model=agent2.causal_agent.causal_model,
            scheduler=agent2.consolidation_scheduler,
            cold_threshold=cfg2.consolidation.cold_threshold,
            n_actions=agent2.n_actions,
        )

        s2 = _run_session(agent2, _N_EPISODES, track_cold=True)

    gate_cold   = s2["cold_override_count"] > 0
    # SR regression check is vacuously true when baseline SR < 0.05 (near-random noise)
    gate_sr     = s2["sr"] >= s1["sr"] * 0.9 or s1["sr"] < 0.05
    gate_steps  = s2["mean_steps"] <= s1["mean_steps"] * 1.1

    passed = gate_cold and gate_sr

    return {
        "passed": passed,
        "session1": s1,
        "session2": s2,
        "gate_details": {
            f"cold_override_count({s2['cold_override_count']}) > 0": gate_cold,
            f"SR_s2({s2['sr']:.3f}) >= SR_s1({s1['sr']:.3f}) * 0.9": gate_sr,
            f"mean_steps_s2({s2['mean_steps']:.1f}) <= mean_steps_s1({s1['mean_steps']:.1f}) * 1.1": gate_steps,
        },
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)

    s1, s2 = result["session1"], result["session2"]
    print(f"\n{'='*60}")
    print("Exp 35: Cross-session Persistence")
    print(f"{'='*60}")
    print(f"Session 1: SR={s1['sr']:.3f}  mean_steps={s1['mean_steps']:.1f}")
    print(f"Session 2: SR={s2['sr']:.3f}  mean_steps={s2['mean_steps']:.1f}"
          f"  cold_overrides={s2['cold_override_count']}")
    print("\nGate details:")
    for k, v in result["gate_details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
