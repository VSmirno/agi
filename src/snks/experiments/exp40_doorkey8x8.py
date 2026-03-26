"""Experiment 40: DoorKey-8x8 Solution — Stage 17.

Tests whether the agent can solve the original DoorKey-8x8 task (key → door → goal).
In exp29–31, success_rate was ~0% because GOAL_SEEKING never activated (chicken-and-egg,
fixed in Stage 15). Now: bootstrap goal_sks on EmptyRoom-5x5, then transfer to DoorKey-8x8.

Protocol:
    Phase 1 — Bootstrap (50 episodes, EmptyRoom-5x5):
        Agent explores with curiosity. First success → goal_sks set.
        Causal model of movement is built.

    Phase 2 — Transfer (200 episodes, DoorKey-8x8):
        goal_sks from Phase 1 transferred to new environment.
        Agent applies goal-seeking and stochastic planner to DoorKey.

Gate:
    goal_seeking_activations > 0          # GOAL_SEEKING actually triggered
    success_rate_phase2 >= 0.05           # >= 5% on DoorKey-8x8
"""
from __future__ import annotations

import sys

import numpy as np

from snks.agent.agent import _perceptual_hash
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
    HACPredictionConfig,
    PipelineConfig,
    ReplayConfig,
    SKSConfig,
)
from snks.env.causal_grid import make_level

_PHASE1_EPS = 50
_PHASE2_EPS = 200
_MAX_STEPS = 200

SUCCESS_RATE_GATE = 0.05


def _build_config(device: str) -> EmbodiedAgentConfig:
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
            goal_cost_threshold=0.10,
            hysteresis_cycles=4,
        ),
    )
    dcam = DcamConfig(hac_dim=256, lsh_n_tables=8, lsh_n_bits=8, episodic_capacity=1000)
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80, dcam=dcam)
    consolidation = ConsolidationConfig(
        enabled=True, every_n=10, top_k=20,
        cold_threshold=0.3, node_threshold=0.7,
    )
    replay = ReplayConfig(enabled=True, top_k=5, n_steps=30)
    return EmbodiedAgentConfig(
        causal=causal,
        consolidation=consolidation,
        replay=replay,
        use_stochastic_planner=True,
        n_plan_samples=8,
        max_plan_depth=5,
        goal_cost_value=1.0,
        plan_min_confidence=0.05,
    )


def _make_env(kind: str, size: int, max_steps: int):
    import minigrid  # noqa: F401
    import gymnasium
    if kind == "empty":
        return gymnasium.make(f"MiniGrid-Empty-{size}x{size}-v0",
                              max_episode_steps=max_steps)
    return make_level("DoorKey", size=size, max_steps=max_steps)


def _img(obs) -> np.ndarray:
    return obs["image"] if isinstance(obs, dict) else obs


def run(device: str = "cpu") -> dict:
    config = _build_config(device)
    agent  = EmbodiedAgent(config)

    goal_seeking_steps = 0
    goal_sks_set_ep: int | None = None

    # ── Phase 1: Bootstrap on EmptyRoom-5x5 ────────────────────────────────
    env1 = _make_env("empty", size=5, max_steps=_MAX_STEPS)
    phase1_successes = 0

    for ep in range(_PHASE1_EPS):
        _obs, _ = env1.reset(seed=ep)
        obs  = _img(_obs)
        done = False
        step = 0

        while not done and step < _MAX_STEPS:
            action = agent.step(obs)

            result = agent.causal_agent.pipeline.last_cycle_result
            if result is not None and result.configurator_action is not None:
                if result.configurator_action.mode == "goal_seeking":
                    goal_seeking_steps += 1

            _obs_next, _, terminated, truncated, _ = env1.step(action)
            obs_next = _img(_obs_next)
            done  = terminated or truncated
            step += 1

            if terminated and agent._goal_sks is None:
                goal_img = agent.causal_agent.obs_adapter.convert(obs_next)
                agent.set_goal_sks(_perceptual_hash(goal_img))
                goal_sks_set_ep = ep

            agent.observe_result(obs_next)
            obs = obs_next

        if terminated:
            phase1_successes += 1
        agent.end_episode()

    env1.close()

    # ── Phase 2: Transfer to DoorKey-8x8 ───────────────────────────────────
    env2 = _make_env("doorkey", size=8, max_steps=_MAX_STEPS)
    phase2_successes = 0
    phase2_steps: list[int] = []

    for ep in range(_PHASE2_EPS):
        _obs, _ = env2.reset(seed=ep)
        obs  = _img(_obs)
        done = False
        step = 0

        while not done and step < _MAX_STEPS:
            action = agent.step(obs)

            result = agent.causal_agent.pipeline.last_cycle_result
            if result is not None and result.configurator_action is not None:
                if result.configurator_action.mode == "goal_seeking":
                    goal_seeking_steps += 1

            _obs_next, _, terminated, truncated, _ = env2.step(action)
            obs_next = _img(_obs_next)
            done  = terminated or truncated
            step += 1
            agent.observe_result(obs_next)
            obs = obs_next

        if terminated:
            phase2_successes += 1
            phase2_steps.append(step)
        agent.end_episode()

    env2.close()

    sr2 = phase2_successes / _PHASE2_EPS
    mean_steps2 = float(np.mean(phase2_steps)) if phase2_steps else float(_MAX_STEPS)

    gate_mode = goal_seeking_steps > 0
    gate_sr   = sr2 >= SUCCESS_RATE_GATE
    passed    = gate_mode and gate_sr

    return {
        "passed": passed,
        "phase1": {
            "episodes":     _PHASE1_EPS,
            "n_successes":  phase1_successes,
            "goal_sks_set_ep": goal_sks_set_ep,
        },
        "phase2": {
            "episodes":       _PHASE2_EPS,
            "n_successes":    phase2_successes,
            "success_rate":   round(sr2, 4),
            "mean_steps":     round(mean_steps2, 1),
        },
        "goal_seeking_steps": goal_seeking_steps,
        "gate_details": {
            "goal_seeking_steps > 0 [PRIMARY]": gate_mode,
            f"success_rate_phase2({sr2:.4f}) >= {SUCCESS_RATE_GATE}": gate_sr,
        },
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)

    p1 = result["phase1"]
    p2 = result["phase2"]
    print(f"\n{'='*60}")
    print("Exp 40: DoorKey-8x8 Solution")
    print(f"{'='*60}")
    print(f"Phase 1 (EmptyRoom-5x5):  n_ep={p1['episodes']}  "
          f"successes={p1['n_successes']}  goal_sks_ep={p1['goal_sks_set_ep']}")
    print(f"Phase 2 (DoorKey-8x8):    n_ep={p2['episodes']}  "
          f"SR={p2['success_rate']:.1%}  mean_steps={p2['mean_steps']:.0f}")
    print(f"GOAL_SEEKING steps total: {result['goal_seeking_steps']}")
    print("\nGate details:")
    for k, v in result["gate_details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
