"""Experiment 33: HAC Episodic Buffer vs Bundle (Stage 15).

Demonstrates that EpisodicHACPredictor maintains better prediction quality
than single-bundle HACPredictionEngine at steps 50-200 (bundle capacity overflow).

Protocol:
    Two agent configs:
        bundle:   HACPredictionEngine (decay=0.95) — current default
        episodic: EpisodicHACPredictor (K=32)      — new Stage 15

    Environment: MiniGrid-Empty-8x8-v0
    N=500, 20 episodes, max_steps=200

    Every 10 steps: record winner_pe (cosine distance predicted vs actual).

Gate:
    mean_pe(episodic, steps 50-200) <= mean_pe(bundle, steps 50-200)
    mean_pe(episodic, steps 50-200) <= 0.49  (better than random 0.5)
"""

from __future__ import annotations

import sys

import numpy as np

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


_N_EPISODES = 20
_MAX_STEPS  = 200
_RECORD_INTERVAL = 10   # record PE every N steps
_EVAL_START_STEP  = 50  # steps to ignore (warm-up)


def _build_config(device: str, use_episodic: bool) -> EmbodiedAgentConfig:
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
    hac_pred = HACPredictionConfig(
        enabled=True,
        memory_decay=0.95,
        use_episodic_buffer=use_episodic,
        episodic_capacity=32,
    )
    pipeline = PipelineConfig(
        daf=daf,
        encoder=encoder,
        sks=sks,
        steps_per_cycle=100,
        device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        hac_prediction=hac_pred,
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(enabled=True),
    )
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80)
    return EmbodiedAgentConfig(causal=causal)


def _make_env(size: int = 8):
    import gymnasium
    return gymnasium.make(f"MiniGrid-Empty-{size}x{size}-v0",
                          max_episode_steps=_MAX_STEPS)


def _img(obs) -> np.ndarray:
    return obs["image"] if isinstance(obs, dict) else obs


def _run_variant(device: str, use_episodic: bool) -> dict:
    """Run one variant and collect per-step winner_pe."""
    config = _build_config(device, use_episodic)
    agent  = EmbodiedAgent(config)
    env    = _make_env(size=8)

    # pe_by_step[s] = list of PE values recorded at global step s (within each episode)
    pe_records: list[tuple[int, float]] = []  # (episode_step, pe)

    for ep in range(_N_EPISODES):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False
        step = 0

        while not done and step < _MAX_STEPS:
            action = agent.step(obs)

            _obs_next, _reward, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated
            step += 1

            agent.observe_result(obs_next)

            # Record winner_pe from last cycle result
            if step % _record_interval_for_step(step) == 0:
                result = agent.causal_agent.pipeline.last_cycle_result
                if result is not None and result.winner_pe > 0.0:
                    pe_records.append((step, result.winner_pe))

            obs = obs_next

    env.close()

    # Split into warm-up and eval windows
    eval_pes = [pe for (s, pe) in pe_records if s >= _EVAL_START_STEP]
    warmup_pes = [pe for (s, pe) in pe_records if s < _EVAL_START_STEP]

    mean_pe_eval   = float(np.mean(eval_pes))   if eval_pes   else 0.5
    mean_pe_warmup = float(np.mean(warmup_pes)) if warmup_pes else 0.5

    return {
        "mean_pe_eval":   round(mean_pe_eval, 4),
        "mean_pe_warmup": round(mean_pe_warmup, 4),
        "n_records":      len(pe_records),
        "n_eval_records": len(eval_pes),
    }


def _record_interval_for_step(step: int) -> int:
    """Return interval so that step % interval == 0 for multiples of _RECORD_INTERVAL."""
    return _RECORD_INTERVAL


def run(device: str = "cpu") -> dict:
    """Run both variants and compare PE.

    Returns:
        Dict with keys: passed, bundle, episodic, gate_details.
    """
    # Collect per-step PE directly (simpler approach — track every cycle)
    results = {}
    for variant, use_episodic in [("bundle", False), ("episodic", True)]:
        config = _build_config(device, use_episodic)
        agent  = EmbodiedAgent(config)
        env    = _make_env(size=8)

        pe_records: list[tuple[int, float]] = []

        for ep in range(_N_EPISODES):
            _obs, _ = env.reset(seed=ep)
            obs = _img(_obs)
            done = False
            step = 0

            while not done and step < _MAX_STEPS:
                action = agent.step(obs)
                _obs_next, _r, terminated, truncated, _ = env.step(action)
                obs_next = _img(_obs_next)
                done = terminated or truncated
                step += 1

                agent.observe_result(obs_next)

                if step % _RECORD_INTERVAL == 0:
                    result = agent.causal_agent.pipeline.last_cycle_result
                    if result is not None and result.winner_pe > 0.0:
                        pe_records.append((step, result.winner_pe))

                obs = obs_next

        env.close()

        eval_pes   = [pe for (s, pe) in pe_records if s >= _EVAL_START_STEP]
        warmup_pes = [pe for (s, pe) in pe_records if s < _EVAL_START_STEP]

        results[variant] = {
            "mean_pe_eval":   round(float(np.mean(eval_pes))   if eval_pes   else 0.5, 4),
            "mean_pe_warmup": round(float(np.mean(warmup_pes)) if warmup_pes else 0.5, 4),
            "n_eval_records": len(eval_pes),
            "n_total_records": len(pe_records),
        }

    b = results["bundle"]
    e = results["episodic"]

    gate_better   = e["mean_pe_eval"] <= b["mean_pe_eval"]
    gate_absolute = e["mean_pe_eval"] <= 0.49

    passed = gate_better and gate_absolute

    return {
        "passed":  passed,
        "bundle":  b,
        "episodic": e,
        "gate_details": {
            f"episodic_pe({e['mean_pe_eval']:.4f}) <= bundle_pe({b['mean_pe_eval']:.4f})": gate_better,
            f"episodic_pe({e['mean_pe_eval']:.4f}) <= 0.49":                               gate_absolute,
        },
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)

    b = result["bundle"]
    e = result["episodic"]
    print(f"\n{'='*60}")
    print(f"Exp 33: HAC Episodic Buffer vs Bundle")
    print(f"{'='*60}")
    print(f"Bundle   (steps >= {_EVAL_START_STEP}): mean_pe={b['mean_pe_eval']:.4f}  "
          f"warmup_pe={b['mean_pe_warmup']:.4f}  n={b['n_eval_records']}")
    print(f"Episodic (steps >= {_EVAL_START_STEP}): mean_pe={e['mean_pe_eval']:.4f}  "
          f"warmup_pe={e['mean_pe_warmup']:.4f}  n={e['n_eval_records']}")
    print(f"\nGate details:")
    for k, v in result["gate_details"].items():
        mark = "PASS" if v else "FAIL"
        print(f"  [{mark}] {k}: {v}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    sys.exit(0 if result["passed"] else 1)
