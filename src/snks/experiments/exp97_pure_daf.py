"""Experiment 97: Pure DAF Agent (Stage 38).

Tests PureDafAgent — ONLY DAF pipeline, no scaffolding.
Compares against random baseline to prove DAF learns.

Gates:
    exp97a: DoorKey-5x5 success_rate >= 0.10 (pure DAF, no grid access)
    exp97b: causal_modulations > 0 (STDP weights modulated by reward)
    exp97c: Empty-8x8 navigation >= 0.30 (navigate to visible goal)
    exp97d: env_agnostic runs without error on multiple envs
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig


def _detect_device() -> str:
    """Auto-detect best device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_config(n_actions: int = 7, small: bool | None = None) -> PureDafConfig:
    """Create config for experiments.

    small=None: auto-detect (GPU → full, CPU → small)
    small=True: 1000 nodes, CPU
    small=False: 50K nodes, GPU
    """
    cfg = PureDafConfig()
    cfg.n_actions = n_actions
    cfg.max_episode_steps = 100

    if small is None:
        small = _detect_device() == "cpu"

    if small:
        cfg.causal.pipeline.daf.num_nodes = 2000
        cfg.causal.pipeline.daf.avg_degree = 15
        cfg.causal.pipeline.daf.device = "cpu"
        cfg.causal.pipeline.daf.disable_csr = True
        # Faster dt for CPU: 50x speedup, FHN still stable
        cfg.causal.pipeline.daf.dt = 0.005
        cfg.causal.pipeline.steps_per_cycle = 200
        cfg.causal.pipeline.encoder.image_size = 32
        cfg.causal.pipeline.encoder.pool_h = 5
        cfg.causal.pipeline.encoder.pool_w = 5
        cfg.causal.pipeline.encoder.n_orientations = 4
        cfg.causal.pipeline.encoder.n_frequencies = 2
        # n_filters = 64, actual sdr = 64 * 5 * 5 = 1600
        cfg.causal.pipeline.encoder.sdr_size = 1600
        cfg.causal.pipeline.sks.min_cluster_size = 3
        cfg.causal.pipeline.sks.coherence_mode = "cofiring"
        cfg.causal.pipeline.sks.top_k = 200
        cfg.causal.motor_sdr_size = 200
        # Lower I_base so only SDR-driven nodes spike
        cfg.causal.pipeline.daf.fhn_I_base = 0.3
        cfg.causal.pipeline.daf.coupling_strength = 0.05
    else:
        # Full GPU config: 50K nodes
        cfg.causal.pipeline.daf.device = "auto"
        cfg.causal.pipeline.daf.disable_csr = True  # avoid HIP issues on AMD
        # dt=0.001, 500 steps → 0.5s model time (enough for FHN spiking)
        cfg.causal.pipeline.daf.dt = 0.001
        cfg.causal.pipeline.steps_per_cycle = 500
        cfg.causal.pipeline.sks.coherence_mode = "cofiring"
        cfg.causal.pipeline.sks.top_k = 2000
        cfg.causal.pipeline.daf.fhn_I_base = 0.3
        cfg.causal.pipeline.daf.coupling_strength = 0.05

    cfg.exploration_epsilon = 0.7 if small else 0.3
    cfg.reward_scale = 3.0
    cfg.trace_length = 5
    cfg.n_sim_steps = 3 if small else 10
    return cfg


def _run_random_baseline(env, n_episodes: int, max_steps: int) -> float:
    """Run random agent for baseline comparison."""
    successes = 0
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = np.random.randint(0, env.n_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        if total_reward > 0:
            successes += 1
    return successes / n_episodes


def exp97a_doorkey():
    """Exp 97a: PureDafAgent on DoorKey-5x5.

    Gate: success_rate >= 0.10 (random baseline ~0.02)
    """
    print("\n--- Exp 97a: PureDafAgent on DoorKey-5x5 ---")

    try:
        from snks.env.adapter import MiniGridAdapter
        env = MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed")
        return {"status": "SKIP", "reason": "MiniGrid not installed"}

    cfg = _make_config(n_actions=env.n_actions)
    is_cpu = cfg.causal.pipeline.daf.device == "cpu" or not __import__('torch').cuda.is_available()
    n_episodes = 10 if is_cpu else 50
    max_steps = 100 if is_cpu else 500
    cfg.max_episode_steps = max_steps
    agent = PureDafAgent(cfg)

    results = agent.run_training(env, n_episodes=n_episodes, max_steps=max_steps)

    successes = sum(1 for r in results if r.success)
    success_rate = successes / n_episodes
    mean_steps = np.mean([r.steps for r in results])
    mean_pe = np.mean([r.mean_pe for r in results])
    causal_stats = results[-1].causal_stats if results else {}

    # Random baseline
    random_rate = _run_random_baseline(env, n_episodes, max_steps)

    print(f"  Pure DAF:  success={success_rate:.3f} ({successes}/{n_episodes})")
    print(f"  Random:    success={random_rate:.3f}")
    print(f"  Mean steps: {mean_steps:.1f}")
    print(f"  Mean PE:   {mean_pe:.3f}")
    print(f"  Causal:    {causal_stats}")
    print(f"  Gate (>=0.10): {'PASS' if success_rate >= 0.10 else 'FAIL'}")

    return {
        "success_rate": success_rate,
        "random_baseline": random_rate,
        "mean_steps": float(mean_steps),
        "mean_pe": float(mean_pe),
        "causal_stats": causal_stats,
        "gate_pass": success_rate >= 0.10,
    }


def exp97b_causal_learning():
    """Exp 97b: Verify STDP weight modulation happens on reward.

    Gate: causal_modulations > 0
    """
    print("\n--- Exp 97b: Causal Learning via Reward-Modulated STDP ---")

    try:
        from snks.env.adapter import MiniGridAdapter
        env = MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed")
        return {"status": "SKIP", "reason": "MiniGrid not installed"}

    cfg = _make_config(n_actions=env.n_actions)
    is_cpu = cfg.causal.pipeline.daf.device == "cpu" or not __import__('torch').cuda.is_available()
    n_episodes = 10 if is_cpu else 30
    max_steps = 100 if is_cpu else 500
    cfg.max_episode_steps = max_steps
    agent = PureDafAgent(cfg)

    results = agent.run_training(env, n_episodes=n_episodes, max_steps=max_steps)
    causal_stats = agent._causal.stats

    modulations = causal_stats.get("total_modulations", 0)
    total_reward = causal_stats.get("total_reward", 0.0)

    print(f"  Modulations: {modulations}")
    print(f"  Total reward received: {total_reward:.3f}")
    print(f"  Gate (modulations > 0): {'PASS' if modulations > 0 else 'FAIL'}")

    return {
        "modulations": modulations,
        "total_reward": total_reward,
        "gate_pass": modulations > 0,
    }


def exp97c_empty_navigation():
    """Exp 97c: Navigate to goal in Empty-8x8.

    Gate: success_rate >= 0.30 (simpler env, should be easier)
    """
    print("\n--- Exp 97c: PureDafAgent on Empty-8x8 ---")

    try:
        from snks.env.adapter import MiniGridAdapter
        env = MiniGridAdapter("MiniGrid-Empty-8x8-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed")
        return {"status": "SKIP", "reason": "MiniGrid not installed"}

    cfg = _make_config(n_actions=env.n_actions)
    is_cpu = cfg.causal.pipeline.daf.device == "cpu" or not __import__('torch').cuda.is_available()
    n_episodes = 10 if is_cpu else 50
    max_steps = 100 if is_cpu else 200
    cfg.max_episode_steps = max_steps
    agent = PureDafAgent(cfg)

    results = agent.run_training(env, n_episodes=n_episodes, max_steps=max_steps)

    successes = sum(1 for r in results if r.success)
    success_rate = successes / n_episodes
    random_rate = _run_random_baseline(env, n_episodes, max_steps)

    print(f"  Pure DAF:  success={success_rate:.3f} ({successes}/{n_episodes})")
    print(f"  Random:    success={random_rate:.3f}")
    print(f"  Gate (>=0.30): {'PASS' if success_rate >= 0.30 else 'FAIL'}")

    return {
        "success_rate": success_rate,
        "random_baseline": random_rate,
        "gate_pass": success_rate >= 0.30,
    }


def exp97d_env_agnostic():
    """Exp 97d: Same agent runs on multiple env types without error.

    Gate: all runs complete without exception
    """
    print("\n--- Exp 97d: Environment-Agnostic Test ---")

    from snks.env.adapter import ArrayEnvAdapter

    # Test 1: Array-based dummy env
    class CounterEnv:
        def __init__(self):
            self._count = 0
        def reset(self, seed=None):
            self._count = 0
            return np.zeros(16)
        def step(self, action):
            self._count += 1
            state = np.full(16, self._count / 20.0)
            reward = 1.0 if self._count >= 10 else 0.0
            done = self._count >= 10
            return state, reward, done, False, {}

    env1 = ArrayEnvAdapter(CounterEnv(), n_actions=3, name="counter")
    cfg = _make_config(n_actions=3)
    cfg.max_episode_steps = 15
    agent = PureDafAgent(cfg)

    errors = []
    try:
        result1 = agent.run_episode(env1, max_steps=15)
        print(f"  CounterEnv: steps={result1.steps} reward={result1.reward:.1f} OK")
    except Exception as e:
        errors.append(f"CounterEnv: {e}")
        print(f"  CounterEnv: ERROR {e}")

    # Test 2: MiniGrid if available
    try:
        from snks.env.adapter import MiniGridAdapter
        env2 = MiniGridAdapter("MiniGrid-Empty-5x5-v0")
        cfg2 = _make_config(n_actions=env2.n_actions)
        cfg2.max_episode_steps = 20
        agent2 = PureDafAgent(cfg2)
        result2 = agent2.run_episode(env2, max_steps=20)
        print(f"  MiniGrid:   steps={result2.steps} reward={result2.reward:.1f} OK")
    except ImportError as e:
        print(f"  MiniGrid:   SKIP (not installed: {e})")
    except Exception as e:
        errors.append(f"MiniGrid: {e}")
        print(f"  MiniGrid:   ERROR {e}")

    gate_pass = len(errors) == 0
    print(f"  Gate (no errors): {'PASS' if gate_pass else 'FAIL'}")

    return {
        "errors": errors,
        "gate_pass": gate_pass,
    }


def main():
    print("=" * 60)
    print("Experiment 97: Pure DAF Agent (Stage 38)")
    print("=" * 60)
    print("NO scaffolding: no GridPerception, no BFS, no hardcoded SKS")
    print("ONLY DAF: oscillators → STDP → coherence → action")
    print()

    results = {}

    # Run all sub-experiments
    results["97d"] = exp97d_env_agnostic()
    results["97b"] = exp97b_causal_learning()
    results["97a"] = exp97a_doorkey()
    results["97c"] = exp97c_empty_navigation()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, res in results.items():
        status = res.get("status", "")
        if status == "SKIP":
            print(f"  Exp {key}: SKIP — {res.get('reason', '')}")
        else:
            gate = "PASS" if res.get("gate_pass", False) else "FAIL"
            print(f"  Exp {key}: {gate}")

    all_pass = all(
        r.get("gate_pass", False) or r.get("status") == "SKIP"
        for r in results.values()
    )
    print(f"\nOverall: {'ALL GATES PASS' if all_pass else 'SOME GATES FAIL'}")
    return results


if __name__ == "__main__":
    main()
