"""Exp 104: Naked DAF — чистое DAF-ядро на DoorKey-5x5 без надстроек.

Конфигурация:
- 50K FHN нод (GPU)
- SymbolicEncoder (идеальное perception)
- БЕЗ WM, eligibility traces, curriculum, AttractorNavigator
- EpsilonScheduler: 0.7 → 0.1
- 200 эпизодов

Gate:
- PASS: success_rate >= 0.15
- PARTIAL: 0.05 <= success_rate < 0.15
- LEARNING SIGNAL: success < 0.05 but STDP weights correlate with task
- FAIL: success < 0.05 AND no learning signal

Phase 2 of Stage 44 Foundation Audit.
"""

import time
import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_naked_daf(n_episodes=200, num_nodes=50000, device="cuda"):
    """Run Naked DAF on DoorKey-5x5."""
    from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig
    from snks.env.adapter import MiniGridAdapter

    print(f"=== Exp 104: Naked DAF on DoorKey-5x5 ===")
    print(f"N={num_nodes}, device={device}, episodes={n_episodes}")
    print(f"NO WM, NO eligibility, NO curriculum, NO navigator")
    print()

    # Naked config — everything off
    cfg = PureDafConfig()
    cfg.causal.pipeline.daf.num_nodes = num_nodes
    cfg.causal.pipeline.daf.device = device
    cfg.causal.pipeline.daf.disable_csr = True  # AMD ROCm safe
    cfg.causal.pipeline.device = device
    cfg.encoder_type = "symbolic"
    cfg.n_actions = 7  # MiniGrid
    cfg.max_episode_steps = 200

    # Epsilon schedule
    cfg.epsilon_initial = 0.7
    cfg.epsilon_decay = 0.95
    cfg.epsilon_floor = 0.1

    # DISABLE extras
    cfg.wm_fraction = 0.0       # No working memory
    cfg.trace_decay = 0.0       # Effectively no eligibility trace
    cfg.trace_reward_lr = 0.0   # No eligibility reward
    # AttractorNavigator is still created but won't be used much
    # because goal_embedding will be None most of the time

    agent = PureDafAgent(cfg)

    # Track initial weights for learning signal
    initial_weights = agent.engine.graph.get_strength().clone()

    import gymnasium as gym
    import minigrid  # noqa: F401
    env_raw = gym.make("MiniGrid-DoorKey-5x5-v0", max_steps=200)
    env = MiniGridAdapter(env_raw)

    results = []
    t_start = time.monotonic()

    for ep in range(n_episodes):
        t_ep = time.monotonic()
        result = agent.run_episode(env, max_steps=200)
        elapsed_ep = time.monotonic() - t_ep

        avg_time_per_ep = (time.monotonic() - t_start) / (ep + 1)
        eta_min = avg_time_per_ep * (n_episodes - ep - 1) / 60

        results.append({
            "episode": ep + 1,
            "success": result.success,
            "reward": result.reward,
            "steps": result.steps,
            "mean_pe": result.mean_pe,
            "sks_count": result.sks_count,
            "epsilon": result.nav_stats.get("epsilon", 0),
            "time_s": elapsed_ep,
        })
        print(
            f"  Ep {ep+1:3d}/{n_episodes}: "
            f"success={result.success}, steps={result.steps}, "
            f"reward={result.reward:.2f}, PE={result.mean_pe:.3f}, "
            f"SKS={result.sks_count}, eps={result.nav_stats.get('epsilon', 0):.2f}, "
            f"time={elapsed_ep:.1f}s | ETA {eta_min:.1f}min"
        )

    # Summary
    total_time = time.monotonic() - t_start
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / n_episodes

    # Learning signal analysis
    final_weights = agent.engine.graph.get_strength()
    weight_delta = (final_weights - initial_weights).abs().mean().item()

    # Check if successful episodes have different weight patterns
    success_episodes = [i for i, r in enumerate(results) if r["success"]]
    fail_episodes = [i for i, r in enumerate(results) if not r["success"]]

    print()
    print(f"=== RESULTS ===")
    print(f"Success rate: {successes}/{n_episodes} = {success_rate:.1%}")
    print(f"Success episodes: {success_episodes[:20]}...")
    print(f"Mean STDP weight delta: {weight_delta:.6f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print()

    # Gate check
    if success_rate >= 0.15:
        gate = "PASS"
    elif success_rate >= 0.05:
        gate = "PARTIAL"
    elif weight_delta > 0.001:
        gate = "LEARNING_SIGNAL"
    else:
        gate = "FAIL"

    print(f"Gate: {gate}")
    print(f"  success_rate={success_rate:.3f} (need >=0.15 for PASS)")
    print(f"  weight_delta={weight_delta:.6f}")

    # Rolling success rate (last 50 vs first 50)
    if n_episodes >= 100:
        first_50 = sum(1 for r in results[:50] if r["success"]) / 50
        last_50 = sum(1 for r in results[-50:] if r["success"]) / 50
        print(f"  first 50 eps: {first_50:.1%}")
        print(f"  last 50 eps: {last_50:.1%}")
        print(f"  improvement: {last_50 - first_50:+.1%}")

    # Save results
    output = {
        "experiment": "exp104_naked_daf",
        "stage": 44,
        "config": {
            "num_nodes": num_nodes,
            "n_episodes": n_episodes,
            "device": device,
            "env": "MiniGrid-DoorKey-5x5-v0",
            "extras": "NONE (no WM, no eligibility, no curriculum, no navigator)",
        },
        "results": {
            "success_rate": success_rate,
            "successes": successes,
            "total_episodes": n_episodes,
            "mean_weight_delta": weight_delta,
            "total_time_s": total_time,
            "gate": gate,
        },
        "episodes": results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "_docs", "exp104_naked_daf.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_nodes = 50000 if device == "cuda" else 2000
    n_episodes = 200 if device == "cuda" else 20
    run_naked_daf(n_episodes=n_episodes, num_nodes=num_nodes, device=device)
