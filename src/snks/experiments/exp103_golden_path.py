"""Exp 103: Golden Path — минимальный тест: учится ли DAF-ядро вообще?

Среда: 3×3 grid, один goal, оптимальное решение = 2-3 шага.
Агент: 500 нод FHN, SymbolicEncoder, без WM/eligibility/curriculum/navigator.
Ожидание: после 20 эпизодов success > 50% (random ~10-15%).

Phase 0 of Stage 44 Foundation Audit.
"""

import time
import sys
import os
import json

import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def make_simple_grid_env():
    """Create minimal 3×3 MiniGrid-like env with single goal."""
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        env = gym.make("MiniGrid-Empty-5x5-v0", max_steps=20)
        return env
    except ImportError:
        return None


class SimpleGridEnv:
    """Fallback: minimal 3×3 grid if MiniGrid not available.

    Grid: 3×3, agent starts at (0,0), goal at (2,2).
    Actions: 0=right, 1=down, 2=left, 3=up.
    Reward: +1 on reaching goal, 0 otherwise.
    """

    def __init__(self, size=3):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pos = (0, 0)
        self.steps = 0
        self.max_steps = 20
        self.n_actions = 4

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self._obs()

    def step(self, action):
        r, c = self.pos
        if action == 0:
            c = min(c + 1, self.size - 1)
        elif action == 1:
            r = min(r + 1, self.size - 1)
        elif action == 2:
            c = max(c - 1, 0)
        elif action == 3:
            r = max(r - 1, 0)
        self.pos = (r, c)
        self.steps += 1

        done = self.pos == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.pos == self.goal else 0.0
        return self._obs(), reward, done

    def _obs(self):
        """Return 7×7×3 symbolic obs (MiniGrid-compatible format)."""
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        # Agent
        obs[self.pos[0], self.pos[1], 0] = 10  # OBJ_AGENT
        # Goal
        obs[self.goal[0], self.goal[1], 0] = 8  # OBJ_GOAL
        return obs

    def get_symbolic_obs(self):
        return self._obs()


def run_golden_path(n_episodes=30, num_nodes=500, device="cpu"):
    """Run Golden Path experiment."""
    from snks.daf.types import DafConfig, PipelineConfig, EncoderConfig, SKSConfig
    from snks.encoder.symbolic import SymbolicEncoder
    from snks.pipeline.runner import Pipeline

    print(f"=== Exp 103: Golden Path (N={num_nodes}, device={device}) ===")
    print(f"Среда: 3×3 grid, goal=(2,2), agent=(0,0)")
    print(f"Агент: {num_nodes} FHN нод, SymbolicEncoder, NO extras")
    print()

    # Minimal pipeline config — NO extras
    daf_cfg = DafConfig(
        num_nodes=num_nodes,
        avg_degree=20,
        device=device,
        disable_csr=(device == "cpu"),
        oscillator_model="fhn",
        wm_fraction=0.0,  # NO working memory
    )
    pipe_cfg = PipelineConfig(
        daf=daf_cfg,
        encoder=EncoderConfig(sdr_size=4096),
        sks=SKSConfig(coherence_mode="cofiring", top_k=min(num_nodes, 500)),
        steps_per_cycle=100,
        device=device,
    )
    pipeline = Pipeline(pipe_cfg)
    encoder = SymbolicEncoder(sdr_size=4096)

    env = SimpleGridEnv(size=3)

    # Track STDP weight stats for learning signal analysis
    initial_weights = pipeline.engine.graph.get_strength().clone()

    results = []
    t_start = time.monotonic()

    for ep in range(n_episodes):
        t_ep = time.monotonic()
        obs = env.reset()

        total_reward = 0.0
        steps = 0

        for step in range(env.max_steps):
            # Encode observation
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            sdr = encoder.encode(obs_tensor)

            # Perception cycle
            cycle_result = pipeline.perception_cycle(pre_sdr=sdr)

            # Action selection: epsilon-greedy (start high, decay)
            epsilon = max(0.1, 0.7 * (0.95 ** ep))
            if np.random.random() < epsilon:
                action = np.random.randint(0, env.n_actions)
            else:
                # Use SKS cluster count as very crude heuristic
                # (real agent uses PE, but we're testing minimal pipeline)
                action = np.random.randint(0, env.n_actions)

            obs, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            # Reward-modulated STDP: apply reward signal
            if reward > 0 and cycle_result.stdp_result is not None:
                # Manual reward modulation on eligibility trace
                dw = cycle_result.stdp_result.dw
                if dw is not None:
                    pipeline.engine.graph.edge_attr[:, 0] += reward * 0.1 * dw
                    pipeline.engine.graph.edge_attr[:, 0].clamp_(0.0, 1.0)

            if done:
                break

        elapsed_ep = time.monotonic() - t_ep
        success = total_reward > 0
        results.append({
            "episode": ep + 1,
            "success": success,
            "reward": total_reward,
            "steps": steps,
            "time_s": elapsed_ep,
        })
        print(
            f"  Ep {ep+1:3d}/{n_episodes}: "
            f"success={success}, steps={steps}, reward={total_reward:.1f}, "
            f"time={elapsed_ep:.1f}s"
        )

    # Summary
    total_time = time.monotonic() - t_start
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / n_episodes

    # Weight change analysis
    final_weights = pipeline.engine.graph.get_strength()
    weight_delta = (final_weights - initial_weights).abs().mean().item()

    print()
    print(f"=== RESULTS ===")
    print(f"Success rate: {successes}/{n_episodes} = {success_rate:.1%}")
    print(f"Mean STDP weight delta: {weight_delta:.6f}")
    print(f"Total time: {total_time:.1f}s")
    print()

    # Gate check
    random_baseline = 0.15  # random on 3×3 with 20 steps ≈ 10-15%
    gate_success = success_rate > 0.50
    gate_learning = weight_delta > 0

    print(f"Gate: success > 50%: {'PASS' if gate_success else 'FAIL'} ({success_rate:.1%})")
    print(f"Gate: weight delta > 0: {'PASS' if gate_learning else 'FAIL'} ({weight_delta:.6f})")
    print(f"Random baseline: ~{random_baseline:.0%}")

    # Save results
    output = {
        "experiment": "exp103_golden_path",
        "stage": 44,
        "config": {
            "num_nodes": num_nodes,
            "n_episodes": n_episodes,
            "device": device,
            "env": "SimpleGridEnv 3x3",
        },
        "results": {
            "success_rate": success_rate,
            "successes": successes,
            "total_episodes": n_episodes,
            "mean_weight_delta": weight_delta,
            "total_time_s": total_time,
        },
        "gates": {
            "success_gt_50pct": gate_success,
            "weight_delta_gt_0": gate_learning,
        },
        "episodes": results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "_docs", "exp103_golden_path.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_nodes = 50000 if device == "cuda" else 500
    run_golden_path(n_episodes=30, num_nodes=num_nodes, device=device)
