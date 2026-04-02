"""Exp 105: VSA World Model experiments for Stage 45.

Sub-experiments:
  105a: VSA encoding accuracy (unbinding test)
  105b: SDM prediction accuracy (seen transitions)
  105c: WorldModelAgent on DoorKey-5x5 (primary gate)
  105d: Comparison with PureDafAgent baseline (3% random)

CPU experiments: 105a, 105b, 105c (small scale)
GPU experiments: 105c (full 200ep), 105d (deferred to minipc)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from snks.agent.vsa_world_model import (
    SDMMemory,
    VSACodebook,
    VSAEncoder,
    WorldModelAgent,
    WorldModelConfig,
)


# ──────────────────────────────────────────────
# Shared env helpers
# ──────────────────────────────────────────────

class DoorKeyEnv:
    """Simplified DoorKey-5x5: agent must pick up key, open door, reach goal.

    Layout (5x5 inner grid, 7x7 with walls):
      Agent at (1,1), Key at (1,3), Door at (2,3) or (3,1), Goal at (3,3).
      Actions: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done.
      Simplified to 4 movement actions for this env.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self.size = 5
        self.n_actions = 7
        self.max_steps = 200
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.agent_pos = [1, 1]
        self.agent_dir = 0  # 0=right, 1=down, 2=left, 3=up
        self.key_pos = [1, 3]
        self.has_key = False
        self.door_pos = [2, 2]
        self.door_open = False
        self.goal_pos = [3, 3]
        self.steps = 0
        self.key_picked = False
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = 0.0

        if action == 0:  # turn left
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:  # turn right
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:  # forward
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                # Check door blocking
                if [nr, nc] == self.door_pos and not self.door_open:
                    pass  # blocked
                else:
                    self.agent_pos = [nr, nc]
        elif action == 3:  # pickup
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                self.key_picked = True
        elif action == 5:  # toggle (open door)
            # Check if facing door
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if [fr, fc] == self.door_pos and self.has_key and not self.door_open:
                self.door_open = True

        # Check goal
        terminated = False
        if self.agent_pos == self.goal_pos:
            reward = 1.0 - 0.9 * (self.steps / self.max_steps)
            terminated = True

        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        # Walls around border
        for i in range(7):
            obs[0, i, 0] = 2
            obs[6, i, 0] = 2
            obs[i, 0, 0] = 2
            obs[i, 6, 0] = 2
        # Map inner coords to obs coords (+1 for wall border)
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir

        if not self.key_picked:
            kr, kc = self.key_pos[0] + 1, self.key_pos[1] + 1
            obs[kr, kc, 0] = 5
            obs[kr, kc, 1] = 1  # green

        dr, dc = self.door_pos[0] + 1, self.door_pos[1] + 1
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = 0 if self.door_open else 2

        gr, gc = self.goal_pos[0] + 1, self.goal_pos[1] + 1
        obs[gr, gc, 0] = 8

        # Has key indicator
        if self.has_key:
            obs[ar, ac, 1] = 5  # carrying

        return obs


# ──────────────────────────────────────────────
# Exp 105a: VSA encoding accuracy
# ──────────────────────────────────────────────

def run_exp105a(n_obs: int = 100) -> dict:
    """Test VSA encoding: encode obs, unbind each role, check filler recovery."""
    print("=== Exp 105a: VSA Encoding Accuracy ===")

    cb = VSACodebook(dim=512, seed=42)
    enc = VSAEncoder(cb)
    env = DoorKeyEnv(seed=42)

    correct = 0
    total = 0

    for i in range(n_obs):
        env.reset(seed=i)
        # Random actions to get diverse states
        for _ in range(np.random.randint(0, 20)):
            env.step(np.random.randint(0, 7))
        obs = env._obs()
        state = enc.encode(obs)

        # Try unbinding agent_pos
        ar, ac = env.agent_pos
        expected_filler = cb.filler(f"pos_{ar + 1}_{ac + 1}")
        recovered = cb.bind(state, cb.role("agent_pos"))
        sim = cb.similarity(recovered, expected_filler)

        if sim > 0.55:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    gate_pass = accuracy >= 0.90

    print(f"  Unbinding accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Gate (≥90%): {'PASS' if gate_pass else 'FAIL'}")
    return {"accuracy": accuracy, "correct": correct, "total": total, "gate": gate_pass}


# ──────────────────────────────────────────────
# Exp 105b: SDM prediction accuracy
# ──────────────────────────────────────────────

def run_exp105b(n_transitions: int = 500) -> dict:
    """Write transitions, read back predictions, measure similarity."""
    print("\n=== Exp 105b: SDM Prediction Accuracy ===")

    cb = VSACodebook(dim=512, seed=42)
    enc = VSAEncoder(cb)
    sdm = SDMMemory(n_locations=5000, dim=512, seed=42)
    env = DoorKeyEnv(seed=42)

    # Collect and write transitions
    transitions = []
    for ep in range(50):
        obs = env.reset(seed=ep)
        state = enc.encode(obs)
        for _ in range(20):
            action = np.random.randint(0, 7)
            obs, reward, term, trunc, _ = env.step(action)
            next_state = enc.encode(obs)
            action_vsa = cb.action(action)
            sdm.write(state, action_vsa, next_state, reward)
            transitions.append((state.clone(), action, next_state.clone()))
            state = next_state
            if term or trunc:
                break

    # Read back subset
    similarities = []
    sample_size = min(n_transitions, len(transitions))
    indices = np.random.RandomState(42).choice(len(transitions), sample_size, replace=False)
    for idx in indices:
        s, a, ns = transitions[idx]
        pred, conf = sdm.read_next(s, cb.action(a))
        if conf > 0:
            sim = cb.similarity(pred, ns)
            similarities.append(sim)

    mean_sim = np.mean(similarities) if similarities else 0.0
    gate_pass = mean_sim >= 0.6

    print(f"  Transitions written: {len(transitions)}")
    print(f"  Predictions tested: {len(similarities)}")
    print(f"  Mean similarity: {mean_sim:.3f}")
    print(f"  Gate (≥0.6): {'PASS' if gate_pass else 'FAIL'}")
    return {
        "n_transitions": len(transitions),
        "n_tested": len(similarities),
        "mean_similarity": mean_sim,
        "gate": gate_pass,
    }


# ──────────────────────────────────────────────
# Exp 105c: WorldModelAgent on DoorKey-5x5
# ──────────────────────────────────────────────

def run_exp105c(n_episodes: int = 200, use_minigrid: bool = False) -> dict:
    """WorldModelAgent on DoorKey-5x5 with causal planning.

    Phase 1 (explore): random actions, fill SDM with transitions
    Phase 2 (plan): forward beam search through SDM world model
    """
    print(f"\n=== Exp 105c: WorldModelAgent DoorKey-5x5 ({n_episodes} episodes) ===")

    explore_eps = 50
    config = WorldModelConfig(
        dim=512,
        n_locations=10000,
        n_actions=7,
        min_confidence=0.05,
        epsilon=0.15,
        max_episode_steps=200,
        explore_episodes=explore_eps,
        plan_depth=3,
        beam_width=3,
    )
    agent = WorldModelAgent(config)
    print(f"  Explore phase: {explore_eps} episodes (random)")
    print(f"  Plan phase: {n_episodes - explore_eps} episodes (beam search depth={config.plan_depth})")

    env = DoorKeyEnv(seed=42)

    results = []
    t0 = time.time()

    for ep in range(n_episodes):
        ep_t0 = time.time()

        obs = env.reset(seed=ep % 50)  # recycle seeds for learning

        success, steps, reward = agent.run_episode(
            env, max_steps=config.max_episode_steps,
        )
        results.append({"success": success, "steps": steps, "reward": reward})

        ep_time = time.time() - ep_t0
        phase = "EXPLORE" if ep < explore_eps else "PLAN"
        if (ep + 1) % 10 == 0 or ep == 0:
            recent = results[max(0, ep - 9):]
            sr = sum(1 for r in recent if r["success"]) / len(recent)
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
            print(
                f"  Ep {ep + 1}/{n_episodes} [{phase}]: "
                f"success_rate(10)={sr:.1%}, "
                f"steps={steps}, reward={reward:.3f}, "
                f"SDM writes={agent.sdm.n_writes}, "
                f"time={ep_time:.1f}s | ETA {eta:.0f}s",
                flush=True,
            )

    total_success = sum(1 for r in results if r["success"])
    success_rate = total_success / n_episodes
    # Plan phase only (after explore)
    plan_results = results[explore_eps:]
    plan_success = sum(1 for r in plan_results if r["success"])
    plan_rate = plan_success / len(plan_results) if plan_results else 0.0
    explore_success = sum(1 for r in results[:explore_eps] if r["success"])

    # Learning trend: first half of plan vs last half of plan
    plan_first = plan_results[:len(plan_results) // 2]
    plan_last = plan_results[len(plan_results) // 2:]
    first_rate = sum(1 for r in plan_first if r["success"]) / max(len(plan_first), 1)
    last_rate = sum(1 for r in plan_last if r["success"]) / max(len(plan_last), 1)

    gate_primary = plan_rate >= 0.15
    gate_stretch = plan_rate >= 0.30
    gate_learning = last_rate > first_rate

    elapsed = time.time() - t0
    print(f"\n  Explore phase: {explore_success}/{explore_eps} = {explore_success / explore_eps:.1%}")
    print(f"  Plan phase: {plan_success}/{len(plan_results)} = {plan_rate:.1%}")
    print(f"  Plan first half: {first_rate:.1%}")
    print(f"  Plan last half: {last_rate:.1%}")
    print(f"  Learning trend: {'YES' if gate_learning else 'NO'}")
    print(f"  Gate primary (≥15% plan phase): {'PASS' if gate_primary else 'FAIL'}")
    print(f"  Gate stretch (≥30% plan phase): {'PASS' if gate_stretch else 'FAIL'}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "success_rate": success_rate,
        "plan_rate": plan_rate,
        "total_success": total_success,
        "plan_success": plan_success,
        "n_episodes": n_episodes,
        "explore_success": explore_success,
        "first_plan_rate": first_rate,
        "last_plan_rate": last_rate,
        "gate_primary": gate_primary,
        "gate_stretch": gate_stretch,
        "gate_learning": gate_learning,
        "elapsed_s": elapsed,
    }


class _SymbolicEnvWrapper:
    """Wraps MiniGridAdapter to return symbolic obs for WorldModelAgent."""
    def __init__(self, adapter):
        self._adapter = adapter

    def reset(self, seed=None):
        self._adapter.reset(seed=seed)
        return self._adapter.get_symbolic_obs()

    def step(self, action):
        _, reward, term, trunc, info = self._adapter.step(action)
        return self._adapter.get_symbolic_obs(), reward, term, trunc, info


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    import json

    print("=" * 60)
    print("  Exp 105: VSA World Model — Stage 45")
    print("=" * 60)

    results = {}

    # 105a: encoding accuracy
    results["105a"] = run_exp105a(n_obs=100)

    # 105b: SDM prediction
    results["105b"] = run_exp105b(n_transitions=500)

    # 105c: DoorKey-5x5
    results["105c"] = run_exp105c(n_episodes=200)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, res in results.items():
        if "gate" in res:
            status = "PASS" if res["gate"] else "FAIL"
            if not res["gate"]:
                all_pass = False
        elif "gate_primary" in res:
            status = "PASS" if res["gate_primary"] else "FAIL"
            if not res["gate_primary"]:
                all_pass = False
        print(f"  {name}: {status}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Save results
    os.makedirs("_docs", exist_ok=True)
    with open("_docs/exp105_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to _docs/exp105_results.json")

    return results


if __name__ == "__main__":
    main()
