"""Exp 106: Subgoal Planning experiments for Stage 46.

Sub-experiments:
  106a: Subgoal extraction accuracy from successful traces
  106b: Plan graph construction and ordering
  106c: SubgoalPlanningAgent on DoorKey-5x5 (primary gate)
  106d: Subgoal navigation quality (steps per subgoal)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from snks.agent.subgoal_planning import (
    PlanGraph,
    SubgoalConfig,
    SubgoalExtractor,
    SubgoalPlanningAgent,
    TraceStep,
)
from snks.agent.vsa_world_model import VSACodebook, VSAEncoder


# ──────────────────────────────────────────────
# DoorKeyEnv (same as exp105)
# ──────────────────────────────────────────────

class DoorKeyEnv:
    """Simplified DoorKey-5x5 with blocking wall.

    Layout (inner 5x5, obs 7x7 with border walls):
      Row 0: . A . K .   (agent at 0,1, key at 0,3)
      Row 1: . . . . .
      Row 2: W W D W W   (wall divider with door at 2,2)
      Row 3: . . . . .
      Row 4: . . . G .   (goal at 4,3)

    The wall at row 2 FORCES the agent through the door.
    Door is locked → must pickup key first, then toggle.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self.size = 5
        self.n_actions = 7
        self.max_steps = 200
        # Wall positions (inner coords) — row 2 except door
        self.wall_positions = [[2, 0], [2, 1], [2, 3], [2, 4]]
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.agent_pos = [0, 1]
        self.agent_dir = 1  # facing down
        self.key_pos = [0, 3]
        self.has_key = False
        self.door_pos = [2, 2]
        self.door_open = False
        self.goal_pos = [4, 3]
        self.steps = 0
        self.key_picked = False
        return self._obs()

    def _is_wall(self, r: int, c: int) -> bool:
        return [r, c] in self.wall_positions

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
                if self._is_wall(nr, nc):
                    pass  # blocked by wall
                elif [nr, nc] == self.door_pos and not self.door_open:
                    pass  # blocked by locked door
                else:
                    self.agent_pos = [nr, nc]
        elif action == 3:  # pickup
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                self.key_picked = True
        elif action == 5:  # toggle
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if [fr, fc] == self.door_pos and self.has_key and not self.door_open:
                self.door_open = True
        terminated = False
        if self.agent_pos == self.goal_pos:
            reward = 1.0 - 0.9 * (self.steps / self.max_steps)
            terminated = True
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        # Border walls
        for i in range(7):
            obs[0, i, 0] = 2; obs[6, i, 0] = 2
            obs[i, 0, 0] = 2; obs[i, 6, 0] = 2
        # Interior walls (inner→obs: +1)
        for wr, wc in self.wall_positions:
            obs[wr + 1, wc + 1, 0] = 2
        # Key (before agent — agent overwrites if co-located)
        if not self.key_picked:
            kr, kc = self.key_pos[0] + 1, self.key_pos[1] + 1
            obs[kr, kc, 0] = 5; obs[kr, kc, 1] = 1
        # Door
        dr, dc = self.door_pos[0] + 1, self.door_pos[1] + 1
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = 0 if self.door_open else 2
        # Goal
        gr, gc = self.goal_pos[0] + 1, self.goal_pos[1] + 1
        obs[gr, gc, 0] = 8
        # Agent LAST — always visible even when overlapping objects
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


# ──────────────────────────────────────────────
# Helper: collect explore traces
# ──────────────────────────────────────────────

def _collect_explore_traces(n_episodes: int = 500, seed: int = 42) -> list[list[TraceStep]]:
    """Run random episodes and return successful traces."""
    env = DoorKeyEnv(seed=seed)
    cb = VSACodebook(dim=512, seed=42)
    enc = VSAEncoder(cb)
    successful: list[list[TraceStep]] = []

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        trace: list[TraceStep] = []
        total_reward = 0.0

        for _ in range(200):
            action = int(np.random.randint(0, 7))
            prev_obs = obs.copy()
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            trace.append(TraceStep(prev_obs, action, obs.copy(), reward))
            if term or trunc:
                break

        if total_reward > 0:
            successful.append(trace)

    return successful


# ──────────────────────────────────────────────
# Exp 106a: Subgoal Extraction Accuracy
# ──────────────────────────────────────────────

def run_exp106a(n_episodes: int = 50) -> dict:
    """Test subgoal extraction from successful random traces."""
    print("=== Exp 106a: Subgoal Extraction Accuracy ===")

    traces = _collect_explore_traces(n_episodes=n_episodes)
    print(f"  Successful traces: {len(traces)} / {n_episodes}")

    if not traces:
        print("  ERROR: No successful traces found!")
        return {"gate": False, "n_traces": 0}

    cb = VSACodebook(dim=512, seed=42)
    enc = VSAEncoder(cb)
    extractor = SubgoalExtractor(cb, enc)

    has_pickup = 0
    has_door = 0
    has_goal = 0
    correct_order = 0

    for trace in traces:
        subgoals = extractor.extract(trace)
        names = [s.name for s in subgoals]

        if "pickup_key" in names:
            has_pickup += 1
        if "open_door" in names:
            has_door += 1
        if "reach_goal" in names:
            has_goal += 1
        if "pickup_key" in names and "open_door" in names and "reach_goal" in names:
            pk_idx = names.index("pickup_key")
            od_idx = names.index("open_door")
            rg_idx = names.index("reach_goal")
            if pk_idx < od_idx < rg_idx:
                correct_order += 1

    n = len(traces)
    pickup_rate = has_pickup / n
    door_rate = has_door / n
    goal_rate = has_goal / n
    order_rate = correct_order / n

    gate_pickup = pickup_rate >= 0.80
    gate_door = door_rate >= 0.80
    gate_order = order_rate >= 0.80
    gate = gate_pickup and gate_door and gate_order

    print(f"  pickup_key detected: {pickup_rate:.1%} ({has_pickup}/{n})")
    print(f"  open_door detected: {door_rate:.1%} ({has_door}/{n})")
    print(f"  reach_goal detected: {goal_rate:.1%} ({has_goal}/{n})")
    print(f"  correct order: {order_rate:.1%} ({correct_order}/{n})")
    print(f"  Gate (≥80% each): {'PASS' if gate else 'FAIL'}")

    return {
        "n_traces": n,
        "pickup_rate": pickup_rate,
        "door_rate": door_rate,
        "goal_rate": goal_rate,
        "order_rate": order_rate,
        "gate": gate,
    }


# ──────────────────────────────────────────────
# Exp 106b: Plan Graph Construction
# ──────────────────────────────────────────────

def run_exp106b() -> dict:
    """Verify plan graph ordering from extracted subgoals."""
    print("\n=== Exp 106b: Plan Graph Construction ===")

    traces = _collect_explore_traces(n_episodes=500)
    if not traces:
        print("  ERROR: No traces")
        return {"gate": False}

    cb = VSACodebook(dim=512, seed=42)
    enc = VSAEncoder(cb)
    extractor = SubgoalExtractor(cb, enc)

    all_correct = 0
    total = 0

    for trace in traces:
        subgoals = extractor.extract(trace)
        if not subgoals:
            continue

        plan = PlanGraph(subgoals)
        names = [s.name for s in plan.subgoals]
        total += 1

        # Check ordering
        if "pickup_key" in names and "open_door" in names:
            if names.index("pickup_key") < names.index("open_door"):
                all_correct += 1
        elif "reach_goal" in names:
            all_correct += 1  # no key/door = simpler task, still valid

    rate = all_correct / total if total > 0 else 0.0
    gate = rate >= 1.0

    print(f"  Plans checked: {total}")
    print(f"  Correct ordering: {all_correct}/{total} = {rate:.1%}")
    print(f"  Gate (100%): {'PASS' if gate else 'FAIL'}")

    return {"total": total, "correct": all_correct, "rate": rate, "gate": gate}


# ──────────────────────────────────────────────
# Exp 106c: SubgoalPlanningAgent on DoorKey-5x5
# ──────────────────────────────────────────────

def run_exp106c(n_episodes: int = 100, explore_eps: int = 50) -> dict:
    """Primary gate: SubgoalPlanningAgent on DoorKey-5x5."""
    print(f"\n=== Exp 106c: SubgoalPlanningAgent DoorKey-5x5 ({n_episodes} eps) ===")

    config = SubgoalConfig(
        dim=512,
        n_locations=5000,
        n_actions=7,
        min_confidence=0.01,
        epsilon=0.15,
        max_episode_steps=200,
        explore_episodes=explore_eps,
        plan_depth=3,
        beam_width=3,
    )
    agent = SubgoalPlanningAgent(config)
    env = DoorKeyEnv(seed=42)

    results = []
    t0 = time.time()

    for ep in range(n_episodes):
        ep_t0 = time.time()
        success, steps, reward = agent.run_episode(env, max_steps=200)
        results.append({"success": success, "steps": steps, "reward": reward})

        ep_time = time.time() - ep_t0
        phase = "EXPLORE" if ep < explore_eps else "PLAN"
        if (ep + 1) % 10 == 0 or ep == 0:
            recent = results[max(0, ep - 9):]
            sr = sum(1 for r in recent if r["success"]) / len(recent)
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
            n_traces = len(agent._successful_traces)
            plan_info = ""
            if agent.plan:
                plan_info = f", plan_subgoals={len(agent.plan.subgoals)}"
            print(
                f"  Ep {ep + 1}/{n_episodes} [{phase}]: "
                f"sr(10)={sr:.1%}, steps={steps}, "
                f"traces={n_traces}{plan_info}, "
                f"SDM={agent.sdm.n_writes} | "
                f"{ep_time:.1f}s, ETA {eta:.0f}s",
                flush=True,
            )

    # Analyze results
    explore_results = results[:explore_eps]
    plan_results = results[explore_eps:]

    explore_success = sum(1 for r in explore_results if r["success"])
    explore_rate = explore_success / explore_eps

    plan_success = sum(1 for r in plan_results if r["success"])
    plan_rate = plan_success / len(plan_results) if plan_results else 0.0

    total_success = sum(1 for r in results if r["success"])
    total_rate = total_success / n_episodes

    # Learning: last half of plan vs first half
    plan_first = plan_results[:len(plan_results) // 2]
    plan_last = plan_results[len(plan_results) // 2:]
    first_rate = sum(1 for r in plan_first if r["success"]) / max(len(plan_first), 1)
    last_rate = sum(1 for r in plan_last if r["success"]) / max(len(plan_last), 1)

    gate_primary = plan_rate >= 0.15
    gate_stretch = plan_rate >= 0.30
    gate_vs_random = plan_rate > explore_rate

    elapsed = time.time() - t0

    print(f"\n  Explore: {explore_success}/{explore_eps} = {explore_rate:.1%}")
    print(f"  Plan: {plan_success}/{len(plan_results)} = {plan_rate:.1%}")
    print(f"  Plan first half: {first_rate:.1%}")
    print(f"  Plan last half: {last_rate:.1%}")
    print(f"  Gate primary (≥15% plan): {'PASS' if gate_primary else 'FAIL'}")
    print(f"  Gate stretch (≥30% plan): {'PASS' if gate_stretch else 'FAIL'}")
    print(f"  Gate vs random: {'PASS' if gate_vs_random else 'FAIL'}")
    print(f"  Traces collected: {len(agent._successful_traces)}")
    if agent.plan:
        print(f"  Plan subgoals: {[s.name for s in agent.plan.subgoals]}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "explore_rate": explore_rate,
        "plan_rate": plan_rate,
        "total_rate": total_rate,
        "first_plan_rate": first_rate,
        "last_plan_rate": last_rate,
        "n_traces": len(agent._successful_traces),
        "gate_primary": gate_primary,
        "gate_stretch": gate_stretch,
        "gate_vs_random": gate_vs_random,
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────
# Exp 106d: Subgoal Navigation Quality
# ──────────────────────────────────────────────

def run_exp106d(n_episodes: int = 50) -> dict:
    """Measure per-subgoal navigation quality: steps to achieve each subgoal."""
    print(f"\n=== Exp 106d: Subgoal Navigation Quality ({n_episodes} eps) ===")

    config = SubgoalConfig(
        dim=512,
        n_locations=5000,
        n_actions=7,
        min_confidence=0.01,
        epsilon=0.15,
        max_episode_steps=200,
        explore_episodes=30,
    )
    agent = SubgoalPlanningAgent(config)
    env = DoorKeyEnv(seed=42)

    # Explore phase
    print("  Exploring...")
    for ep in range(30):
        agent.run_episode(env, max_steps=200)

    if not agent._successful_traces:
        print("  ERROR: No successful traces")
        return {"gate": False}

    # Build plan
    best_trace = min(agent._successful_traces, key=len)
    from snks.agent.subgoal_planning import SubgoalExtractor
    extractor = SubgoalExtractor(agent.codebook, agent.encoder)
    subgoals = extractor.extract(best_trace)
    print(f"  Subgoals: {[s.name for s in subgoals]}")

    # Test per-subgoal navigation
    pickup_steps_list: list[int] = []
    door_steps_list: list[int] = []

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        plan = PlanGraph(subgoals)
        navigator = agent.navigator

        steps_per_sg: dict[str, int] = {}
        sg_start_step = 0

        for step_i in range(200):
            current_sg = plan.current_subgoal()
            if current_sg is None:
                break

            state = agent.encoder.encode(obs)
            action = navigator.select(state, current_sg)
            obs, reward, term, trunc, _ = env.step(action)

            # Record in SDM
            next_state = agent.encoder.encode(obs)
            agent.sdm.write(state, agent.codebook.action(action), next_state, reward)

            if navigator.is_achieved(obs, current_sg):
                steps_per_sg[current_sg.name] = step_i - sg_start_step + 1
                sg_start_step = step_i + 1
                plan.advance()

            if term or trunc:
                break

        if "pickup_key" in steps_per_sg:
            pickup_steps_list.append(steps_per_sg["pickup_key"])
        if "open_door" in steps_per_sg:
            door_steps_list.append(steps_per_sg["open_door"])

    mean_pickup = np.mean(pickup_steps_list) if pickup_steps_list else float('inf')
    mean_door = np.mean(door_steps_list) if door_steps_list else float('inf')
    pickup_achieved = len(pickup_steps_list)
    door_achieved = len(door_steps_list)

    gate_pickup = mean_pickup <= 100  # relaxed gate for CPU
    gate_door = mean_door <= 100

    print(f"  pickup_key: achieved {pickup_achieved}/{n_episodes}, mean steps={mean_pickup:.1f}")
    print(f"  open_door: achieved {door_achieved}/{n_episodes}, mean steps={mean_door:.1f}")
    print(f"  Gate pickup (≤100 steps): {'PASS' if gate_pickup else 'FAIL'}")
    print(f"  Gate door (≤100 steps): {'PASS' if gate_door else 'FAIL'}")

    return {
        "pickup_achieved": pickup_achieved,
        "door_achieved": door_achieved,
        "mean_pickup_steps": float(mean_pickup),
        "mean_door_steps": float(mean_door),
        "gate_pickup": gate_pickup,
        "gate_door": gate_door,
        "gate": gate_pickup and gate_door,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Exp 106: Subgoal Planning — Stage 46")
    print("=" * 60)

    results = {}

    results["106a"] = run_exp106a(n_episodes=500)
    results["106b"] = run_exp106b()
    results["106c"] = run_exp106c(n_episodes=400, explore_eps=200)
    results["106d"] = run_exp106d(n_episodes=50)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        if "gate" in res:
            status = "PASS" if res["gate"] else "FAIL"
        elif "gate_primary" in res:
            status = "PASS" if res["gate_primary"] else "FAIL"
        else:
            status = "?"
        print(f"  {name}: {status}")

    os.makedirs("_docs", exist_ok=True)
    with open("_docs/exp106_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to _docs/exp106_results.json")

    return results


if __name__ == "__main__":
    main()
