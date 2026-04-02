"""Exp 107: Wall-aware navigation experiments for Stage 47.

Sub-experiments:
  107a: BFS pathfinding accuracy on random layouts
  107b: SubgoalPlanningAgent with BFS on random DoorKey-5x5 (primary gate)
  107c: BFS vs heuristic navigation comparison
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from snks.agent.pathfinding import GridPathfinder
from snks.agent.subgoal_planning import (
    PlanGraph,
    SubgoalConfig,
    SubgoalExtractor,
    SubgoalPlanningAgent,
    TraceStep,
    _extract_symbolic,
)
from snks.agent.vsa_world_model import VSACodebook, VSAEncoder


# ──────────────────────────────────────────────
# RandomDoorKeyEnv
# ──────────────────────────────────────────────

class RandomDoorKeyEnv:
    """DoorKey-5x5 with randomized layout per episode.

    Each reset() generates a new layout:
    - Wall-divider row: random from {1, 2, 3}
    - Door column: random in wall-divider
    - Key: random position above wall
    - Agent start: random position above wall (not on key)
    - Goal: random position below wall
    """

    def __init__(self, size: int = 5, seed: int | None = None):
        self.size = size
        self.n_actions = 7
        self.max_steps = 200
        self.rng = np.random.RandomState(seed)
        self.wall_row: int = 2
        self.wall_positions: list[list[int]] = []
        self.agent_pos: list[int] = [0, 0]
        self.agent_dir: int = 0
        self.key_pos: list[int] = [0, 0]
        self.has_key: bool = False
        self.key_picked: bool = False
        self.door_pos: list[int] = [0, 0]
        self.door_open: bool = False
        self.goal_pos: list[int] = [0, 0]
        self.steps: int = 0
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self._generate_layout()
        self.has_key = False
        self.key_picked = False
        self.door_open = False
        self.steps = 0
        return self._obs()

    def _generate_layout(self) -> None:
        """Generate a random but solvable DoorKey layout."""
        # Wall divider row (inner coords): not first or last row
        self.wall_row = self.rng.randint(1, self.size - 1)  # 1..size-2

        # Door position in wall divider
        door_col = self.rng.randint(0, self.size)
        self.door_pos = [self.wall_row, door_col]

        # Wall positions: entire wall_row except door
        self.wall_positions = []
        for c in range(self.size):
            if c != door_col:
                self.wall_positions.append([self.wall_row, c])

        # Available positions above wall (rows 0..wall_row-1)
        above = [(r, c) for r in range(self.wall_row) for c in range(self.size)]
        # Available positions below wall (rows wall_row+1..size-1)
        below = [(r, c) for r in range(self.wall_row + 1, self.size) for c in range(self.size)]

        # Key: random above wall
        idx = self.rng.randint(0, len(above))
        self.key_pos = list(above[idx])

        # Agent: random above wall, not on key
        available_above = [p for p in above if list(p) != self.key_pos]
        idx = self.rng.randint(0, len(available_above))
        self.agent_pos = list(available_above[idx])

        # Agent direction: random
        self.agent_dir = self.rng.randint(0, 4)

        # Goal: random below wall
        idx = self.rng.randint(0, len(below))
        self.goal_pos = list(below[idx])

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
                    pass
                elif [nr, nc] == self.door_pos and not self.door_open:
                    pass
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
        # Key
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
        # Agent LAST
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


# ──────────────────────────────────────────────
# Exp 107a: BFS Pathfinding Accuracy
# ──────────────────────────────────────────────

def run_exp107a(n_layouts: int = 100) -> dict:
    """Test BFS pathfinding on random layouts — all must be solvable."""
    print(f"=== Exp 107a: BFS Pathfinding ({n_layouts} layouts) ===")

    pf = GridPathfinder()
    env = RandomDoorKeyEnv()

    paths_found = 0
    path_lengths: list[int] = []

    for seed in range(n_layouts):
        obs = env.reset(seed=seed)
        agent_obs = (env.agent_pos[0] + 1, env.agent_pos[1] + 1)
        goal_obs = (env.goal_pos[0] + 1, env.goal_pos[1] + 1)
        key_obs = (env.key_pos[0] + 1, env.key_pos[1] + 1)

        # Path to key (no door needed)
        path_to_key = pf.find_path(obs, agent_obs, key_obs)
        # Path from key to goal (through door — allow_door=True)
        path_to_goal = pf.find_path(obs, key_obs, goal_obs, allow_door=True)

        if path_to_key is not None and path_to_goal is not None:
            paths_found += 1
            total_len = len(path_to_key) + len(path_to_goal) - 1
            path_lengths.append(total_len)

    rate = paths_found / n_layouts
    mean_len = np.mean(path_lengths) if path_lengths else float('inf')
    gate = rate >= 1.0

    print(f"  Paths found: {paths_found}/{n_layouts} = {rate:.1%}")
    print(f"  Mean path length: {mean_len:.1f}")
    print(f"  Gate (100%): {'PASS' if gate else 'FAIL'}")

    return {
        "paths_found": paths_found,
        "total": n_layouts,
        "rate": rate,
        "mean_path_length": float(mean_len),
        "gate": gate,
    }


# ──────────────────────────────────────────────
# Exp 107b: SubgoalPlanningAgent with BFS on random layouts
# ──────────────────────────────────────────────

def run_exp107b(n_layouts: int = 200, plan_eps: int = 50) -> dict:
    """Primary gate: BFS-enhanced SubgoalPlanningAgent on random DoorKey-5x5.

    Uses obs-based planning (build_plan_from_obs) — no explore phase needed.
    Each layout gets plan_eps episodes of pure planning with BFS navigation.
    """
    print(f"\n=== Exp 107b: Random DoorKey-5x5 ({n_layouts} layouts, {plan_eps} plan eps each) ===")

    results_per_layout: list[dict] = []
    t0 = time.time()

    for layout_seed in range(n_layouts):
        env = RandomDoorKeyEnv(seed=layout_seed)

        config = SubgoalConfig(
            dim=512,
            n_locations=5000,
            n_actions=7,
            min_confidence=0.01,
            epsilon=0.05,
            max_episode_steps=200,
            explore_episodes=0,  # No explore — build plan from observation
        )
        agent = SubgoalPlanningAgent(config)

        plan_successes = 0
        plan_steps: list[int] = []

        for ep in range(plan_eps):
            success, steps, reward = agent.run_episode(env, max_steps=200)
            if success:
                plan_successes += 1
                plan_steps.append(steps)

        plan_rate = plan_successes / plan_eps
        mean_steps = np.mean(plan_steps) if plan_steps else 200
        results_per_layout.append({
            "seed": layout_seed,
            "plan_rate": plan_rate,
            "mean_steps": float(mean_steps),
            "has_plan": agent.plan is not None,
        })

        ep_time = time.time() - t0
        if (layout_seed + 1) % 20 == 0 or layout_seed == 0:
            layouts_done = layout_seed + 1
            solved = sum(1 for r in results_per_layout if r["plan_rate"] > 0.5)
            eta = ep_time / layouts_done * (n_layouts - layouts_done)
            print(
                f"  Layout {layouts_done}/{n_layouts}: "
                f"solved(>50%)={solved}/{layouts_done}, "
                f"mean_steps={mean_steps:.0f}, "
                f"ETA {eta:.0f}s",
                flush=True,
            )

    # Aggregate
    solved_80 = sum(1 for r in results_per_layout if r["plan_rate"] >= 0.8)
    solved_50 = sum(1 for r in results_per_layout if r["plan_rate"] > 0.5)
    solved_any = sum(1 for r in results_per_layout if r["plan_rate"] > 0)
    mean_plan_rate = np.mean([r["plan_rate"] for r in results_per_layout])
    mean_steps_all = np.mean([r["mean_steps"] for r in results_per_layout])
    no_plan = sum(1 for r in results_per_layout if not r["has_plan"])

    gate = solved_80 / n_layouts >= 0.80  # ≥80% of layouts solved at ≥80% rate

    elapsed = time.time() - t0

    print(f"\n  Solved (≥80% plan): {solved_80}/{n_layouts} = {solved_80 / n_layouts:.1%}")
    print(f"  Solved (>50% plan): {solved_50}/{n_layouts} = {solved_50 / n_layouts:.1%}")
    print(f"  Solved (any plan): {solved_any}/{n_layouts} = {solved_any / n_layouts:.1%}")
    print(f"  No plan built: {no_plan}/{n_layouts}")
    print(f"  Mean plan rate: {mean_plan_rate:.1%}")
    print(f"  Mean steps (solved): {mean_steps_all:.1f}")
    print(f"  Gate (≥80% layouts at ≥80%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "n_layouts": n_layouts,
        "solved_80": solved_80,
        "solved_50": solved_50,
        "solved_any": solved_any,
        "no_plan": no_plan,
        "mean_plan_rate": float(mean_plan_rate),
        "mean_steps": float(mean_steps_all),
        "gate": gate,
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────
# Exp 107c: BFS vs Heuristic comparison
# ──────────────────────────────────────────────

def run_exp107c(n_layouts: int = 50, explore_eps: int = 100) -> dict:
    """Compare BFS navigation vs heuristic on random layouts."""
    print(f"\n=== Exp 107c: BFS vs Heuristic ({n_layouts} layouts) ===")

    bfs_wins = 0
    heuristic_wins = 0
    ties = 0

    for layout_seed in range(n_layouts):
        env = RandomDoorKeyEnv(seed=layout_seed)

        # BFS agent
        config_bfs = SubgoalConfig(
            dim=512, n_locations=5000, n_actions=7,
            min_confidence=0.01, epsilon=0.15,
            explore_episodes=explore_eps,
        )
        agent_bfs = SubgoalPlanningAgent(config_bfs)

        # Heuristic agent (same but without BFS — we'll patch navigator)
        config_heur = SubgoalConfig(
            dim=512, n_locations=5000, n_actions=7,
            min_confidence=0.01, epsilon=0.15,
            explore_episodes=explore_eps,
        )
        agent_heur = SubgoalPlanningAgent(config_heur)
        agent_heur.navigator._use_bfs = False  # disable BFS

        for agent in [agent_bfs, agent_heur]:
            for ep in range(explore_eps + 50):
                agent.run_episode(env, max_steps=200)

        bfs_rate = sum(1 for ep in range(50)
                       for _ in [None]
                       if agent_bfs._successful_traces) / 50 if agent_bfs.plan else 0
        # Simplified: just compare plan existence and trace count
        bfs_traces = len(agent_bfs._successful_traces)
        heur_traces = len(agent_heur._successful_traces)

        if bfs_traces > heur_traces:
            bfs_wins += 1
        elif heur_traces > bfs_traces:
            heuristic_wins += 1
        else:
            ties += 1

    gate = bfs_wins > heuristic_wins

    print(f"  BFS wins: {bfs_wins}")
    print(f"  Heuristic wins: {heuristic_wins}")
    print(f"  Ties: {ties}")
    print(f"  Gate (BFS > heuristic): {'PASS' if gate else 'FAIL'}")

    return {
        "bfs_wins": bfs_wins,
        "heuristic_wins": heuristic_wins,
        "ties": ties,
        "gate": gate,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Exp 107: Wall-aware Navigation — Stage 47")
    print("=" * 60)

    results = {}
    results["107a"] = run_exp107a(n_layouts=200)
    results["107b"] = run_exp107b(n_layouts=200, plan_eps=50)
    # Skip 107c for now — 107b is the primary gate
    # results["107c"] = run_exp107c(n_layouts=50)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        status = "PASS" if res.get("gate", False) else "FAIL"
        print(f"  {name}: {status}")

    os.makedirs("_docs", exist_ok=True)
    with open("_docs/exp107_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to _docs/exp107_results.json")

    return results


if __name__ == "__main__":
    main()
