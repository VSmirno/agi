"""Exp 109: InstructedAgent — language-guided DoorKey (Stage 51 gate).

Gate: ≥70% success on 200 random DoorKey-5x5 with text instruction.
CPU-only (BFS pathfinding, no GPU needed).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from snks.agent.instructed_agent import InstructedAgent

# --- RandomDoorKeyEnv (copy from exp107) ---


class RandomDoorKeyEnv:
    """DoorKey-5x5 with randomized layout per episode."""

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
        self.wall_row = self.rng.randint(1, self.size - 1)
        door_col = self.rng.randint(0, self.size)
        self.door_pos = [self.wall_row, door_col]
        self.wall_positions = []
        for c in range(self.size):
            if c != door_col:
                self.wall_positions.append([self.wall_row, c])

        above = [(r, c) for r in range(self.wall_row) for c in range(self.size)]
        below = [(r, c) for r in range(self.wall_row + 1, self.size)
                 for c in range(self.size)]

        idx = self.rng.randint(0, len(above))
        self.key_pos = list(above[idx])
        available = [p for p in above if list(p) != self.key_pos]
        idx = self.rng.randint(0, len(available))
        self.agent_pos = list(available[idx])
        self.agent_dir = self.rng.randint(0, 4)
        idx = self.rng.randint(0, len(below))
        self.goal_pos = list(below[idx])

    def _is_wall(self, r: int, c: int) -> bool:
        return [r, c] in self.wall_positions

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = 0.0
        if action == 0:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if not self._is_wall(nr, nc):
                    if [nr, nc] != self.door_pos or self.door_open:
                        self.agent_pos = [nr, nc]
        elif action == 3:
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                self.key_picked = True
        elif action == 5:
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if [fr, fc] == self.door_pos and self.has_key and not self.door_open:
                self.door_open = True

        terminated = self.agent_pos == self.goal_pos
        if terminated:
            reward = 1.0 - 0.9 * (self.steps / self.max_steps)
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        for i in range(7):
            obs[0, i, 0] = 2; obs[6, i, 0] = 2
            obs[i, 0, 0] = 2; obs[i, 6, 0] = 2
        for wr, wc in self.wall_positions:
            obs[wr + 1, wc + 1, 0] = 2
        if not self.key_picked:
            kr, kc = self.key_pos[0] + 1, self.key_pos[1] + 1
            obs[kr, kc, 0] = 5; obs[kr, kc, 1] = 1
        dr, dc = self.door_pos[0] + 1, self.door_pos[1] + 1
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = 0 if self.door_open else 2
        gr, gc = self.goal_pos[0] + 1, self.goal_pos[1] + 1
        obs[gr, gc, 0] = 8
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


# --- Experiments ---


def run_exp109a(n_episodes: int = 200) -> dict:
    """Full instruction: 'pick up the key then open the door then go to the goal'."""
    instruction = "pick up the key then open the door then go to the goal"
    print(f"=== Exp 109a: Full DoorKey instruction ({n_episodes} episodes) ===")
    print(f"Instruction: {instruction}")

    agent = InstructedAgent(epsilon=0.05)
    env = RandomDoorKeyEnv()

    successes = 0
    steps_list: list[int] = []
    t0 = time.time()

    for ep in range(n_episodes):
        env_seed = ep + 1000  # different seeds from exp107
        obs = env.reset(seed=env_seed)
        subgoals = agent.set_instruction(instruction)
        ok = agent.build_plan(obs)

        if not ok:
            steps_list.append(0)
            continue

        success = False
        for step_i in range(200):
            current_sg = agent.plan.current_subgoal()
            if current_sg is not None:
                if agent._is_subgoal_achieved(obs, current_sg):
                    agent.plan.advance()

            action = agent.step(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if current_sg is not None:
                current_sg2 = agent.plan.current_subgoal()
                if current_sg2 is not None:
                    if agent._is_subgoal_achieved(obs, current_sg2):
                        agent.plan.advance()

            if terminated:
                if reward > 0:
                    success = True
                break
            if truncated:
                break

        if success:
            successes += 1
        steps_list.append(step_i + 1)

        if (ep + 1) % 50 == 0:
            rate = successes / (ep + 1)
            elapsed = time.time() - t0
            print(f"  ep {ep+1}/{n_episodes}: {rate:.1%} success, "
                  f"mean steps {np.mean(steps_list):.1f}, "
                  f"{elapsed:.1f}s")

    elapsed = time.time() - t0
    rate = successes / n_episodes
    result = {
        "instruction": instruction,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": rate,
        "mean_steps": float(np.mean(steps_list)),
        "elapsed_s": round(elapsed, 1),
        "gate": "PASS" if rate >= 0.70 else "FAIL",
    }
    return result


def run_exp109b(n_episodes: int = 50) -> dict:
    """Partial instruction: 'pick up the key' (key pickup only)."""
    instruction = "pick up the key"
    print(f"\n=== Exp 109b: Partial instruction ({n_episodes} episodes) ===")
    print(f"Instruction: {instruction}")

    agent = InstructedAgent(epsilon=0.05)
    env = RandomDoorKeyEnv()

    successes = 0
    for ep in range(n_episodes):
        success, steps = agent.run_episode(env, instruction, max_steps=100)
        if success:
            successes += 1

    rate = successes / n_episodes
    result = {
        "instruction": instruction,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": rate,
        "gate": "PASS" if rate >= 0.70 else "FAIL",
    }
    return result


def run_exp109c(n_episodes: int = 50) -> dict:
    """Variant instructions (same semantics, different wording)."""
    variants = [
        "pick up the key then open the door then go to the goal",
        "pick up the red key then toggle the door then go to the goal",
    ]
    print(f"\n=== Exp 109c: Instruction variants ({n_episodes} episodes each) ===")

    results = {}
    for instr in variants:
        agent = InstructedAgent(epsilon=0.05)
        env = RandomDoorKeyEnv()
        successes = 0
        for ep in range(n_episodes):
            env.reset(seed=ep + 2000)
            subgoals = agent.set_instruction(instr)
            ok = agent.build_plan(env._obs())

            if not ok:
                continue

            success = False
            obs = env._obs()
            for step_i in range(200):
                sg = agent.plan.current_subgoal()
                if sg and agent._is_subgoal_achieved(obs, sg):
                    agent.plan.advance()
                action = agent.step(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                if sg:
                    sg2 = agent.plan.current_subgoal()
                    if sg2 and agent._is_subgoal_achieved(obs, sg2):
                        agent.plan.advance()
                if terminated and reward > 0:
                    success = True
                    break
                if truncated:
                    break
            if success:
                successes += 1

        rate = successes / n_episodes
        results[instr] = {"successes": successes, "rate": rate}
        print(f"  '{instr[:50]}...' → {rate:.1%}")

    return results


def print_results(r109a: dict, r109b: dict, r109c: dict) -> None:
    print("\n" + "=" * 60)
    print("Exp 109: InstructedAgent — Language-Guided DoorKey (Stage 51)")
    print("=" * 60)
    print(f"\n109a Full instruction: {r109a['success_rate']:.1%} "
          f"({r109a['successes']}/{r109a['n_episodes']})")
    print(f"  Gate ≥70%: {r109a['gate']}")
    print(f"  Mean steps: {r109a['mean_steps']:.1f}")
    print(f"  Time: {r109a['elapsed_s']}s")
    print(f"\n109b Partial (key only): {r109b['success_rate']:.1%} "
          f"({r109b['successes']}/{r109b['n_episodes']})")
    print(f"  Gate ≥70%: {r109b['gate']}")
    print(f"\n109c Variants:")
    for instr, data in r109c.items():
        print(f"  '{instr[:50]}...' → {data['rate']:.1%}")


if __name__ == "__main__":
    r109a = run_exp109a(200)
    r109b = run_exp109b(50)
    r109c = run_exp109c(50)
    print_results(r109a, r109b, r109c)
