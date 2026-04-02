"""Stage 52: Integration test — language-guided MultiRoom-N3 + DoorKey regression.

Gate criteria:
- exp110a: ≥50% random MultiRoom-N3 with "go to the goal" (200 layouts)
- exp110b: ≥50% random MultiRoom-N3 with variant instructions (200 layouts)
- exp110c: ≥90% random DoorKey-5x5 with full instruction (200 layouts, regression)
"""

from __future__ import annotations

import time

import numpy as np

from snks.agent.integration_agent import IntegrationAgent
from snks.agent.multi_room_nav import MultiRoomEnvWrapper


def run_exp110a(n_episodes: int = 200, seed_base: int = 9000) -> dict:
    """MultiRoom-N3 + 'go to the goal'."""
    agent = IntegrationAgent(epsilon=0.0)
    successes = 0
    total_steps = 0
    steps_list = []

    t0 = time.time()
    for i in range(n_episodes):
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6, max_steps=300)
        obs = env.reset(seed=seed_base + i)

        # Run with language instruction
        success, steps = _run_multiroom_episode(
            agent, env, obs, "go to the goal", max_steps=300
        )
        if success:
            successes += 1
        total_steps += steps
        steps_list.append(steps)

    elapsed = time.time() - t0
    rate = successes / n_episodes
    mean_steps = np.mean(steps_list) if steps_list else 0
    p95_steps = np.percentile(steps_list, 95) if steps_list else 0

    return {
        "experiment": "110a",
        "description": "MultiRoom-N3 + 'go to the goal'",
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": rate,
        "gate": 0.50,
        "pass": rate >= 0.50,
        "mean_steps": float(mean_steps),
        "p95_steps": float(p95_steps),
        "time_s": elapsed,
    }


def run_exp110b(n_episodes: int = 200, seed_base: int = 9500) -> dict:
    """MultiRoom-N3 with variant instructions."""
    instructions = [
        "go to the goal",
        "open the door then go to the goal",
        "toggle the door then go to the goal",
    ]
    agent = IntegrationAgent(epsilon=0.0)
    successes = 0

    t0 = time.time()
    for i in range(n_episodes):
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6, max_steps=300)
        obs = env.reset(seed=seed_base + i)
        instr = instructions[i % len(instructions)]

        success, steps = _run_multiroom_episode(
            agent, env, obs, instr, max_steps=300
        )
        if success:
            successes += 1

    elapsed = time.time() - t0
    rate = successes / n_episodes

    return {
        "experiment": "110b",
        "description": "MultiRoom-N3 + variant instructions",
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": rate,
        "gate": 0.50,
        "pass": rate >= 0.50,
        "time_s": elapsed,
    }


class RandomDoorKeyEnv:
    """DoorKey-5x5 with randomized layout (from exp109)."""

    def __init__(self, size: int = 5, seed: int | None = None):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.wall_row = 2
        self.wall_positions: list[list[int]] = []
        self.agent_pos = [0, 0]
        self.agent_dir = 0
        self.key_pos = [0, 0]
        self.has_key = False
        self.key_picked = False
        self.door_pos = [0, 0]
        self.door_open = False
        self.goal_pos = [0, 0]
        self.steps = 0
        self.max_steps = 200
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
                if [nr, nc] not in self.wall_positions:
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
            reward = 1.0
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
        obs[ar, ac, 0] = 10; obs[ar, ac, 2] = self.agent_dir
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


def run_exp110c(n_episodes: int = 200, seed_base: int = 8000) -> dict:
    """DoorKey-5x5 regression — full instruction."""
    agent = IntegrationAgent(epsilon=0.05)
    successes = 0
    total_steps = 0

    t0 = time.time()
    instruction = "pick up the key then open the door then go to the goal"

    for i in range(n_episodes):
        env = RandomDoorKeyEnv(size=5, seed=seed_base + i)
        # Reset env first, then run through IntegrationAgent
        success, steps = agent.run_episode(env, instruction, max_steps=200)
        if success:
            successes += 1
        total_steps += steps

    elapsed = time.time() - t0
    rate = successes / n_episodes

    return {
        "experiment": "110c",
        "description": "DoorKey-5x5 regression (full instruction)",
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": rate,
        "gate": 0.90,
        "pass": rate >= 0.90,
        "mean_steps": total_steps / max(n_episodes, 1),
        "time_s": elapsed,
    }


def _run_multiroom_episode(
    agent: IntegrationAgent,
    env: MultiRoomEnvWrapper,
    obs: np.ndarray,
    instruction: str,
    max_steps: int = 300,
) -> tuple[bool, int]:
    """Run one MultiRoom episode with language instruction.

    Uses IntegrationAgent's multiroom strategy: BFS + reactive door toggle.
    """
    subgoals = agent.parse_instruction(instruction)
    if not subgoals:
        return False, 0

    for step_i in range(max_steps):
        action = agent.multi_room_nav.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            return reward > 0, step_i + 1
        if truncated:
            return False, step_i + 1

    return False, max_steps


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 52: Integration Test — Language-Guided Navigation")
    print("=" * 60)

    for run_fn in [run_exp110a, run_exp110b, run_exp110c]:
        result = run_fn()
        status = "PASS" if result["pass"] else "FAIL"
        print(f"\n--- {result['experiment']}: {result['description']} ---")
        print(f"  Success: {result['successes']}/{result['n_episodes']} "
              f"= {result['success_rate']:.1%}")
        print(f"  Gate: ≥{result['gate']:.0%} → {status}")
        if "mean_steps" in result:
            print(f"  Mean steps: {result['mean_steps']:.1f}")
        print(f"  Time: {result['time_s']:.1f}s")
