"""Stage 49: Multi-Room Navigation — BFS + reactive door toggling.

Navigate multi-room MiniGrid environments by:
1. BFS pathfinding from agent to goal (treating closed doors as passable)
2. Following the path
3. When facing a closed door — toggle it, then continue
4. Re-plan after each door toggle
"""

from __future__ import annotations

import numpy as np
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.wrappers import FullyObsWrapper

from snks.agent.pathfinding import GridPathfinder

# MiniGrid object types
OBJ_EMPTY = 1
OBJ_WALL = 2
OBJ_DOOR = 4
OBJ_KEY = 5
OBJ_GOAL = 8
OBJ_AGENT = 10

# MiniGrid direction deltas: dir → (dr, dc)
DIR_DELTAS: dict[int, tuple[int, int]] = {
    0: (0, 1),   # right
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (-1, 0),  # up
}

# MiniGrid actions
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5
ACT_DONE = 6


def find_objects(obs: np.ndarray) -> dict:
    """Extract agent, doors, and goal from full grid observation.

    Returns dict with:
        agent_pos: (row, col)
        agent_dir: int (0=right, 1=down, 2=left, 3=up)
        goal_pos: (row, col) or None
        doors: list of (row, col, state) where state: 0=open, 1=closed, 2=locked
    """
    agent_pos = None
    agent_dir = 0
    goal_pos = None
    doors: list[tuple[int, int, int]] = []

    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            obj = int(obs[r, c, 0])
            if obj == OBJ_AGENT:
                agent_pos = (r, c)
                agent_dir = int(obs[r, c, 2])
            elif obj == OBJ_GOAL:
                goal_pos = (r, c)
            elif obj == OBJ_DOOR:
                doors.append((r, c, int(obs[r, c, 2])))

    return {
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "goal_pos": goal_pos,
        "doors": doors,
    }


def is_facing_closed_door(obs: np.ndarray, agent_pos: tuple[int, int],
                           agent_dir: int) -> bool:
    """Check if the cell directly in front of agent is a closed door."""
    dr, dc = DIR_DELTAS[agent_dir]
    fr, fc = agent_pos[0] + dr, agent_pos[1] + dc
    if 0 <= fr < obs.shape[0] and 0 <= fc < obs.shape[1]:
        return int(obs[fr, fc, 0]) == OBJ_DOOR and int(obs[fr, fc, 2]) == 1
    return False


class MultiRoomNavigator:
    """Navigate multi-room environments using BFS + reactive door toggling.

    Strategy:
    1. BFS from agent to goal with allow_door=True (closed doors passable)
    2. Follow path: turn toward next cell, move forward
    3. When facing a closed door: toggle it
    4. After toggle: re-plan from new position
    """

    def __init__(self, epsilon: float = 0.05):
        self.pathfinder = GridPathfinder()
        self.epsilon = epsilon

    def select_action(self, obs: np.ndarray) -> int:
        """Select action given full grid observation."""
        objs = find_objects(obs)
        agent_pos = objs["agent_pos"]
        agent_dir = objs["agent_dir"]
        goal_pos = objs["goal_pos"]

        if agent_pos is None or goal_pos is None:
            return ACT_FORWARD  # fallback

        # If facing a closed door → toggle it
        if is_facing_closed_door(obs, agent_pos, agent_dir):
            return ACT_TOGGLE

        # BFS to goal (closed doors are passable)
        path = self.pathfinder.find_path(obs, agent_pos, goal_pos, allow_door=True)
        if path is None or len(path) <= 1:
            # No path or already at goal — try random
            return int(np.random.randint(0, 3))

        # Get action sequence from path
        actions = self.pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]

        return ACT_FORWARD

    def run_episode(self, env: MultiRoomEnvWrapper, obs: np.ndarray,
                    max_steps: int = 500) -> tuple[bool, int, float]:
        """Run one episode. Returns (success, steps, total_reward)."""
        total_reward = 0.0

        for step in range(max_steps):
            # Epsilon exploration
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                action = int(np.random.randint(0, 3))  # left/right/forward only
            else:
                action = self.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                return total_reward > 0, step + 1, total_reward

        return False, max_steps, total_reward


class MultiRoomEnvWrapper:
    """Wrap MiniGrid MultiRoom with FullyObsWrapper for full grid observation."""

    def __init__(self, n_rooms: int = 3, max_room_size: int = 6,
                 max_steps: int = 300):
        base = MultiRoomEnv(
            minNumRooms=n_rooms,
            maxNumRooms=n_rooms,
            maxRoomSize=max_room_size,
            max_steps=max_steps,
        )
        self.env = FullyObsWrapper(base)
        self.n_rooms = n_rooms

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, info = self.env.reset(seed=seed)
        # Transpose from MiniGrid (col, row, 3) to standard (row, col, 3)
        return obs["image"].transpose(1, 0, 2)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        return obs["image"].transpose(1, 0, 2), reward, term, trunc, info

    @property
    def unwrapped(self):
        return self.env.unwrapped
