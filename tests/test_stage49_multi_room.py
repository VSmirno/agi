"""Stage 49: Multi-Room Navigation tests."""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.pathfinding import GridPathfinder


# --- Helper: build a simple multi-room grid observation ---

def make_multiroom_obs(
    width: int = 15,
    height: int = 15,
    walls: list[tuple[int, int]] | None = None,
    doors: list[tuple[int, int, int]] | None = None,  # (r, c, state)
    agent_pos: tuple[int, int] = (1, 1),
    agent_dir: int = 0,
    goal_pos: tuple[int, int] = (13, 13),
    key_pos: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create a synthetic multi-room observation grid.

    Object types: 0=empty, 2=wall, 4=door, 5=key, 8=goal, 10=agent
    Door states: 0=open, 1=closed, 2=locked
    """
    obs = np.zeros((height, width, 3), dtype=np.uint8)

    # Border walls
    for r in range(height):
        obs[r, 0] = [2, 5, 0]
        obs[r, width - 1] = [2, 5, 0]
    for c in range(width):
        obs[0, c] = [2, 5, 0]
        obs[height - 1, c] = [2, 5, 0]

    # Interior walls
    if walls:
        for r, c in walls:
            obs[r, c] = [2, 5, 0]

    # Doors
    if doors:
        for r, c, state in doors:
            obs[r, c] = [4, 4, state]  # yellow door

    # Agent
    obs[agent_pos[0], agent_pos[1]] = [10, 0, agent_dir]

    # Goal
    obs[goal_pos[0], goal_pos[1]] = [8, 1, 0]

    # Key (optional)
    if key_pos:
        obs[key_pos[0], key_pos[1]] = [5, 4, 0]

    return obs


def make_two_room_obs() -> np.ndarray:
    """Two rooms separated by a vertical wall at col=7, door at (5, 7)."""
    walls = [(r, 7) for r in range(1, 14) if r != 5]
    doors = [(5, 7, 1)]  # closed door
    return make_multiroom_obs(
        width=15, height=15,
        walls=walls, doors=doors,
        agent_pos=(3, 3), agent_dir=0,
        goal_pos=(3, 11),
    )


def make_three_room_obs() -> np.ndarray:
    """Three rooms: walls at col=5 and col=10, doors at (3,5) and (3,10)."""
    walls = []
    for c in [5, 10]:
        for r in range(1, 14):
            if r != 3:
                walls.append((r, c))
    doors = [(3, 5, 1), (3, 10, 1)]  # two closed doors
    return make_multiroom_obs(
        width=15, height=15,
        walls=walls, doors=doors,
        agent_pos=(2, 2), agent_dir=0,
        goal_pos=(2, 12),
    )


# --- GridPathfinder tests for multi-room ---

class TestPathfinderMultiRoom:
    """BFS pathfinding through multi-room layouts."""

    def test_path_through_closed_door_allow(self):
        """BFS finds path through closed door when allow_door=True."""
        obs = make_two_room_obs()
        pf = GridPathfinder()
        path = pf.find_path(obs, (3, 3), (3, 11), allow_door=True)
        assert path is not None
        assert path[0] == (3, 3)
        assert path[-1] == (3, 11)
        # Path must go through door at (5, 7)
        assert (5, 7) in path

    def test_path_blocked_by_locked_door_no_allow(self):
        """BFS can't find path through locked door when allow_door=False."""
        obs = make_two_room_obs()
        obs[5, 7] = [4, 4, 2]  # locked door (state=2)
        pf = GridPathfinder()
        path = pf.find_path(obs, (3, 3), (3, 11), allow_door=False)
        assert path is None

    def test_path_through_open_door(self):
        """BFS finds path through open door regardless of allow_door."""
        obs = make_two_room_obs()
        obs[5, 7] = [4, 4, 0]  # open door
        pf = GridPathfinder()
        path = pf.find_path(obs, (3, 3), (3, 11), allow_door=False)
        assert path is not None

    def test_path_through_three_rooms(self):
        """BFS finds path through two doors in three-room layout."""
        obs = make_three_room_obs()
        pf = GridPathfinder()
        path = pf.find_path(obs, (2, 2), (2, 12), allow_door=True)
        assert path is not None
        assert path[0] == (2, 2)
        assert path[-1] == (2, 12)
        # Must pass through both doors
        assert (3, 5) in path
        assert (3, 10) in path

    def test_path_25x25_grid(self):
        """BFS works on 25x25 grids (MultiRoom size)."""
        obs = np.zeros((25, 25, 3), dtype=np.uint8)
        # Border walls
        for r in range(25):
            obs[r, 0] = [2, 5, 0]
            obs[r, 24] = [2, 5, 0]
        for c in range(25):
            obs[0, c] = [2, 5, 0]
            obs[24, c] = [2, 5, 0]
        # Wall at col=12 with door at (12, 12)
        for r in range(1, 24):
            if r != 12:
                obs[r, 12] = [2, 5, 0]
        obs[12, 12] = [4, 4, 1]  # closed door

        pf = GridPathfinder()
        path = pf.find_path(obs, (5, 5), (5, 18), allow_door=True)
        assert path is not None
        assert (12, 12) in path


# --- MultiRoomNavigator tests ---

class TestMultiRoomNavigator:
    """Multi-room navigation agent."""

    def test_import(self):
        """MultiRoomNavigator is importable."""
        from snks.agent.multi_room_nav import MultiRoomNavigator
        nav = MultiRoomNavigator()
        assert nav is not None

    def test_find_objects(self):
        """find_objects correctly locates agent, doors, goal."""
        from snks.agent.multi_room_nav import find_objects
        obs = make_three_room_obs()
        objs = find_objects(obs)
        assert objs["agent_pos"] == (2, 2)
        assert objs["agent_dir"] == 0
        assert objs["goal_pos"] == (2, 12)
        assert len(objs["doors"]) == 2
        assert (3, 5) in [(d[0], d[1]) for d in objs["doors"]]
        assert (3, 10) in [(d[0], d[1]) for d in objs["doors"]]

    def test_facing_closed_door(self):
        """Detect when agent faces a closed door."""
        from snks.agent.multi_room_nav import is_facing_closed_door
        obs = make_two_room_obs()
        # Agent at (3,3) facing right — not facing door
        assert not is_facing_closed_door(obs, (3, 3), 0)

        # Agent at (5,6) facing right — facing door at (5,7)
        obs2 = make_two_room_obs()
        obs2[3, 3] = [0, 0, 0]  # remove agent from old pos
        obs2[5, 6] = [10, 0, 0]  # agent facing right
        assert is_facing_closed_door(obs2, (5, 6), 0)

    def test_facing_open_door_returns_false(self):
        """Don't toggle an already-open door."""
        from snks.agent.multi_room_nav import is_facing_closed_door
        obs = make_two_room_obs()
        obs[5, 7] = [4, 4, 0]  # open door
        obs[3, 3] = [0, 0, 0]
        obs[5, 6] = [10, 0, 0]
        assert not is_facing_closed_door(obs, (5, 6), 0)

    def test_select_action_toggle(self):
        """Agent selects toggle when facing closed door."""
        from snks.agent.multi_room_nav import MultiRoomNavigator
        nav = MultiRoomNavigator(epsilon=0.0)
        obs = make_two_room_obs()
        obs[3, 3] = [0, 0, 0]
        obs[5, 6] = [10, 0, 0]  # facing right toward door at (5,7)
        action = nav.select_action(obs)
        assert action == 5  # toggle

    def test_select_action_navigate(self):
        """Agent navigates toward goal when no door in front."""
        from snks.agent.multi_room_nav import MultiRoomNavigator
        nav = MultiRoomNavigator(epsilon=0.0)
        obs = make_two_room_obs()
        # Agent at (3,3) facing right, goal at (3,11), door at (5,7)
        action = nav.select_action(obs)
        # Should be a navigation action (turn or forward), not toggle
        assert action in (0, 1, 2)

    def test_direction_deltas(self):
        """Direction → delta mapping is correct."""
        from snks.agent.multi_room_nav import DIR_DELTAS
        assert DIR_DELTAS[0] == (0, 1)   # right
        assert DIR_DELTAS[1] == (1, 0)   # down
        assert DIR_DELTAS[2] == (0, -1)  # left
        assert DIR_DELTAS[3] == (-1, 0)  # up


# --- Environment wrapper tests ---

class TestMultiRoomEnvWrapper:
    """Wrapper for MiniGrid MultiRoom environments."""

    def test_wrapper_reset(self):
        """Wrapper returns numpy obs on reset."""
        from snks.agent.multi_room_nav import MultiRoomEnvWrapper
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
        obs = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape[2] == 3
        assert obs.shape[0] == obs.shape[1]  # square grid

    def test_wrapper_step(self):
        """Wrapper returns correct step tuple."""
        from snks.agent.multi_room_nav import MultiRoomEnvWrapper
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
        obs = env.reset(seed=42)
        obs2, reward, term, trunc, info = env.step(2)  # forward
        assert isinstance(obs2, np.ndarray)
        assert obs2.shape == obs.shape

    def test_wrapper_has_goal(self):
        """Environment has a goal object."""
        from snks.agent.multi_room_nav import MultiRoomEnvWrapper, find_objects
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
        obs = env.reset(seed=42)
        objs = find_objects(obs)
        assert objs["goal_pos"] is not None

    def test_wrapper_has_doors(self):
        """Environment has closed doors."""
        from snks.agent.multi_room_nav import MultiRoomEnvWrapper, find_objects
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
        obs = env.reset(seed=42)
        objs = find_objects(obs)
        assert len(objs["doors"]) >= 1


# --- Integration: run_episode ---

class TestRunEpisode:
    """Full episode execution on MultiRoom."""

    def test_run_episode_two_rooms(self):
        """Agent can solve a two-room layout."""
        from snks.agent.multi_room_nav import MultiRoomNavigator, MultiRoomEnvWrapper
        env = MultiRoomEnvWrapper(n_rooms=2, max_room_size=6)
        nav = MultiRoomNavigator(epsilon=0.0)
        obs = env.reset(seed=42)
        success, steps, reward = nav.run_episode(env, obs, max_steps=500)
        assert success, f"Failed two-room layout in {steps} steps"
        assert steps <= 500

    def test_run_episode_three_rooms(self):
        """Agent can solve a three-room layout."""
        from snks.agent.multi_room_nav import MultiRoomNavigator, MultiRoomEnvWrapper
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
        nav = MultiRoomNavigator(epsilon=0.0)
        obs = env.reset(seed=42)
        success, steps, reward = nav.run_episode(env, obs, max_steps=500)
        assert success, f"Failed three-room layout in {steps} steps"

    def test_run_episode_deterministic(self):
        """Same seed → same result."""
        from snks.agent.multi_room_nav import MultiRoomNavigator, MultiRoomEnvWrapper
        results = []
        for _ in range(2):
            env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
            nav = MultiRoomNavigator(epsilon=0.0)
            obs = env.reset(seed=123)
            success, steps, reward = nav.run_episode(env, obs, max_steps=500)
            results.append((success, steps))
        assert results[0] == results[1]

    def test_run_multiple_seeds(self):
        """Agent succeeds on multiple random seeds."""
        from snks.agent.multi_room_nav import MultiRoomNavigator, MultiRoomEnvWrapper
        successes = 0
        n_seeds = 20
        for seed in range(n_seeds):
            env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
            nav = MultiRoomNavigator(epsilon=0.0)
            obs = env.reset(seed=seed)
            success, steps, reward = nav.run_episode(env, obs, max_steps=500)
            if success:
                successes += 1
        rate = successes / n_seeds
        assert rate >= 0.6, f"Success rate {rate:.0%} < 60% on {n_seeds} seeds"
