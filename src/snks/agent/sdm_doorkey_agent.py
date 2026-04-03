"""Stage 58: SDM Retrofit — Learned DoorKey agent with partial observation.

СНКС pipeline: obs 7×7 → SpatialMap → AbstractStateEncoder (VSA) → SDM → planning.
First learned stage after symbolic drift (Stages 47-57).

Two-phase operation:
1. Exploration: FrontierExplorer navigates, SDM records transitions
2. Planning: SDM reward lookahead + trace matching selects actions

Symbolic reflexes (toggle doors, pickup keys) remain as low-level primitives.
High-level planning (where to go, what order) is learned via SDM.
"""

from __future__ import annotations

import numpy as np
import torch

from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    SpatialMap,
    OBJ_DOOR,
    OBJ_EMPTY,
    OBJ_GOAL,
    OBJ_KEY,
    OBJ_WALL,
)
from snks.agent.vsa_world_model import (
    BackwardChainPlanner,
    SDMMemory,
    SDMPlanner,
    VSACodebook,
)

# MiniGrid actions
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5


class AbstractStateEncoder:
    """Encode DoorKey abstract state features into compact VSA vector.

    Encodes high-level features (has_key, door_state, relative positions)
    instead of raw 7×7 grid — enables SDM to generalize across layouts.
    """

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode(self, agent_row: int, agent_col: int,
               has_key: bool, door_state: str,
               key_known: bool, door_known: bool, goal_known: bool,
               exploration_pct: float,
               key_pos: tuple[int, int] | None = None,
               door_pos: tuple[int, int] | None = None,
               goal_pos: tuple[int, int] | None = None) -> torch.Tensor:
        """Encode abstract state → 512-dim binary VSA vector."""
        facts: list[torch.Tensor] = []

        # Agent position (quantized to grid cell)
        facts.append(self.cb.bind(
            self.cb.role("agent_row"), self.cb.filler(f"r{agent_row}")
        ))
        facts.append(self.cb.bind(
            self.cb.role("agent_col"), self.cb.filler(f"c{agent_col}")
        ))

        # Inventory state
        facts.append(self.cb.bind(
            self.cb.role("has_key"), self.cb.filler(f"key_{has_key}")
        ))

        # Door state
        facts.append(self.cb.bind(
            self.cb.role("door_state"), self.cb.filler(f"door_{door_state}")
        ))

        # Knowledge state (what's been discovered)
        facts.append(self.cb.bind(
            self.cb.role("key_known"), self.cb.filler(f"kk_{key_known}")
        ))
        facts.append(self.cb.bind(
            self.cb.role("door_known"), self.cb.filler(f"dk_{door_known}")
        ))
        facts.append(self.cb.bind(
            self.cb.role("goal_known"), self.cb.filler(f"gk_{goal_known}")
        ))

        # Exploration progress (quantized to 10% bins)
        exp_bin = int(exploration_pct * 10)
        facts.append(self.cb.bind(
            self.cb.role("explored"), self.cb.filler(f"exp_{exp_bin}")
        ))

        # Relative distances to known objects (quantized)
        if key_known and key_pos is not None:
            dist = abs(agent_row - key_pos[0]) + abs(agent_col - key_pos[1])
            dist_bin = min(dist, 10)
            facts.append(self.cb.bind(
                self.cb.role("key_dist"), self.cb.filler(f"kd_{dist_bin}")
            ))

        if door_known and door_pos is not None:
            dist = abs(agent_row - door_pos[0]) + abs(agent_col - door_pos[1])
            dist_bin = min(dist, 10)
            facts.append(self.cb.bind(
                self.cb.role("door_dist"), self.cb.filler(f"dd_{dist_bin}")
            ))

        if goal_known and goal_pos is not None:
            dist = abs(agent_row - goal_pos[0]) + abs(agent_col - goal_pos[1])
            dist_bin = min(dist, 10)
            facts.append(self.cb.bind(
                self.cb.role("goal_dist"), self.cb.filler(f"gd_{dist_bin}")
            ))

        return self.cb.bundle(facts)


class SDMDoorKeyAgent:
    """Learned DoorKey agent using СНКС pipeline.

    Exploration phase: FrontierExplorer + SDM recording.
    Planning phase: SDM-based action selection + trace matching.
    Symbolic reflexes: auto-toggle doors, auto-pickup keys.
    """

    def __init__(self, grid_width: int = 5, grid_height: int = 5,
                 dim: int = 512, n_locations: int = 5000,
                 explore_episodes: int = 50,
                 epsilon: float = 0.15,
                 device: torch.device | str | None = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()

        self.codebook = VSACodebook(dim=dim, device=self.device)
        self.encoder = AbstractStateEncoder(self.codebook)
        self.sdm = SDMMemory(n_locations=n_locations, dim=dim, device=self.device)

        self.sdm_planner = SDMPlanner(
            sdm=self.sdm, codebook=self.codebook,
            n_actions=6, min_confidence=0.05, epsilon=epsilon,
        )
        self.backward_planner = BackwardChainPlanner(
            forward_sdm=self.sdm, codebook=self.codebook,
            n_actions=6, backward_depth=5,
            min_confidence=0.05, epsilon=epsilon,
        )

        self.explore_episodes = explore_episodes
        self._episode_count = 0
        self._exploring = True

        # State tracking (subgoal level)
        self._prev_state_vsa: torch.Tensor | None = None
        self._prev_subgoal: int | None = None
        self._episode_trace: list[tuple[torch.Tensor, int]] = []

        # Episode-level state
        self._has_key = False
        self._door_state = "locked"

    def reset_episode(self) -> None:
        """Reset per-episode state (not SDM — that persists)."""
        self.spatial_map.reset()
        self._prev_state_vsa = None
        self._prev_subgoal = None
        self._episode_trace = []
        self._has_key = False
        self._door_state = "locked"

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int,
                      has_key: bool, door_state: str) -> int:
        """Select action using СНКС pipeline."""
        # Update perception
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        self._has_key = has_key
        self._door_state = door_state

        # Symbolic reflexes — low-level primitives (not recorded as subgoals)
        reflex = self._check_reflexes(obs_7x7, has_key)
        if reflex is not None:
            return reflex

        # Encode abstract state via VSA
        state_vsa = self._encode_current_state(agent_row, agent_col, has_key, door_state)

        # Choose subgoal: exploration heuristic or SDM-based planning
        if self._exploring:
            subgoal = self._heuristic_subgoal()
        else:
            subgoal = self._sdm_select_subgoal(state_vsa)

        # Record subgoal-level transition in SDM
        self._record_subgoal_transition(
            agent_row, agent_col, has_key, door_state, subgoal
        )

        # Execute subgoal via symbolic navigation
        return self._execute_subgoal(subgoal, agent_row, agent_col, agent_dir)

    def _check_reflexes(self, obs_7x7: np.ndarray, has_key: bool) -> int | None:
        """Symbolic reflexes: toggle doors, pickup keys."""
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        front_color = int(obs_7x7[3, 5, 1])

        # Facing closed unlocked door → toggle
        if front_obj == OBJ_DOOR and front_state == 1:
            return ACT_TOGGLE

        # Facing locked door with key → toggle
        if front_obj == OBJ_DOOR and front_state == 2 and has_key:
            return ACT_TOGGLE

        # Facing key and not carrying → pickup
        if front_obj == OBJ_KEY and not has_key:
            return ACT_PICKUP

        return None

    def _encode_current_state(self, agent_row: int, agent_col: int,
                              has_key: bool, door_state: str) -> torch.Tensor:
        """Encode current state through AbstractStateEncoder."""
        objs = self.spatial_map.find_objects()
        key_pos = objs.get("key_pos")
        door_pos = objs.get("door_pos")
        goal_pos = objs.get("goal_pos")

        explored = self.spatial_map.explored.sum()
        total = self.spatial_map.width * self.spatial_map.height
        exp_pct = float(explored) / max(total, 1)

        return self.encoder.encode(
            agent_row=agent_row, agent_col=agent_col,
            has_key=has_key, door_state=door_state,
            key_known=key_pos is not None,
            door_known=door_pos is not None,
            goal_known=goal_pos is not None,
            exploration_pct=exp_pct,
            key_pos=key_pos, door_pos=door_pos, goal_pos=goal_pos,
        )

    def _explore_action(self, agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Frontier-guided exploration."""
        return self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

    # Subgoal IDs for SDM planning
    SG_EXPLORE = 0
    SG_GOTO_KEY = 1
    SG_GOTO_DOOR = 2
    SG_GOTO_GOAL = 3

    def _plan_action(self, state_vsa: torch.Tensor,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        """SDM-based subgoal planning + symbolic navigation.

        SDM selects WHAT to do (subgoal). FrontierExplorer/BFS handles HOW.
        This separation lets SDM operate at the right abstraction level.
        """
        subgoal = self._sdm_select_subgoal(state_vsa)
        return self._execute_subgoal(
            subgoal, agent_row, agent_col, agent_dir
        )

    def _sdm_select_subgoal(self, state_vsa: torch.Tensor) -> int:
        """Query SDM for best subgoal given current state."""
        best_sg = self.SG_EXPLORE
        best_score = -float("inf")

        for sg in (self.SG_EXPLORE, self.SG_GOTO_KEY, self.SG_GOTO_DOOR, self.SG_GOTO_GOAL):
            sg_vsa = self.codebook.action(sg)  # reuse action codebook for subgoals
            reward = self.sdm.read_reward(state_vsa, sg_vsa)
            _, conf = self.sdm.read_next(state_vsa, sg_vsa)
            if conf >= 0.01:
                score = reward * conf
                if score > best_score:
                    best_score = score
                    best_sg = sg

        # If no confidence anywhere, use heuristic fallback
        if best_score <= 0:
            return self._heuristic_subgoal()

        return best_sg

    def _heuristic_subgoal(self) -> int:
        """Fallback when SDM has no signal: simple priority ordering."""
        objs = self.spatial_map.find_objects()
        if not self._has_key:
            if objs.get("key_pos") is not None:
                return self.SG_GOTO_KEY
            return self.SG_EXPLORE
        if self._door_state in ("locked", "closed"):
            if objs.get("door_pos") is not None:
                return self.SG_GOTO_DOOR
            return self.SG_EXPLORE
        if objs.get("goal_pos") is not None:
            return self.SG_GOTO_GOAL
        return self.SG_EXPLORE

    def _execute_subgoal(self, subgoal: int,
                         agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Navigate toward subgoal target using BFS/frontier explorer."""
        objs = self.spatial_map.find_objects()

        target_pos = None
        if subgoal == self.SG_GOTO_KEY:
            target_pos = objs.get("key_pos")
        elif subgoal == self.SG_GOTO_DOOR:
            target_pos = objs.get("door_pos")
        elif subgoal == self.SG_GOTO_GOAL:
            target_pos = objs.get("goal_pos")

        if target_pos is None:
            return self._explore_action(agent_row, agent_col, agent_dir)

        # Navigate to adjacent cell of target
        adj = self._find_adjacent_walkable(target_pos, agent_row, agent_col)
        if adj is None:
            return self._explore_action(agent_row, agent_col, agent_dir)

        dr = abs(agent_row - target_pos[0])
        dc = abs(agent_col - target_pos[1])
        if dr + dc == 1:
            # Adjacent — turn to face target
            return self._turn_toward(target_pos[0], target_pos[1],
                                     agent_row, agent_col, agent_dir)

        return self._navigate_to(adj[0], adj[1], agent_row, agent_col, agent_dir)

    def _find_adjacent_walkable(self, pos: tuple[int, int],
                                agent_row: int, agent_col: int) -> tuple[int, int] | None:
        obs = self.spatial_map.to_obs()
        # Mark objects as walls for pathfinding
        for t in (OBJ_KEY, 6, 7):  # key, ball, box
            mask = obs[:, :, 0] == t
            obs[mask, 0] = OBJ_WALL
        best = None
        best_dist = float("inf")
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.spatial_map.height and 0 <= nc < self.spatial_map.width):
                continue
            if int(obs[nr, nc, 0]) in (OBJ_WALL,):
                continue
            path = self.pathfinder.find_path(obs, (agent_row, agent_col), (nr, nc), allow_door=True)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (nr, nc)
        return best

    def _navigate_to(self, tr: int, tc: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        if agent_row == tr and agent_col == tc:
            return int(np.random.randint(0, 3))
        obs = self.spatial_map.to_obs()
        path = self.pathfinder.find_path(obs, (agent_row, agent_col), (tr, tc), allow_door=True)
        if path is None or len(path) <= 1:
            return self._explore_action(agent_row, agent_col, agent_dir)
        actions = self.pathfinder.path_to_actions(path, agent_dir)
        return actions[0] if actions else ACT_FORWARD

    def _turn_toward(self, tr: int, tc: int,
                     ar: int, ac: int, agent_dir: int) -> int:
        dr, dc = tr - ar, tc - ac
        if dc > 0: need = 0
        elif dr > 0: need = 1
        elif dc < 0: need = 2
        else: need = 3
        if need == agent_dir:
            return ACT_FORWARD
        diff = (need - agent_dir) % 4
        return ACT_RIGHT if diff <= 2 else ACT_LEFT

    def _record_subgoal_transition(self, agent_row: int, agent_col: int,
                                   has_key: bool, door_state: str,
                                   subgoal: int, reward: float = 0.0) -> None:
        """Record subgoal-level transition in SDM.

        Key insight: SDM stores (abstract_state, subgoal) → (next_abstract_state, reward).
        This is at the subgoal level, not at the raw action level.
        """
        state_vsa = self._encode_current_state(agent_row, agent_col, has_key, door_state)

        if self._prev_state_vsa is not None and self._prev_subgoal is not None:
            sg_vsa = self.codebook.action(self._prev_subgoal)
            self.sdm.write(self._prev_state_vsa, sg_vsa, state_vsa, reward)
            self._episode_trace.append((self._prev_state_vsa.clone(), self._prev_subgoal))

        self._prev_state_vsa = state_vsa
        self._prev_subgoal = subgoal

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       has_key: bool, door_state: str, reward: float) -> None:
        """Update state after action execution."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # If reward received, write amplified reward signal to SDM
        if reward > 0 and self._prev_state_vsa is not None and self._prev_subgoal is not None:
            sg_vsa = self.codebook.action(self._prev_subgoal)
            next_state = self._encode_current_state(agent_row, agent_col, has_key, door_state)
            for _ in range(10):  # amplify reward signal
                self.sdm.write(self._prev_state_vsa, sg_vsa, next_state, reward)

    def _episode_done(self, success: bool) -> None:
        """Called at end of episode."""
        if success and self._episode_trace:
            self.backward_planner.record_success_trace(list(self._episode_trace))

        self._episode_count += 1
        self._exploring = self._episode_count < self.explore_episodes
        self.reset_episode()


class SDMDoorKeyEnv:
    """Wrapper for MiniGrid DoorKey-5x5 with partial observation.

    Provides agent position, carrying state, and door state info
    needed for AbstractStateEncoder.
    """

    def __init__(self, size: int = 5, max_steps: int = 200):
        from minigrid.envs.doorkey import DoorKeyEnv
        self._env = DoorKeyEnv(size=size, max_steps=max_steps)
        self.grid_width = size
        self.grid_height = size

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, int, int, bool, str]:
        """Reset → (obs_7x7, agent_col, agent_row, agent_dir, has_key, door_state)."""
        obs, info = self._env.reset(seed=seed)
        return self._extract(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, int, int, int, bool, str]:
        """Step → (obs_7x7, reward, term, trunc, col, row, dir, has_key, door_state)."""
        obs, reward, term, trunc, info = self._env.step(action)
        img, col, row, d, has_key, door_state = self._extract(obs)
        return img, float(reward), term, trunc, col, row, d, has_key, door_state

    def _extract(self, obs) -> tuple[np.ndarray, int, int, int, bool, str]:
        uw = self._env
        img = obs["image"]
        pos = uw.agent_pos
        has_key = uw.carrying is not None

        # Find door state
        door_state = "locked"
        g = uw.grid
        for i in range(g.width):
            for j in range(g.height):
                cell = g.get(i, j)
                if cell is not None and cell.type == "door":
                    if cell.is_open:
                        door_state = "open"
                    elif cell.is_locked:
                        door_state = "locked"
                    else:
                        door_state = "closed"
                    break

        return img, int(pos[0]), int(pos[1]), int(uw.agent_dir), has_key, door_state

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env
