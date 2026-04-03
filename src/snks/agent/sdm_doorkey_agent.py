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

        # State tracking
        self._prev_state_vsa: torch.Tensor | None = None
        self._prev_action: int | None = None
        self._episode_trace: list[tuple[torch.Tensor, int]] = []

        # Episode-level state
        self._has_key = False
        self._door_state = "locked"

    def reset_episode(self) -> None:
        """Reset per-episode state (not SDM — that persists)."""
        self.spatial_map.reset()
        self._prev_state_vsa = None
        self._prev_action = None
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

        # Symbolic reflexes — low-level primitives
        reflex = self._check_reflexes(obs_7x7, has_key)
        if reflex is not None:
            self._record_action(agent_row, agent_col, has_key, door_state, reflex)
            return reflex

        # Encode abstract state via VSA
        state_vsa = self._encode_current_state(agent_row, agent_col, has_key, door_state)

        # Choose action: exploration or SDM-based planning
        if self._exploring:
            action = self._explore_action(agent_row, agent_col, agent_dir)
        else:
            action = self._plan_action(state_vsa, agent_row, agent_col, agent_dir)

        self._record_action(agent_row, agent_col, has_key, door_state, action)
        return action

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

    def _plan_action(self, state_vsa: torch.Tensor,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        """SDM-based planning with trace matching fallback."""
        # Try backward chain planner (trace matching)
        action = self.backward_planner.select(state_vsa)

        # If backward planner returned random (no confident match),
        # try SDM reward lookahead
        # Check if SDM has confidence for this state
        best_conf = 0.0
        for a in range(6):
            _, conf = self.sdm.read_next(state_vsa, self.codebook.action(a))
            best_conf = max(best_conf, conf)

        if best_conf < 0.05:
            # SDM has no knowledge about this state — fall back to exploration
            return self._explore_action(agent_row, agent_col, agent_dir)

        # Use SDM planner for action selection
        return self.sdm_planner.select(state_vsa)

    def _record_action(self, agent_row: int, agent_col: int,
                       has_key: bool, door_state: str, action: int) -> None:
        """Record transition in SDM."""
        state_vsa = self._encode_current_state(agent_row, agent_col, has_key, door_state)

        if self._prev_state_vsa is not None and self._prev_action is not None:
            action_vsa = self.codebook.action(self._prev_action)
            # Write to SDM: (prev_state, prev_action) → current_state
            self.sdm.write(self._prev_state_vsa, action_vsa, state_vsa, 0.0)
            # Record for backward chaining
            self.backward_planner.record_transition(
                self._prev_state_vsa, self._prev_action, state_vsa, 0.0,
            )
            self._episode_trace.append((self._prev_state_vsa.clone(), self._prev_action))

        self._prev_state_vsa = state_vsa
        self._prev_action = action

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       has_key: bool, door_state: str, reward: float) -> None:
        """Update state after action execution."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # If reward received, write reward signal to SDM for last transition
        if reward > 0 and self._prev_state_vsa is not None and self._prev_action is not None:
            action_vsa = self.codebook.action(self._prev_action)
            next_state = self._encode_current_state(agent_row, agent_col, has_key, door_state)
            # Overwrite with reward signal
            for _ in range(5):  # amplify reward signal
                self.sdm.write(self._prev_state_vsa, action_vsa, next_state, reward)

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
