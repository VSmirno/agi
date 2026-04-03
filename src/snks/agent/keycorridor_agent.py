"""Stage 57: KeyCorridorAgent — Long subgoal chains with prerequisite-graph planning.

Handles BabyAI KeyCorridor and BlockedUnlockPickup environments where the agent
must complete 5+ sequential subgoals (explore → find key → pickup → navigate to
locked door → open → find goal → pickup goal) under 7×7 partial observation.

Architecture:
- ChainPlanner: builds prerequisite chain from SpatialMap state
- KeyCorridorAgent: executes chain using BFS navigation + automatic door handling
- KeyCorridorEnv: wrapper providing agent position and carrying state
"""

from __future__ import annotations

import re

import numpy as np

from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    SpatialMap,
    OBJ_DOOR,
    OBJ_EMPTY,
    OBJ_KEY,
    OBJ_WALL,
)

# MiniGrid actions
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5

# MiniGrid object type IDs
OBJ_BALL = 6
OBJ_BOX = 7

# Direction deltas: dir → (d_row, d_col)
DIR_DR = {0: 0, 1: 1, 2: 0, 3: -1}
DIR_DC = {0: 1, 1: 0, 2: -1, 3: 0}

TYPE_NAMES = {"key": OBJ_KEY, "ball": OBJ_BALL, "box": OBJ_BOX, "door": OBJ_DOOR}

_PICKUP_RE = re.compile(r"pick up (?:the |a )?(\w+)")
_OPEN_RE = re.compile(r"open (?:the |a )?(\w+)")


class MissionAnalyzer:
    """Parse BabyAI mission strings into (action, target_type, target_color)."""

    def analyze(self, mission: str) -> tuple[str, int, int | None]:
        """Return (action, obj_type_id, color_id_or_None)."""
        m = _PICKUP_RE.search(mission)
        if m:
            obj_name = m.group(1)
            return ("pickup", TYPE_NAMES.get(obj_name, OBJ_BALL), None)

        m = _OPEN_RE.search(mission)
        if m:
            return ("open", OBJ_DOOR, None)

        # Fallback: try to pickup ball
        return ("pickup", OBJ_BALL, None)


class Subgoal:
    """A single subgoal in the prerequisite chain."""

    __slots__ = ("name", "target_pos", "target_type", "target_color")

    def __init__(self, name: str, target_pos: tuple[int, int] | None = None,
                 target_type: int | None = None, target_color: int | None = None):
        self.name = name
        self.target_pos = target_pos
        self.target_type = target_type
        self.target_color = target_color

    def __repr__(self) -> str:
        return f"Subgoal({self.name}, pos={self.target_pos}, type={self.target_type}, color={self.target_color})"


class ChainPlanner:
    """Build prerequisite chain from map state and goal.

    The chain is built reactively — each call to build_chain() re-evaluates
    the SpatialMap and produces an updated ordered list of subgoals.
    """

    def __init__(self, goal: tuple[str, int, int | None], spatial_map: SpatialMap):
        self.goal_action, self.goal_type, self.goal_color = goal
        self.spatial_map = spatial_map

    def build_chain(self) -> list[Subgoal]:
        """Build ordered subgoal chain from current map state.

        Strategy (backward chaining):
        1. Find goal object → need to reach it → GOTO_GOAL + PICKUP_GOAL
        2. Is goal behind locked door? → need to open it → OPEN_DOOR
        3. Need key for locked door → GOTO_KEY + PICKUP_KEY
        4. If key not found → EXPLORE
        """
        chain: list[Subgoal] = []

        # Find goal object
        goal_pos = self._find_goal()

        # Find locked doors and matching keys
        locked_doors = self._find_locked_doors()
        keys = self._find_keys()

        if not goal_pos and not locked_doors:
            # Nothing found yet — explore
            return [Subgoal("EXPLORE")]

        # If there's a locked door, we need key → open chain
        if locked_doors:
            door_pos, door_color = locked_doors[0]  # first locked door

            # Find matching key
            matching_key = None
            for key_pos, key_color in keys:
                if key_color == door_color:
                    matching_key = (key_pos, key_color)
                    break

            if matching_key:
                key_pos, key_color = matching_key
                chain.append(Subgoal("GOTO_KEY", key_pos, OBJ_KEY, key_color))
                chain.append(Subgoal("PICKUP_KEY", key_pos, OBJ_KEY, key_color))
            else:
                # Key not found yet — explore to find it
                chain.append(Subgoal("EXPLORE_FOR_KEY", target_color=door_color))

            chain.append(Subgoal("GOTO_LOCKED_DOOR", door_pos, OBJ_DOOR, door_color))
            chain.append(Subgoal("OPEN_DOOR", door_pos, OBJ_DOOR, door_color))

        if goal_pos:
            chain.append(Subgoal("GOTO_GOAL", goal_pos, self.goal_type, self.goal_color))
            chain.append(Subgoal("PICKUP_GOAL", goal_pos, self.goal_type, self.goal_color))
        else:
            # Goal not found — explore after prerequisites
            chain.append(Subgoal("EXPLORE_FOR_GOAL"))
            chain.append(Subgoal("GOTO_GOAL", target_type=self.goal_type))
            chain.append(Subgoal("PICKUP_GOAL", target_type=self.goal_type))

        # If chain is empty (shouldn't happen), explore
        if not chain:
            chain = [Subgoal("EXPLORE")]

        return chain

    def _find_goal(self) -> tuple[int, int] | None:
        """Find goal object on map."""
        if self.goal_color is not None:
            return self.spatial_map.find_object_by_type_color(self.goal_type, self.goal_color)
        return self.spatial_map.find_object(self.goal_type)

    def _find_locked_doors(self) -> list[tuple[tuple[int, int], int]]:
        """Find all locked doors. Returns [(pos, color), ...]."""
        result = []
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                if (int(self.spatial_map.grid[r, c, 0]) == OBJ_DOOR and
                        int(self.spatial_map.grid[r, c, 2]) == 2):  # locked
                    color = int(self.spatial_map.grid[r, c, 1])
                    result.append(((r, c), color))
        return result

    def _find_keys(self) -> list[tuple[tuple[int, int], int]]:
        """Find all keys on map. Returns [(pos, color), ...]."""
        result = []
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                if int(self.spatial_map.grid[r, c, 0]) == OBJ_KEY:
                    color = int(self.spatial_map.grid[r, c, 1])
                    result.append(((r, c), color))
        return result


class KeyCorridorAgent:
    """Agent for KeyCorridor and BlockedUnlockPickup environments.

    Maintains a prerequisite chain of subgoals and executes them in order,
    rebuilding the chain when new objects are discovered. Handles unlocked
    doors automatically during navigation.
    """

    def __init__(self, grid_width: int, grid_height: int, mission: str):
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()

        analyzer = MissionAnalyzer()
        self._goal = analyzer.analyze(mission)

        self.phase = "EXPLORE"
        self._carrying = False
        self._carrying_type: int | None = None
        self._carrying_color: int | None = None
        self._locked_door_color: int | None = None
        self._locked_door_pos: tuple[int, int] | None = None

        self.subgoals_completed = 0
        self._completed_names: list[str] = []
        self._last_known_objects: int = 0

    def reset(self, mission: str) -> None:
        self.spatial_map.reset()
        analyzer = MissionAnalyzer()
        self._goal = analyzer.analyze(mission)
        self.phase = "EXPLORE"
        self._carrying = False
        self._carrying_type = None
        self._carrying_color = None
        self._locked_door_color = None
        self._locked_door_pos = None
        self.subgoals_completed = 0
        self._completed_names = []
        self._last_known_objects = 0

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int) -> int:
        # Update spatial map
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        if self._carrying:
            self.spatial_map.grid[agent_row, agent_col, 0] = OBJ_EMPTY
            self.spatial_map.grid[agent_row, agent_col, 1] = 0
            self.spatial_map.grid[agent_row, agent_col, 2] = 0

        # Priority 1: handle doors/objects in front of agent
        front_action = self._handle_front_cell(obs_7x7, agent_row, agent_col, agent_dir)
        if front_action is not None:
            return front_action

        # Update phase based on state
        self._update_phase()

        # Execute current phase
        if self.phase == "EXPLORE":
            return self._act_explore(agent_row, agent_col, agent_dir)
        elif self.phase == "GOTO_KEY":
            return self._act_goto_key(agent_row, agent_col, agent_dir, obs_7x7)
        elif self.phase == "GOTO_LOCKED_DOOR":
            return self._act_goto_locked_door(agent_row, agent_col, agent_dir)
        elif self.phase == "GOTO_GOAL":
            return self._act_goto_goal(agent_row, agent_col, agent_dir, obs_7x7)
        elif self.phase == "DROP_KEY":
            return self._act_drop_key(agent_row, agent_col, agent_dir)
        else:
            return self._act_explore(agent_row, agent_col, agent_dir)

    def _handle_front_cell(self, obs_7x7: np.ndarray,
                           agent_row: int, agent_col: int,
                           agent_dir: int) -> int | None:
        """Handle immediate interactions with front cell."""
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        front_color = int(obs_7x7[3, 5, 1])

        # Closed unlocked door → toggle (navigation primitive)
        if front_obj == OBJ_DOOR and front_state == 1:
            return ACT_TOGGLE

        # Locked door + have matching key → toggle
        if (front_obj == OBJ_DOOR and front_state == 2 and
                self._carrying and self._carrying_type == OBJ_KEY and
                self._carrying_color == front_color):
            self._complete_subgoal("OPEN_DOOR")
            return ACT_TOGGLE

        # Key in front and not carrying anything → pickup
        if (front_obj == OBJ_KEY and not self._carrying and
                self._should_pickup_key(front_color)):
            self._complete_subgoal("PICKUP_KEY")
            return ACT_PICKUP

        # Goal object in front, not carrying, prereqs met → pickup
        if (front_obj == self._goal[1] and not self._carrying and
                self._prereqs_met()):
            self._complete_subgoal("PICKUP_GOAL")
            return ACT_PICKUP

        # In DROP_KEY phase: if facing empty cell → drop key
        if (self.phase == "DROP_KEY" and self._carrying and
                front_obj == OBJ_EMPTY):
            fr = agent_row + DIR_DR[agent_dir]
            fc = agent_col + DIR_DC[agent_dir]
            if (0 <= fr < self.spatial_map.height and
                    0 <= fc < self.spatial_map.width):
                self._complete_subgoal("DROP_KEY")
                return ACT_DROP

        return None

    def _should_pickup_key(self, key_color: int) -> bool:
        """Should we pick up this key?"""
        # If door already opened, never pick up keys
        if "OPEN_DOOR" in self._completed_names:
            return False
        # If we know which locked door color we need, match it
        if self._locked_door_color is not None:
            return key_color == self._locked_door_color
        # If we don't know yet, pick up any key (explore phase)
        return True

    def _prereqs_met(self) -> bool:
        """Are all prerequisites met to pick up the goal?"""
        # All locked doors opened (or none existed)
        if "OPEN_DOOR" in self._completed_names:
            return True
        planner = ChainPlanner(self._goal, self.spatial_map)
        locked = planner._find_locked_doors()
        return len(locked) == 0

    def _update_phase(self) -> None:
        """Update phase based on map state and carrying.

        MiniGrid 3.0: key is NOT consumed when opening a locked door.
        Track door-opened via _completed_names, not carrying state.
        """
        planner = ChainPlanner(self._goal, self.spatial_map)
        locked_doors = planner._find_locked_doors()
        keys = planner._find_keys()
        goal_pos = planner._find_goal()

        # Track locked door info (only if not yet opened)
        if locked_doors and "OPEN_DOOR" not in self._completed_names:
            self._locked_door_pos, self._locked_door_color = locked_doors[0]

        # Phase transitions — priority order
        if "DROP_KEY" in self._completed_names and not self._carrying:
            # Key dropped → go pickup goal
            if goal_pos is not None:
                self.phase = "GOTO_GOAL"
            else:
                self.phase = "EXPLORE"
        elif "OPEN_DOOR" in self._completed_names:
            # Door opened → navigate to goal, then drop key if carrying
            if goal_pos is not None:
                if self._carrying:
                    # Check if adjacent to goal — if so, drop key
                    gr, gc = goal_pos
                    # Use _goal_adjacent check from GOTO_GOAL
                    self.phase = "GOTO_GOAL"  # navigate to goal even while carrying
                else:
                    self.phase = "GOTO_GOAL"
            else:
                self.phase = "EXPLORE"
        elif self._carrying and self._carrying_type == OBJ_KEY:
            # Have key, door not yet opened → go to locked door
            if self._locked_door_pos is not None:
                if self.phase != "GOTO_LOCKED_DOOR":
                    self._complete_subgoal("GOTO_KEY")
                self.phase = "GOTO_LOCKED_DOOR"
            else:
                self.phase = "EXPLORE"  # explore to find locked door
        elif locked_doors:
            # Need key for locked door
            matching_key = None
            for kp, kc in keys:
                if kc == locked_doors[0][1]:
                    matching_key = (kp, kc)
                    break
            if matching_key:
                self.phase = "GOTO_KEY"
            else:
                self.phase = "EXPLORE"  # explore to find key
        elif goal_pos is not None:
            # No locked doors, goal visible → go to it
            self.phase = "GOTO_GOAL"
        else:
            self.phase = "EXPLORE"

    def _act_explore(self, agent_row: int, agent_col: int, agent_dir: int) -> int:
        return self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

    def _act_goto_key(self, agent_row: int, agent_col: int, agent_dir: int,
                      obs_7x7: np.ndarray) -> int:
        """Navigate to key position."""
        planner = ChainPlanner(self._goal, self.spatial_map)
        keys = planner._find_keys()

        target_key = None
        if self._locked_door_color is not None:
            for kp, kc in keys:
                if kc == self._locked_door_color:
                    target_key = kp
                    break
        elif keys:
            target_key = keys[0][0]

        if target_key is None:
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Check if adjacent
        dr = abs(agent_row - target_key[0])
        dc = abs(agent_col - target_key[1])
        if dr + dc == 1:
            # Face key and pickup
            return self._turn_toward(target_key[0], target_key[1],
                                     agent_row, agent_col, agent_dir)

        adj = self._find_adjacent_walkable(target_key, agent_row, agent_col)
        if adj is None:
            return self._act_explore(agent_row, agent_col, agent_dir)
        return self._navigate_to(adj[0], adj[1], agent_row, agent_col, agent_dir)

    def _act_goto_locked_door(self, agent_row: int, agent_col: int,
                              agent_dir: int) -> int:
        """Navigate to cell adjacent to locked door."""
        if self._locked_door_pos is None:
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Check if adjacent
        dr = abs(agent_row - self._locked_door_pos[0])
        dc = abs(agent_col - self._locked_door_pos[1])
        if dr + dc == 1:
            # Face door and toggle
            return self._turn_toward(self._locked_door_pos[0], self._locked_door_pos[1],
                                     agent_row, agent_col, agent_dir)

        adj = self._find_adjacent_walkable(self._locked_door_pos, agent_row, agent_col)
        if adj is None:
            return self._act_explore(agent_row, agent_col, agent_dir)
        return self._navigate_to(adj[0], adj[1], agent_row, agent_col, agent_dir)

    def _act_goto_goal(self, agent_row: int, agent_col: int, agent_dir: int,
                       obs_7x7: np.ndarray) -> int:
        """Navigate to goal object. If carrying key and adjacent, switch to DROP_KEY."""
        planner = ChainPlanner(self._goal, self.spatial_map)
        goal_pos = planner._find_goal()

        if goal_pos is None:
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Goal objects (ball, box) can't be walked onto — navigate to adjacent
        dr = abs(agent_row - goal_pos[0])
        dc = abs(agent_col - goal_pos[1])
        if dr + dc == 1:
            # Adjacent to goal
            if self._carrying:
                # Need to drop key first
                self._complete_subgoal("GOTO_GOAL")
                self.phase = "DROP_KEY"
                return self._act_drop_key(agent_row, agent_col, agent_dir)
            return self._turn_toward(goal_pos[0], goal_pos[1],
                                     agent_row, agent_col, agent_dir)

        adj = self._find_adjacent_walkable(goal_pos, agent_row, agent_col)
        if adj is None:
            return self._act_explore(agent_row, agent_col, agent_dir)
        return self._navigate_to(adj[0], adj[1], agent_row, agent_col, agent_dir)

    def _act_drop_key(self, agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Drop key to free hands for goal pickup.

        Strategy: find an empty cell adjacent to agent, face it, drop.
        Prefer cells that don't block path to goal.
        """
        # Check if current facing direction has an empty cell
        fr = agent_row + DIR_DR[agent_dir]
        fc = agent_col + DIR_DC[agent_dir]
        if (0 <= fr < self.spatial_map.height and
                0 <= fc < self.spatial_map.width):
            cell = int(self.spatial_map.grid[fr, fc, 0])
            if cell in (OBJ_EMPTY, 1):  # empty or floor
                return ACT_DROP

        # Try other directions — find empty adjacent cell and turn to face it
        for try_dir in range(4):
            if try_dir == agent_dir:
                continue
            tr = agent_row + DIR_DR[try_dir]
            tc = agent_col + DIR_DC[try_dir]
            if (0 <= tr < self.spatial_map.height and
                    0 <= tc < self.spatial_map.width):
                cell = int(self.spatial_map.grid[tr, tc, 0])
                if cell in (OBJ_EMPTY, 1):
                    diff = (try_dir - agent_dir) % 4
                    return ACT_RIGHT if diff <= 2 else ACT_LEFT

        # No empty cell found — move away and try again
        return ACT_FORWARD

    def _find_adjacent_walkable(self, pos: tuple[int, int],
                                agent_row: int, agent_col: int) -> tuple[int, int] | None:
        """Find walkable cell adjacent to pos, closest to agent."""
        obs = self._pathfinding_obs()
        best = None
        best_dist = float("inf")

        for ddr, ddc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ar, ac = pos[0] + ddr, pos[1] + ddc
            if not (0 <= ar < self.spatial_map.height and 0 <= ac < self.spatial_map.width):
                continue
            obj = int(obs[ar, ac, 0])
            if obj in (OBJ_WALL,):
                continue
            # Objects (key, ball, box) block movement
            orig_obj = int(self.spatial_map.grid[ar, ac, 0])
            if orig_obj in (OBJ_KEY, OBJ_BALL, OBJ_BOX):
                continue
            path = self.pathfinder.find_path(
                obs, (agent_row, agent_col), (ar, ac), allow_door=True
            )
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (ar, ac)

        return best

    def _navigate_to(self, target_row: int, target_col: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        if agent_row == target_row and agent_col == target_col:
            return int(np.random.randint(0, 3))

        obs = self._pathfinding_obs()
        path = self.pathfinder.find_path(
            obs, (agent_row, agent_col), (target_row, target_col),
            allow_door=True
        )
        if path is None or len(path) <= 1:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )

        actions = self.pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return ACT_FORWARD

    def _pathfinding_obs(self) -> np.ndarray:
        """Create observation for BFS. Objects are walls, doors are passable."""
        obs = self.spatial_map.to_obs()
        for obj_type in (OBJ_KEY, OBJ_BALL, OBJ_BOX):
            mask = obs[:, :, 0] == obj_type
            obs[mask, 0] = OBJ_WALL
        return obs

    def _turn_toward(self, target_row: int, target_col: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        dr = target_row - agent_row
        dc = target_col - agent_col
        need_dir = self._dir_from_delta(dr, dc)
        if need_dir is None or need_dir == agent_dir:
            return ACT_FORWARD
        diff = (need_dir - agent_dir) % 4
        return ACT_RIGHT if diff <= 2 else ACT_LEFT

    @staticmethod
    def _dir_from_delta(dr: int, dc: int) -> int | None:
        if dc > 0:
            return 0
        if dr > 0:
            return 1
        if dc < 0:
            return 2
        if dr < 0:
            return 3
        return None

    def update_carrying(self, obj_type: int, obj_color: int) -> None:
        if not self._carrying:
            self._complete_subgoal("PICKUP_KEY" if obj_type == OBJ_KEY else "PICKUP_ITEM")
        self._carrying = True
        self._carrying_type = obj_type
        self._carrying_color = obj_color
        # Remove picked-up object from map
        if obj_type == OBJ_KEY:
            pos = self.spatial_map.find_object_by_type_color(obj_type, obj_color)
            if pos is not None:
                self.spatial_map.grid[pos[0], pos[1], 0] = OBJ_EMPTY
                self.spatial_map.grid[pos[0], pos[1], 1] = 0
                self.spatial_map.grid[pos[0], pos[1], 2] = 0

    def clear_carrying(self) -> None:
        self._carrying = False
        self._carrying_type = None
        self._carrying_color = None

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       reward: float) -> None:
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        if self._carrying:
            self.spatial_map.grid[agent_row, agent_col, 0] = OBJ_EMPTY
            self.spatial_map.grid[agent_row, agent_col, 1] = 0
            self.spatial_map.grid[agent_row, agent_col, 2] = 0

    def _complete_subgoal(self, name: str) -> None:
        if name not in self._completed_names:
            self._completed_names.append(name)
            self.subgoals_completed += 1


class KeyCorridorEnv:
    """Wrapper for BabyAI KeyCorridor and BlockedUnlockPickup environments."""

    def __init__(self, env_name: str = "BabyAI-KeyCorridorS4R3-v0"):
        import minigrid
        minigrid.register_minigrid_envs()
        import gymnasium as gym
        self._env = gym.make(env_name)
        self._mission: str = ""
        self.grid_width: int = 0
        self.grid_height: int = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, int, int, str]:
        """Reset → (obs_7x7, agent_col, agent_row, agent_dir, mission)."""
        obs, info = self._env.reset(seed=seed)
        uw = self._env.unwrapped
        self._mission = obs["mission"]
        self.grid_width = uw.grid.width
        self.grid_height = uw.grid.height
        pos = uw.agent_pos
        return (obs["image"], int(pos[0]), int(pos[1]), int(uw.agent_dir),
                self._mission)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, int, int, int]:
        """Step → (obs_7x7, reward, term, trunc, agent_col, agent_row, agent_dir)."""
        obs, reward, term, trunc, info = self._env.step(action)
        uw = self._env.unwrapped
        pos = uw.agent_pos
        return (obs["image"], float(reward), term, trunc,
                int(pos[0]), int(pos[1]), int(uw.agent_dir))

    @property
    def carrying_type_color(self) -> tuple[int, int] | None:
        """Return (type_id, color_id) of carried object, or None."""
        uw = self._env.unwrapped
        obj = uw.carrying
        if obj is None:
            return None
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        return (OBJECT_TO_IDX[obj.type], COLOR_TO_IDX[obj.color])

    @property
    def carrying(self):
        return self._env.unwrapped.carrying

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
