"""Stage 61: Demo-Guided Agent.

Agent that uses CausalWorldModel (Stage 60) for planning in MiniGrid
environments with partial observability. Learns causal rules from
demonstrations, explores layout via FrontierExplorer, builds plans
via backward chaining, executes via BFS navigation.

Components:
- CausalPlanner: backward chaining → executable subgoals with grid positions
- SubgoalExecutor: state machine (EXPLORE → NAVIGATE → INTERACT)
- DemoGuidedAgent: full agent combining exploration, planning, execution
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from snks.agent.causal_world_model import CausalWorldModel
from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    SpatialMap,
    OBJ_DOOR,
    OBJ_GOAL,
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

# MiniGrid color indices
COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
COLOR_PURPLE = 3
COLOR_YELLOW = 4
COLOR_GREY = 5

COLOR_NAMES = {
    0: "red", 1: "green", 2: "blue",
    3: "purple", 4: "yellow", 5: "grey",
}
COLOR_IDS = {v: k for k, v in COLOR_NAMES.items()}


@dataclass
class ExecutableSubgoal:
    """A subgoal bound to grid position and action."""
    name: str
    target_pos: tuple[int, int] | None  # (row, col), None if not yet found
    action_at_target: int | None        # ACT_PICKUP, ACT_TOGGLE, or None
    precondition: str | None            # "adjacent", "has_key", etc.
    target_obj_type: int | None = None  # OBJ_KEY, OBJ_DOOR, OBJ_GOAL
    target_color: int | None = None     # color index for color-specific search


class ExecutorState(Enum):
    EXPLORE = auto()
    NAVIGATE = auto()
    INTERACT = auto()


class CausalPlanner:
    """Converts causal chains into executable subgoals with grid positions.

    Uses CausalWorldModel.query_chain() for backward chaining,
    then binds each abstract step to concrete grid positions from SpatialMap.
    """

    def __init__(self, causal_model: CausalWorldModel):
        self.model = causal_model

    def plan(self, goal: str, door_color_name: str,
             spatial_map: SpatialMap) -> list[ExecutableSubgoal]:
        """Generate executable subgoals for a goal.

        Args:
            goal: high-level goal (e.g. "pass_locked_door")
            door_color_name: color name of the target door
            spatial_map: current spatial map with known positions
        """
        chain = self.model.query_chain(goal, door_color_name)
        if not chain or chain == ["cannot_plan"]:
            return []

        # Determine needed key color via causal model
        key_color_name = self.model.query_precondition("open", door_color_name)
        key_color_id = COLOR_IDS.get(key_color_name)
        door_color_id = COLOR_IDS.get(door_color_name)

        subgoals: list[ExecutableSubgoal] = []

        for step in chain:
            if step == "find_key":
                key_pos = None
                if key_color_id is not None:
                    key_pos = spatial_map.find_object_by_type_color(
                        OBJ_KEY, key_color_id
                    )
                if key_pos is None:
                    key_pos = spatial_map.find_object(OBJ_KEY)
                subgoals.append(ExecutableSubgoal(
                    name="find_key",
                    target_pos=key_pos,
                    action_at_target=None,
                    precondition=None,
                    target_obj_type=OBJ_KEY,
                    target_color=key_color_id,
                ))

            elif step == "pickup_key":
                key_pos = None
                if key_color_id is not None:
                    key_pos = spatial_map.find_object_by_type_color(
                        OBJ_KEY, key_color_id
                    )
                if key_pos is None:
                    key_pos = spatial_map.find_object(OBJ_KEY)
                subgoals.append(ExecutableSubgoal(
                    name="pickup_key",
                    target_pos=key_pos,
                    action_at_target=ACT_PICKUP,
                    precondition="adjacent",
                    target_obj_type=OBJ_KEY,
                    target_color=key_color_id,
                ))

            elif step == "open_door":
                door_pos = None
                if door_color_id is not None:
                    door_pos = spatial_map.find_object_by_type_color(
                        OBJ_DOOR, door_color_id
                    )
                if door_pos is None:
                    door_pos = spatial_map.find_object(OBJ_DOOR)
                subgoals.append(ExecutableSubgoal(
                    name="open_door",
                    target_pos=door_pos,
                    action_at_target=ACT_TOGGLE,
                    precondition="has_key",
                    target_obj_type=OBJ_DOOR,
                    target_color=door_color_id,
                ))

            elif step == "pass_through":
                goal_pos = spatial_map.find_object(OBJ_GOAL)
                subgoals.append(ExecutableSubgoal(
                    name="pass_through",
                    target_pos=goal_pos,
                    action_at_target=None,
                    precondition=None,
                    target_obj_type=OBJ_GOAL,
                ))

        return subgoals

    def replan(self, subgoals: list[ExecutableSubgoal],
               spatial_map: SpatialMap) -> list[ExecutableSubgoal]:
        """Update target positions of unresolved subgoals from current map."""
        for sg in subgoals:
            if sg.target_pos is not None:
                continue
            if sg.target_obj_type is not None:
                if sg.target_color is not None:
                    pos = spatial_map.find_object_by_type_color(
                        sg.target_obj_type, sg.target_color
                    )
                else:
                    pos = spatial_map.find_object(sg.target_obj_type)
                sg.target_pos = pos
        return subgoals


class SubgoalExecutor:
    """Executes subgoals one by one via state machine.

    States: EXPLORE → NAVIGATE → INTERACT
    """

    def __init__(self, spatial_map: SpatialMap,
                 explorer: FrontierExplorer,
                 pathfinder: GridPathfinder):
        self.spatial_map = spatial_map
        self.explorer = explorer
        self.pathfinder = pathfinder
        self.state = ExecutorState.EXPLORE

    def select_action(self, subgoal: ExecutableSubgoal,
                      agent_row: int, agent_col: int,
                      agent_dir: int, obs_7x7: np.ndarray) -> int:
        """Select action to progress toward current subgoal."""
        # Update state based on subgoal info
        self._update_state(subgoal, agent_row, agent_col, obs_7x7)

        if self.state == ExecutorState.EXPLORE:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )

        if self.state == ExecutorState.INTERACT:
            return self._interact(subgoal, agent_row, agent_col, agent_dir)

        # NAVIGATE
        return self._navigate(subgoal, agent_row, agent_col, agent_dir)

    def _update_state(self, subgoal: ExecutableSubgoal,
                      agent_row: int, agent_col: int,
                      obs_7x7: np.ndarray) -> None:
        """Determine current executor state."""
        if subgoal.target_pos is None:
            self.state = ExecutorState.EXPLORE
            return

        tr, tc = subgoal.target_pos

        # For pickup/toggle: need to be adjacent
        if subgoal.action_at_target is not None:
            dist = abs(agent_row - tr) + abs(agent_col - tc)
            if dist == 1:
                self.state = ExecutorState.INTERACT
            else:
                self.state = ExecutorState.NAVIGATE
        else:
            # For pass_through/find: just navigate to position
            if agent_row == tr and agent_col == tc:
                self.state = ExecutorState.INTERACT
            else:
                self.state = ExecutorState.NAVIGATE

    def _navigate(self, subgoal: ExecutableSubgoal,
                  agent_row: int, agent_col: int,
                  agent_dir: int) -> int:
        """BFS navigate toward subgoal target."""
        tr, tc = subgoal.target_pos  # type: ignore

        # For interaction subgoals, navigate to adjacent cell
        if subgoal.action_at_target is not None:
            adj = self._best_adjacent(tr, tc, agent_row, agent_col)
            if adj is not None:
                tr, tc = adj

        if agent_row == tr and agent_col == tc:
            return int(np.random.randint(0, 3))

        obs = self.spatial_map.to_obs()
        allow_door = subgoal.name in ("open_door", "pass_through")
        path = self.pathfinder.find_path(
            obs, (agent_row, agent_col), (tr, tc), allow_door=allow_door
        )
        if path is None:
            path = self.pathfinder.find_path(
                obs, (agent_row, agent_col), (tr, tc), allow_door=True
            )
        if path is None or len(path) <= 1:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )

        actions = self.pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return ACT_FORWARD

    def _interact(self, subgoal: ExecutableSubgoal,
                  agent_row: int, agent_col: int,
                  agent_dir: int) -> int:
        """Turn to face target, then execute action."""
        if subgoal.action_at_target is None:
            # Just need to be here (pass_through/find)
            return ACT_FORWARD

        tr, tc = subgoal.target_pos  # type: ignore
        dr = tr - agent_row
        dc = tc - agent_col

        need_dir = _dir_from_delta(dr, dc)
        if need_dir is not None and need_dir != agent_dir:
            diff = (need_dir - agent_dir) % 4
            return ACT_RIGHT if diff <= 2 else ACT_LEFT

        # Facing target — execute action
        return subgoal.action_at_target

    def _best_adjacent(self, target_row: int, target_col: int,
                       agent_row: int, agent_col: int
                       ) -> tuple[int, int] | None:
        """Find best reachable cell adjacent to target."""
        obs = self.spatial_map.to_obs()
        best = None
        best_dist = float("inf")

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ar, ac = target_row + dr, target_col + dc
            if not (0 <= ar < self.spatial_map.height
                    and 0 <= ac < self.spatial_map.width):
                continue
            obj = int(obs[ar, ac, 0])
            if obj in (OBJ_WALL, OBJ_DOOR, OBJ_KEY):
                continue
            path = self.pathfinder.find_path(
                obs, (agent_row, agent_col), (ar, ac), allow_door=True
            )
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (ar, ac)

        return best

    def is_subgoal_achieved(self, subgoal: ExecutableSubgoal,
                            obs_7x7: np.ndarray,
                            has_key: bool, door_open: bool) -> bool:
        """Check if subgoal is achieved."""
        if subgoal.name == "find_key":
            return subgoal.target_pos is not None
        if subgoal.name == "pickup_key":
            return has_key
        if subgoal.name == "open_door":
            return door_open
        if subgoal.name == "pass_through":
            return False  # detected by env termination
        return False


class DemoGuidedAgent:
    """Agent that learns causal rules from demos, then plans and executes.

    Lifecycle:
    1. learn_from_demos() — teach causal rules
    2. reset() — start new episode
    3. select_action() loop — explore → plan → execute
    """

    def __init__(self, grid_width: int, grid_height: int,
                 causal_dim: int = 512, seed: int = 42):
        self.causal_model = CausalWorldModel(
            dim=causal_dim, seed=seed
        )
        self.planner = CausalPlanner(self.causal_model)
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()
        self.executor = SubgoalExecutor(
            self.spatial_map, self.explorer, self.pathfinder
        )

        self._has_key = False
        self._door_open = False
        self._plan: list[ExecutableSubgoal] = []
        self._current_sg_idx = 0
        self._trained = False

        # Stats
        self.explore_steps = 0
        self.execute_steps = 0
        self.plan_ready = False

    def learn_from_demos(self, colors: list[str]) -> None:
        """Teach causal rules from demonstration colors."""
        self.causal_model.learn_all_rules(colors)
        self._trained = True

    def reset(self) -> None:
        """Reset for new episode."""
        self.spatial_map.reset()
        self._has_key = False
        self._door_open = False
        self._locked_door_pos = None
        self._plan = []
        self._current_sg_idx = 0
        self.explore_steps = 0
        self.execute_steps = 0
        self.plan_ready = False

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int,
                      agent_dir: int) -> int:
        """Main action selection."""
        # Update spatial map
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Check immediate interactions
        action = self._check_immediate(obs_7x7, agent_row, agent_col, agent_dir)
        if action is not None:
            return action

        # Try to build/update plan
        if not self._plan:
            self._try_plan()

        if self._plan:
            self.planner.replan(self._plan, self.spatial_map)

        # Execute current subgoal or explore
        if self._plan and self._current_sg_idx < len(self._plan):
            sg = self._plan[self._current_sg_idx]
            if sg.target_pos is None:
                # Target not yet found — explore (avoid locked doors)
                self.explore_steps += 1
                return self._explore_avoiding_locked(
                    agent_row, agent_col, agent_dir
                )
            self.execute_steps += 1
            return self.executor.select_action(
                sg, agent_row, agent_col, agent_dir, obs_7x7
            )

        # No plan yet — explore
        self.explore_steps += 1
        return self._explore_avoiding_locked(
            agent_row, agent_col, agent_dir
        )

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int,
                       agent_dir: int, reward: float,
                       carrying: bool) -> None:
        """Update state after action execution."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        prev_has_key = self._has_key
        self._has_key = carrying

        # Detect door open
        self._detect_door_state()

        # Check subgoal advancement
        if self._plan and self._current_sg_idx < len(self._plan):
            sg = self._plan[self._current_sg_idx]
            if self.executor.is_subgoal_achieved(
                sg, obs_7x7, self._has_key, self._door_open
            ):
                self._current_sg_idx += 1

    def _try_plan(self) -> None:
        """Attempt to build a plan from current map knowledge."""
        # Find a LOCKED door (state=2), not just any door
        door_pos = self._find_locked_door()
        if door_pos is None:
            return

        self._locked_door_pos = door_pos

        # Get door color
        door_color_id = int(self.spatial_map.grid[door_pos[0], door_pos[1], 1])
        door_color_name = COLOR_NAMES.get(door_color_id)
        if door_color_name is None:
            return

        self._plan = self.planner.plan(
            "pass_locked_door", door_color_name, self.spatial_map
        )
        if self._plan:
            self._current_sg_idx = 0
            self.plan_ready = True

    def _find_locked_door(self) -> tuple[int, int] | None:
        """Find position of a locked door (state=2) in spatial map."""
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                obj = int(self.spatial_map.grid[r, c, 0])
                state = int(self.spatial_map.grid[r, c, 2])
                if obj == OBJ_DOOR and state == 2:
                    return (r, c)
        return None

    def _check_immediate(self, obs_7x7: np.ndarray,
                         agent_row: int, agent_col: int,
                         agent_dir: int) -> int | None:
        """Check for immediate interaction opportunities."""
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        front_color = int(obs_7x7[3, 5, 1])

        # Facing key and should pick it up
        if front_obj == OBJ_KEY and not self._has_key:
            # Check if this is the right key (color match)
            if self._should_pickup_key(front_color):
                return ACT_PICKUP

        # Facing locked door with key — toggle
        if front_obj == OBJ_DOOR and front_state == 2 and self._has_key:
            door_color_name = COLOR_NAMES.get(front_color, "unknown")
            # Verify we have the right key via causal model
            if self._trained:
                # In DoorKey there's only one key, so always toggle
                return ACT_TOGGLE
            return ACT_TOGGLE

        # Facing closed (not locked) door — toggle
        if front_obj == OBJ_DOOR and front_state == 1:
            return ACT_TOGGLE

        # Not facing a door, but adjacent to a closed door — turn toward it
        turn = self._turn_toward_closed_door(agent_row, agent_col, agent_dir)
        if turn is not None:
            return turn

        return None

    def _turn_toward_closed_door(self, agent_row: int, agent_col: int,
                                 agent_dir: int) -> int | None:
        """If adjacent to a closed (not locked) door, turn to face it."""
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = agent_row + dr, agent_col + dc
            if not (0 <= nr < self.spatial_map.height
                    and 0 <= nc < self.spatial_map.width):
                continue
            if not self.spatial_map.explored[nr, nc]:
                continue
            obj = int(self.spatial_map.grid[nr, nc, 0])
            state = int(self.spatial_map.grid[nr, nc, 2])
            if obj == OBJ_DOOR and state == 1:  # closed, not locked
                need_dir = _dir_from_delta(dr, dc)
                if need_dir is not None and need_dir != agent_dir:
                    diff = (need_dir - agent_dir) % 4
                    return ACT_RIGHT if diff <= 2 else ACT_LEFT
        return None

    def _should_pickup_key(self, key_color_id: int) -> bool:
        """Check if we should pick up this key based on causal model."""
        if not self._trained:
            return True

        # Find the locked door specifically
        door_pos = self._find_locked_door() if not hasattr(self, '_locked_door_pos') or self._locked_door_pos is None else self._locked_door_pos
        if door_pos is None:
            # No locked door found yet — in DoorKey there's only one door,
            # so check any door; in LockedRoom, skip wrong keys
            door_pos = self.spatial_map.find_object(OBJ_DOOR)
            if door_pos is None:
                return True  # no door found yet, pick up any key

        door_color_id = int(self.spatial_map.grid[door_pos[0], door_pos[1], 1])
        door_color_name = COLOR_NAMES.get(door_color_id, "unknown")
        key_color_name = COLOR_NAMES.get(key_color_id, "unknown")

        return self.causal_model.query_color_match(key_color_name, door_color_name)

    def _explore_avoiding_locked(self, agent_row: int, agent_col: int,
                                  agent_dir: int) -> int:
        """Explore while treating locked doors as walls.

        Temporarily masks locked doors in the spatial map so
        FrontierExplorer's BFS won't route through them.
        """
        # Find locked doors and temporarily mark as walls
        locked_cells: list[tuple[int, int, int, int, int]] = []
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                obj = int(self.spatial_map.grid[r, c, 0])
                state = int(self.spatial_map.grid[r, c, 2])
                if obj == OBJ_DOOR and state == 2:
                    color = int(self.spatial_map.grid[r, c, 1])
                    locked_cells.append((r, c, obj, color, state))
                    self.spatial_map.grid[r, c, 0] = OBJ_WALL

        action = self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

        # Restore locked doors
        for r, c, obj, color, state in locked_cells:
            self.spatial_map.grid[r, c, 0] = obj
            self.spatial_map.grid[r, c, 1] = color
            self.spatial_map.grid[r, c, 2] = state

        return action

    def _detect_door_state(self) -> None:
        """Detect if the locked door is now open."""
        if not hasattr(self, '_locked_door_pos') or self._locked_door_pos is None:
            return
        r, c = self._locked_door_pos
        if self.spatial_map.explored[r, c]:
            state = int(self.spatial_map.grid[r, c, 2])
            if state == 0:  # open
                self._door_open = True

    def get_stats(self) -> dict:
        """Return episode statistics."""
        return {
            "explore_steps": self.explore_steps,
            "execute_steps": self.execute_steps,
            "plan_ready": self.plan_ready,
            "subgoals_completed": self._current_sg_idx,
            "total_subgoals": len(self._plan),
            "causal_model_stats": self.causal_model.get_stats(),
        }


def _dir_from_delta(dr: int, dc: int) -> int | None:
    """Convert (dr, dc) to MiniGrid direction."""
    if dc > 0:
        return 0  # right
    if dr > 0:
        return 1  # down
    if dc < 0:
        return 2  # left
    if dr < 0:
        return 3  # up
    return None
