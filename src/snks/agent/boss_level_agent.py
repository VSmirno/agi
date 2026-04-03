"""Stage 62: BossLevel Agent.

Extends DemoGuidedAgent to handle BabyAI BossLevel — 22x22 grids,
5 object types (key, door, ball, box, wall), compound missions,
and multi-step subgoal chains learned from Bot demonstrations.

Key differences from Stage 61 DemoGuidedAgent:
- MissionModel for mission→subgoal sequence mapping
- Generic CausalPlanner (no hardcoded 4-step chain)
- Extended SubgoalExecutor (DROP, GO_TO, inventory tracking)
- Object types: Ball + Box support in spatial map
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snks.agent.causal_world_model import CausalWorldModel
from snks.agent.demo_guided_agent import (
    ACT_DROP,
    ACT_FORWARD,
    ACT_LEFT,
    ACT_PICKUP,
    ACT_RIGHT,
    ACT_TOGGLE,
    COLOR_IDS,
    COLOR_NAMES,
    ExecutableSubgoal,
    ExecutorState,
    _dir_from_delta,
)
from snks.agent.mission_model import (
    MissionModel,
    Subgoal,
    SG_DROP,
    SG_GO_TO,
    SG_OPEN,
    SG_PICK_UP,
    SG_PUT_NEXT_TO,
)
from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    OBJ_BALL,
    OBJ_BOX,
    OBJ_DOOR,
    OBJ_KEY,
    OBJ_WALL,
    SpatialMap,
)

# Map string object names to spatial map type IDs
OBJ_NAME_TO_ID = {
    "key": OBJ_KEY,
    "door": OBJ_DOOR,
    "ball": OBJ_BALL,
    "box": OBJ_BOX,
}


class BossLevelAgent:
    """Agent for BabyAI BossLevel environments.

    Lifecycle:
    1. train(demos) — learn causal rules + mission→subgoal mappings
    2. reset(mission) — start new episode with mission string
    3. select_action() loop — explore → plan → execute
    """

    def __init__(self, grid_width: int = 22, grid_height: int = 22,
                 causal_dim: int = 512, seed: int = 42):
        self.causal_model = CausalWorldModel(dim=causal_dim, seed=seed)
        self.mission_model = MissionModel(dim=causal_dim, seed=seed + 50)
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()

        self._trained = False
        self._mission = ""
        self._plan: list[ExecutableSubgoal] = []
        self._current_sg_idx = 0
        self._carrying: tuple[str, str] | None = None  # (type, color)
        self._completed_subgoals: list[bool] = []

        # Anti-stuck tracking
        self._last_pos: tuple[int, int] | None = None
        self._stuck_count = 0

        # Stats
        self.explore_steps = 0
        self.execute_steps = 0

    def train(self, demos: list[dict],
              colors: list[str] | None = None) -> dict:
        """Train from Bot demonstration episodes.

        Args:
            demos: list of demo dicts from generate_bosslevel_demos.py
            colors: colors for causal model. If None, extract from demos.
        """
        if colors is None:
            colors = list({
                sg.get("color", "")
                for d in demos
                for sg in d.get("subgoals_extracted", [])
                if sg.get("color")
            })
            if not colors:
                colors = ["red", "green", "blue", "purple", "yellow", "grey"]

        self.causal_model.learn_all_rules(colors)
        n_learned = self.mission_model.train_from_demos(demos)
        self._trained = True

        return {"colors": colors, "demos_learned": n_learned}

    def reset(self, mission: str = "") -> None:
        """Reset for new episode."""
        self.spatial_map.reset()
        self._mission = mission
        self._carrying = None
        self._plan = []
        self._current_sg_idx = 0
        self._completed_subgoals = []
        self.explore_steps = 0
        self.execute_steps = 0

        # Generate plan from mission
        if mission and self._trained:
            self._build_plan(mission)

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int,
                      agent_dir: int) -> int:
        """Main action selection loop."""
        # Update spatial map
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Check immediate interactions (open closed doors for navigation)
        # But skip if current subgoal targets this door specifically
        if not self._is_targeting_front_door(obs_7x7, agent_row, agent_col, agent_dir):
            action = self._check_immediate_nav(obs_7x7, agent_row, agent_col, agent_dir)
            if action is not None:
                return action

        # Check if current "open" subgoal needs a key we don't have
        self._maybe_insert_key_subgoals()

        # Execute current subgoal
        if self._plan and self._current_sg_idx < len(self._plan):
            sg = self._plan[self._current_sg_idx]

            # Try to resolve target position
            self._resolve_target(sg)

            if sg.target_pos is None:
                # Target not yet found — explore
                self.explore_steps += 1
                return self._explore(agent_row, agent_col, agent_dir)

            # Anti-stuck for execution too
            current_pos = (agent_row, agent_col)
            if current_pos == self._last_pos:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_pos = current_pos

            if self._stuck_count >= 3:
                self._stuck_count = 0
                return ACT_LEFT if np.random.random() > 0.5 else ACT_RIGHT

            self.execute_steps += 1
            return self._execute_subgoal(sg, agent_row, agent_col, agent_dir, obs_7x7)

        # No plan or all subgoals done — explore (safety net)
        self.explore_steps += 1
        return self._explore(agent_row, agent_col, agent_dir)

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int,
                       agent_dir: int, reward: float,
                       carrying_type: str | None,
                       carrying_color: str | None) -> None:
        """Update state after action."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Update inventory
        if carrying_type is not None:
            self._carrying = (carrying_type, carrying_color or "")
        else:
            self._carrying = None

        # Check subgoal advancement
        if self._plan and self._current_sg_idx < len(self._plan):
            sg = self._plan[self._current_sg_idx]
            if self._is_subgoal_achieved(sg, agent_row, agent_col):
                self._completed_subgoals.append(True)
                self._current_sg_idx += 1

    def _is_targeting_front_door(self, obs_7x7: np.ndarray,
                                agent_row: int, agent_col: int,
                                agent_dir: int) -> bool:
        """Check if current subgoal targets the door we're facing."""
        if not self._plan or self._current_sg_idx >= len(self._plan):
            return False
        sg = self._plan[self._current_sg_idx]
        if sg.name not in ("open",):
            return False
        front_obj = int(obs_7x7[3, 5, 0])
        if front_obj != OBJ_DOOR:
            return False
        if sg.target_pos is None:
            return False
        # Check if front cell matches target
        dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][agent_dir]
        front_r, front_c = agent_row + dr, agent_col + dc
        return sg.target_pos == (front_r, front_c)

    def _maybe_insert_key_subgoals(self) -> None:
        """If current subgoal is 'open' a locked door, insert key-fetch subgoals."""
        if not self._plan or self._current_sg_idx >= len(self._plan):
            return
        sg = self._plan[self._current_sg_idx]
        if sg.name != "open" or sg.target_pos is None:
            return

        # Check if door is locked
        r, c = sg.target_pos
        if not self.spatial_map.explored[r, c]:
            return
        state = int(self.spatial_map.grid[r, c, 2])
        if state != 2:  # not locked
            return

        # Already carrying a key? Check color match
        if self._carrying and self._carrying[0] == "key":
            door_color_id = int(self.spatial_map.grid[r, c, 1])
            door_color_name = COLOR_NAMES.get(door_color_id, "")
            key_color = self._carrying[1]
            if self.causal_model.query_color_match(key_color, door_color_name):
                return  # Have the right key, proceed

        # Need to fetch a key — check if we already inserted key subgoals
        if (self._current_sg_idx > 0 and
                self._plan[self._current_sg_idx - 1].name == "pick_up" and
                self._plan[self._current_sg_idx - 1].target_obj_type == OBJ_KEY):
            return  # Already have key-fetch subgoals, they must have failed

        # Find the right key color
        door_color_id = int(self.spatial_map.grid[r, c, 1])
        door_color_name = COLOR_NAMES.get(door_color_id, "")
        key_color_name = self.causal_model.query_precondition("open", door_color_name)
        key_color_id = COLOR_IDS.get(key_color_name)

        # Insert drop (if carrying something) + go_to key + pick_up key
        inserts = []
        if self._carrying is not None:
            inserts.append(ExecutableSubgoal(
                name="drop", target_pos=(0, 0),
                action_at_target=ACT_DROP, precondition=None,
            ))

        key_pos = None
        if key_color_id is not None:
            key_pos = self.spatial_map.find_object_by_type_color(OBJ_KEY, key_color_id)
        if key_pos is None:
            key_pos = self.spatial_map.find_object(OBJ_KEY)

        inserts.append(ExecutableSubgoal(
            name="go_to", target_pos=key_pos,
            action_at_target=None, precondition="adjacent",
            target_obj_type=OBJ_KEY, target_color=key_color_id,
        ))
        inserts.append(ExecutableSubgoal(
            name="pick_up", target_pos=key_pos,
            action_at_target=ACT_PICKUP, precondition="adjacent",
            target_obj_type=OBJ_KEY, target_color=key_color_id,
        ))

        # Insert before current subgoal
        for i, ins in enumerate(inserts):
            self._plan.insert(self._current_sg_idx + i, ins)

    # ── Plan building ──

    def _build_plan(self, mission: str) -> None:
        """Build executable plan from mission via MissionModel + CausalPlanner."""
        subgoals = self.mission_model.retrieve(mission)
        if not subgoals:
            return

        executable = []
        for sg in subgoals:
            ex = self._subgoal_to_executable(sg)
            if ex is not None:
                executable.append(ex)

        self._plan = executable
        self._current_sg_idx = 0
        self._completed_subgoals = []

    def _subgoal_to_executable(self, sg: Subgoal) -> ExecutableSubgoal | None:
        """Convert MissionModel subgoal to executable subgoal with grid binding."""
        obj_type_id = OBJ_NAME_TO_ID.get(sg.obj)
        color_id = COLOR_IDS.get(sg.color) if sg.color else None

        # Target position resolved lazily via _resolve_target during execution
        target_pos = None

        if sg.type == SG_GO_TO:
            return ExecutableSubgoal(
                name="go_to",
                target_pos=target_pos,
                action_at_target=None,  # adjacency = done
                precondition="adjacent",
                target_obj_type=obj_type_id,
                target_color=color_id,
            )
        elif sg.type == SG_PICK_UP:
            return ExecutableSubgoal(
                name="pick_up",
                target_pos=target_pos,
                action_at_target=ACT_PICKUP,
                precondition="adjacent",
                target_obj_type=obj_type_id,
                target_color=color_id,
            )
        elif sg.type == SG_OPEN:
            return ExecutableSubgoal(
                name="open",
                target_pos=target_pos,
                action_at_target=ACT_TOGGLE,
                precondition="adjacent",
                target_obj_type=obj_type_id,
                target_color=color_id,
            )
        elif sg.type == SG_DROP:
            return ExecutableSubgoal(
                name="drop",
                target_pos=(0, 0),  # drop at current position
                action_at_target=ACT_DROP,
                precondition=None,
            )
        elif sg.type == SG_PUT_NEXT_TO:
            # Target = obj2 (navigate to it, then drop)
            obj2_type_id = OBJ_NAME_TO_ID.get(sg.obj2)
            color2_id = COLOR_IDS.get(sg.color2) if sg.color2 else None
            target_pos2 = None
            if obj2_type_id is not None:
                if color2_id is not None:
                    target_pos2 = self.spatial_map.find_object_by_type_color(
                        obj2_type_id, color2_id
                    )
                if target_pos2 is None:
                    target_pos2 = self.spatial_map.find_object(obj2_type_id)
            return ExecutableSubgoal(
                name="put_next_to",
                target_pos=target_pos2,
                action_at_target=ACT_DROP,
                precondition="carrying_and_adjacent",
                target_obj_type=obj2_type_id,
                target_color=color2_id,
            )

        return None

    def _resolve_target(self, sg: ExecutableSubgoal) -> None:
        """Try to resolve target position from spatial map.

        When multiple matching objects exist, picks the nearest to agent.
        """
        if sg.target_pos is not None:
            return
        if sg.name == "drop":
            return
        if sg.target_obj_type is not None:
            sg.target_pos = self._find_nearest_object(
                sg.target_obj_type, sg.target_color
            )

    def _find_nearest_object(self, obj_type: int,
                             color: int | None) -> tuple[int, int] | None:
        """Find the nearest matching object to the agent."""
        # Get all matching positions
        type_match = self.spatial_map.grid[:, :, 0] == obj_type
        if color is not None:
            color_match = self.spatial_map.grid[:, :, 1] == color
            mask = type_match & color_match
        else:
            mask = type_match

        positions = np.argwhere(mask)
        if len(positions) == 0:
            return None

        if len(positions) == 1:
            return int(positions[0, 0]), int(positions[0, 1])

        # Find nearest to agent's last known position
        if self._last_pos is not None:
            ar, ac = self._last_pos
        else:
            ar, ac = self.spatial_map.height // 2, self.spatial_map.width // 2

        best = None
        best_dist = float("inf")
        for pos in positions:
            d = abs(int(pos[0]) - ar) + abs(int(pos[1]) - ac)
            if d < best_dist:
                best_dist = d
                best = (int(pos[0]), int(pos[1]))
        return best

    # ── Subgoal execution ──

    def _execute_subgoal(self, sg: ExecutableSubgoal,
                         agent_row: int, agent_col: int,
                         agent_dir: int, obs_7x7: np.ndarray) -> int:
        """Execute a single subgoal."""
        if sg.name == "drop":
            return ACT_DROP

        tr, tc = sg.target_pos  # type: ignore

        # Check if we need interaction (pickup/toggle/drop)
        if sg.action_at_target is not None:
            dist = abs(agent_row - tr) + abs(agent_col - tc)
            if dist == 1:
                return self._face_and_act(tr, tc, agent_row, agent_col,
                                          agent_dir, sg.action_at_target)
            # Navigate to adjacent cell
            return self._navigate_to_adjacent(
                tr, tc, agent_row, agent_col, agent_dir, sg
            )

        # GO_TO: adjacency + facing = done, navigate there
        dist = abs(agent_row - tr) + abs(agent_col - tc)
        if dist <= 1:
            # Face the target
            return self._face_and_act(tr, tc, agent_row, agent_col,
                                      agent_dir, ACT_FORWARD)
        return self._navigate_to(tr, tc, agent_row, agent_col, agent_dir, sg)

    def _navigate_to(self, tr: int, tc: int,
                     agent_row: int, agent_col: int,
                     agent_dir: int, sg: ExecutableSubgoal) -> int:
        """BFS navigate to target position."""
        obs = self.spatial_map.to_obs()
        path = self.pathfinder.find_path(
            obs, (agent_row, agent_col), (tr, tc), allow_door=True
        )
        if path is None or len(path) <= 1:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )
        actions = self.pathfinder.path_to_actions(path, agent_dir)
        return actions[0] if actions else ACT_FORWARD

    def _navigate_to_adjacent(self, tr: int, tc: int,
                              agent_row: int, agent_col: int,
                              agent_dir: int,
                              sg: ExecutableSubgoal) -> int:
        """Navigate to best adjacent cell of target."""
        adj = self._best_adjacent(tr, tc, agent_row, agent_col)
        if adj is not None:
            return self._navigate_to(adj[0], adj[1], agent_row, agent_col,
                                     agent_dir, sg)
        return self._navigate_to(tr, tc, agent_row, agent_col, agent_dir, sg)

    def _face_and_act(self, tr: int, tc: int,
                      agent_row: int, agent_col: int,
                      agent_dir: int, action: int) -> int:
        """Turn to face target and execute action."""
        dr = tr - agent_row
        dc = tc - agent_col
        need_dir = _dir_from_delta(dr, dc)
        if need_dir is not None and need_dir != agent_dir:
            diff = (need_dir - agent_dir) % 4
            return ACT_RIGHT if diff <= 2 else ACT_LEFT
        return action

    def _best_adjacent(self, target_row: int, target_col: int,
                       agent_row: int, agent_col: int
                       ) -> tuple[int, int] | None:
        """Find best reachable adjacent cell to target."""
        obs = self.spatial_map.to_obs()
        best = None
        best_dist = float("inf")

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ar, ac = target_row + dr, target_col + dc
            if not (0 <= ar < self.spatial_map.height
                    and 0 <= ac < self.spatial_map.width):
                continue
            obj = int(obs[ar, ac, 0])
            if obj in (OBJ_WALL,):
                continue
            path = self.pathfinder.find_path(
                obs, (agent_row, agent_col), (ar, ac), allow_door=True
            )
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (ar, ac)

        return best

    # ── Subgoal achievement ──

    def _is_subgoal_achieved(self, sg: ExecutableSubgoal,
                             agent_row: int, agent_col: int) -> bool:
        """Check if current subgoal is completed."""
        if sg.name == "go_to":
            if sg.target_pos is None:
                return False
            tr, tc = sg.target_pos
            # Adjacent to target (can't stand ON objects in MiniGrid)
            return abs(agent_row - tr) + abs(agent_col - tc) <= 1

        if sg.name == "pick_up":
            # Achieved when we're carrying the target object
            if self._carrying is None:
                return False
            if sg.target_color is not None:
                expected_color = COLOR_NAMES.get(sg.target_color, "")
                return self._carrying[1] == expected_color
            return True  # picked up something

        if sg.name == "open":
            # Door toggled — check spatial map
            if sg.target_pos is None:
                return False
            r, c = sg.target_pos
            if self.spatial_map.explored[r, c]:
                state = int(self.spatial_map.grid[r, c, 2])
                return state == 0  # open
            return False

        if sg.name == "drop":
            return self._carrying is None

        if sg.name == "put_next_to":
            # Object dropped near target
            return self._carrying is None

        return False

    # ── Navigation helpers ──

    def _check_immediate_nav(self, obs_7x7: np.ndarray,
                             agent_row: int, agent_col: int,
                             agent_dir: int) -> int | None:
        """Open closed (non-locked) doors for navigation, and
        handle immediate interactions with mission objects."""
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        front_color = int(obs_7x7[3, 5, 1])

        # If current subgoal is pick_up and we're facing the target — pick up
        if self._plan and self._current_sg_idx < len(self._plan):
            sg = self._plan[self._current_sg_idx]
            if (sg.name == "pick_up" and sg.action_at_target == ACT_PICKUP
                    and front_obj == sg.target_obj_type):
                if sg.target_color is None or front_color == sg.target_color:
                    return ACT_PICKUP

        # Facing closed (not locked) door — toggle to open for navigation
        if front_obj == OBJ_DOOR and front_state == 1:
            return ACT_TOGGLE

        # Facing locked door with matching key — toggle
        if (front_obj == OBJ_DOOR and front_state == 2 and
                self._carrying and self._carrying[0] == "key"):
            door_color_name = COLOR_NAMES.get(front_color, "")
            if self.causal_model.query_color_match(self._carrying[1], door_color_name):
                return ACT_TOGGLE

        # Turn toward closed door if adjacent
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
            if obj == OBJ_DOOR and state == 1:
                need_dir = _dir_from_delta(dr, dc)
                if need_dir is not None and need_dir != agent_dir:
                    diff = (need_dir - agent_dir) % 4
                    return ACT_RIGHT if diff <= 2 else ACT_LEFT
        return None

    def _explore(self, agent_row: int, agent_col: int,
                 agent_dir: int) -> int:
        """Explore toward nearest frontier with anti-stuck fallback.

        Unlike Stage 61 which masked locked doors as walls, BossLevel
        exploration goes through all doors (most are just closed, not locked).
        When the agent hits a locked door, _check_immediate_nav handles it
        (toggles if carrying key, otherwise anti-stuck turns away).
        """
        current_pos = (agent_row, agent_col)
        if current_pos == self._last_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._last_pos = current_pos

        if self._stuck_count >= 3:
            self._stuck_count = 0
            return ACT_LEFT if np.random.random() > 0.5 else ACT_RIGHT

        return self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

    def get_stats(self) -> dict:
        """Return episode statistics."""
        return {
            "explore_steps": self.explore_steps,
            "execute_steps": self.execute_steps,
            "mission": self._mission,
            "subgoals_completed": sum(self._completed_subgoals),
            "total_subgoals": len(self._plan),
            "plan": [sg.name for sg in self._plan],
            "current_sg_idx": self._current_sg_idx,
        }
