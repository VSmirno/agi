"""Stage 46: Subgoal Planning — extract subgoals from traces, build plan graph, navigate.

Components:
- SubgoalExtractor: detect key events (pickup, toggle, goal) from episode traces
- PlanGraph: ordered chain of subgoals with advancement
- SubgoalNavigator: SDM-based action selection toward current subgoal
- SubgoalPlanningAgent: full agent with explore→extract→plan loop
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from snks.agent.pathfinding import GridPathfinder
from snks.agent.vsa_world_model import (
    SDMMemory,
    VSACodebook,
    VSAEncoder,
    WorldModelAgent,
    WorldModelConfig,
)


@dataclass
class TraceStep:
    """One step of an episode trace."""
    obs_before: np.ndarray
    action: int
    obs_after: np.ndarray
    reward: float


@dataclass
class Subgoal:
    """A subgoal extracted from a trace."""
    name: str
    target_state: torch.Tensor      # VSA vector after achievement
    precondition_state: torch.Tensor  # VSA vector before achievement
    detection_type: str              # "symbolic" | "vsa_diff"


@dataclass
class SubgoalConfig(WorldModelConfig):
    vsa_diff_threshold: float = 0.7
    achievement_threshold: float = 0.75
    max_subgoals: int = 5
    use_best_trace: bool = True


class SubgoalExtractor:
    """Extract subgoals from successful episode traces."""

    OBJ_KEY = 5
    OBJ_DOOR = 4
    OBJ_GOAL = 8
    OBJ_AGENT = 10

    def __init__(self, codebook: VSACodebook, encoder: VSAEncoder,
                 vsa_diff_threshold: float = 0.7):
        self.cb = codebook
        self.enc = encoder
        self.vsa_diff_threshold = vsa_diff_threshold

    def extract(self, trace: list[TraceStep]) -> list[Subgoal]:
        """Extract ordered subgoals from a successful trace."""
        if not trace:
            return []

        subgoals: list[Subgoal] = []
        seen_names: set[str] = set()

        for step in trace:
            events = self._detect_symbolic(step)
            for name, detection_type in events:
                if name not in seen_names:
                    seen_names.add(name)
                    target = self.enc.encode(step.obs_after)
                    precond = self.enc.encode(step.obs_before)
                    subgoals.append(Subgoal(name, target, precond, detection_type))

        # Always append reach_goal if not already detected and trace has reward
        if "reach_goal" not in seen_names and trace[-1].reward > 0:
            target = self.enc.encode(trace[-1].obs_after)
            precond = self.enc.encode(trace[-1].obs_before)
            subgoals.append(Subgoal("reach_goal", target, precond, "symbolic"))

        return subgoals

    def _detect_symbolic(self, step: TraceStep) -> list[tuple[str, str]]:
        """Detect symbolic events from observation changes."""
        events: list[tuple[str, str]] = []

        # Pickup key: agent carrying key in obs_after but not in obs_before
        carrying_before = self._agent_carrying(step.obs_before)
        carrying_after = self._agent_carrying(step.obs_after)
        if not carrying_before and carrying_after:
            events.append(("pickup_key", "symbolic"))

        # Open door: door locked in obs_before, open in obs_after
        door_locked_before = self._door_locked(step.obs_before)
        door_open_after = self._door_open(step.obs_after)
        if door_locked_before and door_open_after:
            events.append(("open_door", "symbolic"))

        # Reach goal: reward > 0
        if step.reward > 0:
            events.append(("reach_goal", "symbolic"))

        return events

    def _has_object(self, obs: np.ndarray, obj_type: int) -> bool:
        return bool(np.any(obs[:, :, 0] == obj_type))

    def _agent_carrying(self, obs: np.ndarray) -> bool:
        """Check if agent is carrying an item (color channel = 5)."""
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if int(obs[r, c, 0]) == self.OBJ_AGENT:
                    return int(obs[r, c, 1]) == 5
        return False

    def _door_locked(self, obs: np.ndarray) -> bool:
        door_mask = obs[:, :, 0] == self.OBJ_DOOR
        if not np.any(door_mask):
            return False
        door_states = obs[:, :, 2][door_mask]
        return bool(np.any(door_states == 2))

    def _door_open(self, obs: np.ndarray) -> bool:
        door_mask = obs[:, :, 0] == self.OBJ_DOOR
        if not np.any(door_mask):
            return False
        door_states = obs[:, :, 2][door_mask]
        return bool(np.any(door_states == 0))


class PlanGraph:
    """Ordered chain of subgoals with state tracking."""

    def __init__(self, subgoals: list[Subgoal]):
        self.subgoals = subgoals
        self.current_idx = 0

    def current_subgoal(self) -> Subgoal | None:
        if self.current_idx >= len(self.subgoals):
            return None
        return self.subgoals[self.current_idx]

    def advance(self) -> bool:
        """Advance to next subgoal. Returns True if plan complete."""
        self.current_idx += 1
        return self.current_idx >= len(self.subgoals)

    def reset(self):
        self.current_idx = 0


@dataclass
class SymbolicState:
    """Symbolic state extracted from observation for matching."""
    agent_row: int
    agent_col: int
    agent_dir: int
    has_key: bool
    door_open: bool

    def distance(self, other: SymbolicState) -> int:
        """Manhattan distance + direction mismatch + state mismatch."""
        d = abs(self.agent_row - other.agent_row) + abs(self.agent_col - other.agent_col)
        if self.agent_dir != other.agent_dir:
            d += 1
        if self.has_key != other.has_key:
            d += 2  # key state matters a lot
        if self.door_open != other.door_open:
            d += 2
        return d


def _extract_symbolic(obs: np.ndarray) -> SymbolicState:
    """Extract symbolic state from 7x7x3 observation."""
    agent_row, agent_col, agent_dir = 0, 0, 0
    has_key_vis = False
    door_open = False

    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            obj_type = int(obs[r, c, 0])
            if obj_type == 10:  # agent
                agent_row, agent_col = r, c
                agent_dir = int(obs[r, c, 2])
            elif obj_type == 5:  # key
                has_key_vis = True
            elif obj_type == 4:  # door
                door_open = int(obs[r, c, 2]) == 0

    # has_key = carrying (key not visible on grid + agent color indicator)
    has_key = not has_key_vis and int(obs[agent_row, agent_col, 1]) == 5

    return SymbolicState(agent_row, agent_col, agent_dir, has_key, door_open)


class SubgoalNavigator:
    """Navigate toward a subgoal using symbolic trace-based matching.

    Strategy: match current symbolic state (position, direction, inventory)
    to trace segment states, and replay the action from closest match.
    Falls back to SDM prediction if no trace segment available.
    """

    OBJ_KEY = 5
    OBJ_DOOR = 4

    def __init__(self, sdm: SDMMemory, codebook: VSACodebook,
                 encoder: VSAEncoder, n_actions: int = 7,
                 epsilon: float = 0.15, min_confidence: float = 0.01):
        self.sdm = sdm
        self.cb = codebook
        self.enc = encoder
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.min_confidence = min_confidence
        # Target positions per subgoal: (row, col, special_action_or_None)
        self._target_positions: dict[str, tuple[int, int, int | None]] = {}
        # Trace segments (kept for compatibility)
        self._trace_segments: dict[str, list[tuple[SymbolicState, int]]] = {}
        # BFS pathfinding (Stage 47)
        self._pathfinder = GridPathfinder()
        self._cached_path: list[int] | None = None
        self._cached_target: tuple[int, int] | None = None
        self._use_bfs: bool = True

    def set_trace_segments(self, segments: dict[str, list[tuple[SymbolicState, int]]]) -> None:
        """Set trace segments for each subgoal."""
        self._trace_segments = segments

    def set_target_positions(self, targets: dict[str, tuple[int, int, int | None]]) -> None:
        """Set target positions for subgoal navigation.

        targets: dict mapping subgoal name → (target_row, target_col, special_action)
        special_action: action to execute at target (3=pickup, 5=toggle, None=just arrive)
        """
        self._target_positions = targets

    def select(self, current_state: torch.Tensor, subgoal: Subgoal,
               current_obs: np.ndarray | None = None) -> int:
        """Select action via position-based navigation toward subgoal target."""
        # Epsilon exploration
        if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        # Strategy 1: position-based navigation using subgoal target
        if current_obs is not None:
            target_info = self._target_positions.get(subgoal.name)
            if target_info is not None:
                current_sym = _extract_symbolic(current_obs)
                target_row, target_col, special_action = target_info

                # At target position — execute special action
                if current_sym.agent_row == target_row and current_sym.agent_col == target_col:
                    if special_action == 5:
                        # Toggle: must face the door first
                        door_pos = self._find_door(current_obs)
                        if door_pos is not None:
                            dr, dc = door_pos[0] - target_row, door_pos[1] - target_col
                            need_dir = self._dir_from_delta(dr, dc)
                            if need_dir is not None and current_sym.agent_dir != need_dir:
                                diff = (need_dir - current_sym.agent_dir) % 4
                                return 1 if diff <= 2 else 0
                        return 5  # toggle
                    if special_action is not None:
                        return special_action
                    return int(torch.randint(0, self.n_actions, (1,)).item())

                # Navigate toward target: BFS pathfinding (wall-aware)
                if self._use_bfs:
                    return self._navigate_bfs(current_obs, current_sym,
                                              target_row, target_col, subgoal)
                return self._navigate_toward(current_sym, target_row, target_col)

        # Strategy 2: SDM prediction (fallback)
        target = subgoal.target_state
        best_sim = -1.0
        best_action = -1

        for a_idx in range(self.n_actions):
            action_vsa = self.cb.action(a_idx)
            pred_next, conf = self.sdm.read_next(current_state, action_vsa)
            if conf < self.min_confidence:
                continue
            sim = self.cb.similarity(pred_next, target)
            if sim > best_sim:
                best_sim = sim
                best_action = a_idx

        if best_action < 0:
            return int(torch.randint(0, self.n_actions, (1,)).item())
        return best_action

    def _find_door(self, obs: np.ndarray) -> tuple[int, int] | None:
        """Find door position in observation."""
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if int(obs[r, c, 0]) == self.OBJ_DOOR:
                    return (r, c)
        return None

    @staticmethod
    def _dir_from_delta(dr: int, dc: int) -> int | None:
        """Convert (dr, dc) delta to direction index."""
        if dc > 0:
            return 0  # right
        if dr > 0:
            return 1  # down
        if dc < 0:
            return 2  # left
        if dr < 0:
            return 3  # up
        return None

    def _navigate_toward(self, current: SymbolicState, target_r: int, target_c: int) -> int:
        """Turn to face target, then move forward.

        Direction mapping: 0=right(+col), 1=down(+row), 2=left(-col), 3=up(-row)
        Actions: 0=turn_left, 1=turn_right, 2=forward
        """
        dr = target_r - current.agent_row
        dc = target_c - current.agent_col

        # Determine desired direction
        if abs(dr) >= abs(dc):
            want_dir = 1 if dr > 0 else 3  # down or up
        else:
            want_dir = 0 if dc > 0 else 2  # right or left

        if current.agent_dir == want_dir:
            return 2  # forward

        # Turn shortest way
        diff = (want_dir - current.agent_dir) % 4
        return 1 if diff <= 2 else 0  # right or left turn

    def _navigate_bfs(self, obs: np.ndarray, current_sym: SymbolicState,
                      target_row: int, target_col: int,
                      subgoal: Subgoal) -> int:
        """BFS-based wall-aware navigation toward target position.

        Recomputes path each step (BFS on 7x7 is <1ms).
        Returns first action of the optimal path.
        """
        start = (current_sym.agent_row, current_sym.agent_col)
        goal = (target_row, target_col)

        # For open_door subgoal: door might still be locked, use allow_door
        allow_door = subgoal.name == "open_door"
        path = self._pathfinder.find_path(obs, start, goal,
                                          allow_door=allow_door)
        if path is None:
            path = self._pathfinder.find_path(obs, start, goal,
                                              allow_door=True)
        if path is None or len(path) <= 1:
            return self._navigate_toward(current_sym, target_row, target_col)

        actions = self._pathfinder.path_to_actions(path, current_sym.agent_dir)
        if actions:
            return actions[0]

        return self._navigate_toward(current_sym, target_row, target_col)

    def is_achieved(self, obs: np.ndarray, subgoal: Subgoal) -> bool:
        """Check if subgoal is achieved in current observation."""
        if subgoal.name == "pickup_key":
            return self._agent_carrying_key(obs)
        elif subgoal.name == "open_door":
            return self._door_is_open(obs)
        elif subgoal.name == "reach_goal":
            return False  # detected by env termination/reward

        # VSA fallback for unknown subgoals
        current_vsa = self.enc.encode(obs)
        return self.cb.similarity(current_vsa, subgoal.target_state) > 0.75

    def _agent_carrying_key(self, obs: np.ndarray) -> bool:
        """Check if agent is carrying a key (color channel = 5 at agent position)."""
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if int(obs[r, c, 0]) == 10:  # agent
                    return int(obs[r, c, 1]) == 5  # carrying indicator
        return False

    def _key_visible(self, obs: np.ndarray) -> bool:
        return bool(np.any(obs[:, :, 0] == self.OBJ_KEY))

    def _door_is_open(self, obs: np.ndarray) -> bool:
        door_mask = obs[:, :, 0] == self.OBJ_DOOR
        if not np.any(door_mask):
            return False
        return bool(np.any(obs[:, :, 2][door_mask] == 0))


class SubgoalPlanningAgent(WorldModelAgent):
    """WorldModelAgent extended with subgoal extraction and chained navigation.

    Explore phase: random actions, fill SDM, collect successful traces.
    Plan phase: extract subgoals from best trace, navigate to each in sequence.
    """

    def __init__(self, config: SubgoalConfig):
        super().__init__(config)
        self.sg_config = config
        self.extractor = SubgoalExtractor(
            self.codebook, self.encoder,
            vsa_diff_threshold=config.vsa_diff_threshold,
        )
        self.navigator = SubgoalNavigator(
            self.sdm, self.codebook, self.encoder,
            n_actions=config.n_actions,
            epsilon=config.epsilon,
            min_confidence=config.min_confidence,
        )
        self.plan: PlanGraph | None = None
        self._successful_traces: list[list[TraceStep]] = []
        self._current_trace: list[TraceStep] = []

    def build_plan_from_obs(self, obs: np.ndarray) -> bool:
        """Build plan directly from observation by scanning for key objects.

        Skips explore phase — constructs subgoals and target positions from
        the visible objects in the grid. Returns True if plan built successfully.
        """
        OBJ_KEY = 5
        OBJ_DOOR = 4
        OBJ_GOAL = 8

        key_pos = None
        door_pos = None
        goal_pos = None

        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                obj = int(obs[r, c, 0])
                if obj == OBJ_KEY:
                    key_pos = (r, c)
                elif obj == OBJ_DOOR:
                    door_pos = (r, c)
                elif obj == OBJ_GOAL:
                    goal_pos = (r, c)

        if not all([key_pos, door_pos, goal_pos]):
            return False

        # Build subgoals with dummy VSA vectors (not used for BFS navigation)
        dummy = torch.zeros(self.sg_config.dim)
        subgoals = [
            Subgoal("pickup_key", dummy, dummy, "symbolic"),
            Subgoal("open_door", dummy, dummy, "symbolic"),
            Subgoal("reach_goal", dummy, dummy, "symbolic"),
        ]
        self.plan = PlanGraph(subgoals)

        # Find door-adjacent cell for toggle position
        pf = GridPathfinder()
        agent_sym = _extract_symbolic(obs)
        best_door_adj = None
        best_dist = float('inf')
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ar, ac = door_pos[0] + dr, door_pos[1] + dc
            if 0 <= ar < obs.shape[0] and 0 <= ac < obs.shape[1]:
                if int(obs[ar, ac, 0]) not in (2,):  # not a wall
                    path = pf.find_path(obs, (agent_sym.agent_row, agent_sym.agent_col),
                                        (ar, ac))
                    if path and len(path) < best_dist:
                        best_dist = len(path)
                        best_door_adj = (ar, ac)

        if best_door_adj is None:
            best_door_adj = (door_pos[0] - 1, door_pos[1])  # fallback: above door

        targets = {
            "pickup_key": (key_pos[0], key_pos[1], 3),
            "open_door": (best_door_adj[0], best_door_adj[1], 5),
            "reach_goal": (goal_pos[0], goal_pos[1], None),
        }
        self.navigator.set_target_positions(targets)
        return True

    def run_episode(self, env, max_steps: int = 200) -> tuple[bool, int, float]:
        self._exploring = self._episode_count < self.config.explore_episodes
        self._current_trace = []

        obs = env.reset()
        total_reward = 0.0
        self._prev_state = None
        self._prev_action = None
        self._prev_obs = obs.copy()

        # Set goal from initial observation
        if self._episode_count == 0:
            self.set_goal_from_obs(obs)

        # Build plan from observation if no successful traces yet (Stage 47: obs-based planning)
        if self.plan is None and self._episode_count >= self.config.explore_episodes:
            self.build_plan_from_obs(obs)

        if self._exploring:
            return self._run_explore_episode(env, obs, max_steps)
        else:
            return self._run_plan_episode(env, obs, max_steps)

    def _run_explore_episode(self, env, obs: np.ndarray,
                             max_steps: int) -> tuple[bool, int, float]:
        """Explore: random actions, record transitions and traces."""
        total_reward = 0.0

        for step_i in range(max_steps):
            action = int(torch.randint(0, self.config.n_actions, (1,)).item())
            state = self.encoder.encode(obs)
            self._prev_state = state
            self._prev_action = action

            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.encoder.encode(obs)
            total_reward += reward

            # Record in SDM
            action_vsa = self.codebook.action(action)
            self.sdm.write(state, action_vsa, next_state, reward)

            # Record trace step
            self._current_trace.append(TraceStep(prev_obs, action, obs.copy(), reward))

            if terminated or truncated:
                if total_reward > 0:
                    self._successful_traces.append(list(self._current_trace))
                self._episode_count += 1
                return total_reward > 0, step_i + 1, total_reward

        self._episode_count += 1
        return total_reward > 0, max_steps, total_reward

    def _run_plan_episode(self, env, obs: np.ndarray,
                          max_steps: int) -> tuple[bool, int, float]:
        """Plan: extract subgoals if needed, navigate to each in sequence."""
        # Build plan from best successful trace
        if self.plan is None and self._successful_traces:
            best_trace = self._select_best_trace()
            subgoals = self.extractor.extract(best_trace)
            if subgoals:
                self.plan = PlanGraph(subgoals)
                # Extract target positions for each subgoal
                targets = self._extract_target_positions(best_trace, subgoals)
                self.navigator.set_target_positions(targets)

        # Fallback if no plan available
        if self.plan is None:
            return self._run_explore_episode(env, obs, max_steps)

        self.plan.reset()
        total_reward = 0.0

        for step_i in range(max_steps):
            current_subgoal = self.plan.current_subgoal()

            state = self.encoder.encode(obs)

            if current_subgoal is None:
                # Plan complete but env not terminated — random walk
                action = int(torch.randint(0, self.config.n_actions, (1,)).item())
            else:
                action = self.navigator.select(state, current_subgoal, current_obs=obs)

            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.encoder.encode(obs)
            total_reward += reward

            # Record in SDM (keep learning during plan phase too)
            action_vsa = self.codebook.action(action)
            self.sdm.write(state, action_vsa, next_state, reward)

            # Record trace
            self._current_trace.append(TraceStep(prev_obs, action, obs.copy(), reward))

            # Check subgoal achievement
            if current_subgoal is not None:
                if self.navigator.is_achieved(obs, current_subgoal):
                    self.plan.advance()

            if terminated or truncated:
                if total_reward > 0:
                    self._successful_traces.append(list(self._current_trace))
                self._episode_count += 1
                return total_reward > 0, step_i + 1, total_reward

        self._episode_count += 1
        return total_reward > 0, max_steps, total_reward

    def _select_best_trace(self) -> list[TraceStep]:
        """Select the shortest successful trace for planning."""
        if self.sg_config.use_best_trace:
            return min(self._successful_traces, key=len)
        return self._successful_traces[-1]

    def _extract_target_positions(self, trace: list[TraceStep],
                                  subgoals: list[Subgoal]) -> dict[str, tuple[int, int, int | None]]:
        """Extract target position and action for each subgoal from trace.

        For each subgoal, find the trace step where it was achieved and extract:
        - Target position: where the agent was when the subgoal was achieved
        - Special action: the action used to achieve it (pickup=3, toggle=5, None)

        For "open_door": target is one cell before the door (agent must face door).
        """
        targets: dict[str, tuple[int, int, int | None]] = {}

        for sg in subgoals:
            if sg.name == "reach_goal":
                # For reach_goal: find goal position from any trace step
                for step in trace:
                    goal_pos = self._find_goal_in_obs(step.obs_before)
                    if goal_pos is not None:
                        targets[sg.name] = (goal_pos[0], goal_pos[1], None)
                        break
                continue

            for step in trace:
                if self.navigator.is_achieved(step.obs_after, sg):
                    sym_before = _extract_symbolic(step.obs_before)
                    if sg.name == "pickup_key":
                        targets[sg.name] = (sym_before.agent_row, sym_before.agent_col, 3)
                    elif sg.name == "open_door":
                        targets[sg.name] = (sym_before.agent_row, sym_before.agent_col, 5)
                    break

        return targets

    @staticmethod
    def _find_goal_in_obs(obs: np.ndarray) -> tuple[int, int] | None:
        """Find goal position (type 8) in observation."""
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if int(obs[r, c, 0]) == 8:
                    return (r, c)
        return None
