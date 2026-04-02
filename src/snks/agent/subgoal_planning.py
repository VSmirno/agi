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

        # Pickup key: key present in obs_before, absent in obs_after
        key_before = self._has_object(step.obs_before, self.OBJ_KEY)
        key_after = self._has_object(step.obs_after, self.OBJ_KEY)
        if key_before and not key_after:
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


class SubgoalNavigator:
    """Navigate toward a subgoal using trace-based action replay.

    Strategy: match current state to states in the trace segment for the
    current subgoal, and replay the action from the most similar state.
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
        # Trace segments per subgoal: list of (state_vsa, action)
        self._trace_segments: dict[str, list[tuple[torch.Tensor, int]]] = {}

    def set_trace_segments(self, segments: dict[str, list[tuple[torch.Tensor, int]]]) -> None:
        """Set trace segments for each subgoal (extracted from successful trace)."""
        self._trace_segments = segments

    def select(self, current_state: torch.Tensor, subgoal: Subgoal) -> int:
        """Select action via trace matching or SDM fallback."""
        # Epsilon exploration
        if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        # Strategy 1: trace-based replay
        segment = self._trace_segments.get(subgoal.name, [])
        if segment:
            best_sim = -1.0
            best_action = -1
            for trace_state, trace_action in segment:
                sim = self.cb.similarity(current_state, trace_state)
                if sim > best_sim:
                    best_sim = sim
                    best_action = trace_action
            if best_action >= 0 and best_sim > 0.55:
                return best_action

        # Strategy 2: SDM prediction toward target state
        target = subgoal.target_state
        best_sim = -1.0
        best_action = -1
        any_confident = False

        for a_idx in range(self.n_actions):
            action_vsa = self.cb.action(a_idx)
            pred_next, conf = self.sdm.read_next(current_state, action_vsa)
            if conf < self.min_confidence:
                continue
            any_confident = True
            sim = self.cb.similarity(pred_next, target)
            if sim > best_sim:
                best_sim = sim
                best_action = a_idx

        if not any_confident or best_action < 0:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        return best_action

    def is_achieved(self, obs: np.ndarray, subgoal: Subgoal) -> bool:
        """Check if subgoal is achieved in current observation."""
        if subgoal.name == "pickup_key":
            return not self._key_visible(obs)
        elif subgoal.name == "open_door":
            return self._door_is_open(obs)
        elif subgoal.name == "reach_goal":
            return False  # detected by env termination/reward

        # VSA fallback for unknown subgoals
        current_vsa = self.enc.encode(obs)
        return self.cb.similarity(current_vsa, subgoal.target_state) > 0.75

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
                # Segment trace by subgoal boundaries
                segments = self._segment_trace(best_trace, subgoals)
                self.navigator.set_trace_segments(segments)

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
                action = self.navigator.select(state, current_subgoal)

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

    def _segment_trace(self, trace: list[TraceStep],
                       subgoals: list[Subgoal]) -> dict[str, list[tuple[torch.Tensor, int]]]:
        """Split trace into segments per subgoal.

        For each subgoal, find the trace steps leading up to its achievement.
        Returns dict mapping subgoal name → list of (state_vsa, action).
        """
        segments: dict[str, list[tuple[torch.Tensor, int]]] = {}
        subgoal_names = [sg.name for sg in subgoals]

        # Find boundary indices in trace (where each subgoal is achieved)
        boundaries: list[int] = []
        for sg in subgoals:
            for i, step in enumerate(trace):
                if self.navigator.is_achieved(step.obs_after, sg):
                    boundaries.append(i)
                    break
            else:
                # Subgoal not found — use end of trace
                boundaries.append(len(trace) - 1)

        # Build segments: trace from previous boundary to this boundary
        prev_boundary = 0
        for sg_idx, sg in enumerate(subgoals):
            end = boundaries[sg_idx]
            segment_steps: list[tuple[torch.Tensor, int]] = []
            for i in range(prev_boundary, end + 1):
                state_vsa = self.encoder.encode(trace[i].obs_before)
                segment_steps.append((state_vsa, trace[i].action))
            segments[sg.name] = segment_steps
            prev_boundary = end + 1

        return segments
