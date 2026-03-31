"""GoalAgent: autonomous multi-step goal composition (Stage 25).

Backward chaining from final goal → sub-goals → execute → learn.
Uses BabyAIExecutor as primitive, CausalWorldModel for causal reasoning,
and exploration fallback when causal links are unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.babyai_executor import BabyAIExecutor
from snks.language.blocking_analyzer import BlockingAnalyzer, SubGoal
from snks.language.causal_learner import CausalLearner
from snks.language.grid_navigator import GridNavigator, PathStatus
from snks.language.grid_perception import GridPerception, SKS_KEY_HELD
from snks.language.grounding_map import GroundingMap

# Terminal actions.
ACT_PICKUP = 3
ACT_TOGGLE = 5

ACTION_NAME_TO_ID = {"pickup": ACT_PICKUP, "toggle": ACT_TOGGLE}


@dataclass
class EpisodeResult:
    """Result of a GoalAgent episode."""

    success: bool = False
    reward: float = 0.0
    steps_taken: int = 0
    subgoals_identified: list[str] = field(default_factory=list)
    explored: bool = False
    error: str = ""


class GoalAgent:
    """Autonomous goal-driven agent with backward chaining and causal learning.

    Owns persistent CausalWorldModel and GroundingMap across episodes.
    """

    def __init__(
        self,
        env,
        grounding_map: GroundingMap | None = None,
        causal_model: CausalWorldModel | None = None,
    ) -> None:
        self._env = env
        self._gmap = grounding_map or GroundingMap()
        self._perception = GridPerception(self._gmap)
        self._navigator = GridNavigator()
        self._executor = BabyAIExecutor(env, self._perception)
        self._analyzer = BlockingAnalyzer()

        if causal_model is None:
            config = CausalAgentConfig(causal_min_observations=1)
            causal_model = CausalWorldModel(config)
        self._causal_model = causal_model
        self._learner = CausalLearner(causal_model)

    @property
    def causal_model(self) -> CausalWorldModel:
        return self._causal_model

    def run_episode(self, instruction: str, max_steps: int = 200) -> EpisodeResult:
        """Run one episode with backward chaining + exploration."""
        result = EpisodeResult()
        steps = 0
        total_reward = 0.0
        max_retries = 3  # retry backward chaining after exploration

        for attempt in range(max_retries):
            # Perceive current state.
            uw = self._env.unwrapped
            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
            )

            # Find the goal object (last noun in instruction, usually "goal").
            goal_obj = self._perception.find_object("goal")
            if goal_obj is None:
                result.error = "goal object not found"
                return result

            # Try to reach goal.
            path_result = self._navigator.plan_path_ex(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                goal_obj.pos, stop_adjacent=False,
            )

            if path_result.status == PathStatus.ALREADY_THERE:
                result.success = True
                result.reward = total_reward
                result.steps_taken = steps
                return result

            if path_result.status == PathStatus.OK:
                # Navigate to goal.
                for action in path_result.actions:
                    if steps >= max_steps:
                        break
                    self._learner.before_action(current_sks)
                    obs, reward, terminated, truncated, info = self._env.step(action)
                    steps += 1
                    total_reward += reward
                    carrying = getattr(uw, "carrying", None)
                    current_sks = self._perception.perceive(
                        uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                        carrying=carrying,
                    )
                    self._learner.after_action(action, current_sks)
                    if terminated:
                        result.success = total_reward > 0
                        result.reward = total_reward
                        result.steps_taken = steps
                        return result

                result.success = total_reward > 0
                result.reward = total_reward
                result.steps_taken = steps
                return result

            # Path BLOCKED — backward chaining.
            blocker = self._analyzer.find_blocker(
                uw.grid, tuple(uw.agent_pos), goal_obj.pos,
            )
            if blocker is None:
                result.error = "path blocked but no blocker found"
                result.steps_taken = steps
                return result

            resolution = self._analyzer.suggest_resolution(
                blocker, self._causal_model, current_sks,
            )

            if resolution is not None:
                # Execute sub-goal chain.
                subgoals = self._flatten_subgoals(resolution)
                result.subgoals_identified = [
                    f"{sg.action} {sg.target_word}" for sg in subgoals
                ]
                for sg in subgoals:
                    ok, sg_steps, sg_reward = self._execute_subgoal(
                        sg, max_steps - steps,
                    )
                    steps += sg_steps
                    total_reward += sg_reward
                    if not ok or steps >= max_steps:
                        break
                # After sub-goals, retry reaching goal (loop continues).
                continue

            # No resolution — explore.
            result.explored = True
            exp_steps = self._explore(max_steps - steps)
            steps += exp_steps
            # After exploration, retry backward chaining.

        result.reward = total_reward
        result.steps_taken = steps
        result.success = total_reward > 0
        return result

    def _execute_subgoal(
        self, subgoal: SubGoal, max_steps: int,
    ) -> tuple[bool, int, float]:
        """Execute a single sub-goal. Returns (success, steps, reward)."""
        uw = self._env.unwrapped
        carrying = getattr(uw, "carrying", None)
        current_sks = self._perception.perceive(
            uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
        )

        # Find the target object.
        target_obj = self._perception.find_object(subgoal.target_word)
        if target_obj is None:
            return False, 0, 0.0

        # Navigate to target (stop adjacent for interaction).
        nav_actions = self._navigator.plan_path(
            uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
            target_obj.pos, stop_adjacent=True,
        )

        steps = 0
        total_reward = 0.0
        for action in nav_actions:
            if steps >= max_steps:
                return False, steps, total_reward
            self._learner.before_action(current_sks)
            obs, reward, terminated, truncated, info = self._env.step(action)
            steps += 1
            total_reward += reward
            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )
            self._learner.after_action(action, current_sks)
            if terminated:
                return True, steps, total_reward

        # Execute terminal action.
        action_id = ACTION_NAME_TO_ID.get(subgoal.action)
        if action_id is not None and steps < max_steps:
            self._learner.before_action(current_sks)
            obs, reward, terminated, truncated, info = self._env.step(action_id)
            steps += 1
            total_reward += reward
            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )
            self._learner.after_action(action_id, current_sks)

        return True, steps, total_reward

    def _explore(self, max_steps: int) -> int:
        """Try interactions with all interactive objects. Returns steps taken."""
        uw = self._env.unwrapped
        objects = sorted(
            self._perception.objects,
            key=lambda o: abs(o.pos[0] - uw.agent_pos[0]) + abs(o.pos[1] - uw.agent_pos[1]),
        )

        steps = 0
        for obj in objects[:8]:
            if steps >= max_steps:
                break

            # Navigate to object.
            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )
            nav_actions = self._navigator.plan_path(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                obj.pos, stop_adjacent=True,
            )
            for action in nav_actions:
                if steps >= max_steps:
                    break
                self._env.step(action)
                steps += 1

            # Try pickup and toggle.
            for action_id in [ACT_PICKUP, ACT_TOGGLE]:
                if steps >= max_steps:
                    break
                carrying = getattr(uw, "carrying", None)
                current_sks = self._perception.perceive(
                    uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                    carrying=carrying,
                )
                self._learner.before_action(current_sks)
                obs, reward, terminated, truncated, info = self._env.step(action_id)
                steps += 1
                carrying = getattr(uw, "carrying", None)
                new_sks = self._perception.perceive(
                    uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                    carrying=carrying,
                )
                self._learner.after_action(action_id, new_sks)

        return steps

    @staticmethod
    def _flatten_subgoals(subgoal: SubGoal) -> list[SubGoal]:
        """Flatten recursive SubGoal chain into ordered list (prerequisites first)."""
        chain: list[SubGoal] = []
        current: SubGoal | None = subgoal
        while current is not None:
            chain.append(current)
            current = current.prerequisite
        chain.reverse()
        return chain
