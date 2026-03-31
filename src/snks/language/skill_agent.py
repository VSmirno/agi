"""SkillAgent: GoalAgent with skill-first execution (Stage 27)."""

from __future__ import annotations

from dataclasses import dataclass, field

from snks.agent.causal_model import CausalWorldModel
from snks.language.goal_agent import GoalAgent, EpisodeResult, ACT_PICKUP, ACT_TOGGLE, ACTION_NAME_TO_ID
from snks.language.grid_perception import SKS_DOOR_OPEN, SKS_KEY_HELD
from snks.language.grid_navigator import PathStatus
from snks.language.grounding_map import GroundingMap
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


@dataclass
class SkillEpisodeResult(EpisodeResult):
    """Extends EpisodeResult with skill tracking."""

    skills_used: list[str] = field(default_factory=list)
    skills_total: int = 0


class SkillAgent(GoalAgent):
    """GoalAgent extended with skill-first execution.

    Priority: composite skill → primitive skill → backward chaining → explore.
    After each episode, extracts new skills from causal model.
    """

    def __init__(
        self,
        env,
        skill_library: SkillLibrary | None = None,
        grounding_map: GroundingMap | None = None,
        causal_model: CausalWorldModel | None = None,
    ) -> None:
        super().__init__(env, grounding_map=grounding_map, causal_model=causal_model)
        self._library = skill_library or SkillLibrary()

    @property
    def library(self) -> SkillLibrary:
        return self._library

    def run_episode(self, instruction: str, max_steps: int = 300) -> SkillEpisodeResult:
        """Skill-first episode execution."""
        result = SkillEpisodeResult()
        steps = 0
        total_reward = 0.0
        max_retries = 5

        for attempt in range(max_retries):
            uw = self._env.unwrapped
            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
            )

            # Find goal.
            goal_obj = self._perception.find_object("goal")
            if goal_obj is None:
                result.error = "goal object not found"
                break

            # Try direct path.
            path_result = self._navigator.plan_path_ex(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                goal_obj.pos, stop_adjacent=False,
            )

            if path_result.status == PathStatus.ALREADY_THERE:
                result.success = True
                break

            if path_result.status == PathStatus.OK:
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
                        break
                if result.success or steps >= max_steps:
                    break
                continue

            # Path BLOCKED — try ONE skill first, then backward chaining.
            # Skills are generic (no target_pos), so try only once per attempt.
            # If skill fails or doesn't resolve blocker, fall through to
            # backward chaining which has position-specific targeting.
            goal_sks = frozenset({SKS_DOOR_OPEN})
            applicable = self._library.find_applicable(current_sks, goal_sks)

            skill_used = False
            if applicable and attempt == 0:
                # Try best applicable skill on FIRST attempt only.
                skill = applicable[0]
                sks_before = set(current_sks)
                if skill.is_composite:
                    ok, s, r = self._try_composite_skill(skill, max_steps - steps)
                else:
                    ok, s, r = self._try_primitive_skill(skill, max_steps - steps)
                steps += s
                total_reward += r
                result.skills_used.append(skill.name)
                skill.attempt_count += 1

                # Check if state actually changed.
                uw_check = self._env.unwrapped
                carrying_check = getattr(uw_check, "carrying", None)
                sks_after = self._perception.perceive(
                    uw_check.grid, tuple(uw_check.agent_pos), int(uw_check.agent_dir),
                    carrying=carrying_check,
                )
                if ok and sks_after != sks_before:
                    skill.success_count += 1
                    skill_used = True

            if skill_used:
                continue

            # Fallback: backward chaining (from parent GoalAgent).
            blocker = self._analyzer.find_blocker(
                uw.grid, tuple(uw.agent_pos), goal_obj.pos,
            )
            if blocker is None:
                result.error = "path blocked but no blocker found"
                break

            resolution = self._analyzer.suggest_resolution(
                blocker, self._causal_model, current_sks,
            )

            if resolution is not None:
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
                continue

            # Last resort: explore.
            result.explored = True
            exp_steps = self._explore(max_steps - steps)
            steps += exp_steps

        result.reward = total_reward
        result.steps_taken = steps
        if not result.error:
            result.success = result.success or total_reward > 0
        result.skills_total = len(result.skills_used)

        # Post-episode: extract skills.
        self._after_episode()

        return result

    def _try_primitive_skill(
        self, skill: Skill, max_steps: int,
    ) -> tuple[bool, int, float]:
        """Execute a primitive skill: navigate to target + terminal action."""
        uw = self._env.unwrapped
        carrying = getattr(uw, "carrying", None)
        current_sks = self._perception.perceive(
            uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
        )

        target_obj = self._perception.find_object(skill.target_word)
        if target_obj is None:
            return False, 0, 0.0

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

        # Terminal action.
        if skill.terminal_action is not None and steps < max_steps:
            self._learner.before_action(current_sks)
            obs, reward, terminated, truncated, info = self._env.step(skill.terminal_action)
            steps += 1
            total_reward += reward
            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )
            self._learner.after_action(skill.terminal_action, current_sks)

        # Verify: check if effects achieved.
        achieved = skill.effects <= frozenset(current_sks)
        return achieved, steps, total_reward

    def _try_composite_skill(
        self, skill: Skill, max_steps: int,
    ) -> tuple[bool, int, float]:
        """Execute a composite skill by running sub-skills in order."""
        total_steps = 0
        total_reward = 0.0

        for sub_name in skill.sub_skills:
            sub_skill = self._library.get(sub_name)
            if sub_skill is None:
                return False, total_steps, total_reward

            if sub_skill.is_composite:
                ok, s, r = self._try_composite_skill(sub_skill, max_steps - total_steps)
            else:
                ok, s, r = self._try_primitive_skill(sub_skill, max_steps - total_steps)
            total_steps += s
            total_reward += r
            if not ok:
                return False, total_steps, total_reward

        return True, total_steps, total_reward

    def _after_episode(self) -> None:
        """Extract skills from updated causal model."""
        self._library.extract_from_causal_model(self._causal_model)
        self._library.compose_skills()
