"""CuriosityAgent: SkillAgent with curiosity-driven exploration (Stage 29).

Overrides _explore() to prefer novel states over random wandering.
Curiosity intrinsic reward supplements external reward during learning.
"""

from __future__ import annotations

import random

from snks.language.curiosity_module import CuriosityModule
from snks.language.goal_agent import ACT_PICKUP, ACT_TOGGLE
from snks.language.skill_agent import SkillAgent, SkillEpisodeResult


# Navigation actions (turn left, turn right, forward).
_NAV_ACTIONS = [0, 1, 2]


class CuriosityAgent(SkillAgent):
    """SkillAgent with count-based curiosity exploration.

    During _explore(), instead of trying all objects exhaustively,
    chooses actions that maximize expected intrinsic reward (novelty).
    """

    def __init__(self, env, curiosity_across_episodes: bool = False, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self._curiosity = CuriosityModule()
        self._curiosity_across_episodes = curiosity_across_episodes
        self._total_intrinsic_reward: float = 0.0

    @property
    def curiosity(self) -> CuriosityModule:
        return self._curiosity

    def run_episode(self, instruction: str, max_steps: int = 300) -> SkillEpisodeResult:
        """Run episode; observe curiosity at each step including normal navigation.

        Wraps the environment so every step call records curiosity.
        """
        if not self._curiosity_across_episodes:
            self._curiosity.reset()
        self._total_intrinsic_reward = 0.0

        # Wrap step to auto-observe curiosity.
        _orig_step = self._env.step

        def _curious_step(action):
            result = _orig_step(action)
            uw = self._env.unwrapped
            carrying = getattr(uw, "carrying", None)
            sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )
            key = CuriosityModule.make_key(sks, tuple(uw.agent_pos))
            r_int = self._curiosity.observe(key)
            self._total_intrinsic_reward += r_int
            return result

        self._env.step = _curious_step
        self._curiosity_wrapper_active = True
        try:
            episode_result = super().run_episode(instruction, max_steps=max_steps)
        finally:
            self._env.step = _orig_step
            self._curiosity_wrapper_active = False

        return episode_result

    def _explore(self, max_steps: int) -> int:
        """Curiosity-guided exploration: prefer novel states.

        Uses 1-step lookahead: for each navigation action, estimate
        intrinsic reward of resulting state and pick best action.
        Falls back to interaction with nearby objects.
        """
        uw = self._env.unwrapped
        steps = 0

        # Phase 1: Curiosity-guided navigation (move toward novel cells).
        # Cap at nav_budget to leave room for Phase 2 (object interaction).
        nav_budget = min(max_steps, 60)
        for _ in range(nav_budget):
            if steps >= nav_budget:
                break

            carrying = getattr(uw, "carrying", None)
            current_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )

            # Score each navigation action by expected novelty.
            best_action = self._most_novel_action(
                uw, current_sks, _NAV_ACTIONS,
            )

            self._learner.before_action(current_sks)
            # Note: if wrapper is active, _curious_step handles curiosity observation.
            # If not (direct _explore call), we observe manually below.
            obs, reward, terminated, truncated, info = self._env.step(best_action)
            steps += 1

            carrying = getattr(uw, "carrying", None)
            new_sks = self._perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                carrying=carrying,
            )
            self._learner.after_action(best_action, new_sks)

            # Observe curiosity only if wrapper NOT active (direct call).
            if not getattr(self, "_curiosity_wrapper_active", False):
                key = CuriosityModule.make_key(new_sks, tuple(uw.agent_pos))
                r_int = self._curiosity.observe(key)
                self._total_intrinsic_reward += r_int

            if terminated:
                return steps

        # Phase 2: Interaction with nearest objects (same as parent).
        if steps < max_steps:
            objects = sorted(
                self._perception.objects,
                key=lambda o: abs(o.pos[0] - uw.agent_pos[0]) + abs(o.pos[1] - uw.agent_pos[1]),
            )
            for obj in objects[:4]:
                if steps >= max_steps:
                    break
                carrying = getattr(uw, "carrying", None)
                current_sks = self._perception.perceive(
                    uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
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

                for action_id in [ACT_PICKUP, ACT_TOGGLE]:
                    if steps >= max_steps:
                        break
                    carrying = getattr(uw, "carrying", None)
                    current_sks = self._perception.perceive(
                        uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
                    )
                    self._learner.before_action(current_sks)
                    obs, reward, terminated, truncated, info = self._env.step(action_id)
                    steps += 1
                    carrying = getattr(uw, "carrying", None)
                    new_sks = self._perception.perceive(
                        uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
                    )
                    self._learner.after_action(action_id, new_sks)
                    if terminated:
                        return steps

        return steps

    def _most_novel_action(
        self,
        uw,
        current_sks: set[int],
        actions: list[int],
    ) -> int:
        """Return action with highest expected novelty using multi-step lookahead.

        Simulates N steps ahead for each starting action and sums novelty scores.
        This allows the agent to "look around corners" and prefer paths that
        lead to novel regions rather than dead ends.
        Falls back to random if all actions score equally.
        """
        LOOKAHEAD = 5
        best_action = random.choice(actions)
        best_score = -1.0

        for action in actions:
            # Simulate LOOKAHEAD steps starting with this action.
            pos, d = self._simulate_nav_action(
                uw, tuple(uw.agent_pos), int(uw.agent_dir), action,
            )
            score = self._curiosity.intrinsic_reward(
                CuriosityModule.make_key(current_sks, pos)
            )
            # Continue forward from simulated position.
            for _ in range(LOOKAHEAD - 1):
                next_pos, next_d = self._simulate_nav_action(uw, pos, d, 2)  # forward
                if next_pos == pos:
                    # Blocked — try turning.
                    next_pos, next_d = self._simulate_nav_action(uw, pos, d, 0)
                pos, d = next_pos, next_d
                score += self._curiosity.intrinsic_reward(
                    CuriosityModule.make_key(current_sks, pos)
                ) * (0.9 ** _)  # discount
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @staticmethod
    def _simulate_nav_action(
        uw, agent_pos: tuple[int, int], agent_dir: int, action: int,
    ) -> tuple[tuple[int, int], int]:
        """Simulate result of turn/forward action on (pos, dir) without env step.

        Returns (new_pos, new_dir).
        Actions: 0=turn_left, 1=turn_right, 2=forward.
        Directions: 0=right, 1=down, 2=left, 3=up.
        """
        if action == 0:  # turn left
            return agent_pos, (agent_dir - 1) % 4
        if action == 1:  # turn right
            return agent_pos, (agent_dir + 1) % 4
        if action == 2:  # forward
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][agent_dir]
            nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
            # Check if walkable (basic check — wall/door).
            cell = uw.grid.get(nx, ny)
            if cell is None or cell.type in ("empty", "goal"):
                return (nx, ny), agent_dir
            # Not walkable — stay.
            return agent_pos, agent_dir
        # Unknown action.
        return agent_pos, agent_dir
