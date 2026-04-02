"""Stage 52: IntegrationAgent — unified language-guided agent for DoorKey + MultiRoom.

Combines:
- LanguageGrounder (Stage 50): instruction → subgoals
- InstructedAgent (Stage 51): DoorKey navigation with subgoal chain
- MultiRoomNavigator (Stage 49): BFS + reactive door toggle for multi-room

Detects environment type from observation and selects appropriate strategy.
"""

from __future__ import annotations

import numpy as np

from snks.agent.instructed_agent import InstructedAgent
from snks.agent.multi_room_nav import (
    MultiRoomNavigator,
    find_objects,
    is_facing_closed_door,
)
from snks.agent.pathfinding import GridPathfinder
from snks.agent.vsa_world_model import VSACodebook
from snks.language.language_grounder import LanguageGrounder

OBJ_KEY = 5
OBJ_DOOR = 4
OBJ_GOAL = 8
OBJ_AGENT = 10


class IntegrationAgent:
    """Unified agent: language instruction → env detection → adaptive navigation.

    For DoorKey envs (has key): delegates to InstructedAgent (subgoal chain + BFS).
    For MultiRoom envs (no key): uses MultiRoomNavigator strategy (BFS + reactive toggle).
    """

    def __init__(self, codebook: VSACodebook | None = None, epsilon: float = 0.0):
        if codebook is None:
            codebook = VSACodebook(dim=512, seed=42)
        self.codebook = codebook
        self.grounder = LanguageGrounder(codebook)
        self.instructed = InstructedAgent(codebook, epsilon=epsilon)
        self.multi_room_nav = MultiRoomNavigator(epsilon=epsilon)
        self.pathfinder = GridPathfinder()

    def parse_instruction(self, instruction: str) -> list[str]:
        """Parse instruction into subgoal names."""
        return self.grounder.to_subgoals(instruction)

    def detect_env_type(self, obs: np.ndarray) -> str:
        """Detect environment type from observation.

        Returns "doorkey" if key is visible, "multiroom" otherwise.
        """
        has_key = bool(np.any(obs[:, :, 0] == OBJ_KEY))
        if has_key:
            return "doorkey"
        return "multiroom"

    def run_episode(
        self, env, instruction: str, max_steps: int = 500
    ) -> tuple[bool, int]:
        """Run one episode with language instruction.

        Returns (success, steps).
        """
        subgoals = self.parse_instruction(instruction)
        if not subgoals:
            return False, 0

        obs = env.reset() if not hasattr(env, '_last_obs') else env.reset()
        env_type = self.detect_env_type(obs)

        if env_type == "doorkey":
            return self.instructed.run_episode(env, instruction, max_steps)
        else:
            return self._run_multiroom(env, obs, subgoals, max_steps)

    def _run_multiroom(
        self, env, obs: np.ndarray, subgoals: list[str], max_steps: int
    ) -> tuple[bool, int]:
        """Run MultiRoom episode using BFS + reactive door toggle.

        For "reach_goal": navigate to goal with allow_door=True, toggle doors reactively.
        For "open_door" + "reach_goal": same strategy (doors opened on the way).
        """
        # Find goal position
        objs = find_objects(obs)
        if objs["goal_pos"] is None:
            return False, 0

        for step_i in range(max_steps):
            action = self.multi_room_nav.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                return reward > 0, step_i + 1
            if truncated:
                return False, step_i + 1

        return False, max_steps
