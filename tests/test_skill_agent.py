"""Tests for language/skill_agent.py — SkillAgent (Stage 27)."""

import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.grid_perception import (
    SKS_DOOR_LOCKED,
    SKS_DOOR_OPEN,
    SKS_KEY_HELD,
    SKS_KEY_PRESENT,
    SKS_GOAL_PRESENT,
)
from snks.language.skill import Skill
from snks.language.skill_agent import SkillAgent, SkillEpisodeResult
from snks.language.skill_library import SkillLibrary


class TestSkillEpisodeResult:
    def test_extends_episode_result(self):
        r = SkillEpisodeResult()
        assert not r.success
        assert r.skills_used == []
        assert r.skills_total == 0

    def test_with_skills(self):
        r = SkillEpisodeResult(
            success=True,
            steps_taken=10,
            skills_used=["pickup_key", "toggle_door"],
            skills_total=2,
        )
        assert r.success
        assert len(r.skills_used) == 2


class TestSkillAgentInit:
    def test_init_with_empty_library(self):
        import gymnasium as gym
        import minigrid
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        env.reset(seed=0)
        agent = SkillAgent(env)
        assert len(agent._library.skills) == 0
        env.close()

    def test_init_with_library(self):
        import gymnasium as gym
        import minigrid
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        env.reset(seed=0)
        lib = SkillLibrary()
        lib.register(Skill(
            name="pickup_key",
            preconditions=frozenset({SKS_KEY_PRESENT}),
            effects=frozenset({SKS_KEY_HELD}),
            terminal_action=3,
            target_word="key",
        ))
        agent = SkillAgent(env, skill_library=lib)
        assert len(agent._library.skills) == 1
        env.close()


class TestSkillAgentExecution:
    def test_solves_doorkey_with_skills(self):
        """SkillAgent should solve DoorKey using extracted skills."""
        import gymnasium as gym
        import minigrid

        # Train to build causal model
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env.reset(seed=0)
        agent = SkillAgent(env)

        # First episode: learns causal links + extracts skills
        result = agent.run_episode(obs["mission"], max_steps=200)
        assert result.success

        # Second episode: should use skills
        env2 = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs2, _ = env2.reset(seed=1)
        agent._env = env2
        agent._executor._env = env2

        result2 = agent.run_episode(obs2["mission"], max_steps=200)
        assert result2.success
        env.close()
        env2.close()

    def test_extracts_skills_after_episode(self):
        import gymnasium as gym
        import minigrid

        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env.reset(seed=0)
        agent = SkillAgent(env)

        assert len(agent._library.skills) == 0
        agent.run_episode(obs["mission"], max_steps=200)
        # After first episode, skills should be extracted
        assert len(agent._library.skills) >= 2
        env.close()
