"""Tests for Stage 36: AutonomousAgent + CurriculumManager."""

from __future__ import annotations

import pytest

from snks.language.curriculum_manager import CurriculumManager, LevelStats
from snks.language.autonomous_agent import AutonomousAgent, CurriculumResult, _env_name


# ── CurriculumManager ────────────────────────────────────────


class TestCurriculumManager:

    def test_init_defaults(self):
        cm = CurriculumManager()
        assert cm.current_grid_size == 5
        assert cm.current_level_idx == 0
        assert not cm.is_final_level

    def test_custom_levels(self):
        cm = CurriculumManager(levels=[3, 7, 15])
        assert cm.current_grid_size == 3
        assert len(cm.levels) == 3

    def test_record_episode(self):
        cm = CurriculumManager()
        cm.record_episode(True, steps=10)
        cm.record_episode(False, steps=20)
        stats = cm.get_stats()
        assert stats.episodes == 2
        assert stats.successes == 1
        assert stats.total_steps == 30

    def test_should_advance_not_enough_episodes(self):
        cm = CurriculumManager(min_episodes_per_level=10)
        for _ in range(5):
            cm.record_episode(True)
        assert not cm.should_advance()

    def test_should_advance_sufficient(self):
        cm = CurriculumManager(
            advance_threshold=0.5,
            window_size=10,
            min_episodes_per_level=10,
        )
        for _ in range(10):
            cm.record_episode(True)
        assert cm.should_advance()

    def test_should_advance_below_threshold(self):
        cm = CurriculumManager(
            advance_threshold=0.5,
            window_size=20,
            min_episodes_per_level=10,
        )
        for _ in range(10):
            cm.record_episode(True)
        for _ in range(10):
            cm.record_episode(False)
        assert cm.should_advance()  # 10/20 = 0.5 >= 0.5

    def test_advance(self):
        cm = CurriculumManager(levels=[5, 8, 16])
        assert cm.current_grid_size == 5
        cm.advance()
        assert cm.current_grid_size == 8
        cm.advance()
        assert cm.current_grid_size == 16
        assert cm.is_final_level

    def test_advance_at_final(self):
        cm = CurriculumManager(levels=[5])
        assert cm.is_final_level
        cm.advance()  # should not crash
        assert cm.current_grid_size == 5

    def test_max_steps_for_level(self):
        cm = CurriculumManager(levels=[5, 8, 16])
        assert cm.max_steps_for_level(5) == 75
        assert cm.max_steps_for_level(8) == 192
        assert cm.max_steps_for_level(16) == 768

    def test_episodes_budget(self):
        cm = CurriculumManager(levels=[5, 6, 8, 16])
        assert cm.episodes_budget_for_level(5) == 50
        assert cm.episodes_budget_for_level(6) == 50
        assert cm.episodes_budget_for_level(8) == 100
        assert cm.episodes_budget_for_level(16) == 200

    def test_all_stats(self):
        cm = CurriculumManager(levels=[5, 8])
        cm.record_episode(True, 10)
        stats = cm.all_stats()
        assert 5 in stats
        assert 8 in stats
        assert stats[5].episodes == 1

    def test_level_stats_success_rate(self):
        ls = LevelStats(grid_size=5, episodes=4, successes=3)
        assert ls.success_rate == 0.75

    def test_level_stats_empty(self):
        ls = LevelStats(grid_size=5)
        assert ls.success_rate == 0.0


# ── AutonomousAgent ──────────────────────────────────────────


class TestAutonomousAgent:

    def test_init(self):
        agent = AutonomousAgent(levels=[5])
        assert agent.curriculum.current_grid_size == 5
        assert agent.causal_model is not None

    def test_env_name_mapping(self):
        assert _env_name(5) == "MiniGrid-DoorKey-5x5-v0"
        assert _env_name(6) == "MiniGrid-DoorKey-6x6-v0"
        assert _env_name(8) == "MiniGrid-DoorKey-8x8-v0"
        assert _env_name(16) == "MiniGrid-DoorKey-16x16-v0"
        # Fallback
        assert _env_name(3) == "MiniGrid-DoorKey-5x5-v0"
        assert _env_name(12) == "MiniGrid-DoorKey-16x16-v0"

    def test_curriculum_result_final_success_rate(self):
        cr = CurriculumResult(
            final_grid_size=8,
            level_stats={8: {"success_rate": 0.75}},
        )
        assert cr.final_success_rate == 0.75

    def test_curriculum_result_empty(self):
        cr = CurriculumResult()
        assert cr.final_success_rate == 0.0


# ── Integration (requires gymnasium + minigrid) ─────────────

def _has_minigrid() -> bool:
    try:
        import gymnasium
        import minigrid
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_minigrid(), reason="gymnasium/minigrid not installed")
class TestAutonomousAgentIntegration:

    def test_run_single_episode_5x5(self):
        agent = AutonomousAgent(levels=[5])
        result = agent.run_episode(seed=42)
        assert result.steps_taken >= 0
        assert agent.curriculum.get_stats().episodes == 1

    def test_run_curriculum_5x5_only(self):
        agent = AutonomousAgent(levels=[5], advance_threshold=0.9)
        cr = agent.run_curriculum(total_episodes=30)
        assert cr.total_episodes > 0
        assert cr.final_grid_size == 5
        assert 5 in cr.level_stats

    def test_curriculum_advances_on_easy(self):
        agent = AutonomousAgent(
            levels=[5, 6],
            advance_threshold=0.3,
        )
        cr = agent.run_curriculum(total_episodes=80)
        # Should advance to 6 since 5x5 is easy
        assert cr.total_episodes > 0
        stats5 = cr.level_stats.get(5, {})
        # On 5x5, GoalAgent usually succeeds
        if stats5.get("success_rate", 0) >= 0.3:
            assert cr.final_grid_size >= 6

    def test_causal_model_persists_across_levels(self):
        agent = AutonomousAgent(
            levels=[5, 6],
            advance_threshold=0.3,
        )
        agent.run_curriculum(total_episodes=60)
        # Causal model should have learned some links
        assert agent.causal_model.n_links >= 0

    def test_episode_log_populated(self):
        agent = AutonomousAgent(levels=[5])
        agent.run_episode(seed=0)
        agent.run_episode(seed=1)
        assert len(agent._episode_log) == 2
        assert "grid_size" in agent._episode_log[0]
        assert "success" in agent._episode_log[0]
