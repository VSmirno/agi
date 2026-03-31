"""Tests for Stage 32: Meta-Learning."""

from __future__ import annotations

import pytest

from snks.language.meta_learner import (
    EpisodeResult,
    MetaLearner,
    StrategyConfig,
    TaskProfile,
)


@pytest.fixture
def learner() -> MetaLearner:
    return MetaLearner(adaptation_rate=0.1)


class TestTaskProfile:
    def test_defaults(self):
        p = TaskProfile()
        assert p.has_demos is False
        assert p.known_skills == 0
        assert p.state_coverage == 0.0

    def test_custom(self):
        p = TaskProfile(has_demos=True, n_demos=3, known_skills=5)
        assert p.n_demos == 3
        assert p.known_skills == 5


class TestStrategySelection:
    def test_few_shot_when_demos_available(self, learner: MetaLearner):
        profile = TaskProfile(has_demos=True, n_demos=2, known_skills=0)
        config = learner.select_strategy(profile)
        assert config.strategy == "few_shot"
        assert "demos" in config.reason or "few-shot" in config.reason

    def test_skill_when_knowledge_available(self, learner: MetaLearner):
        profile = TaskProfile(known_skills=3, causal_links=10, state_coverage=0.5)
        config = learner.select_strategy(profile)
        assert config.strategy == "skill"
        assert config.use_analogy is True

    def test_curiosity_when_low_coverage(self, learner: MetaLearner):
        profile = TaskProfile(state_coverage=0.1, known_skills=0, causal_links=0)
        config = learner.select_strategy(profile)
        assert config.strategy == "curiosity"
        assert config.curiosity_epsilon >= 0.3

    def test_curiosity_when_high_prediction_error(self, learner: MetaLearner):
        profile = TaskProfile(
            state_coverage=0.5, known_skills=1, causal_links=2,
            mean_prediction_error=0.8,
        )
        config = learner.select_strategy(profile)
        assert config.strategy == "curiosity"

    def test_explore_fallback(self, learner: MetaLearner):
        profile = TaskProfile(
            state_coverage=0.5, known_skills=1, causal_links=2,
            mean_prediction_error=0.3,
        )
        config = learner.select_strategy(profile)
        assert config.strategy == "explore"

    def test_few_shot_overrides_skills(self, learner: MetaLearner):
        """Few-shot should be chosen even if coverage is low, as long as demos exist."""
        profile = TaskProfile(
            has_demos=True, n_demos=1, known_skills=0,
            state_coverage=0.1,
        )
        config = learner.select_strategy(profile)
        assert config.strategy == "few_shot"

    def test_skill_overrides_curiosity(self, learner: MetaLearner):
        """Skill strategy when knowledge is sufficient, even with moderate coverage."""
        profile = TaskProfile(
            known_skills=5, causal_links=20, state_coverage=0.4,
            mean_prediction_error=0.3,
        )
        config = learner.select_strategy(profile)
        assert config.strategy == "skill"


class TestAdaptation:
    def test_increase_epsilon_on_skill_failure(self, learner: MetaLearner):
        initial_eps = learner.current_epsilon
        profile = TaskProfile(known_skills=3, causal_links=10)
        result = EpisodeResult(success=False, steps=50)
        learner.adapt(profile, result)
        assert learner.current_epsilon > initial_eps

    def test_decrease_epsilon_on_quick_success(self, learner: MetaLearner):
        initial_eps = learner.current_epsilon
        profile = TaskProfile(known_skills=3, causal_links=10, state_coverage=0.7)
        result = EpisodeResult(success=True, steps=10, new_states_discovered=3)
        learner.adapt(profile, result)
        assert learner.current_epsilon < initial_eps

    def test_stagnation_increases_exploration(self, learner: MetaLearner):
        initial_eps = learner.current_epsilon
        profile = TaskProfile(state_coverage=0.2)
        result = EpisodeResult(
            success=False, steps=60, new_states_discovered=0,
        )
        learner.adapt(profile, result)
        assert learner.current_epsilon > initial_eps

    def test_successful_skills_tighten_analogy(self, learner: MetaLearner):
        initial_thr = learner.current_analogy_threshold
        profile = TaskProfile(known_skills=3, causal_links=10)
        result = EpisodeResult(success=True, steps=20, skills_used=2)
        learner.adapt(profile, result)
        assert learner.current_analogy_threshold > initial_thr

    def test_history_recorded(self, learner: MetaLearner):
        profile = TaskProfile(known_skills=3, causal_links=10)
        result = EpisodeResult(success=True, steps=20)
        learner.adapt(profile, result)
        assert len(learner.history) == 1

    def test_epsilon_bounded(self, learner: MetaLearner):
        """Epsilon should stay within [0.05, 0.5]."""
        profile = TaskProfile(known_skills=3, causal_links=10)
        # Many failures
        for _ in range(20):
            result = EpisodeResult(success=False, steps=50)
            learner.adapt(profile, result)
        assert learner.current_epsilon <= 0.5

        # Many quick successes
        for _ in range(50):
            result = EpisodeResult(success=True, steps=5)
            learner.adapt(profile, result)
        assert learner.current_epsilon >= 0.05

    def test_analogy_threshold_bounded(self, learner: MetaLearner):
        profile = TaskProfile(known_skills=3, causal_links=10)
        # Many skill failures → threshold should decrease but stay >= 0.4
        for _ in range(20):
            result = EpisodeResult(success=False, steps=50)
            learner.adapt(profile, result)
        assert learner.current_analogy_threshold >= 0.4


class TestReset:
    def test_reset_clears_state(self, learner: MetaLearner):
        profile = TaskProfile(known_skills=3, causal_links=10)
        learner.adapt(profile, EpisodeResult(success=False, steps=50))
        learner.reset()
        assert learner.current_epsilon == 0.2
        assert learner.current_analogy_threshold == 0.7
        assert len(learner.history) == 0


class TestMultiEpisodeSequence:
    def test_adaptation_trajectory(self, learner: MetaLearner):
        """Simulate a realistic multi-episode sequence."""
        # Episode 1: New task, no knowledge → curiosity
        p1 = TaskProfile(state_coverage=0.0)
        c1 = learner.select_strategy(p1)
        assert c1.strategy == "curiosity"

        # Episode 2: After some exploration → still curiosity
        p2 = TaskProfile(state_coverage=0.2, causal_links=3)
        r1 = EpisodeResult(success=False, steps=60, new_states_discovered=10)
        c2 = learner.adapt(p2, r1)
        assert c2.strategy == "curiosity"

        # Episode 3: Knowledge accumulated → skill
        p3 = TaskProfile(state_coverage=0.6, known_skills=3, causal_links=12)
        r2 = EpisodeResult(success=False, steps=40, new_states_discovered=5)
        c3 = learner.adapt(p3, r2)
        assert c3.strategy == "skill"

        # Episode 4: Skill success → exploit more
        p4 = TaskProfile(state_coverage=0.8, known_skills=4, causal_links=15)
        r3 = EpisodeResult(success=True, steps=15, skills_used=3)
        c4 = learner.adapt(p4, r3)
        assert c4.strategy == "skill"
        assert c4.curiosity_epsilon < c1.curiosity_epsilon
