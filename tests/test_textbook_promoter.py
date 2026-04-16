"""Stage 88: Unit tests for TextbookPromoter + HypothesisTracker.initial + PostMortemLearner.from_promoted."""

from __future__ import annotations

import pytest
from pathlib import Path

from snks.agent.death_hypothesis import DeathHypothesis, HypothesisTracker
from snks.agent.post_mortem import PostMortemLearner
from snks.agent.textbook_promoter import TextbookPromoter, PROMOTE_SUPPORT_RATE, PROMOTE_MIN_OBSERVED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h(cause: str, vital: str, n_supporting: int, n_observed: int) -> DeathHypothesis:
    from snks.agent.death_hypothesis import _HYPOTHESIS_THRESHOLDS
    return DeathHypothesis(
        cause=cause,
        vital=vital,
        threshold=_HYPOTHESIS_THRESHOLDS[vital],
        n_supporting=n_supporting,
        n_observed=n_observed,
    )


# ---------------------------------------------------------------------------
# TextbookPromoter.should_promote
# ---------------------------------------------------------------------------

class TestShouldPromote:
    def test_pass_at_boundary(self):
        p = TextbookPromoter()
        h = _make_h("zombie", "drink", n_supporting=3, n_observed=5)
        # support_rate = 0.6, n_observed = 5 → should pass
        assert p.should_promote(h)

    def test_fail_support_rate_below(self):
        p = TextbookPromoter()
        # n_supporting=2, n_observed=5 → rate=0.4 < 0.5
        h = _make_h("zombie", "drink", n_supporting=2, n_observed=5)
        assert not p.should_promote(h)

    def test_fail_support_rate_exact_boundary(self):
        p = TextbookPromoter()
        # n_supporting=3, n_observed=6 → rate=0.5 exactly → should pass
        h = _make_h("zombie", "drink", n_supporting=3, n_observed=6)
        assert p.should_promote(h)

    def test_fail_n_observed_below(self):
        p = TextbookPromoter()
        # n_observed=4 < 5 → fail
        h = _make_h("zombie", "drink", n_supporting=4, n_observed=4)
        assert not p.should_promote(h)

    def test_fail_n_observed_zero(self):
        p = TextbookPromoter()
        h = _make_h("zombie", "drink", n_supporting=0, n_observed=0)
        assert not p.should_promote(h)


# ---------------------------------------------------------------------------
# TextbookPromoter.save + load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        p = TextbookPromoter()
        path = tmp_path / "promoted.yaml"
        h1 = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        h2 = _make_h("dehydration", "food", n_supporting=5, n_observed=5)
        p.save([h1, h2], path)

        loaded = p.load(path)
        assert len(loaded) == 2
        keys = {(h.cause, h.vital) for h in loaded}
        assert ("zombie", "drink") in keys
        assert ("dehydration", "food") in keys

        by_key = {(h.cause, h.vital): h for h in loaded}
        assert by_key[("zombie", "drink")].n_observed == 13
        assert by_key[("zombie", "drink")].n_supporting == 7
        assert by_key[("dehydration", "food")].n_observed == 5

    def test_load_missing_file_returns_empty(self, tmp_path):
        p = TextbookPromoter()
        result = p.load(tmp_path / "nonexistent.yaml")
        assert result == []

    def test_load_empty_file_returns_empty(self, tmp_path):
        p = TextbookPromoter()
        path = tmp_path / "empty.yaml"
        path.write_text("hypotheses: []\n")
        assert p.load(path) == []

    def test_merge_higher_n_observed_wins(self, tmp_path):
        p = TextbookPromoter()
        path = tmp_path / "promoted.yaml"
        existing = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        p.save([existing], path)

        # New entry with lower n_observed — existing should win
        weaker = _make_h("zombie", "drink", n_supporting=4, n_observed=7)
        p.save([weaker], path)

        loaded = p.load(path)
        assert len(loaded) == 1
        assert loaded[0].n_observed == 13

    def test_merge_lower_n_observed_loses(self, tmp_path):
        p = TextbookPromoter()
        path = tmp_path / "promoted.yaml"
        existing = _make_h("zombie", "drink", n_supporting=4, n_observed=7)
        p.save([existing], path)

        # New entry with higher n_observed — new should win
        stronger = _make_h("zombie", "drink", n_supporting=9, n_observed=15)
        p.save([stronger], path)

        loaded = p.load(path)
        assert len(loaded) == 1
        assert loaded[0].n_observed == 15

    def test_merge_adds_new_key(self, tmp_path):
        p = TextbookPromoter()
        path = tmp_path / "promoted.yaml"
        h1 = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        p.save([h1], path)

        h2 = _make_h("dehydration", "drink", n_supporting=5, n_observed=5)
        p.save([h2], path)

        loaded = p.load(path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# PostMortemLearner.from_promoted
# ---------------------------------------------------------------------------

class TestFromPromoted:
    def test_empty_list_gives_defaults(self):
        learner = PostMortemLearner.from_promoted([])
        assert learner.drink_threshold == pytest.approx(3.0)
        assert learner.food_threshold == pytest.approx(3.0)
        assert learner.health_weight == pytest.approx(1.0)

    def test_drink_threshold_raised(self):
        # zombie+drink rate=0.54 → bump=1.08 → drink_threshold=4.08
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        learner = PostMortemLearner.from_promoted([h])
        assert learner.drink_threshold == pytest.approx(4.08, abs=0.01)

    def test_food_threshold_raised(self):
        h = _make_h("starvation", "food", n_supporting=5, n_observed=5)
        learner = PostMortemLearner.from_promoted([h])
        # rate=1.0, bump=2.0 → food_threshold=5.0
        assert learner.food_threshold == pytest.approx(5.0)

    def test_health_weight_raised_for_zombie(self):
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        learner = PostMortemLearner.from_promoted([h])
        # health_weight = max(1.0, 1.0+0.54) = 1.54
        assert learner.health_weight == pytest.approx(1.54, abs=0.01)

    def test_health_weight_raised_for_skeleton(self):
        h = _make_h("skeleton", "drink", n_supporting=6, n_observed=10)
        learner = PostMortemLearner.from_promoted([h])
        assert learner.health_weight == pytest.approx(1.6, abs=0.01)

    def test_dehydration_does_not_raise_health_weight(self):
        h = _make_h("dehydration", "drink", n_supporting=5, n_observed=5)
        learner = PostMortemLearner.from_promoted([h])
        assert learner.health_weight == pytest.approx(1.0)

    def test_two_hypotheses_ordered(self):
        # Pass zombie+drink first, then dehydration+drink — dehydration has higher rate
        h1 = _make_h("zombie", "drink", n_supporting=7, n_observed=13)     # rate=0.54
        h2 = _make_h("dehydration", "drink", n_supporting=5, n_observed=5) # rate=1.0
        learner = PostMortemLearner.from_promoted([h1, h2])
        # After h1: drink_threshold=4.08; after h2: max(4.08, 5.0)=5.0
        assert learner.drink_threshold == pytest.approx(5.0)
        # health_weight only bumped by zombie (cause=zombie → +0.54), dehydration doesn't qualify
        assert learner.health_weight == pytest.approx(1.54, abs=0.01)

    def test_bump_capped_at_2(self):
        h = _make_h("zombie", "food", n_supporting=10, n_observed=10)  # rate=1.0
        learner = PostMortemLearner.from_promoted([h])
        # bump = min(1.0 * 2.0, 2.0) = 2.0 → food_threshold = 5.0
        assert learner.food_threshold == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# HypothesisTracker with initial=
# ---------------------------------------------------------------------------

class TestHypothesisTrackerInitial:
    def test_empty_initial_same_as_default(self):
        t1 = HypothesisTracker()
        t2 = HypothesisTracker(initial=[])
        assert t1.all_hypotheses() == t2.all_hypotheses()

    def test_promoted_immediately_in_all_hypotheses(self):
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        tracker = HypothesisTracker(initial=[h])
        all_h = tracker.all_hypotheses()
        keys = {(x.cause, x.vital) for x in all_h}
        assert ("zombie", "drink") in keys

    def test_promoted_immediately_verifiable(self):
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        tracker = HypothesisTracker(initial=[h])
        assert tracker.n_verifiable() >= 1

    def test_active_hypothesis_returns_promoted_before_first_death(self):
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        tracker = HypothesisTracker(initial=[h])
        active = tracker.active_hypothesis()
        assert active is not None
        assert active.cause == "zombie"
        assert active.vital == "drink"

    def test_promoted_persists_after_record_for_unseen_key(self):
        # Promoted: zombie+drink. Record a death from starvation+food.
        # zombie+drink should still be in all_hypotheses (unseen key preserved).
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        tracker = HypothesisTracker(initial=[h])

        # Record a death attributed to starvation with food < threshold
        tracker.record(attribution={"starvation": 1.0}, vitals_at_death={"food": 0.5, "drink": 8.0})

        keys = {(x.cause, x.vital) for x in tracker.all_hypotheses()}
        assert ("zombie", "drink") in keys

    def test_live_entry_replaces_promoted_for_same_key(self):
        # Promoted: zombie+drink (n_obs=13). Record 5 zombie deaths in new gen.
        # After 5 records, the live-derived zombie+drink entry should be present
        # and zombie+drink should NOT be the promoted version (it gets replaced).
        h = _make_h("zombie", "drink", n_supporting=7, n_observed=13)
        tracker = HypothesisTracker(initial=[h])

        for _ in range(5):
            tracker.record(
                attribution={"zombie": 1.0},
                vitals_at_death={"drink": 2.0, "food": 5.0},
            )

        zombie_drink = [
            x for x in tracker.all_hypotheses()
            if x.cause == "zombie" and x.vital == "drink"
        ]
        assert len(zombie_drink) == 1
        # The live-derived entry has n_observed from _records (5), not the promoted 13
        assert zombie_drink[0].n_observed == 5
