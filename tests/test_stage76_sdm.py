"""Stage 76 Phase 3: EpisodicSDM tests.

Covers:
- Episode dataclass
- FIFO buffer write / overflow / len
- recall (linear-scan popcount overlap)
- score_actions (deficit × delta aggregation)
- select_action (softmax selection)
- count_similar (bootstrap gate)
- Performance smoke (10K episodes × 4096 bits within budget)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from snks.agent.perception import HomeostaticTracker
from snks.memory.episodic_sdm import (
    Episode,
    EpisodicSDM,
    score_actions,
    select_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sdr(n_bits: int = 4096, n_active: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    bits = np.zeros(n_bits, dtype=bool)
    indices = rng.choice(n_bits, n_active, replace=False)
    bits[indices] = True
    return bits


def make_episode(
    action: str,
    body_delta: dict[str, int] | None = None,
    state_seed: int = 0,
    next_state_seed: int = 1,
    step: int = 0,
) -> Episode:
    return Episode(
        state_sdr=make_sdr(seed=state_seed),
        action=action,
        next_state_sdr=make_sdr(seed=next_state_seed),
        body_delta=body_delta or {},
        step=step,
    )


def make_tracker_with_obs(observed_max: dict[str, int]) -> HomeostaticTracker:
    t = HomeostaticTracker()
    t.observed_max = dict(observed_max)
    return t


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


class TestEpisode:
    def test_create(self):
        state = make_sdr(seed=1)
        next_state = make_sdr(seed=2)
        ep = Episode(
            state_sdr=state,
            action="move_right",
            next_state_sdr=next_state,
            body_delta={"health": -1},
            step=5,
        )
        assert ep.action == "move_right"
        assert ep.step == 5
        assert ep.body_delta == {"health": -1}
        assert np.array_equal(ep.state_sdr, state)


# ---------------------------------------------------------------------------
# Buffer mechanics
# ---------------------------------------------------------------------------


class TestBuffer:
    def test_empty_buffer(self):
        sdm = EpisodicSDM(capacity=100)
        assert len(sdm) == 0
        assert sdm.recall(make_sdr()) == []

    def test_write_grows_len(self):
        sdm = EpisodicSDM(capacity=100)
        for i in range(5):
            sdm.write(make_episode("noop", step=i))
        assert len(sdm) == 5

    def test_capacity_enforced(self):
        sdm = EpisodicSDM(capacity=3)
        for i in range(10):
            sdm.write(make_episode("noop", step=i))
        assert len(sdm) == 3  # never exceeds capacity

    def test_fifo_overwrite_order(self):
        """When full, oldest entries are replaced by newest."""
        sdm = EpisodicSDM(capacity=3)
        for i in range(5):
            sdm.write(make_episode("noop", step=i))
        # After writing 0..4 into capacity-3 buffer, expected steps are {2,3,4}
        stored_steps = {ep.step for ep in sdm._buffer}
        assert stored_steps == {2, 3, 4}


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_empty_buffer_returns_empty(self):
        sdm = EpisodicSDM(capacity=100)
        assert sdm.recall(make_sdr()) == []

    def test_top_k_respected(self):
        sdm = EpisodicSDM(capacity=100)
        for i in range(10):
            sdm.write(make_episode("noop", state_seed=i, step=i))
        results = sdm.recall(make_sdr(seed=0), top_k=3)
        assert len(results) == 3

    def test_most_similar_comes_first(self):
        """The exact-match SDR should yield max overlap."""
        sdm = EpisodicSDM(capacity=100)
        # Write 5 random episodes + the target
        for i in range(1, 6):
            sdm.write(make_episode("noop", state_seed=i, step=i))
        target_sdr = make_sdr(seed=42)
        target_ep = Episode(
            state_sdr=target_sdr,
            action="target_action",
            next_state_sdr=make_sdr(seed=43),
            body_delta={},
            step=99,
        )
        sdm.write(target_ep)

        results = sdm.recall(target_sdr, top_k=1)
        assert len(results) == 1
        overlap, ep = results[0]
        assert ep.action == "target_action"
        assert overlap == target_sdr.sum()  # perfect self-match

    def test_sorted_descending(self):
        sdm = EpisodicSDM(capacity=100)
        for i in range(10):
            sdm.write(make_episode("noop", state_seed=i, step=i))
        results = sdm.recall(make_sdr(seed=0), top_k=10)
        overlaps = [ov for ov, _ in results]
        assert overlaps == sorted(overlaps, reverse=True)


# ---------------------------------------------------------------------------
# Bootstrap gate — count_similar
# ---------------------------------------------------------------------------


class TestCountSimilar:
    def test_empty_buffer_zero(self):
        sdm = EpisodicSDM(capacity=100)
        assert sdm.count_similar(make_sdr()) == 0

    def test_zero_popcount_query(self):
        sdm = EpisodicSDM(capacity=100)
        sdm.write(make_episode("noop"))
        empty_query = np.zeros(4096, dtype=bool)
        assert sdm.count_similar(empty_query) == 0

    def test_self_match_counted(self):
        sdm = EpisodicSDM(capacity=100)
        sdr = make_sdr(seed=7)
        sdm.write(Episode(
            state_sdr=sdr,
            action="noop",
            next_state_sdr=make_sdr(seed=8),
            body_delta={},
            step=0,
        ))
        # Query = same SDR → overlap = popcount → ≥ threshold
        assert sdm.count_similar(sdr, threshold_ratio=0.5) == 1

    def test_distinct_random_below_threshold(self):
        """Random SDRs don't match at 50% overlap."""
        sdm = EpisodicSDM(capacity=100)
        for i in range(20):
            sdm.write(make_episode("noop", state_seed=i + 100, step=i))
        # Query totally unrelated
        query = make_sdr(seed=999)
        count = sdm.count_similar(query, threshold_ratio=0.5)
        assert count == 0


# ---------------------------------------------------------------------------
# score_actions
# ---------------------------------------------------------------------------


class TestScoreActions:
    def test_empty_recalled_empty_scores(self):
        tracker = make_tracker_with_obs({"health": 9})
        assert score_actions([], {"health": 5}, tracker) == {}

    def test_deficit_weighted_recovery(self):
        """When health is low, actions that raised health get positive score."""
        tracker = make_tracker_with_obs({"health": 9, "food": 9})
        recalled = [
            (100, Episode(
                state_sdr=make_sdr(),
                action="eat_cow",
                next_state_sdr=make_sdr(),
                body_delta={"health": +4},  # restorative
                step=0,
            )),
            (90, Episode(
                state_sdr=make_sdr(),
                action="attack_zombie",
                next_state_sdr=make_sdr(),
                body_delta={"health": -3},  # harmful
                step=1,
            )),
        ]
        current_body = {"health": 2, "food": 9}
        scores = score_actions(recalled, current_body, tracker)
        assert scores["eat_cow"] > scores["attack_zombie"]
        # Specific values: deficit_health = 9-2 = 7
        #   eat_cow: 7 * 4 = 28
        #   attack: 7 * -3 = -21
        assert scores["eat_cow"] == 28
        assert scores["attack_zombie"] == -21

    def test_no_deficit_zero_contribution(self):
        """When all body stats are at max, deficits are 0 → all scores 0."""
        tracker = make_tracker_with_obs({"health": 9, "food": 9})
        recalled = [
            (100, Episode(
                state_sdr=make_sdr(),
                action="a",
                next_state_sdr=make_sdr(),
                body_delta={"health": 2, "food": 1},
                step=0,
            )),
        ]
        current_body = {"health": 9, "food": 9}  # at max
        scores = score_actions(recalled, current_body, tracker)
        assert scores["a"] == 0

    def test_unknown_variable_ignored(self):
        """Delta for variable not tracked doesn't contribute."""
        tracker = make_tracker_with_obs({"health": 9})  # only health observed
        recalled = [
            (100, Episode(
                state_sdr=make_sdr(),
                action="a",
                next_state_sdr=make_sdr(),
                body_delta={"health": 2, "mana": 100},  # mana unknown
                step=0,
            )),
        ]
        scores = score_actions(recalled, {"health": 5}, tracker)
        # Only health contributes: deficit=4, delta=2 → 8
        assert scores["a"] == 8

    def test_multiple_episodes_averaged(self):
        """Score per action is mean over episodes of that action."""
        tracker = make_tracker_with_obs({"health": 9})
        recalled = [
            (100, Episode(make_sdr(), "a", make_sdr(), {"health": 2}, 0)),
            (100, Episode(make_sdr(), "a", make_sdr(), {"health": 4}, 1)),
        ]
        scores = score_actions(recalled, {"health": 5}, tracker)
        # deficit = 4; contributions = 8, 16; mean = 12
        assert scores["a"] == 12

    def test_deficit_sign_emerges_from_data(self):
        """Test that 'restorative' signal emerges when deltas match deficits.

        No hardcoded "higher is better". The sign comes from deficit × delta.
        """
        tracker = make_tracker_with_obs({"energy": 9})
        # When energy is low (deficit 7), +delta is scored positive
        recalled = [
            (100, Episode(make_sdr(), "sleep", make_sdr(), {"energy": +5}, 0)),
            (100, Episode(make_sdr(), "fight", make_sdr(), {"energy": -3}, 1)),
        ]
        scores = score_actions(recalled, {"energy": 2}, tracker)
        assert scores["sleep"] > 0
        assert scores["fight"] < 0


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------


class TestSelectAction:
    def test_empty_returns_none(self):
        assert select_action({}) is None

    def test_higher_score_more_likely(self):
        """Action with highest score should be sampled most often.

        Use modest score differences so softmax isn't fully peaked.
        """
        scores = {"a": 2.0, "b": 1.0, "c": 0.0}
        rng = np.random.RandomState(0)
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(2000):
            action = select_action(scores, temperature=1.0, rng=rng)
            counts[action] += 1
        # Expected probs ≈ [0.665, 0.245, 0.09] → ~[1330, 490, 180]
        assert counts["a"] > counts["b"] > counts["c"]

    def test_greedy_at_low_temperature(self):
        """Very low temperature → always picks argmax."""
        scores = {"a": 1.0, "b": 2.0, "c": 0.5}
        rng = np.random.RandomState(0)
        picks = {select_action(scores, temperature=0.001, rng=rng) for _ in range(20)}
        assert picks == {"b"}

    def test_uniform_at_high_temperature(self):
        """Very high temperature → roughly uniform selection."""
        scores = {"a": 10.0, "b": -10.0, "c": 5.0}
        rng = np.random.RandomState(42)
        counts = {"a": 0, "b": 0, "c": 0}
        n = 3000
        for _ in range(n):
            counts[select_action(scores, temperature=1000.0, rng=rng)] += 1
        # Each should be roughly n/3 = 1000, allow ±200
        for k in counts:
            assert 800 <= counts[k] <= 1200, f"{k}={counts[k]} not near 1000"

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            select_action({"a": 1.0}, temperature=0.0)

    def test_deterministic_with_seeded_rng(self):
        scores = {"a": 1.0, "b": 2.0, "c": 0.5}
        rng1 = np.random.RandomState(7)
        rng2 = np.random.RandomState(7)
        seq1 = [select_action(scores, 1.0, rng1) for _ in range(10)]
        seq2 = [select_action(scores, 1.0, rng2) for _ in range(10)]
        assert seq1 == seq2


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_recall_10k_buffer_fast(self):
        """10K episodes × 4096 bits: recall within ~200ms is acceptable."""
        sdm = EpisodicSDM(capacity=10_000)
        for i in range(10_000):
            sdm.write(make_episode("noop", state_seed=i, step=i))
        query = make_sdr(seed=500)
        t0 = time.perf_counter()
        _ = sdm.recall(query, top_k=20)
        elapsed = time.perf_counter() - t0
        # Lenient bound for CI variability; spec target ~50ms, we allow 500ms
        assert elapsed < 0.5, f"recall took {elapsed*1000:.1f}ms"
