"""Stage 79 — unit tests for SurpriseAccumulator + ContextKey.

Covers:
  - quartile_for boundary correctness
  - ContextKey.from_state quartile mapping
  - ContextKey hashability + equality
  - ContextKey.coarsen() collapses body_quartiles
  - SurpriseAccumulator.observe writes both L2 and L1 buckets
  - Sliding-window eviction at MAX_BUCKET_SIZE
  - stats() reports correct counts
"""

from __future__ import annotations

from snks.learning.surprise_accumulator import (
    BODY_ORDER,
    MAX_BUCKET_SIZE,
    QUARTILE_BOUNDARIES,
    ContextKey,
    SurpriseAccumulator,
    SurpriseRecord,
    quartile_for,
)


# ---------------------------------------------------------------------------
# quartile_for
# ---------------------------------------------------------------------------


def test_quartile_for_boundary_values():
    # Boundaries are 2.5 / 5.0 / 7.5
    assert quartile_for(0.0) == 0
    assert quartile_for(2.4) == 0
    assert quartile_for(2.5) == 1  # boundary inclusive of upper bucket
    assert quartile_for(4.9) == 1
    assert quartile_for(5.0) == 2
    assert quartile_for(7.4) == 2
    assert quartile_for(7.5) == 3
    assert quartile_for(9.0) == 3


def test_quartile_for_clamps_extremes():
    assert quartile_for(-1.0) == 0
    assert quartile_for(100.0) == 3


def test_quartile_for_canonical_conjunctive_case():
    # Stage 78a's conjunctive rule fires when food == 0 OR drink == 0.
    # Both must land in quartile 0 for L2 buckets to capture them as
    # the same context.
    assert quartile_for(0.0) == 0
    # Full body is quartile 3 (used as the "non-starving" sleep case)
    assert quartile_for(9.0) == 3


# ---------------------------------------------------------------------------
# ContextKey
# ---------------------------------------------------------------------------


def test_context_key_from_state_basic():
    key = ContextKey.from_state(
        visible={"tree", "cow"},
        body={"health": 9.0, "food": 0.0, "drink": 5.0, "energy": 7.0},
        action="sleep",
    )
    assert key.visible == frozenset({"tree", "cow"})
    assert key.body_quartiles == (3, 0, 2, 2)
    assert key.action == "sleep"


def test_context_key_is_hashable_and_equality_holds():
    a = ContextKey.from_state({"tree"}, {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0}, "do")
    b = ContextKey.from_state({"tree"}, {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0}, "do")
    c = ContextKey.from_state({"tree"}, {"health": 5.0, "food": 0.0, "drink": 5.0, "energy": 5.0}, "do")

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # Both can live in a dict
    d = {a: 1}
    d[b] = 2  # overwrites a's entry because a == b
    assert len(d) == 1
    d[c] = 3
    assert len(d) == 2


def test_context_key_coarsen_zeros_body_quartiles():
    full = ContextKey.from_state(
        visible={"skeleton"},
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        action="move_left",
    )
    coarse = full.coarsen()
    assert coarse.visible == full.visible
    assert coarse.action == full.action
    assert coarse.body_quartiles == (0, 0, 0, 0)
    assert coarse != full
    assert coarse.is_l1()
    assert not full.is_l1()


def test_context_key_l1_self_coarsen_is_idempotent():
    l1 = ContextKey(visible=frozenset({"tree"}), body_quartiles=(0, 0, 0, 0), action="do")
    assert l1.is_l1()
    again = l1.coarsen()
    assert again == l1


def test_context_key_visible_order_independent():
    a = ContextKey.from_state({"tree", "cow"}, {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0}, "do")
    b = ContextKey.from_state({"cow", "tree"}, {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0}, "do")
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# SurpriseAccumulator
# ---------------------------------------------------------------------------


def _mk_l2_key() -> ContextKey:
    return ContextKey.from_state(
        visible={"cow"},
        body={"health": 5.0, "food": 0.0, "drink": 5.0, "energy": 5.0},
        action="do",
    )


def test_accumulator_observe_writes_both_l2_and_l1():
    acc = SurpriseAccumulator()
    key = _mk_l2_key()
    assert not key.is_l1()  # food=0 → quartile 0 ≠ all-zero quartiles in this case (health is q2, drink q2, energy q2)

    acc.observe(
        context=key,
        predicted={"health": 0.04, "food": 0.0, "drink": 0.0, "energy": 0.0},
        actual={"health": -0.067, "food": 0.0, "drink": 0.0, "energy": 0.0},
        tick_id=0,
    )

    # L2 bucket has the record
    assert acc.bucket_size(key) == 1
    record = acc.bucket_records(key)[0]
    assert record.delta["health"] < 0  # actual - predicted = -0.067 - 0.04 = -0.107

    # L1 bucket also has the record
    l1_key = key.coarsen()
    assert acc.bucket_size(l1_key) == 1
    assert acc.bucket_records(l1_key)[0].delta["health"] == record.delta["health"]


def test_accumulator_l1_observation_does_not_double_count_itself():
    acc = SurpriseAccumulator()
    l1_key = ContextKey(visible=frozenset({"tree"}), body_quartiles=(0, 0, 0, 0), action="do")

    acc.observe(
        context=l1_key,
        predicted={"food": 0.0},
        actual={"food": 1.0},
        tick_id=0,
    )

    assert acc.bucket_size(l1_key) == 1
    # No "second L1" was written
    assert acc.stats()["n_buckets"] == 1


def test_accumulator_sliding_window_evicts_oldest():
    cap = 5
    acc = SurpriseAccumulator(max_bucket_size=cap)
    key = _mk_l2_key()

    for i in range(cap + 3):
        acc.observe(
            context=key,
            predicted={"food": 0.0},
            actual={"food": float(i)},
            tick_id=i,
        )

    # L2 bucket capped at `cap`
    assert acc.bucket_size(key) == cap
    records = acc.bucket_records(key)
    # The earliest 3 records (tick_id 0, 1, 2) should have been evicted
    assert records[0].tick_id == 3
    assert records[-1].tick_id == cap + 2


def test_accumulator_iter_buckets_snapshots_independent_of_mutation():
    acc = SurpriseAccumulator()
    key = _mk_l2_key()

    acc.observe(key, {"food": 0.0}, {"food": 1.0}, tick_id=0)
    acc.observe(key, {"food": 0.0}, {"food": 1.0}, tick_id=1)

    snapshots = list(acc.iter_buckets())
    assert len(snapshots) == 2  # L2 + L1 sharing same observations

    # Mutate after iteration started — snapshots should remain valid
    acc.observe(key, {"food": 0.0}, {"food": 5.0}, tick_id=2)
    # Snapshots are plain lists copied at iter time
    for _, recs in snapshots:
        # The snapshot recorded in snapshots should still have only the
        # records from before the third observe() call.
        assert all(r.tick_id <= 1 for r in recs)


def test_accumulator_stats_counts_l1_and_l2():
    acc = SurpriseAccumulator()
    k1 = ContextKey.from_state({"tree"}, {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0}, "do")
    k2 = ContextKey.from_state({"cow"}, {"health": 9.0, "food": 0.0, "drink": 5.0, "energy": 5.0}, "sleep")

    acc.observe(k1, {"food": 0.0}, {"food": 0.0}, tick_id=0)
    acc.observe(k1, {"food": 0.0}, {"food": 0.0}, tick_id=1)
    acc.observe(k2, {"health": 0.04}, {"health": -0.067}, tick_id=2)

    stats = acc.stats()
    # Two distinct L2 keys → also two distinct L1 keys (different visible)
    assert stats["n_buckets"] == 4
    assert stats["n_l2_buckets"] == 2
    assert stats["n_l1_buckets"] == 2
    assert stats["total_observations"] == 3
    # Total records is 6 because each obs feeds both L2 and L1
    assert stats["total_records"] == 6


def test_accumulator_delta_uses_default_body_order():
    """`observe` must populate delta for ALL body vars in BODY_ORDER,
    even if predicted/actual omit some — defaulted to 0.0."""
    acc = SurpriseAccumulator()
    key = _mk_l2_key()

    acc.observe(
        context=key,
        predicted={"health": 0.04},   # food/drink/energy missing
        actual={"health": 0.0},       # food/drink/energy missing
        tick_id=0,
    )

    record = acc.bucket_records(key)[0]
    assert set(record.delta.keys()) == set(BODY_ORDER)
    assert record.delta["health"] == -0.04
    assert record.delta["food"] == 0.0
    assert record.delta["drink"] == 0.0
    assert record.delta["energy"] == 0.0
