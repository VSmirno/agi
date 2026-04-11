"""Stage 79 — unit tests for RuleNursery.

The accumulator writes each L2 observation to BOTH the L2 bucket and
the L1 (coarsened) bucket — that's by design (see the design doc),
because L1 buckets catch body-independent rules and L2 buckets catch
body-conditional rules. This means an L2 observation will, after
sufficient records, produce TWO candidates (one from L2, one from
L1). For the simpler unit tests below we use L1-direct keys
(body_quartiles == (0, 0, 0, 0)) so each `observe` only writes one
bucket, keeping the test focused on a single emit/verify/promote
cycle. The L1+L2 dual-emission interaction is exercised by a
dedicated test at the end.

Covers:
  - _try_emit returns None below MIN_OBS
  - _try_emit returns None when MAD too high (inconsistent records)
  - _try_emit returns None when mean below SIGNIFICANCE_FLOOR
  - _try_emit returns CandidateRule when MIN_OBS + consistent
  - tick() promotes after VERIFY_N matching records
  - tick() rejects when verify mean drifts > VERIFY_TOL
  - Promoted candidate is recorded in promoted_contexts (no re-emit)
  - Rejected candidate is removed from in-flight tracking
  - Promoted candidate writes a LearnedRule to store
  - L1 + L2 dual emission produces two candidates from one stream
"""

from __future__ import annotations

from typing import Any

from snks.agent.learned_rule import LearnedRule
from snks.learning.rule_nursery import CandidateRule, RuleNursery
from snks.learning.surprise_accumulator import (
    BODY_ORDER,
    ContextKey,
    SurpriseAccumulator,
)


# ---------------------------------------------------------------------------
# Stub store — minimal implementation of `add_learned_rule`
# ---------------------------------------------------------------------------


class StubStore:
    """Tiny test double for ConceptStore — captures learned rules."""

    def __init__(self) -> None:
        self.learned_rules: list[Any] = []

    def add_learned_rule(self, rule: Any) -> None:
        self.learned_rules.append(rule)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l1_ctx() -> ContextKey:
    """An L1 (coarse, body-independent) context. Single-bucket — observe()
    writes only to this key, since coarsen() is a no-op for L1 keys."""
    return ContextKey(
        visible=frozenset({"cow"}),
        body_quartiles=(0, 0, 0, 0),
        action="sleep",
    )


def _l2_ctx_food_zero() -> ContextKey:
    """An L2 context with food=quartile-0. Used for the L1+L2 dual
    emission test."""
    return ContextKey(
        visible=frozenset({"cow"}),
        body_quartiles=(2, 0, 2, 2),
        action="sleep",
    )


def _populate(
    acc: SurpriseAccumulator,
    key: ContextKey,
    deltas: list[dict[str, float]],
    start_tick: int = 0,
) -> None:
    """Helper: feed a sequence of (predicted=0, actual=delta) records
    so the accumulator delta equals the desired value."""
    for i, delta in enumerate(deltas):
        acc.observe(
            context=key,
            predicted={var: 0.0 for var in BODY_ORDER},
            actual={var: delta.get(var, 0.0) for var in BODY_ORDER},
            tick_id=start_tick + i,
        )


# ---------------------------------------------------------------------------
# _try_emit gates
# ---------------------------------------------------------------------------


def test_try_emit_below_min_obs_returns_none():
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    key = _l1_ctx()  # uses MIN_OBS_L1 = 5
    # Add 4 records — below MIN_OBS_L1
    _populate(acc, key, [{"health": -0.067}] * 4)
    nursery.tick(acc, store=None, current_tick=0)
    assert nursery.candidates() == []
    assert nursery.stats()["emitted"] == 0


def test_try_emit_high_mad_returns_none():
    """Inconsistent records → MAD high → mean not significant → no emit."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    key = _l1_ctx()
    # Wildly varying health deltas with mean ≈ 0
    deltas = [{"health": v} for v in [+1.0, -1.0, +0.5, -0.5, +0.8, -0.8, +0.2, -0.2, +1.5, -1.5]]
    _populate(acc, key, deltas)
    nursery.tick(acc, store=None, current_tick=0)
    assert nursery.candidates() == []


def test_try_emit_below_significance_floor_returns_none():
    """Mean magnitude < SIGNIFICANCE_FLOOR → no emit even with low MAD."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    key = _l1_ctx()
    # Tight cluster around 0.005 — below SIGNIFICANCE_FLOOR=0.01
    deltas = [{"health": v} for v in [0.005, 0.006, 0.004, 0.005, 0.006,
                                       0.004, 0.005, 0.006, 0.004, 0.005]]
    _populate(acc, key, deltas)
    nursery.tick(acc, store=None, current_tick=0)
    assert nursery.candidates() == []


def test_try_emit_consistent_significant_returns_candidate():
    """Tight cluster around -0.067 → emit (single bucket via L1 key)."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    key = _l1_ctx()
    deltas = [{"health": v} for v in [-0.065, -0.067, -0.069, -0.066, -0.068,
                                       -0.067, -0.067, -0.068, -0.066, -0.067]]
    _populate(acc, key, deltas)
    nursery.tick(acc, store=None, current_tick=0)
    cands = nursery.candidates()
    assert len(cands) == 1
    cand = cands[0]
    assert cand.status == "verifying"
    assert "health" in cand.mean_effect
    assert abs(cand.mean_effect["health"] - (-0.067)) < 0.005
    assert "food" not in cand.mean_effect


def test_try_emit_only_significant_vars_in_effect():
    """When multiple vars are present, only the ones above the gates
    appear in mean_effect."""
    import random
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    key = _l1_ctx()
    rng = random.Random(42)
    deltas = [
        {"health": -0.067 + rng.uniform(-0.001, 0.001),
         "food": rng.uniform(-0.001, 0.001),
         "drink": 0.0,
         "energy": 0.0}
        for _ in range(10)
    ]
    _populate(acc, key, deltas)
    nursery.tick(acc, store=None, current_tick=0)
    cand = nursery.candidates()[0]
    assert "health" in cand.mean_effect
    assert "food" not in cand.mean_effect
    assert "drink" not in cand.mean_effect
    assert "energy" not in cand.mean_effect


# ---------------------------------------------------------------------------
# Verification + promotion
# ---------------------------------------------------------------------------


def test_tick_promotes_candidate_after_verify_n_matching_records():
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    store = StubStore()
    key = _l1_ctx()

    # Phase 1: emit (need MIN_OBS_L1=5 for L1 key)
    initial = [{"health": -0.067}] * RuleNursery.MIN_OBS_L1
    _populate(acc, key, initial, start_tick=0)
    nursery.tick(acc, store, current_tick=0)
    assert len(nursery.candidates()) == 1
    assert nursery.candidates()[0].status == "verifying"

    # Phase 2: feed VERIFY_N more matching records, tick after each
    for i in range(RuleNursery.VERIFY_N):
        _populate(acc, key, [{"health": -0.067}], start_tick=100 + i)
        nursery.tick(acc, store, current_tick=100 + i)

    assert nursery.stats()["promoted"] == 1
    assert nursery.stats()["rejected"] == 0
    assert len(store.learned_rules) == 1
    rule = store.learned_rules[0]
    assert isinstance(rule, LearnedRule)
    assert rule.precondition == key
    assert "health" in rule.effect
    assert abs(rule.effect["health"] - (-0.067)) < 0.005


def test_tick_rejects_when_verify_mean_drifts():
    """Verify records have a different mean than the candidate's mean_effect
    → rejection (mean shift > VERIFY_TOL)."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    store = StubStore()
    key = _l1_ctx()

    # Emit on -0.067
    _populate(acc, key, [{"health": -0.067}] * RuleNursery.MIN_OBS_L1)
    nursery.tick(acc, store, current_tick=0)
    assert len(nursery.candidates()) == 1

    # Verify with WAY different values (-0.20) → rejection
    for i in range(RuleNursery.VERIFY_N):
        _populate(acc, key, [{"health": -0.20}], start_tick=100 + i)
        nursery.tick(acc, store, current_tick=100 + i)

    assert nursery.stats()["promoted"] == 0
    assert nursery.stats()["rejected"] == 1
    assert len(store.learned_rules) == 0
    # Rejected candidate is removed → can be re-emitted later
    assert len(nursery.candidates()) == 0


def test_promoted_context_does_not_reemit():
    """Once promoted, the same context should be suppressed even if more
    surprise records arrive."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    store = StubStore()
    key = _l1_ctx()

    # Promote
    _populate(acc, key, [{"health": -0.067}] * RuleNursery.MIN_OBS_L1)
    nursery.tick(acc, store, current_tick=0)
    for i in range(RuleNursery.VERIFY_N):
        _populate(acc, key, [{"health": -0.067}], start_tick=100 + i)
        nursery.tick(acc, store, current_tick=100 + i)
    assert nursery.stats()["promoted"] == 1

    # Add many more surprises and tick — should NOT emit a second candidate
    _populate(acc, key, [{"health": -0.05}] * 50, start_tick=200)
    nursery.tick(acc, store, current_tick=300)
    assert nursery.stats()["emitted"] == 1  # only the original
    assert nursery.stats()["promoted"] == 1
    assert len(store.learned_rules) == 1  # no duplicate


def test_rejected_candidate_can_re_emit_after_more_data():
    """A rejected candidate's context is NOT permanently blocked.
    New data can lead to a successful emission later."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    key = _l1_ctx()

    # First batch: emit on -0.067
    _populate(acc, key, [{"health": -0.067}] * RuleNursery.MIN_OBS_L1)
    nursery.tick(acc, store=None, current_tick=0)
    assert nursery.stats()["emitted"] == 1
    # Verify with -0.20 → rejection
    for i in range(RuleNursery.VERIFY_N):
        _populate(acc, key, [{"health": -0.20}], start_tick=100 + i)
        nursery.tick(acc, store=None, current_tick=100 + i)
    assert nursery.stats()["rejected"] == 1
    assert len(nursery.candidates()) == 0

    # Now feed more consistent -0.20 records to dominate the bucket
    # (sliding window cap = 100, current bucket has 5 + 10 = 15 records).
    # Add 50 more -0.20 records — bucket will be dominated by -0.20.
    _populate(acc, key, [{"health": -0.20}] * 50, start_tick=200)
    nursery.tick(acc, store=None, current_tick=300)
    # Should re-emit because the bucket mean is now stably -0.20
    assert nursery.stats()["emitted"] == 2


def test_l1_uses_smaller_min_obs_than_l2():
    """L1 key needs only MIN_OBS_L1=5 to emit; L2 needs MIN_OBS_L2=10."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    l1_key = _l1_ctx()
    _populate(acc, l1_key, [{"health": -0.5}] * RuleNursery.MIN_OBS_L1)
    nursery.tick(acc, store=None, current_tick=0)
    assert len(nursery.candidates()) == 1
    cand = nursery.candidates()[0]
    assert cand.context == l1_key


def test_stats_in_flight_count():
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    k1 = _l1_ctx()
    k2 = ContextKey(visible=frozenset({"zombie"}), body_quartiles=(0, 0, 0, 0), action="do")

    _populate(acc, k1, [{"health": -0.067}] * RuleNursery.MIN_OBS_L1)
    _populate(acc, k2, [{"health": -0.5}] * RuleNursery.MIN_OBS_L1)
    nursery.tick(acc, store=None, current_tick=0)
    stats = nursery.stats()
    assert stats["emitted"] == 2
    assert stats["in_flight"] == 2
    assert stats["promoted"] == 0
    assert stats["rejected"] == 0


# ---------------------------------------------------------------------------
# L1 + L2 dual emission interaction
# ---------------------------------------------------------------------------


def test_l2_observation_emits_from_both_l1_and_l2_independently():
    """When L2 observations arrive, they feed both L2 and L1 buckets.
    Both buckets can independently emit candidates if their MIN_OBS is met
    and the data is consistent. The L2 candidate has the body_quartile
    constraint; the L1 candidate is body-agnostic. This is by design —
    Phase 7 in _apply_tick can fire both rules and they compose
    additively without conflict."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    l2_key = _l2_ctx_food_zero()
    l1_key = l2_key.coarsen()
    assert l2_key != l1_key

    # 10 records (= MIN_OBS_L2). L1 is also at 10, well above MIN_OBS_L1=5.
    _populate(acc, l2_key, [{"health": -0.067}] * RuleNursery.MIN_OBS_L2)
    nursery.tick(acc, store=None, current_tick=0)

    cands = nursery.candidates()
    contexts = {c.context for c in cands}
    # Both L2 and L1 contexts emit
    assert l2_key in contexts
    assert l1_key in contexts
    assert len(cands) == 2
    # Both have the same effect (since the data is identical)
    for c in cands:
        assert "health" in c.mean_effect
        assert abs(c.mean_effect["health"] - (-0.067)) < 0.005


def test_l1_inconsistent_l2_consistent_only_l2_emits():
    """When the L1 bucket aggregates samples from MULTIPLE L2 buckets
    with conflicting effects, L1 mean has high MAD → no L1 emission.
    L2 with consistent data still emits. This is the core mechanism
    for the conjunctive case: sleep without starvation gives +0.04
    health (L2: food_q=high), sleep with starvation gives -0.067
    (L2: food_q=0). L1 sees both → noisy. L2 (food_q=0) sees only
    starvation → emits."""
    nursery = RuleNursery()
    acc = SurpriseAccumulator()
    visible = frozenset({"cow"})
    action = "sleep"
    l2_starving = ContextKey(visible=visible, body_quartiles=(2, 0, 2, 2), action=action)
    l2_satiated = ContextKey(visible=visible, body_quartiles=(2, 3, 2, 2), action=action)
    l1_key = l2_starving.coarsen()
    assert l1_key == l2_satiated.coarsen()

    # 10 starving sleep observations: -0.067
    _populate(acc, l2_starving, [{"health": -0.067}] * 10)
    # 10 satiated sleep observations: +0.04
    _populate(acc, l2_satiated, [{"health": +0.04}] * 10, start_tick=100)
    # L1 bucket now has 20 records mixed evenly between -0.067 and +0.04
    # → mean ≈ -0.014, MAD ≈ 0.054 → 2*MAD = 0.108 > |mean| → not significant

    nursery.tick(acc, store=None, current_tick=200)
    cands = nursery.candidates()
    contexts = {c.context for c in cands}
    # L2 starving emits (-0.067 cluster)
    assert l2_starving in contexts
    # L2 satiated emits (+0.04 cluster)
    assert l2_satiated in contexts
    # L1 does NOT emit (mixed → high MAD blocks significance test)
    assert l1_key not in contexts
