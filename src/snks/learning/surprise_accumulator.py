"""Stage 79 — Surprise accumulator for runtime rule induction.

Per-context buckets of (predicted, actual, delta) records. Two-level
keying:

  - L2 (full): (visible_concepts, body_quartiles, action) — captures
    body-state-dependent rules like the conjunctive sleep+starvation
    case from Stage 78a.
  - L1 (coarse): visible+action only (body_quartiles zeroed) — captures
    body-independent rules like skeleton-in-view → health drop.

Every observation feeds BOTH the L2 bucket and the L1 bucket. The
RuleNursery (separate module) consumes accumulator buckets to emit
candidate rules.

This module has no dependency on torch — it is pure Python + numpy +
stdlib so it can be unit-tested without GPU. The accumulator is
deliberately simple: a dict of deques with sliding-window eviction.

Design: docs/superpowers/specs/2026-04-11-stage79-rule-nursery-design.md
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterator


# Body-variable order shared with ConceptStore. Imported lazily inside
# the methods that need it to keep this module dependency-light for
# tests; the canonical definition lives in
# `snks.agent.concept_store.RESIDUAL_BODY_ORDER`.
BODY_ORDER: tuple[str, ...] = ("health", "food", "drink", "energy")


# Quartile boundaries for body values. Body vars in Crafter live in
# [0, 9] (clamped). Boundaries chosen so food=0 (the canonical
# Stage 78a conjunctive case) cleanly lands in quartile 0 and
# food=9 (full) lands in quartile 3.
QUARTILE_BOUNDARIES: tuple[float, float, float] = (2.5, 5.0, 7.5)

# Sliding-window cap per bucket. Older observations age out as
# distribution shifts (e.g. agent moves into a new biome / picks up
# tools that change action availability). 100 observations is enough
# for MIN_OBS_L2=10 with headroom for verification, while bounding
# memory at ~100 × n_active_buckets records.
MAX_BUCKET_SIZE: int = 100


def quartile_for(value: float) -> int:
    """Return quartile index in {0, 1, 2, 3} for a body variable value.

    Boundaries are 2.5 / 5.0 / 7.5 — fixed (not data-derived) so the
    bucketing is deterministic across episodes and easy to test. Any
    value below 0 maps to 0; any value above 9 maps to 3.
    """
    if value < QUARTILE_BOUNDARIES[0]:
        return 0
    if value < QUARTILE_BOUNDARIES[1]:
        return 1
    if value < QUARTILE_BOUNDARIES[2]:
        return 2
    return 3


@dataclass(frozen=True)
class ContextKey:
    """Hashable bucket fingerprint.

    `visible` is the set of concept IDs the agent perceives at the
    moment of acting. `body_quartiles` is a 4-tuple of quartile indices
    for (health, food, drink, energy) — quartile boundaries are fixed.
    `action` is the primitive env action string ('do', 'sleep',
    'move_left', 'place_stone', etc).

    Equality and hashing follow the dataclass-frozen default — two
    keys are equal iff all three fields are equal.
    """

    visible: frozenset[str]
    body_quartiles: tuple[int, int, int, int]
    action: str

    @classmethod
    def from_state(
        cls,
        visible: set[str] | frozenset[str],
        body: dict[str, float],
        action: str,
        body_order: tuple[str, ...] = BODY_ORDER,
    ) -> "ContextKey":
        """Build a ContextKey from a runtime SimState snapshot."""
        quartiles = tuple(
            quartile_for(float(body.get(var, 0.0))) for var in body_order
        )
        # Pad/truncate to exactly 4 to keep the type stable.
        if len(quartiles) != 4:
            quartiles = (quartiles + (0, 0, 0, 0))[:4]
        return cls(
            visible=frozenset(visible),
            body_quartiles=quartiles,  # type: ignore[arg-type]
            action=action,
        )

    def coarsen(self) -> "ContextKey":
        """Return the L1 (body-independent) version of this key.

        L1 collapses the body_quartiles to (0, 0, 0, 0) so all body
        states with the same visible+action map to the same coarse
        bucket. The nursery uses L1 buckets to detect rules that do
        not depend on body state (e.g. zombie nearby → take damage
        regardless of current health), and L2 buckets to detect
        body-state-conditional rules.
        """
        return ContextKey(
            visible=self.visible,
            body_quartiles=(0, 0, 0, 0),
            action=self.action,
        )

    def is_l1(self) -> bool:
        """True iff this is an L1 (coarse) key — body_quartiles all zero."""
        return self.body_quartiles == (0, 0, 0, 0)


@dataclass
class SurpriseRecord:
    """One observed (predicted, actual, delta) triple at a tick.

    `predicted` and `actual` are body-delta dicts (NOT absolute body
    values) — i.e. the change over one env step. `delta` is
    `actual - predicted`, the residual error. `tick_id` is the
    monotone counter from the MPC loop, used for ordering and
    diagnostics.
    """

    context: ContextKey
    predicted: dict[str, float]
    actual: dict[str, float]
    delta: dict[str, float]
    tick_id: int


class SurpriseAccumulator:
    """Per-context bucket store for surprise records.

    Use:
        acc = SurpriseAccumulator()
        acc.observe(context_key, predicted_delta, actual_delta, tick_id=42)
        ...
        for key, records in acc.iter_buckets():
            ...

    Each `observe` call writes the record to the L2 bucket (full
    ContextKey) and ALSO to the L1 bucket (the coarsened version).
    This costs 2× memory but lets the nursery detect rules at either
    granularity without re-processing the data.

    The bucket store uses a deque with `maxlen=max_bucket_size` so
    sliding-window eviction is automatic — once a bucket fills, the
    oldest observation is dropped on each new append.
    """

    def __init__(self, max_bucket_size: int = MAX_BUCKET_SIZE) -> None:
        self.max_bucket_size = max_bucket_size
        self._buckets: dict[ContextKey, deque[SurpriseRecord]] = defaultdict(
            lambda: deque(maxlen=max_bucket_size)
        )
        self._n_observations: int = 0

    def observe(
        self,
        context: ContextKey,
        predicted: dict[str, float],
        actual: dict[str, float],
        tick_id: int,
        body_order: tuple[str, ...] = BODY_ORDER,
    ) -> None:
        """Add an observation to its L2 and L1 buckets.

        `predicted` and `actual` are deltas (changes per env step),
        not absolute body values. The accumulator stores
        `delta = actual - predicted` for each body var in
        `body_order`; vars not present in either dict default to 0.0.
        """
        delta = {
            var: float(actual.get(var, 0.0)) - float(predicted.get(var, 0.0))
            for var in body_order
        }
        record = SurpriseRecord(
            context=context,
            predicted={var: float(predicted.get(var, 0.0)) for var in body_order},
            actual={var: float(actual.get(var, 0.0)) for var in body_order},
            delta=delta,
            tick_id=tick_id,
        )
        self._buckets[context].append(record)
        # Also feed L1 — only if this is an L2 key, otherwise it's
        # already L1 and we'd double-count.
        if not context.is_l1():
            l1_key = context.coarsen()
            self._buckets[l1_key].append(record)
        self._n_observations += 1

    def bucket_size(self, key: ContextKey) -> int:
        """How many records are currently in this bucket."""
        return len(self._buckets.get(key, ()))

    def bucket_records(self, key: ContextKey) -> list[SurpriseRecord]:
        """Snapshot the current records of a bucket as a plain list."""
        return list(self._buckets.get(key, ()))

    def iter_buckets(self) -> Iterator[tuple[ContextKey, list[SurpriseRecord]]]:
        """Iterate over all non-empty buckets and their current records.

        Yields snapshot lists, not the underlying deques, so the
        caller can mutate the accumulator while iterating without
        invalidation.
        """
        # Materialise to avoid mutation-during-iteration if observe() is
        # called while the nursery is iterating.
        items = list(self._buckets.items())
        for key, dq in items:
            yield key, list(dq)

    def stats(self) -> dict[str, Any]:
        """Diagnostic counters for episode-end reporting."""
        sizes = [len(b) for b in self._buckets.values()]
        return {
            "n_buckets": len(self._buckets),
            "n_l2_buckets": sum(1 for k in self._buckets if not k.is_l1()),
            "n_l1_buckets": sum(1 for k in self._buckets if k.is_l1()),
            "total_records": sum(sizes),
            "total_observations": self._n_observations,
            "max_bucket_size": max(sizes) if sizes else 0,
            "median_bucket_size": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
        }
