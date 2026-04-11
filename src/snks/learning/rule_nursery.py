"""Stage 79 — Rule nursery: emit, verify, promote symbolic rules.

Consumes `SurpriseAccumulator` buckets, emits `CandidateRule`s when a
bucket shows a consistent non-zero error, verifies them over a held-out
window of new observations, and promotes survivors to
`ConceptStore.learned_rules` as `LearnedRule` instances.

Pipeline per `tick()`:

  for each bucket in accumulator:
      if not already tracking a candidate for this context:
          if bucket has >= MIN_OBS records:
              try to emit (consistency + significance gates)
  for each in-flight candidate:
      append any new bucket records to the verify window
      if window has >= VERIFY_N records:
          if mean still matches the candidate's emitted effect:
              promote → store.add_learned_rule(...)
          else:
              reject and forget

Constants are `class` attributes on `RuleNursery` so subclasses or
test fixtures can override them without monkey-patching.

Design: docs/superpowers/specs/2026-04-11-stage79-rule-nursery-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from snks.learning.surprise_accumulator import (
    BODY_ORDER,
    ContextKey,
    SurpriseAccumulator,
    SurpriseRecord,
)

if TYPE_CHECKING:
    from snks.agent.concept_store import ConceptStore
    from snks.agent.learned_rule import LearnedRule


@dataclass
class CandidateRule:
    """A rule emitted from a saturated surprise bucket, awaiting verification.

    `mean_effect` only contains body vars whose mean error is
    "significant" (above the noise floor and the MAD-based consistency
    threshold) — vars whose effect is indistinguishable from zero are
    excluded so the candidate represents only the load-bearing
    correction.

    `verify_records` accumulates new observations of the same context
    after emission. Promotion fires when `len(verify_records) >=
    VERIFY_N` AND the mean over verify_records still matches
    `mean_effect` within `VERIFY_TOL` for every var.
    """

    context: ContextKey
    mean_effect: dict[str, float]
    n_obs: int
    mad: dict[str, float]
    status: str  # "verifying" | "promoted" | "rejected"
    verify_records: list[SurpriseRecord] = field(default_factory=list)
    emitted_at_tick: int = 0


class RuleNursery:
    """Emit, verify, and promote candidate rules from a surprise accumulator.

    Use:
        nursery = RuleNursery()
        # ... after each env step:
        accumulator.observe(...)
        nursery.tick(accumulator, store, current_tick=step)

    The nursery holds at most one in-flight candidate per context. If
    a candidate is rejected, it is removed and the same context can
    re-emit later (after more observations accumulate). If a candidate
    is promoted, it stays tracked so the same context will not be
    re-emitted (the promoted rule should now suppress the surprise on
    that bucket).
    """

    # Minimum bucket size before we attempt to emit a candidate.
    # L1 (coarse, body-independent) needs fewer because there are far
    # fewer possible L1 contexts (visible × action ≈ 200) so each one
    # accumulates faster. L2 needs more because there are 256× as many
    # possible buckets and we want stronger evidence before splitting.
    MIN_OBS_L1: int = 5
    MIN_OBS_L2: int = 10

    # Verification window size — how many fresh observations to collect
    # after emission before promoting/rejecting.
    VERIFY_N: int = 10

    # Tolerance: the mean over the verification window must be within
    # VERIFY_TOL of the candidate's emitted mean for every significant
    # var. 0.02 is roughly the per-tick body decay rate, so it permits
    # ordinary noise without permitting drift.
    VERIFY_TOL: float = 0.02

    # MAD-based consistency multiplier. mean must be > MAD_K * MAD to
    # count as significant — i.e. the typical noise around the mean
    # must be smaller than the mean itself.
    MAD_K: float = 2.0

    # Floor on absolute mean magnitude. Anything smaller is treated
    # as noise regardless of the MAD test (avoids false positives on
    # tightly-clustered near-zero distributions).
    SIGNIFICANCE_FLOOR: float = 0.01

    def __init__(self) -> None:
        self._candidates: dict[ContextKey, CandidateRule] = {}
        # Promoted contexts are remembered to suppress re-emission even
        # if the bucket later accumulates new evidence (the existing
        # learned rule should be picking up the slack).
        self._promoted_contexts: set[ContextKey] = set()
        self._stats = {"emitted": 0, "promoted": 0, "rejected": 0}

    # ---- Public API --------------------------------------------------------

    def tick(
        self,
        accumulator: SurpriseAccumulator,
        store: "ConceptStore | None",
        current_tick: int,
    ) -> None:
        """Run one nursery cycle: emit, verify, promote/reject.

        `store` is optional so this method can be unit-tested without
        a full ConceptStore — pass None and inspect `_candidates`
        directly. When `store` is provided, promoted rules are written
        via `store.add_learned_rule(...)`.
        """
        # 1. Try to emit new candidates from saturated buckets we're
        #    not already tracking.
        for key, records in accumulator.iter_buckets():
            if key in self._candidates:
                continue
            if key in self._promoted_contexts:
                continue
            min_obs = self.MIN_OBS_L1 if key.is_l1() else self.MIN_OBS_L2
            if len(records) < min_obs:
                continue
            candidate = self._try_emit(records, key, current_tick)
            if candidate is not None:
                self._candidates[key] = candidate
                self._stats["emitted"] += 1

        # 2. Verify in-flight candidates against any new bucket records
        #    that arrived since emission.
        to_remove: list[ContextKey] = []
        for key, candidate in list(self._candidates.items()):
            if candidate.status != "verifying":
                continue
            current_bucket = accumulator.bucket_records(key)
            already_seen = candidate.n_obs + len(candidate.verify_records)
            new_records = current_bucket[already_seen:]
            for record in new_records:
                candidate.verify_records.append(record)
                if len(candidate.verify_records) >= self.VERIFY_N:
                    self._resolve(candidate, store)
                    if candidate.status == "rejected":
                        to_remove.append(key)
                    break

        # 3. Drop rejected candidates so the same context can be
        #    re-considered with later evidence.
        for key in to_remove:
            del self._candidates[key]

    def stats(self) -> dict[str, Any]:
        """Episode-end diagnostic counters."""
        in_flight = sum(
            1 for c in self._candidates.values() if c.status == "verifying"
        )
        return {
            "emitted": self._stats["emitted"],
            "promoted": self._stats["promoted"],
            "rejected": self._stats["rejected"],
            "in_flight": in_flight,
        }

    def candidates(self) -> list[CandidateRule]:
        """Snapshot of all in-flight candidate rules (for inspection / tests)."""
        return list(self._candidates.values())

    # ---- Emission and verification helpers ---------------------------------

    def _try_emit(
        self,
        records: list[SurpriseRecord],
        key: ContextKey,
        current_tick: int,
    ) -> CandidateRule | None:
        """Compute mean delta + MAD per var. Emit only if at least one
        var passes the (significance + consistency) gates.

        Returns None if no var qualifies.
        """
        deltas_per_var = {
            var: np.array([float(r.delta.get(var, 0.0)) for r in records])
            for var in BODY_ORDER
        }
        mean = {var: float(np.mean(deltas_per_var[var])) for var in BODY_ORDER}
        # MAD = median absolute deviation from mean. Robust to outliers.
        mad = {
            var: float(np.median(np.abs(deltas_per_var[var] - mean[var])))
            for var in BODY_ORDER
        }

        significant: dict[str, bool] = {}
        for var in BODY_ORDER:
            threshold = max(self.MAD_K * mad[var], self.SIGNIFICANCE_FLOOR)
            significant[var] = abs(mean[var]) > threshold

        if not any(significant.values()):
            return None

        return CandidateRule(
            context=key,
            mean_effect={v: mean[v] for v in BODY_ORDER if significant[v]},
            n_obs=len(records),
            mad=dict(mad),
            status="verifying",
            verify_records=[],
            emitted_at_tick=current_tick,
        )

    def _resolve(
        self,
        candidate: CandidateRule,
        store: "ConceptStore | None",
    ) -> None:
        """Decide promote vs reject after the verify window fills."""
        ok = True
        for var, expected in candidate.mean_effect.items():
            observed_mean = float(
                np.mean([r.delta.get(var, 0.0) for r in candidate.verify_records])
            )
            if abs(observed_mean - expected) > self.VERIFY_TOL:
                ok = False
                break

        if ok:
            candidate.status = "promoted"
            self._stats["promoted"] += 1
            self._promoted_contexts.add(candidate.context)
            if store is not None:
                # Lazy import to avoid circular dependency between
                # snks.learning and snks.agent.
                from snks.agent.learned_rule import LearnedRule

                store.add_learned_rule(
                    LearnedRule(
                        precondition=candidate.context,
                        effect=dict(candidate.mean_effect),
                        confidence=0.5,
                        n_observations=candidate.n_obs + len(candidate.verify_records),
                        source="runtime_nursery",
                    )
                )
        else:
            candidate.status = "rejected"
            self._stats["rejected"] += 1
