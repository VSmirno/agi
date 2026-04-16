"""Stage 87: DeathHypothesis + HypothesisTracker — death-relevant curiosity.

A DeathHypothesis correlates a death cause with a vital state at death time:
    "I die from [cause] more often when [vital] < [threshold]"

HypothesisTracker records (cause, vitals_at_death) per episode and derives
verifiable hypotheses. The active hypothesis is passed into CuriosityStimulus
to weight prediction surprise by death relevance.

Design: docs/superpowers/specs/2026-04-15-stage87-death-curiosity-design.md
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snks.agent.vector_sim import VectorTrajectory

# Vital thresholds to test for each vital variable.
_HYPOTHESIS_THRESHOLDS: dict[str, float] = {
    "food": 3.0,
    "drink": 3.0,
    "health": 4.0,
    "energy": 2.0,
}


@dataclass
class DeathHypothesis:
    """Correlational hypothesis: 'I die from [cause] more when [vital] < [threshold]'.

    Verifiable when observed enough times (n_observed >= 3, n_supporting >= 2).
    """

    cause: str
    vital: str
    threshold: float
    n_supporting: int = 0   # episodes: dominant_cause==cause AND vital<threshold at death
    n_observed: int = 0     # total episodes where dominant_cause==cause

    @property
    def is_verifiable(self) -> bool:
        return self.n_observed >= 3 and self.n_supporting >= 2

    @property
    def support_rate(self) -> float:
        if self.n_observed == 0:
            return 0.0
        return self.n_supporting / self.n_observed

    def death_relevance(self, trajectory: "VectorTrajectory") -> float:
        """Bonus multiplier for trajectories that expose the hypothesis condition.

        Returns 1.0 (neutral) to 2.0 (max relevance).
        High relevance when trajectory visits states where vital ≈ threshold.
        """
        if not trajectory.states:
            return 1.0
        min_vital = min(s.body.get(self.vital, 9.0) for s in trajectory.states)
        # proximity ∈ [0, 1]: 1.0 when min_vital == threshold, 0 when |delta| >= 3
        proximity = max(0.0, 1.0 - abs(min_vital - self.threshold) / 3.0)
        return 1.0 + proximity

    def __str__(self) -> str:
        return (
            f"H: die from {self.cause} when {self.vital} < {self.threshold:.1f} "
            f"[{self.n_supporting}/{self.n_observed}, rate={self.support_rate:.2f}]"
        )


class HypothesisTracker:
    """Generates verifiable death hypotheses from per-episode death history.

    Usage:
        tracker = HypothesisTracker()

        # After each episode:
        tracker.record(attribution, vitals_at_death)
        hypothesis = tracker.active_hypothesis()
    """

    def __init__(self, initial: list[DeathHypothesis] | None = None) -> None:
        self._records: list[dict] = []
        self._hypotheses: list[DeathHypothesis] = []
        self._promoted: list[DeathHypothesis] = list(initial or [])
        if self._promoted:
            self._rebuild_hypotheses()  # populate _hypotheses from _promoted immediately

    def record(
        self,
        attribution: dict[str, float],
        vitals_at_death: dict[str, float],
    ) -> None:
        """Record one death episode's attribution + vitals at time of death.

        attribution: from PostMortemAnalyzer.attribute() — {} if agent survived.
        vitals_at_death: DamageEvent.vitals from last damage event, or {} if survived.
        """
        if not attribution:
            return  # agent survived — no death data

        cause = max(attribution, key=attribution.__getitem__)
        self._records.append({"cause": cause, "vitals": vitals_at_death})
        self._rebuild_hypotheses()

    def _rebuild_hypotheses(self) -> None:
        """Re-derive all hypotheses from recorded death events."""
        # counts[cause][vital] = [n_supporting, n_observed]
        counts: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))

        for rec in self._records:
            cause = rec["cause"]
            vitals = rec["vitals"]
            for vital, thr in _HYPOTHESIS_THRESHOLDS.items():
                val = vitals.get(vital, 9.0)
                counts[cause][vital][1] += 1
                if val < thr:
                    counts[cause][vital][0] += 1

        # Index promoted priors for fast lookup.
        promoted_by_key = {(ph.cause, ph.vital): ph for ph in self._promoted}

        hypotheses: list[DeathHypothesis] = []
        for cause, vital_counts in counts.items():
            for vital, (n_sup, n_obs) in vital_counts.items():
                # Merge promoted prior counts into live observations so that the
                # hypothesis stays verifiable from the very first death in a new
                # generation (fixes premature-replacement bug).
                prior = promoted_by_key.get((cause, vital))
                if prior is not None:
                    n_sup += prior.n_supporting
                    n_obs += prior.n_observed
                hypotheses.append(
                    DeathHypothesis(
                        cause=cause,
                        vital=vital,
                        threshold=_HYPOTHESIS_THRESHOLDS[vital],
                        n_supporting=n_sup,
                        n_observed=n_obs,
                    )
                )
        # Carry forward promoted entries for (cause, vital) pairs not yet seen
        # in this generation's records.
        live_keys = {(h.cause, h.vital) for h in hypotheses}
        for ph in self._promoted:
            if (ph.cause, ph.vital) not in live_keys:
                hypotheses.append(ph)

        self._hypotheses = hypotheses

    def active_hypothesis(self) -> DeathHypothesis | None:
        """Return the highest-support verifiable hypothesis, or None.

        Priority: highest support_rate, then highest n_supporting.
        """
        verifiable = [h for h in self._hypotheses if h.is_verifiable]
        if not verifiable:
            return None
        return max(verifiable, key=lambda h: (h.support_rate, h.n_supporting))

    def n_verifiable(self) -> int:
        """Count of distinct verifiable hypotheses formed so far."""
        return sum(1 for h in self._hypotheses if h.is_verifiable)

    def all_hypotheses(self) -> list[DeathHypothesis]:
        """Return all tracked hypotheses (for diagnostics)."""
        return list(self._hypotheses)
