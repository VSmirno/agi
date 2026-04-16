"""Stage 86: Post-Mortem Learning — DamageEvent, attribution, stimulus adaptation.

After each episode, attributed damage causes are used to update StimuliLayer
parameters so future episodes plan more defensively.

Design: docs/superpowers/specs/2026-04-15-stage86-post-mortem-learning-design.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from snks.agent.stimuli import CuriosityStimulus, HomeostasisStimulus, StimuliLayer, SurvivalAversion
from snks.agent.death_hypothesis import DeathHypothesis


@dataclass
class DamageEvent:
    """One health-decrease event observed during an episode."""
    step: int
    health_delta: float                      # always < 0
    vitals: dict[str, float]                 # food, drink, energy at time of event
    nearby_cids: list[tuple[str, int]]       # [(concept_id, manhattan_dist), ...]


class PostMortemAnalyzer:
    """Attribute damage events to causal sources using temporal decay."""

    def attribute(
        self,
        damage_log: list[DamageEvent],
        death_step: int,
        decay: float = 0.02,
    ) -> dict[str, float]:
        """Return attribution dict: source → fraction of total weighted damage.

        Temporal decay: events closer to death receive higher weight.
        Multi-source: one event may have multiple sources; weight split equally.
        Returns {} if damage_log is empty (agent survived to step limit).
        """
        if not damage_log:
            return {}

        # Step 1: compute raw weight per event
        weights = [
            math.exp(-decay * (death_step - ev.step))
            for ev in damage_log
        ]
        total_w = sum(weights)
        if total_w == 0:
            return {}

        # Step 2: attribute weighted mass to sources
        attribution: dict[str, float] = {}

        for ev, w in zip(damage_log, weights):
            sources = self._detect_sources(ev)
            share = w / (total_w * len(sources))
            for src in sources:
                attribution[src] = attribution.get(src, 0.0) + share

        # Values already normalised (sum of shares == total_w / total_w == 1.0)
        return attribution

    # Crafter damage mechanics (from objects.py):
    # - zombie: melee, attacks at dist<=1, moves toward player
    # - skeleton: ranged, shoots Arrow at dist<=5; arrow travels ~1 tile/step,
    #   so skeleton may be at dist 7-10 by the time the arrow hits
    # - cow: does NOT deal damage (moves randomly only)
    _MELEE_RANGE = 6    # zombie — with entity_tracker timing lag
    _RANGED_RANGE = 10  # skeleton — accounts for arrow travel time
    _DAMAGE_DEALERS = {"zombie": _MELEE_RANGE, "skeleton": _RANGED_RANGE}

    @classmethod
    def _detect_sources(cls, ev: DamageEvent) -> list[str]:
        """Return list of causal source labels for this damage event."""
        sources: list[str] = []

        if ev.vitals.get("food", 9.0) < 0.5:
            sources.append("starvation")
        if ev.vitals.get("drink", 9.0) < 0.5:
            sources.append("dehydration")
        for cid, dist in ev.nearby_cids:
            max_dist = cls._DAMAGE_DEALERS.get(cid)
            if max_dist is not None and dist <= max_dist:
                sources.append(cid)

        return sources if sources else ["unknown"]


_THRESHOLD_MIN = 1.0
_THRESHOLD_MAX = 8.0
_WEIGHT_MIN = 0.5
_WEIGHT_MAX = 5.0


@dataclass
class PostMortemLearner:
    """Adapt StimuliLayer parameters between episodes based on death attribution.

    Parameters update in-place each episode; reset on new run (no persistence).
    """
    food_threshold: float = 3.0
    drink_threshold: float = 3.0
    health_weight: float = 1.0
    lr: float = 0.1

    def update(self, attribution: dict[str, float]) -> None:
        """Update parameters based on one episode's attribution."""
        if not attribution:
            return

        if "starvation" in attribution:
            self.food_threshold = self._clamp(
                self.food_threshold + self.lr * attribution["starvation"],
                _THRESHOLD_MIN, _THRESHOLD_MAX,
            )
        if "dehydration" in attribution:
            self.drink_threshold = self._clamp(
                self.drink_threshold + self.lr * attribution["dehydration"],
                _THRESHOLD_MIN, _THRESHOLD_MAX,
            )
        entity_share = sum(
            v for k, v in attribution.items() if k in ("zombie", "skeleton")
        )
        if entity_share > 0:
            self.health_weight = self._clamp(
                self.health_weight + self.lr * entity_share,
                _WEIGHT_MIN, _WEIGHT_MAX,
            )

    @classmethod
    def from_promoted(
        cls,
        hypotheses: list[DeathHypothesis],
        lr: float = 0.1,
    ) -> PostMortemLearner:
        """Create PostMortemLearner with thresholds pre-initialized from promoted hypotheses.

        Formula: threshold = max(default, 3.0 + support_rate * 2.0), capped at _THRESHOLD_MAX.
        This is a conservative overestimate from the default base — update() will pull it back
        within the new generation if it overcorrects.
        """
        learner = cls(lr=lr)
        for h in hypotheses:
            bump = min(h.support_rate * 2.0, 2.0)
            if h.vital == "drink":
                learner.drink_threshold = cls._clamp(
                    max(learner.drink_threshold, 3.0 + bump),
                    _THRESHOLD_MIN, _THRESHOLD_MAX,
                )
            elif h.vital == "food":
                learner.food_threshold = cls._clamp(
                    max(learner.food_threshold, 3.0 + bump),
                    _THRESHOLD_MIN, _THRESHOLD_MAX,
                )
            if h.cause in ("zombie", "skeleton"):
                learner.health_weight = cls._clamp(
                    max(learner.health_weight, 1.0 + h.support_rate),
                    _WEIGHT_MIN, _WEIGHT_MAX,
                )
        return learner

    def build_stimuli(
        self,
        vital_vars: list[str],
        hypothesis: "DeathHypothesis | None" = None,
    ) -> StimuliLayer:
        """Create a new StimuliLayer with current parameters.

        If hypothesis is provided (Stage 87+), adds CuriosityStimulus weighted
        by death_relevance from the active DeathHypothesis.
        """
        stimuli = [
            SurvivalAversion(),
            HomeostasisStimulus(
                vital_vars=vital_vars,
                weight=self.health_weight,
                thresholds={
                    "food": self.food_threshold,
                    "drink": self.drink_threshold,
                },
            ),
        ]
        if hypothesis is not None:
            stimuli.append(CuriosityStimulus(hypothesis=hypothesis))
        return StimuliLayer(stimuli)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))


def dominant_cause(attribution: dict[str, float]) -> str:
    """Return the source with highest attribution weight, or 'alive' if empty."""
    if not attribution:
        return "alive"
    return max(attribution, key=attribution.__getitem__)
