"""CausalWorldModel: learns causal relationships (context, action) → effect."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from snks.daf.types import CausalAgentConfig


@dataclass(frozen=True)
class CausalLink:
    """Directed causal relationship: action in context → effect."""
    action: int                 # which action
    context_sks: frozenset[int]  # which SKS were active (what we see)
    effect_sks: frozenset[int]  # which SKS activated as result
    strength: float             # confidence (0..1)
    count: int                  # how many times observed


def _context_hash(sks: frozenset[int]) -> int:
    """Hash a set of SKS IDs to a 64-bit key for fast lookup."""
    return hash(sks)


# IDs >= this offset are perceptual hashes (stable across episodes).
# See agent.py: _perceptual_hash() uses offset=10_000.
_PERCEPTUAL_HASH_OFFSET = 10_000


def _split_context(sks: set[int], n_bins: int) -> frozenset[int]:
    """Build a context key from SKS IDs, separating stable from unstable IDs.

    Stable IDs (>= _PERCEPTUAL_HASH_OFFSET): perceptual hash pseudo-SKS.
    These are deterministic functions of the image → used as-is.

    Unstable IDs (< _PERCEPTUAL_HASH_OFFSET): DAF cluster IDs that vary
    across runs and episodes → coarsen to n_bins buckets.

    Rationale: different visual scenes (key visible, door open, empty room)
    produce different perceptual hash IDs, giving the causal model enough
    discrimination without relying on noisy DAF cluster IDs.
    """
    if not sks:
        return frozenset()
    stable   = frozenset(s for s in sks if s >= _PERCEPTUAL_HASH_OFFSET)
    unstable = frozenset(s % n_bins for s in sks if s < _PERCEPTUAL_HASH_OFFSET)
    return stable | unstable


def _coarsen_sks(sks: set[int], n_bins: int) -> frozenset[int]:
    """Coarsen all SKS IDs into n_bins buckets.

    Deprecated: use _split_context() instead, which preserves stable
    perceptual-hash IDs and only coarsens noisy DAF cluster IDs.
    Kept for backward compatibility with existing tests.
    """
    if not sks:
        return frozenset()
    return frozenset(s % n_bins for s in sks)


class CausalWorldModel:
    """Learns causal relationships: (context, action) → effect.

    Key difference from PredictionEngine (Stage 3):
    - PredictionEngine: SKS_A → SKS_B (passive sequences)
    - CausalWorldModel: (context + action) → effect (interventional causality)

    Causal links are strengthened ONLY when:
    1. Agent itself performed the action (intervention)
    2. Effect observed AFTER the action
    3. Effect NOT observed without the action (counterfactual control via decay)
    """

    def __init__(self, config: CausalAgentConfig):
        self.config = config
        self._n_bins = config.causal_context_bins
        # (context_hash, action) → {effect_hash: TransitionRecord}
        self._transitions: dict[tuple[int, int], dict[int, _TransitionRecord]] = defaultdict(dict)
        # Track baseline rates: how often each SKS set appears without any specific action
        self._baseline_counts: dict[int, int] = defaultdict(int)
        self._total_observations: int = 0

    def observe_transition(
        self,
        pre_sks: set[int],
        action: int,
        post_sks: set[int],
    ) -> None:
        """Record observation: (context, action) → effect.

        Only the *change* (new SKS not in pre) is considered the effect,
        implementing the counterfactual criterion.
        """
        self._total_observations += 1

        ctx = _split_context(pre_sks, self._n_bins)
        # Effect = symmetric difference: captures both new and disappeared SKS
        # This is more robust than just post-pre, since minor visual changes
        # may shift existing clusters rather than create entirely new ones
        effect = frozenset(post_sks.symmetric_difference(pre_sks))
        ctx_hash = _context_hash(ctx)
        eff_hash = _context_hash(effect)

        key = (ctx_hash, action)

        if eff_hash not in self._transitions[key]:
            self._transitions[key][eff_hash] = _TransitionRecord(
                context_sks=ctx,
                effect_sks=effect,
                count=0,
                total_in_context=0,
            )

        record = self._transitions[key][eff_hash]
        record.count += 1
        record.total_in_context += 1

        # Update baseline: track appearance of effects without specific action attribution
        baseline_hash = _context_hash(frozenset(post_sks))
        self._baseline_counts[baseline_hash] += 1

        # Decay old records
        for k, records in self._transitions.items():
            for h, r in records.items():
                if k != key or h != eff_hash:
                    r.total_in_context = int(r.total_in_context * self.config.causal_decay)

    def predict_effect(
        self,
        context_sks: set[int],
        action: int,
    ) -> tuple[set[int], float]:
        """Predict effect of action in context.

        Returns:
            (predicted_sks, confidence)
        """
        ctx = _split_context(context_sks, self._n_bins)
        ctx_hash = _context_hash(ctx)
        key = (ctx_hash, action)

        records = self._transitions.get(key, {})
        if not records:
            return set(), 0.0

        # Find strongest effect
        best_record = max(records.values(), key=lambda r: r.count)

        if best_record.count < self.config.causal_min_observations:
            return set(best_record.effect_sks), 0.0

        # Confidence: count / total observations in this context
        total = sum(r.total_in_context for r in records.values())
        confidence = best_record.count / max(total, 1)

        return set(best_record.effect_sks), confidence

    def get_causal_links(self, min_confidence: float = 0.3) -> list[CausalLink]:
        """Extract all causal links above threshold."""
        links = []
        for (ctx_hash, action), records in self._transitions.items():
            total = sum(r.total_in_context for r in records.values())
            for eff_hash, record in records.items():
                if record.count < self.config.causal_min_observations:
                    continue
                confidence = record.count / max(total, 1)
                if confidence >= min_confidence:
                    links.append(CausalLink(
                        action=action,
                        context_sks=record.context_sks,
                        effect_sks=record.effect_sks,
                        strength=confidence,
                        count=record.count,
                    ))
        return links

    def get_all_effects_for_action(
        self, context_sks: set[int], action: int
    ) -> list[tuple[set[int], float]]:
        """Get all possible effects with their confidences for an action in context."""
        ctx = _split_context(context_sks, self._n_bins)
        ctx_hash = _context_hash(ctx)
        key = (ctx_hash, action)
        records = self._transitions.get(key, {})
        if not records:
            return []

        total = sum(r.total_in_context for r in records.values())
        results = []
        for record in records.values():
            confidence = record.count / max(total, 1)
            results.append((set(record.effect_sks), confidence))
        return results

    def best_action(
        self,
        context_sks: set[int],
        n_actions: int = 7,
    ) -> tuple[int | None, float]:
        """Return (best_action, confidence) for context_sks.

        Iterates over all actions, returns the one with highest predict_effect
        confidence.  Returns (None, 0.0) if no action has any data.
        """
        best_action_id: int | None = None
        best_conf: float = 0.0
        for a in range(n_actions):
            _, conf = self.predict_effect(context_sks, a)
            if conf > best_conf:
                best_conf = conf
                best_action_id = a
        return best_action_id, best_conf

    @property
    def n_links(self) -> int:
        """Total number of transition records."""
        return sum(len(records) for records in self._transitions.values())


@dataclass
class _TransitionRecord:
    """Internal record of a (context, action) → effect transition."""
    context_sks: frozenset[int]
    effect_sks: frozenset[int]
    count: int
    total_in_context: int
