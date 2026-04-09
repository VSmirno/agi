"""Stage 76 v2: AttentionWeights — per-variable attention over SDR bits.

The v1 evaluation showed memory-based reactive policy matched the scripted
bootstrap baseline (~180 steps survival) but didn't exceed it. Diagnosis:
recall returned episodes with similar body/inventory bits but different
enemy/hazard bits, because the ~5% of SDR bits encoding the dangerous
state are dominated by the other 95% in straight popcount similarity.

AttentionWeights learns, for each body variable V the tracker has
observed, which SDR bits are most predictive of V's changes. At query
time it builds a soft per-bit mask weighted by the current deficit, so
recall becomes biased toward bits that matter for the situation — enemy
bits when health is dropping, food bits when hungry, etc.

Ideology compliance:
- No hardcoded variable list — weights dict grows from tracker observations
- No derived features — weights learned from raw (state_bit × body_delta)
  correlations
- No argmax over drives — contributions are summed, weighted by deficit
- Only tunable scalars (lr, clip, mask_baseline) — whitelisted
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttentionWeights:
    """Per-variable attention over SDR bits, learned via EMA.

    Attributes:
        n_bits: width of the SDR (e.g., 4096).
        lr: learning rate for EMA updates.
        clip: magnitude clip to prevent runaway weights.
        mask_baseline: floor value for the query mask — every bit
            contributes at least this much, preventing zero-similarity
            bits from disappearing entirely.
    """

    n_bits: int
    lr: float = 0.02
    clip: float = 5.0
    mask_baseline: float = 1.0
    _weights: dict[str, np.ndarray] = field(default_factory=dict)

    def update(
        self,
        state_sdr: np.ndarray,
        body_delta: dict[str, int],
    ) -> None:
        """Credit/blame assignment: active bits get attributed to deltas.

        For each variable V with nonzero delta, bits that were ON in the
        current state get their weight nudged by `lr × delta`. Positive
        weights → bit is active when V goes up; negative → active when V
        goes down. Magnitude captures strength of correlation.
        """
        if state_sdr.dtype != np.bool_:
            state_sdr = state_sdr.astype(np.bool_)
        for var, delta in body_delta.items():
            if delta == 0:
                continue
            if var not in self._weights:
                self._weights[var] = np.zeros(self.n_bits, dtype=np.float32)
            w = self._weights[var]
            w[state_sdr] += self.lr * float(delta)
            np.clip(w, -self.clip, self.clip, out=w)

    def query_mask(
        self,
        deficits: dict[str, float],
    ) -> np.ndarray | None:
        """Build a per-bit similarity mask from current deficits.

        mask[i] = baseline + Σ_V (normalized_deficit[V] × |weights[V][i]|)

        Returns None if no deficit is positive or no variables have learned
        weights — caller should fall back to uniform similarity.
        """
        if not deficits or not self._weights:
            return None
        total_deficit = sum(float(d) for d in deficits.values() if d > 0)
        if total_deficit <= 0:
            return None
        mask = np.full(self.n_bits, self.mask_baseline, dtype=np.float32)
        for var, deficit in deficits.items():
            if deficit <= 0 or var not in self._weights:
                continue
            # Absolute magnitude: bits relevant to V regardless of direction
            relevance = np.abs(self._weights[var])
            mask += (float(deficit) / total_deficit) * relevance
        return mask

    def known_variables(self) -> set[str]:
        return set(self._weights.keys())

    def weights_for(self, var: str) -> np.ndarray | None:
        return self._weights.get(var)
