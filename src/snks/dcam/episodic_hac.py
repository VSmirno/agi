"""EpisodicHACPredictor: K-pair episodic buffer for HAC prediction (Stage 15).

Replaces the single-bundle HACPredictionEngine when use_episodic_buffer=True.

Problem with the bundle approach (HACPredictionEngine):
    memory = bundle(bind(e_t, e_{t+1})) accumulated with decay=0.95.
    After ~20 steps the bundle capacity overflows: SNR drops, predictions
    degrade toward random (cosine similarity ≈ 0.5).

Solution:
    Store the last K (prev_aggregate, curr_aggregate) pairs in a deque.
    predict_next() finds the most similar prev to the current aggregate
    using batch cosine similarity (O(K) lookup, K ≤ 64 → negligible).

    Advantages:
    - No capacity overflow: each pair is stored independently.
    - Prediction quality is stable over long episodes.
    - O(K) lookup with K ≤ 64 is faster than FFT unbind for small K.

    Tradeoff vs bundle:
    - Memory: O(K * D) floats vs O(D).
    - Lookup: O(K) vs O(1). Negligible for K ≤ 64.
    - No implicit generalisation: only exact-match pairs are stored.
      (The bundle implicitly superimposes all pairs, providing a form
      of "blurry" generalisation — lost here. Compensated by larger K.)

API contract: identical to HACPredictionEngine (drop-in replacement).
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from snks.dcam.hac import HACEngine


class EpisodicHACPredictor:
    """K-pair episodic buffer predictor.

    Stores the last `capacity` (prev_aggregate, next_aggregate) pairs.
    predict_next() returns the next_aggregate paired with the most
    similar prev_aggregate to the current state.

    Same interface as HACPredictionEngine: observe(), predict_next(),
    reset(), compute_winner_pe().
    """

    def __init__(self, hac: HACEngine, capacity: int = 32) -> None:
        self.hac = hac
        self.capacity = capacity
        # deque of (prev_aggregate, next_aggregate) pairs, unit-norm tensors
        self._pairs: deque[tuple[Tensor, Tensor]] = deque(maxlen=capacity)
        self._prev_embeddings: dict[int, Tensor] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(self, embeddings: dict[int, Tensor]) -> Tensor:
        """Bundle all SKS embeddings into one aggregate vector."""
        vecs = list(embeddings.values())
        if len(vecs) == 1:
            return vecs[0]
        return self.hac.bundle(vecs)

    # ------------------------------------------------------------------
    # Public API (mirrors HACPredictionEngine)
    # ------------------------------------------------------------------

    def observe(self, embeddings: dict[int, Tensor]) -> None:
        """Store (prev_aggregate, curr_aggregate) pair in the buffer.

        Called once per perception cycle with the current SKS embeddings.
        On the first call there is no previous state, so nothing is stored.
        """
        if self._prev_embeddings is not None and embeddings:
            prev_agg = self._aggregate(self._prev_embeddings)
            curr_agg = self._aggregate(embeddings)
            self._pairs.append((prev_agg, curr_agg))
        self._prev_embeddings = dict(embeddings)

    def predict_next(self, embeddings: dict[int, Tensor]) -> Tensor | None:
        """Predict next aggregate via nearest-neighbour lookup in the buffer.

        Finds the stored prev_aggregate most similar (cosine) to the
        current aggregate, then returns its paired next_aggregate.

        Returns:
            Predicted next embedding (hac_dim,), unit norm. None if buffer empty.
        """
        if not self._pairs or not embeddings:
            return None

        curr_agg = self._aggregate(embeddings)

        if len(self._pairs) == 1:
            best_next = self._pairs[0][1]
        else:
            # Stack all prev vectors → (K, D), compute batch cosine similarity
            prevs = torch.stack([p for p, _ in self._pairs])   # (K, D)
            nexts = torch.stack([n for _, n in self._pairs])   # (K, D)
            sims = self.hac.batch_similarity(curr_agg, prevs)  # (K,)
            best_idx = int(sims.argmax().item())
            best_next = nexts[best_idx]

        norm = best_next.norm().clamp(min=1e-8)
        return best_next / norm

    def reset(self) -> None:
        """Clear buffer and prev state (e.g. at episode boundary)."""
        self._pairs.clear()
        self._prev_embeddings = None

    def compute_winner_pe(self, predicted: Tensor, actual_winner_embed: Tensor) -> float:
        """Prediction error as cosine distance.

        PE = (1 - cosine(predicted, actual)) / 2  ∈ [0, 1]
        PE = 0 → perfect prediction
        PE = 0.5 → orthogonal (random baseline)
        PE = 1 → opposite directions
        """
        cos = self.hac.similarity(predicted, actual_winner_embed)
        return float((1.0 - cos) / 2.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Number of pairs currently stored."""
        return len(self._pairs)
