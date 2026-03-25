"""HACPredictionEngine: HAC associative memory predictor (Stage 9).

Predicts the next SKS embedding by storing associations bind(e_t, e_{t+1})
in a bundle (superposition). Prediction = unbind(current, memory).

No backpropagation — uses only HAC bind/unbind/bundle operations.
"""

from __future__ import annotations

from torch import Tensor

from snks.dcam.hac import HACEngine
from snks.daf.types import HACPredictionConfig


class HACPredictionEngine:
    """HAC associative memory for predicting next SKS embedding.

    Memory = bundle of bind(e_t, e_{t+1}) pairs accumulated over time.
    On each observe(): new pairs are added, old memory decays.
    predict_next(): unbind(aggregate_current, memory) → predicted next.
    """

    def __init__(self, hac: HACEngine, config: HACPredictionConfig) -> None:
        self.hac = hac
        self.config = config
        self._memory: Tensor | None = None
        self._prev_embeddings: dict[int, Tensor] | None = None

    def observe(self, embeddings: dict[int, Tensor]) -> None:
        """Update associative memory with new (prev → curr) pairs.

        For each SKS present in both prev and current embeddings,
        bind(prev_embed, curr_embed) is bundled into memory with decay.
        """
        if self._prev_embeddings is not None and embeddings:
            new_pairs: list[Tensor] = []
            for sks_id, curr in embeddings.items():
                if sks_id in self._prev_embeddings:
                    pair = self.hac.bind(self._prev_embeddings[sks_id], curr)
                    new_pairs.append(pair)

            if new_pairs:
                new_bundle = self.hac.bundle(new_pairs) if len(new_pairs) > 1 else new_pairs[0]
                if self._memory is None:
                    self._memory = new_bundle
                else:
                    # Decay old memory and superpose with new associations
                    decayed = self._memory * self.config.memory_decay
                    combined = [decayed, new_bundle]
                    self._memory = self.hac.bundle(combined)

        self._prev_embeddings = dict(embeddings)

    def predict_next(self, embeddings: dict[int, Tensor]) -> Tensor | None:
        """Predict next aggregate embedding via unbind from memory.

        Args:
            embeddings: current SKS embeddings.

        Returns:
            Predicted next embedding (hac_dim,), unit norm. None if memory empty.
        """
        if self._memory is None or not embeddings:
            return None

        vecs = list(embeddings.values())
        aggregate = self.hac.bundle(vecs) if len(vecs) > 1 else vecs[0]
        predicted = self.hac.unbind(aggregate, self._memory)
        norm = predicted.norm().clamp(min=1e-8)
        return predicted / norm

    def compute_winner_pe(self, predicted: Tensor, actual_winner_embed: Tensor) -> float:
        """Compute prediction error as cosine distance in HAC space.

        PE = (1 - cosine(predicted, actual)) / 2  ∈ [0, 1]
        PE = 0 → perfect prediction (identical vectors)
        PE = 0.5 → orthogonal (random)
        PE = 1 → opposite directions
        """
        cos = self.hac.similarity(predicted, actual_winner_embed)
        return float((1.0 - cos) / 2.0)
