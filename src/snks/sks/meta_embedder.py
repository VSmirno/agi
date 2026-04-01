"""MetaEmbedder: EWA-based meta-embedding over HAC aggregates (Stage 10).

Accumulates a "slow" meta-embedding via Exponential Weighted Average of
cycle-level HAC bundles. Effective window ≈ 1/(1-decay) cycles.
"""

from __future__ import annotations

import torch
from torch import Tensor

from snks.daf.types import HierarchicalConfig
from snks.dcam.hac import HACEngine


class MetaEmbedder:
    """Computes EWA meta-embedding over per-cycle HAC aggregates.

    Each cycle:
        cycle_embed = bundle(embeddings.values())
        meta = normalize(decay * meta_prev + (1 - decay) * cycle_embed)

    meta_embed is None until the first non-empty embeddings are seen.
    """

    def __init__(self, hac: HACEngine, config: HierarchicalConfig) -> None:
        self._hac = hac
        self._decay: float = config.meta_decay
        self._meta: Tensor | None = None

    def update(self, embeddings: dict[int, Tensor]) -> Tensor | None:
        """Compute cycle aggregate, update EWA meta-embedding.

        Args:
            embeddings: per-SKS HAC vectors {sks_id: (hac_dim,)}.

        Returns:
            Current meta_embed (unit norm), or None if embeddings empty.
        """
        if not embeddings:
            return self._meta

        vecs = list(embeddings.values())
        cycle_embed = self._hac.bundle(vecs) if len(vecs) > 1 else vecs[0]
        # Normalise cycle_embed to unit sphere
        norm = cycle_embed.norm().clamp(min=1e-8)
        cycle_embed = cycle_embed / norm

        if self._meta is None:
            self._meta = cycle_embed
        else:
            # Ensure same device (Stage 43: symbolic encoder may produce CPU tensors)
            if self._meta.device != cycle_embed.device:
                cycle_embed = cycle_embed.to(self._meta.device)
            raw = self._decay * self._meta + (1.0 - self._decay) * cycle_embed
            norm = raw.norm().clamp(min=1e-8)
            self._meta = raw / norm

        return self._meta

    def get_meta_embed(self) -> Tensor | None:
        """Return current meta-embedding (unit norm), or None if not yet seen data."""
        return self._meta

    def reset(self) -> None:
        """Reset meta-embedding to None.

        Must be called by Agent/experiment at episode boundaries.
        Pipeline does NOT know about episode boundaries and does NOT call reset().

        Example:
            for episode in episodes:
                pipeline.meta_embedder.reset()
                pipeline.l2_predictor.reset()
                for image in episode:
                    result = pipeline.perception_cycle(image)
        """
        self._meta = None
