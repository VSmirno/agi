"""Role-filler parser and embedding resolver for compositional understanding (Stage 20).

RoleFillerParser: chunks + embeddings → sentence_hac via HAC bind/bundle.
EmbeddingResolver: word → HAC embedding (hybrid: GroundingMap cache + DAF fallback).
"""

from __future__ import annotations

from torch import Tensor

from snks.dcam.hac import HACEngine
from snks.language.chunker import Chunk


class RoleFillerParser:
    """Encodes sentences as HAC role-filler structures.

    parse():   chunks → sentence_hac  (bind each filler to its role, bundle)
    extract(): role + sentence_hac → recovered filler
    """

    def __init__(self, hac: HACEngine, roles: dict[str, Tensor]) -> None:
        self.hac = hac
        self.roles = roles

    def parse(
        self, chunks: list[Chunk], embeddings: dict[str, Tensor]
    ) -> Tensor:
        """Encode chunks into a single HAC sentence vector.

        Args:
            chunks: output of chunker.chunk().
            embeddings: mapping chunk.text → HAC unit vector (hac_dim,).

        Returns:
            (hac_dim,) unit vector — the sentence representation.
        """
        bindings: list[Tensor] = []
        for chunk in chunks:
            role_vec = self.roles[chunk.role]
            filler_vec = embeddings[chunk.text]
            bindings.append(self.hac.bind(role_vec, filler_vec))
        return self.hac.bundle(bindings)

    def extract(self, role: str, sentence_hac: Tensor) -> Tensor:
        """Extract filler for a given role from sentence_hac."""
        return self.hac.unbind(self.roles[role], sentence_hac)

    def extract_all(self, sentence_hac: Tensor) -> dict[str, Tensor]:
        """Extract fillers for all known roles."""
        return {name: self.extract(name, sentence_hac) for name in self.roles}


class EmbeddingResolver:
    """Resolves word → HAC embedding using hybrid strategy.

    1. Check GroundingMap for cached SKS → embedding.
    2. Fallback: run DAF perception cycle for unknown words.
    """

    def __init__(
        self,
        grounding_map,  # GroundingMap
        embedder,       # SKSEmbedder
        pipeline=None,  # Pipeline (optional, for DAF fallback)
    ) -> None:
        self.grounding_map = grounding_map
        self.embedder = embedder
        self.pipeline = pipeline
        self._embedding_cache: dict[int, Tensor] = {}

    def resolve(self, word: str, sks_embeddings: dict[int, Tensor] | None = None) -> Tensor | None:
        """Resolve word to HAC embedding.

        Args:
            word: text to resolve.
            sks_embeddings: current cycle's SKS embeddings (sks_id → vec).
                If provided, used for lookup instead of re-computing.

        Returns:
            (hac_dim,) unit vector, or None if word is unknown.
        """
        sks_id = self.grounding_map.word_to_sks(word)
        if sks_id is None:
            return None

        # Check provided embeddings first.
        if sks_embeddings is not None and sks_id in sks_embeddings:
            return sks_embeddings[sks_id]

        # Check local cache.
        if sks_id in self._embedding_cache:
            return self._embedding_cache[sks_id]

        return None

    def cache_embeddings(self, sks_embeddings: dict[int, Tensor]) -> None:
        """Cache SKS embeddings for later resolve() calls."""
        self._embedding_cache.update(sks_embeddings)
