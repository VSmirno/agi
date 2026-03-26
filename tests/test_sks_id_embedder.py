"""Tests for SKSIDEmbedder — deterministic cross-session HAC embeddings."""

from __future__ import annotations

import torch
import pytest

from snks.dcam.consolidation_sched import SKSIDEmbedder
from snks.dcam.hac import HACEngine


@pytest.fixture
def hac():
    return HACEngine(dim=64, device=torch.device("cpu"))


@pytest.fixture
def embedder(hac):
    return SKSIDEmbedder(hac_dim=hac.dim, device=torch.device("cpu"))


class TestSKSIDEmbedder:
    def test_same_id_same_vector(self, embedder):
        """Same SKS ID must always produce the exact same vector."""
        v1 = embedder.embed_id(42)
        v2 = embedder.embed_id(42)
        assert torch.allclose(v1, v2)

    def test_different_ids_different_vectors(self, embedder):
        """Different IDs should produce different vectors."""
        v1 = embedder.embed_id(1)
        v2 = embedder.embed_id(2)
        assert not torch.allclose(v1, v2)

    def test_unit_norm(self, embedder):
        """Embedded vectors should have unit norm."""
        v = embedder.embed_id(100)
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_cross_session_determinism(self, hac):
        """Two independent embedders produce identical vectors for same ID."""
        e1 = SKSIDEmbedder(hac_dim=hac.dim, device=torch.device("cpu"))
        e2 = SKSIDEmbedder(hac_dim=hac.dim, device=torch.device("cpu"))
        assert torch.allclose(e1.embed_id(999), e2.embed_id(999))

    def test_encode_sks_set_empty_returns_none(self, embedder, hac):
        result = embedder.encode_sks_set(set(), hac)
        assert result is None

    def test_encode_sks_set_single(self, embedder, hac):
        result = embedder.encode_sks_set({7}, hac)
        assert result is not None
        assert result.shape == (hac.dim,)

    def test_encode_sks_set_multiple(self, embedder, hac):
        result = embedder.encode_sks_set({1, 2, 3}, hac)
        assert result is not None
        assert result.shape == (hac.dim,)

    def test_re_encode_same_ids_high_similarity(self, hac):
        """Re-encoding same set with a fresh embedder gives similarity > 0.99."""
        e1 = SKSIDEmbedder(hac_dim=hac.dim, device=torch.device("cpu"))
        e2 = SKSIDEmbedder(hac_dim=hac.dim, device=torch.device("cpu"))
        sks = {10, 20, 30}
        v1 = e1.encode_sks_set(sks, hac)
        v2 = e2.encode_sks_set(sks, hac)
        assert hac.similarity(v1, v2) > 0.99
