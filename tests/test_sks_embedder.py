"""Tests for SKSEmbedder (Stage 9)."""

import torch
import pytest

from snks.sks.embedder import SKSEmbedder


@pytest.fixture
def embedder() -> SKSEmbedder:
    return SKSEmbedder(n_nodes=100, hac_dim=128, device="cpu")


class TestSKSEmbedder:

    def test_embed_returns_unit_vectors(self, embedder: SKSEmbedder) -> None:
        clusters = {0: {0, 1, 2, 3}, 1: {10, 11, 12}}
        result = embedder.embed(clusters)
        for sks_id, vec in result.items():
            assert vec.shape == (128,)
            norm = vec.norm().item()
            assert abs(norm - 1.0) < 1e-5, f"SKS {sks_id} norm={norm}, expected 1.0"

    def test_different_clusters_different_embeddings(self, embedder: SKSEmbedder) -> None:
        clusters = {0: {0, 1, 2}, 1: {50, 51, 52}}
        result = embedder.embed(clusters)
        cos = torch.dot(result[0], result[1]).item()
        assert cos < 0.99, f"Different clusters too similar: cosine={cos}"

    def test_same_cluster_same_embedding(self, embedder: SKSEmbedder) -> None:
        clusters = {0: {5, 6, 7, 8}}
        r1 = embedder.embed(clusters)
        r2 = embedder.embed(clusters)
        cos = torch.dot(r1[0], r2[0]).item()
        assert abs(cos - 1.0) < 1e-5, f"Same cluster not deterministic: cosine={cos}"

    def test_empty_clusters_returns_empty(self, embedder: SKSEmbedder) -> None:
        result = embedder.embed({})
        assert result == {}

    def test_item_memory_not_updated(self, embedder: SKSEmbedder) -> None:
        memory_before = embedder._item_memory.clone()
        for _ in range(10):
            embedder.embed({0: {0, 1, 2}, 1: {3, 4, 5}})
        diff = (embedder._item_memory - memory_before).abs().max().item()
        assert diff == 0.0, "item_memory was modified during embed()"

    def test_single_node_cluster(self, embedder: SKSEmbedder) -> None:
        clusters = {0: {42}}
        result = embedder.embed(clusters)
        assert 0 in result
        assert result[0].shape == (128,)
        norm = result[0].norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_large_cluster(self, embedder: SKSEmbedder) -> None:
        clusters = {0: set(range(80))}
        result = embedder.embed(clusters)
        assert 0 in result
        norm = result[0].norm().item()
        assert abs(norm - 1.0) < 1e-5
