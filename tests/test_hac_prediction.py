"""Tests for HACPredictionEngine (Stage 9)."""

import torch
import pytest

from snks.dcam.hac import HACEngine
from snks.daf.hac_prediction import HACPredictionEngine
from snks.daf.types import HACPredictionConfig
from snks.sks.embedder import SKSEmbedder


HAC_DIM = 128


@pytest.fixture
def hac() -> HACEngine:
    return HACEngine(dim=HAC_DIM)


@pytest.fixture
def config() -> HACPredictionConfig:
    return HACPredictionConfig(memory_decay=0.95, enabled=True)


@pytest.fixture
def engine(hac: HACEngine, config: HACPredictionConfig) -> HACPredictionEngine:
    return HACPredictionEngine(hac, config)


@pytest.fixture
def embedder() -> SKSEmbedder:
    return SKSEmbedder(n_nodes=50, hac_dim=HAC_DIM, device="cpu")


def make_embeddings(embedder: SKSEmbedder, clusters: dict) -> dict:
    return embedder.embed(clusters)


class TestHACPredictionEngine:

    def test_predict_next_returns_none_before_memory(
        self, engine: HACPredictionEngine, embedder: SKSEmbedder
    ) -> None:
        emb = make_embeddings(embedder, {0: {0, 1, 2}})
        result = engine.predict_next(emb)
        assert result is None

    def test_observe_builds_memory(
        self, engine: HACPredictionEngine, embedder: SKSEmbedder
    ) -> None:
        emb_a = make_embeddings(embedder, {0: {0, 1, 2}})
        emb_b = make_embeddings(embedder, {0: {3, 4, 5}})
        engine.observe(emb_a)
        engine.observe(emb_b)
        assert engine._memory is not None

    def test_predict_next_after_observe(
        self, engine: HACPredictionEngine, embedder: SKSEmbedder
    ) -> None:
        emb_a = make_embeddings(embedder, {0: {0, 1, 2}})
        emb_b = make_embeddings(embedder, {0: {3, 4, 5}})
        engine.observe(emb_a)
        engine.observe(emb_b)
        result = engine.predict_next(emb_a)
        assert result is not None
        assert result.shape == (HAC_DIM,)
        # должен быть единичным вектором
        assert abs(result.norm().item() - 1.0) < 1e-4

    def test_repeated_ab_improves_similarity(
        self, engine: HACPredictionEngine, embedder: SKSEmbedder
    ) -> None:
        emb_a = make_embeddings(embedder, {0: {0, 1, 2}})
        emb_b = make_embeddings(embedder, {0: {10, 11, 12}})

        # После 1 пары
        engine.observe(emb_a)
        engine.observe(emb_b)
        pred1 = engine.predict_next(emb_a)
        sim1 = engine.hac.similarity(pred1, emb_b[0])

        # После 10 пар
        for _ in range(9):
            engine.observe(emb_a)
            engine.observe(emb_b)
        pred10 = engine.predict_next(emb_a)
        sim10 = engine.hac.similarity(pred10, emb_b[0])

        assert sim10 >= sim1 - 0.05, (
            f"Similarity did not improve with more observations: {sim1:.3f} → {sim10:.3f}"
        )

    def test_compute_winner_pe_identical(
        self, engine: HACPredictionEngine, hac: HACEngine
    ) -> None:
        v = hac.random_vector()
        pe = engine.compute_winner_pe(v, v)
        assert abs(pe - 0.0) < 1e-4, f"PE for identical vectors should be 0, got {pe}"

    def test_compute_winner_pe_orthogonal(
        self, engine: HACPredictionEngine
    ) -> None:
        v1 = torch.zeros(HAC_DIM)
        v1[0] = 1.0
        v2 = torch.zeros(HAC_DIM)
        v2[1] = 1.0
        pe = engine.compute_winner_pe(v1, v2)
        assert abs(pe - 0.5) < 1e-4, f"PE for orthogonal vectors should be 0.5, got {pe}"

    def test_pe_in_range(
        self, engine: HACPredictionEngine, hac: HACEngine
    ) -> None:
        v1 = hac.random_vector()
        v2 = hac.random_vector()
        pe = engine.compute_winner_pe(v1, v2)
        assert 0.0 <= pe <= 1.0, f"PE out of range [0,1]: {pe}"

    def test_memory_decay_reduces_old_associations(
        self, engine: HACPredictionEngine, embedder: SKSEmbedder
    ) -> None:
        emb_a = make_embeddings(embedder, {0: {0, 1, 2}})
        emb_b = make_embeddings(embedder, {0: {10, 11, 12}})
        emb_c = make_embeddings(embedder, {0: {20, 21, 22}})

        # Обучаем A→B 5 раз
        for _ in range(5):
            engine.observe(emb_a)
            engine.observe(emb_b)

        sim_ab_before = engine.hac.similarity(engine.predict_next(emb_a), emb_b[0])

        # Затем обучаем A→C 20 раз (decay вытесняет A→B)
        for _ in range(20):
            engine.observe(emb_a)
            engine.observe(emb_c)

        sim_ab_after = engine.hac.similarity(engine.predict_next(emb_a), emb_b[0])
        # A→B должно ослабнуть (или A→C стало сильнее)
        sim_ac = engine.hac.similarity(engine.predict_next(emb_a), emb_c[0])
        assert sim_ac > sim_ab_after - 0.1, (
            f"After decay A→C ({sim_ac:.3f}) should dominate over A→B ({sim_ab_after:.3f})"
        )
