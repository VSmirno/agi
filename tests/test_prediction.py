"""Tests for PredictionEngine (Stage 3)."""

import torch
import pytest

from snks.daf.prediction import PredictionEngine, CausalEdge
from snks.daf.types import PredictionConfig


@pytest.fixture
def config() -> PredictionConfig:
    return PredictionConfig()


@pytest.fixture
def engine(config: PredictionConfig) -> PredictionEngine:
    return PredictionEngine(config)


class TestPredictionEngineObserve:
    """PredictionEngine.observe — causal graph building."""

    def test_no_crash_on_first_observe(self, engine: PredictionEngine) -> None:
        engine.observe({0, 1})

    def test_repeated_sequence_builds_edge(self, engine: PredictionEngine) -> None:
        """A→B repeatedly → causal edge A→B appears."""
        for _ in range(10):
            engine.observe({0})
            engine.observe({1})
        graph = engine.get_causal_graph()
        assert 0 in graph
        assert 1 in graph[0]
        assert graph[0][1].confidence > 0.0

    def test_no_edge_for_unrelated(self, engine: PredictionEngine) -> None:
        """Non-sequential SKS → no causal edge."""
        engine.observe({0})
        engine.observe({1})
        engine.observe({2})
        # Only one repetition, confidence should be low
        graph = engine.get_causal_graph()
        # With just one occurrence, edge may exist but with low confidence
        if 0 in graph and 1 in graph[0]:
            assert graph[0][1].confidence < 0.5


class TestPredictionEnginePredict:
    """PredictionEngine.predict — next SKS prediction."""

    def test_predict_empty_initially(self, engine: PredictionEngine) -> None:
        assert engine.predict() == set()

    def test_predicts_after_learning(self, engine: PredictionEngine) -> None:
        """After learning A→B, observing A → predicts B."""
        for _ in range(20):
            engine.observe({0})
            engine.observe({1})
        # Now observe A again
        engine.observe({0})
        predicted = engine.predict()
        assert 1 in predicted


class TestPredictionError:
    """PredictionEngine.compute_prediction_error."""

    def test_correct_prediction_zero_pe(self, engine: PredictionEngine) -> None:
        """Predicted matches actual → PE = 0."""
        pe = engine.compute_prediction_error(
            predicted={0}, actual={0}, n_nodes=100,
            sks_clusters={0: {0, 1, 2}, 1: {3, 4, 5}},
        )
        assert pe.shape == (100,)
        assert pe.sum().item() == 0.0

    def test_wrong_prediction_nonzero_pe(self, engine: PredictionEngine) -> None:
        """Predicted ≠ actual → PE > 0 for surprise nodes."""
        pe = engine.compute_prediction_error(
            predicted={0}, actual={1}, n_nodes=100,
            sks_clusters={0: {0, 1, 2}, 1: {3, 4, 5}},
        )
        assert pe.shape == (100,)
        # Nodes in unexpected cluster 1 and missing cluster 0 should have PE
        assert pe.sum().item() > 0.0

    def test_empty_prediction_pe(self, engine: PredictionEngine) -> None:
        """No prediction → PE for all actual SKS nodes."""
        pe = engine.compute_prediction_error(
            predicted=set(), actual={0}, n_nodes=50,
            sks_clusters={0: {0, 1, 2}},
        )
        assert pe[0].item() > 0
        assert pe[1].item() > 0
        assert pe[2].item() > 0


class TestLRModulation:
    """PredictionEngine.get_lr_modulation."""

    def test_shape(self, engine: PredictionEngine) -> None:
        pe = torch.zeros(100)
        lr = engine.get_lr_modulation(pe, alpha=1.0)
        assert lr.shape == (100,)

    def test_zero_pe_gives_ones(self, engine: PredictionEngine) -> None:
        pe = torch.zeros(100)
        lr = engine.get_lr_modulation(pe, alpha=1.0)
        assert torch.allclose(lr, torch.ones(100))

    def test_positive_pe_gives_greater_than_one(self, engine: PredictionEngine) -> None:
        pe = torch.ones(100) * 0.5
        lr = engine.get_lr_modulation(pe, alpha=2.0)
        assert (lr > 1.0).all()
        assert torch.allclose(lr, torch.ones(100) * 2.0)
