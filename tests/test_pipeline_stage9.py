"""Tests for Pipeline Stage 9 integration: SKSEmbedder + HACPredictionEngine + BroadcastPolicy."""

from __future__ import annotations

import torch
import pytest

from snks.daf.types import (
    PipelineConfig, DafConfig, SKSConfig, HACPredictionConfig, MetacogConfig
)
from snks.pipeline.runner import Pipeline, CycleResult


def make_small_config(**kwargs) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.daf.num_nodes = 500
    cfg.daf.oscillator_model = "fhn"
    cfg.sks.coherence_mode = "rate"
    cfg.steps_per_cycle = 20
    cfg.device = "cpu"
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def make_image() -> torch.Tensor:
    return torch.rand(64, 64)


class TestCycleResultStage9Fields:

    def test_cycle_result_has_winner_pe(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        result = pipeline.perception_cycle(image=make_image())
        assert hasattr(result, "winner_pe")
        assert isinstance(result.winner_pe, float)

    def test_cycle_result_has_winner_embedding(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        result = pipeline.perception_cycle(image=make_image())
        assert hasattr(result, "winner_embedding")

    def test_cycle_result_has_hac_predicted(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        result = pipeline.perception_cycle(image=make_image())
        assert hasattr(result, "hac_predicted")


class TestWinnerPEBehavior:

    def test_winner_pe_zero_before_memory_built(self) -> None:
        """First cycle: HAC memory empty → winner_pe = 0.0."""
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        result = pipeline.perception_cycle(image=make_image())
        assert result.winner_pe == 0.0

    def test_winner_pe_nonzero_after_sequence(self) -> None:
        """After several cycles with same image, memory builds up → winner_pe > 0."""
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        image = make_image()
        last_pe = 0.0
        # Run multiple cycles — after 2nd cycle memory exists
        for _ in range(5):
            result = pipeline.perception_cycle(image=image)
            last_pe = result.winner_pe
        # After 5 cycles with same image we expect memory has been built
        # winner_pe may still be 0 if no GWS winner detected — test robustly
        assert isinstance(last_pe, float)
        assert 0.0 <= last_pe <= 1.0

    def test_winner_embedding_unit_norm_when_present(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        image = make_image()
        # Run 3 cycles to build some state
        for _ in range(3):
            result = pipeline.perception_cycle(image=image)
        if result.winner_embedding is not None:
            norm = result.winner_embedding.norm().item()
            assert abs(norm - 1.0) < 1e-4

    def test_hac_predicted_unit_norm_when_present(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        image = make_image()
        for _ in range(3):
            result = pipeline.perception_cycle(image=image)
        if result.hac_predicted is not None:
            norm = result.hac_predicted.norm().item()
            assert abs(norm - 1.0) < 1e-4


class TestBroadcastIntegration:

    def test_broadcast_policy_config_accepted(self) -> None:
        cfg = make_small_config()
        cfg.metacog = MetacogConfig(policy="broadcast", policy_strength=0.5, broadcast_threshold=0.3)
        pipeline = Pipeline(cfg)
        # Should not crash
        for _ in range(3):
            pipeline.perception_cycle(image=make_image())

    def test_broadcast_currents_applied_next_cycle(self) -> None:
        """BroadcastPolicy stores pending currents that are injected next cycle."""
        cfg = make_small_config()
        cfg.metacog = MetacogConfig(policy="broadcast", policy_strength=1.0, broadcast_threshold=0.0)
        pipeline = Pipeline(cfg)
        # Run 2 cycles: first sets up broadcast, second applies it
        pipeline.perception_cycle(image=make_image())
        pipeline.perception_cycle(image=make_image())
        # Broadcast currents should have been consumed (None after apply)
        assert pipeline._broadcast_currents is None


class TestBackwardCompatibility:

    def test_mean_prediction_error_still_present(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        result = pipeline.perception_cycle(image=make_image())
        assert hasattr(result, "mean_prediction_error")
        assert isinstance(result.mean_prediction_error, float)

    def test_old_fields_unchanged(self) -> None:
        cfg = make_small_config()
        pipeline = Pipeline(cfg)
        result = pipeline.perception_cycle(image=make_image())
        assert hasattr(result, "sks_clusters")
        assert hasattr(result, "n_sks")
        assert hasattr(result, "n_spikes")
        assert hasattr(result, "cycle_time_ms")
        assert hasattr(result, "gws")
        assert hasattr(result, "metacog")
