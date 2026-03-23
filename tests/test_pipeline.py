"""Tests for Pipeline (Stage 3)."""

import torch
import pytest
import numpy as np

from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, SKSConfig, PredictionConfig
from snks.pipeline.runner import Pipeline, CycleResult, TrainResult


@pytest.fixture
def small_pipeline_config() -> PipelineConfig:
    """Small config for fast tests."""
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=500,
            avg_degree=10,
            oscillator_model="kuramoto",
            coupling_strength=2.0,
            device="cpu",
        ),
        encoder=EncoderConfig(),
        sks=SKSConfig(top_k=200, dbscan_min_samples=5, min_cluster_size=5),
        prediction=PredictionConfig(),
        steps_per_cycle=50,
        device="cpu",
    )


@pytest.fixture
def pipeline(small_pipeline_config: PipelineConfig) -> Pipeline:
    return Pipeline(small_pipeline_config)


class TestPipelinePerceptionCycle:
    """Pipeline.perception_cycle — single image processing."""

    def test_returns_cycle_result(self, pipeline: Pipeline) -> None:
        img = torch.rand(64, 64)
        result = pipeline.perception_cycle(img)
        assert isinstance(result, CycleResult)

    def test_cycle_result_fields(self, pipeline: Pipeline) -> None:
        img = torch.rand(64, 64)
        result = pipeline.perception_cycle(img)
        assert isinstance(result.sks_clusters, dict)
        assert isinstance(result.n_sks, int)
        assert isinstance(result.mean_prediction_error, float)
        assert isinstance(result.n_spikes, int)
        assert isinstance(result.cycle_time_ms, float)
        assert result.n_sks >= 0
        assert result.cycle_time_ms > 0

    def test_multiple_cycles(self, pipeline: Pipeline) -> None:
        """Multiple cycles don't crash."""
        for _ in range(5):
            img = torch.rand(64, 64)
            result = pipeline.perception_cycle(img)
            assert isinstance(result, CycleResult)


class TestPipelineTrainOnDataset:
    """Pipeline.train_on_dataset — batch training."""

    def test_returns_train_result(self, pipeline: Pipeline) -> None:
        images = torch.rand(10, 64, 64)
        labels = torch.arange(10) % 3
        result = pipeline.train_on_dataset(images, labels, epochs=1)
        assert isinstance(result, TrainResult)

    def test_train_result_fields(self, pipeline: Pipeline) -> None:
        images = torch.rand(10, 64, 64)
        labels = torch.arange(10) % 3
        result = pipeline.train_on_dataset(images, labels, epochs=1)
        assert result.n_cycles == 10
        assert isinstance(result.final_nmi, float)
        assert len(result.mean_pe_history) == 10
        assert len(result.sks_count_history) == 10

    def test_multiple_epochs(self, pipeline: Pipeline) -> None:
        images = torch.rand(5, 64, 64)
        labels = torch.arange(5) % 2
        result = pipeline.train_on_dataset(images, labels, epochs=2)
        assert result.n_cycles == 10  # 5 images × 2 epochs


class TestPipelineIntegration:
    """Integration: full pipeline flow."""

    def test_encoder_to_daf(self, pipeline: Pipeline) -> None:
        """Encoder output feeds into DAF correctly."""
        img = torch.rand(64, 64)
        sdr = pipeline.encoder.encode(img)
        assert sdr.shape == (4096,)
        currents = pipeline.encoder.sdr_to_currents(sdr, pipeline.engine.config.num_nodes)
        assert currents.shape == (500, 8)

    def test_detection_after_steps(self, pipeline: Pipeline) -> None:
        """After DAF stepping, detection produces some output."""
        img = torch.rand(64, 64)
        result = pipeline.perception_cycle(img)
        # At minimum, detection ran without errors
        assert result.n_sks >= 0
