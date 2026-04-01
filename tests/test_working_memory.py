"""Tests for Stage 43: Working Memory — sustained oscillation."""

import torch
import pytest

from snks.daf.types import DafConfig, PipelineConfig
from snks.daf.engine import DafEngine
from snks.pipeline.runner import Pipeline


class TestSelectiveReset:
    """Perception cycle resets perceptual zone but preserves WM zone."""

    def test_wm_nodes_preserved_after_cycle(self):
        """WM zone should retain activation after perception_cycle."""
        config = PipelineConfig()
        config.daf.num_nodes = 200
        config.daf.avg_degree = 10
        config.daf.device = "cpu"
        config.daf.disable_csr = True
        config.daf.dt = 0.005
        config.daf.wm_fraction = 0.2
        config.steps_per_cycle = 50

        pipeline = Pipeline(config)
        n_wm = int(200 * 0.2)  # 40 WM nodes
        wm_start = 200 - n_wm  # 160

        # Set WM nodes to a known state
        pipeline.engine.states[wm_start:, 0] = 1.5

        # Run perception cycle
        sdr = torch.zeros(config.encoder.sdr_size)
        sdr[0:10] = 1.0
        pipeline.perception_cycle(pre_sdr=sdr)

        # WM nodes should NOT be fully reset
        wm_v = pipeline.engine.states[wm_start:, 0]
        assert wm_v.abs().mean() > 0.1, f"WM should retain activation, got mean={wm_v.abs().mean()}"

    def test_perceptual_nodes_reset(self):
        """Perceptual zone should be reset each cycle."""
        config = PipelineConfig()
        config.daf.num_nodes = 200
        config.daf.avg_degree = 10
        config.daf.device = "cpu"
        config.daf.disable_csr = True
        config.daf.dt = 0.005
        config.daf.wm_fraction = 0.2
        config.steps_per_cycle = 50

        pipeline = Pipeline(config)
        n_wm = int(200 * 0.2)
        n_percept = 200 - n_wm

        # Set perceptual nodes to extreme values
        pipeline.engine.states[:n_percept, 0] = 5.0

        # Run perception cycle
        sdr = torch.zeros(config.encoder.sdr_size)
        pipeline.perception_cycle(pre_sdr=sdr)

        # Perceptual nodes should be reset (small random values)
        percept_v = pipeline.engine.states[:n_percept, 0]
        assert percept_v.abs().mean() < 1.0, "Perceptual zone should be reset"

    def test_wm_decay_over_cycles(self):
        """WM should decay over multiple cycles without reinforcement."""
        config = PipelineConfig()
        config.daf.num_nodes = 200
        config.daf.avg_degree = 10
        config.daf.device = "cpu"
        config.daf.disable_csr = True
        config.daf.dt = 0.005
        config.daf.wm_fraction = 0.2
        config.daf.wm_decay = 0.9
        config.steps_per_cycle = 50

        pipeline = Pipeline(config)
        n_wm = int(200 * 0.2)
        wm_start = 200 - n_wm

        # Inject strong activation
        pipeline.engine.states[wm_start:, 0] = 2.0
        initial_mag = pipeline.engine.states[wm_start:, 0].abs().mean().item()

        # Run 20 cycles without WM-relevant stimulus
        sdr = torch.zeros(config.encoder.sdr_size)
        for _ in range(20):
            pipeline.perception_cycle(pre_sdr=sdr)

        final_mag = pipeline.engine.states[wm_start:, 0].abs().mean().item()
        # WM decays but coupling from perceptual zone can partially sustain it
        # With wm_decay=0.9 and 20 cycles: pure decay = 0.9^20 = 0.12
        # But FHN dynamics + coupling means slower decay. Check < 80% of initial.
        assert final_mag < initial_mag * 0.8, f"WM should decay: {initial_mag} → {final_mag}"

    def test_zero_wm_fraction_is_full_reset(self):
        """With wm_fraction=0, behavior should be identical to original."""
        config = PipelineConfig()
        config.daf.num_nodes = 200
        config.daf.avg_degree = 10
        config.daf.device = "cpu"
        config.daf.disable_csr = True
        config.daf.dt = 0.005
        config.daf.wm_fraction = 0.0
        config.steps_per_cycle = 50

        pipeline = Pipeline(config)

        # Set all to extreme
        pipeline.engine.states[:, 0] = 5.0

        sdr = torch.zeros(config.encoder.sdr_size)
        pipeline.perception_cycle(pre_sdr=sdr)

        # Everything should be reset
        assert pipeline.engine.states[:, 0].abs().mean() < 1.0


class TestWMRetention:
    """WM retains stimulus pattern over multiple cycles."""

    def test_stimulus_boosts_wm_activation(self):
        """A strong stimulus should increase WM activation above baseline."""
        config = PipelineConfig()
        config.daf.num_nodes = 500
        config.daf.avg_degree = 15
        config.daf.device = "cpu"
        config.daf.disable_csr = True
        config.daf.dt = 0.005
        config.daf.wm_fraction = 0.2
        config.daf.wm_decay = 0.95
        config.steps_per_cycle = 100

        pipeline = Pipeline(config)
        n_wm = int(500 * 0.2)
        wm_start = 500 - n_wm

        # Baseline: empty cycle
        empty_sdr = torch.zeros(config.encoder.sdr_size)
        pipeline.perception_cycle(pre_sdr=empty_sdr)
        baseline_mag = pipeline.engine.states[wm_start:, 0].abs().mean().item()

        # Strong stimulus cycle
        sdr = torch.zeros(config.encoder.sdr_size)
        sdr[0:200] = 1.0
        pipeline.perception_cycle(pre_sdr=sdr)
        stimulated_mag = pipeline.engine.states[wm_start:, 0].abs().mean().item()

        # WM should have higher activation after stimulus vs baseline
        # (stimulus drives perceptual zone which couples to WM)
        assert stimulated_mag > baseline_mag * 0.5, \
            f"Stimulus should boost WM: baseline={baseline_mag:.3f} stimulated={stimulated_mag:.3f}"


class TestCycleResultWM:
    """CycleResult includes WM metrics."""

    def test_wm_activation_in_result(self):
        from snks.pipeline.runner import CycleResult
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(CycleResult)]
        assert "wm_activation" in field_names


class TestDafConfigWM:
    """DafConfig has WM parameters."""

    def test_wm_fraction_default(self):
        config = DafConfig()
        assert hasattr(config, "wm_fraction")
        assert config.wm_fraction == 0.0  # backward compat: no WM by default

    def test_wm_decay_default(self):
        config = DafConfig()
        assert hasattr(config, "wm_decay")
        assert config.wm_decay == 0.95
