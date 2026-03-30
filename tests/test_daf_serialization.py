"""Tests for DAF serialization: DafEngine.save_state/load_state and
EmbodiedAgent.save_checkpoint/load_checkpoint round-trips."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

from snks.daf.engine import DafEngine
from snks.daf.types import DafConfig

# gymnasium is only present on the minipc (AMD); skip EmbodiedAgent tests locally.
_gymnasium_available = importlib.util.find_spec("gymnasium") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_daf_config() -> DafConfig:
    """Minimal CPU config for fast serialization tests."""
    return DafConfig(
        num_nodes=100,
        state_dim=8,
        avg_degree=8,
        dt=0.001,
        noise_sigma=0.005,
        oscillator_model="fhn",
        device="cpu",
        disable_csr=True,
    )


# ---------------------------------------------------------------------------
# DafEngine.save_state / load_state
# ---------------------------------------------------------------------------

class TestDafEngineSaveLoad:
    """Round-trip tests for DafEngine.save_state() and load_state()."""

    def test_save_creates_file(self, tmp_path):
        """save_state() creates {base}_daf.safetensors."""
        engine = DafEngine(_small_daf_config())
        base = str(tmp_path / "ckpt")
        engine.save_state(base)
        assert os.path.exists(base + "_daf.safetensors")

    def test_states_round_trip(self, tmp_path):
        """Loaded states are identical to saved states."""
        cfg = _small_daf_config()
        engine = DafEngine(cfg)
        engine.step(50)

        base = str(tmp_path / "ckpt")
        engine.save_state(base)

        engine2 = DafEngine(cfg)
        engine2.load_state(base)

        assert torch.allclose(engine.states, engine2.states), (
            "states mismatch after load_state"
        )

    def test_edge_attr_round_trip(self, tmp_path):
        """Loaded edge_attr is identical to saved edge_attr."""
        cfg = _small_daf_config()
        engine = DafEngine(cfg)
        engine.step(50)

        base = str(tmp_path / "ckpt")
        engine.save_state(base)

        engine2 = DafEngine(cfg)
        engine2.load_state(base)

        assert torch.allclose(engine.graph.edge_attr, engine2.graph.edge_attr), (
            "graph.edge_attr mismatch after load_state"
        )

    def test_edge_index_round_trip(self, tmp_path):
        """Loaded edge_index is identical to saved edge_index."""
        cfg = _small_daf_config()
        engine = DafEngine(cfg)
        engine.step(50)

        base = str(tmp_path / "ckpt")
        engine.save_state(base)

        engine2 = DafEngine(cfg)
        engine2.load_state(base)

        assert torch.equal(engine.graph.edge_index, engine2.graph.edge_index), (
            "graph.edge_index mismatch after load_state"
        )

    def test_step_count_round_trip(self, tmp_path):
        """Loaded step_count matches the value at save time."""
        cfg = _small_daf_config()
        engine = DafEngine(cfg)
        engine.step(50)

        assert engine.step_count == 50

        base = str(tmp_path / "ckpt")
        engine.save_state(base)

        engine2 = DafEngine(cfg)
        engine2.load_state(base)

        assert engine2.step_count == 50

    def test_load_missing_file_raises(self, tmp_path):
        """load_state() raises FileNotFoundError for a missing checkpoint."""
        engine = DafEngine(_small_daf_config())
        with pytest.raises(FileNotFoundError, match="_daf.safetensors"):
            engine.load_state(str(tmp_path / "nonexistent"))

    def test_overwrite_preserves_integrity(self, tmp_path):
        """Saving twice to the same path overwrites cleanly."""
        cfg = _small_daf_config()
        engine = DafEngine(cfg)
        base = str(tmp_path / "ckpt")

        engine.step(10)
        engine.save_state(base)

        engine.step(10)
        engine.save_state(base)  # overwrite

        engine2 = DafEngine(cfg)
        engine2.load_state(base)

        assert engine2.step_count == 20
        assert torch.allclose(engine.states, engine2.states)

    def test_engine_still_runs_after_load(self, tmp_path):
        """Engine loaded from checkpoint can continue stepping without error."""
        cfg = _small_daf_config()
        engine = DafEngine(cfg)
        engine.step(20)

        base = str(tmp_path / "ckpt")
        engine.save_state(base)

        engine2 = DafEngine(cfg)
        engine2.load_state(base)
        result = engine2.step(10)  # must not raise

        assert result.states.shape == (cfg.num_nodes, cfg.state_dim)


# ---------------------------------------------------------------------------
# EmbodiedAgent.save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

def _make_small_agent_config():
    """Minimal EmbodiedAgentConfig for serialization tests (no DCAM/consolidation)."""
    from snks.agent.embodied_agent import EmbodiedAgentConfig
    from snks.daf.types import (
        CausalAgentConfig,
        ConfiguratorConfig,
        CostModuleConfig,
        EncoderConfig,
        PipelineConfig,
        SKSConfig,
    )

    pipeline = PipelineConfig(
        daf=DafConfig(
            num_nodes=500,
            avg_degree=8,
            oscillator_model="fhn",
            dt=0.001,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device="cpu",
            disable_csr=True,
        ),
        encoder=EncoderConfig(sdr_size=256, sdr_sparsity=0.04),
        sks=SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5),
        steps_per_cycle=5,
        device="cpu",
        cost_module=CostModuleConfig(enabled=False),
        configurator=ConfiguratorConfig(enabled=False),
    )
    causal = CausalAgentConfig(
        pipeline=pipeline,
        motor_sdr_size=50,
        causal_min_observations=1,
        curiosity_epsilon=0.5,
    )
    return EmbodiedAgentConfig(
        causal=causal,
        use_stochastic_planner=False,
        n_plan_samples=1,
        max_plan_depth=1,
    )


@pytest.mark.skipif(not _gymnasium_available, reason="gymnasium not installed — run on minipc")
class TestEmbodiedAgentCheckpoint:
    """Round-trip tests for EmbodiedAgent.save_checkpoint / load_checkpoint."""

    def _random_obs(self) -> np.ndarray:
        return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_checkpoint_creates_daf_file(self, tmp_path):
        """save_checkpoint() creates {base}_daf.safetensors."""
        from snks.agent.embodied_agent import EmbodiedAgent

        agent = EmbodiedAgent(_make_small_agent_config())
        agent.step(self._random_obs())

        base = str(tmp_path / "agent")
        agent.save_checkpoint(base)

        assert os.path.exists(base + "_daf.safetensors")

    def test_checkpoint_creates_parent_dir(self, tmp_path):
        """save_checkpoint() creates intermediate directories automatically."""
        from snks.agent.embodied_agent import EmbodiedAgent

        agent = EmbodiedAgent(_make_small_agent_config())
        agent.step(self._random_obs())

        nested = str(tmp_path / "sub" / "dir" / "agent")
        agent.save_checkpoint(nested)

        assert os.path.exists(nested + "_daf.safetensors")

    def test_daf_states_round_trip(self, tmp_path):
        """Loaded DAF states match the original agent's states."""
        from snks.agent.embodied_agent import EmbodiedAgent

        agent = EmbodiedAgent(_make_small_agent_config())
        for _ in range(3):
            agent.step(self._random_obs())
            agent.observe_result(self._random_obs())

        base = str(tmp_path / "agent")
        agent.save_checkpoint(base)

        engine_orig = agent.causal_agent.pipeline.engine

        agent2 = EmbodiedAgent(_make_small_agent_config())
        agent2.load_checkpoint(base)

        engine_loaded = agent2.causal_agent.pipeline.engine

        assert torch.allclose(engine_orig.states, engine_loaded.states), (
            "DAF states mismatch after load_checkpoint"
        )

    def test_daf_edge_attr_round_trip(self, tmp_path):
        """Loaded DAF edge_attr matches the original agent's edge_attr."""
        from snks.agent.embodied_agent import EmbodiedAgent

        agent = EmbodiedAgent(_make_small_agent_config())
        for _ in range(3):
            agent.step(self._random_obs())
            agent.observe_result(self._random_obs())

        base = str(tmp_path / "agent")
        agent.save_checkpoint(base)

        engine_orig = agent.causal_agent.pipeline.engine

        agent2 = EmbodiedAgent(_make_small_agent_config())
        agent2.load_checkpoint(base)

        engine_loaded = agent2.causal_agent.pipeline.engine

        assert torch.allclose(
            engine_orig.graph.edge_attr, engine_loaded.graph.edge_attr
        ), "DAF graph.edge_attr mismatch after load_checkpoint"

    def test_agent_can_step_after_load(self, tmp_path):
        """Agent loaded from checkpoint can continue stepping without error."""
        from snks.agent.embodied_agent import EmbodiedAgent

        agent = EmbodiedAgent(_make_small_agent_config())
        agent.step(self._random_obs())

        base = str(tmp_path / "agent")
        agent.save_checkpoint(base)

        agent2 = EmbodiedAgent(_make_small_agent_config())
        agent2.load_checkpoint(base)

        action = agent2.step(self._random_obs())
        assert isinstance(action, int)
        assert 0 <= action < agent2.n_actions
