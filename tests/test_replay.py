"""Tests for ReplayEngine (Stage 16)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import pytest

from snks.agent.transition_buffer import AgentTransitionBuffer
from snks.daf.engine import DafEngine
from snks.daf.stdp import STDP
from snks.daf.types import DafConfig
from snks.dcam.replay import ReplayEngine, ReplayReport


def _make_engine(n=200):
    cfg = DafConfig(
        num_nodes=n,
        avg_degree=4,
        oscillator_model="fhn",
        dt=0.01,
        noise_sigma=0.0,
        fhn_I_base=0.0,
        disable_csr=True,
        device="cpu",
    )
    return DafEngine(cfg, enable_learning=True)


class TestReplayEngine:
    def test_empty_buffer_zero_updates(self):
        engine = _make_engine()
        replay = ReplayEngine(engine, engine.stdp, top_k=5, n_steps=10)
        buf = AgentTransitionBuffer()
        report = replay.replay(buf)
        assert report.stdp_updates == 0
        assert report.n_replayed == 0

    def test_valid_nodes_trigger_stdp(self):
        engine = _make_engine(n=200)
        replay = ReplayEngine(engine, engine.stdp, top_k=3, n_steps=20)
        buf = AgentTransitionBuffer()
        # Add transitions with node IDs within [0, 200)
        buf.add({10, 20, 30}, 1, {40, 50}, importance=1.0)
        buf.add({5, 15}, 2, {25, 35}, importance=0.8)
        report = replay.replay(buf)
        assert report.stdp_updates > 0

    def test_out_of_range_nodes_skipped(self):
        engine = _make_engine(n=100)
        replay = ReplayEngine(engine, engine.stdp, top_k=5, n_steps=10)
        buf = AgentTransitionBuffer()
        # All node IDs >= num_nodes → should be skipped
        buf.add({1000, 2000}, 0, {3000}, importance=1.0)
        report = replay.replay(buf)
        assert report.stdp_updates == 0

    def test_report_type(self):
        engine = _make_engine()
        replay = ReplayEngine(engine, engine.stdp, top_k=2, n_steps=5)
        buf = AgentTransitionBuffer()
        buf.add({1, 2}, 0, {3}, importance=0.5)
        report = replay.replay(buf)
        assert isinstance(report, ReplayReport)
        assert isinstance(report.n_replayed, int)
        assert isinstance(report.stdp_updates, int)

    def test_external_currents_reset_after_step(self):
        """inject_external_currents must be transient: zeroed after step()."""
        engine = _make_engine(n=50)
        engine.inject_external_currents([0, 1, 2], value=1.0)
        engine.step(n_steps=5)
        assert engine._external_currents.abs().sum().item() == pytest.approx(0.0)
