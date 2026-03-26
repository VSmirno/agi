"""Tests for ConsolidationScheduler (Stage 16)."""

from __future__ import annotations

import os
import tempfile

import torch
import pytest

from snks.agent.transition_buffer import AgentTransitionBuffer
from snks.daf.types import DcamConfig
from snks.dcam.consolidation_sched import ConsolidationScheduler
from snks.dcam.world_model import DcamWorldModel


def _make_scheduler(capacity=50, save_path=None):
    cfg = DcamConfig(hac_dim=64, lsh_n_tables=4, lsh_n_bits=8, episodic_capacity=100)
    dcam = DcamWorldModel(cfg, device=torch.device("cpu"))
    buf = AgentTransitionBuffer(capacity=capacity)
    return ConsolidationScheduler(
        agent_buffer=buf,
        dcam=dcam,
        every_n=10,
        top_k=20,
        node_threshold=0.7,
        save_path=save_path,
    ), buf, dcam


class TestConsolidationScheduler:
    def test_maybe_consolidate_no_trigger(self):
        sched, _, _ = _make_scheduler()
        assert sched.maybe_consolidate(0) is None
        assert sched.maybe_consolidate(5) is None

    def test_maybe_consolidate_triggers_at_every_n(self):
        sched, buf, _ = _make_scheduler()
        # Fill buffer with some transitions
        for i in range(15):
            buf.add({i}, i % 5, {i + 1}, importance=0.5)
        result = sched.maybe_consolidate(10)
        assert result is not None

    def test_edges_appear_after_run(self):
        sched, buf, dcam = _make_scheduler()
        for i in range(20):
            buf.add({i * 100}, i % 5, {i * 100 + 1}, importance=0.8)
        result = sched.maybe_consolidate(10)
        assert result is not None
        assert result.n_edges_added > 0
        assert result.total_causal_edges > 0
        assert result.total_nodes > 0

    def test_query_returns_none_on_empty(self):
        sched, _, _ = _make_scheduler()
        action, weight = sched.query({1, 2, 3})
        assert action is None
        assert weight == 0.0

    def test_query_after_consolidation(self):
        sched, buf, _ = _make_scheduler()
        for i in range(20):
            buf.add({i}, 2, {i + 1}, importance=1.0)
        sched.maybe_consolidate(10)
        # After consolidation nodes exist; query may or may not match above threshold
        # but should not raise
        action, weight = sched.query({0, 1})
        assert isinstance(weight, float)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sched_test")
            sched, buf, dcam = _make_scheduler(save_path=path)
            for i in range(20):
                buf.add({i * 50}, i % 5, {i * 50 + 10}, importance=0.9)
            summary = sched.maybe_consolidate(10)
            assert summary is not None

            # Create new scheduler, load state
            cfg2 = DcamConfig(hac_dim=64, lsh_n_tables=4, lsh_n_bits=8, episodic_capacity=100)
            dcam2 = DcamWorldModel(cfg2, device=torch.device("cpu"))
            buf2 = AgentTransitionBuffer()
            sched2 = ConsolidationScheduler(agent_buffer=buf2, dcam=dcam2, save_path=None)
            sched2.load_state(path)

            assert sched2._next_node_id == sched._next_node_id
            assert len(sched2._node_registry) == len(sched._node_registry)
            assert len(sched2._edge_actions) == len(sched._edge_actions)
