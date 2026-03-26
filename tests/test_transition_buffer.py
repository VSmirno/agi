"""Tests for AgentTransitionBuffer."""

from __future__ import annotations

import pytest

from snks.agent.transition_buffer import AgentTransition, AgentTransitionBuffer


class TestAgentTransitionBuffer:
    def test_add_and_len(self):
        buf = AgentTransitionBuffer(capacity=10)
        buf.add({1, 2}, 0, {3, 4}, importance=0.5)
        assert len(buf) == 1

    def test_capacity_eviction(self):
        """Buffer must not exceed capacity; oldest entries are evicted."""
        buf = AgentTransitionBuffer(capacity=3)
        for i in range(5):
            buf.add({i}, i, {i + 1}, importance=float(i))
        assert len(buf) == 3

    def test_get_top_k_by_importance(self):
        buf = AgentTransitionBuffer(capacity=10)
        buf.add({0}, 0, {1}, importance=0.1)
        buf.add({1}, 1, {2}, importance=0.9)
        buf.add({2}, 2, {3}, importance=0.5)
        top = buf.get_top_k(k=2)
        assert top[0].importance == 0.9
        assert top[1].importance == 0.5

    def test_get_top_k_exceeds_len(self):
        buf = AgentTransitionBuffer(capacity=10)
        buf.add({0}, 0, {1}, importance=1.0)
        top = buf.get_top_k(k=100)
        assert len(top) == 1

    def test_fields_preserved(self):
        buf = AgentTransitionBuffer(capacity=5)
        buf.add({10, 11}, 3, {20, 21}, importance=0.7)
        t = buf.get_top_k(k=1)[0]
        assert t.pre_sks == {10, 11}
        assert t.action == 3
        assert t.post_sks == {20, 21}
        assert t.importance == pytest.approx(0.7)

    def test_empty_get_top_k(self):
        buf = AgentTransitionBuffer()
        assert buf.get_top_k(k=5) == []
