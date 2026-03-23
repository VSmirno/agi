"""Tests for DafEngine."""

import torch

from snks.daf.engine import DafEngine, StepResult
from snks.daf.types import DafConfig


class TestEngineInit:
    def test_states_shape(self, device):
        cfg = DafConfig(num_nodes=200, avg_degree=10, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        assert engine.states.shape == (200, 8)
        assert engine.states.device.type == device.type

    def test_graph_created(self, device):
        cfg = DafConfig(num_nodes=200, avg_degree=10, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        assert engine.graph.num_nodes == 200
        assert engine.graph.num_edges > 0


class TestEngineStep:
    def test_step_result_shapes(self, device):
        cfg = DafConfig(num_nodes=100, avg_degree=10, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        result = engine.step(n_steps=10)
        assert isinstance(result, StepResult)
        assert result.states.shape == (100, 8)
        assert result.fired_history.shape == (10, 100)
        assert result.prediction_error.shape == (100,)

    def test_state_changes_after_step(self, device):
        cfg = DafConfig(num_nodes=100, avg_degree=10, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        before = engine.states.clone()
        engine.step(n_steps=5)
        assert not torch.allclose(engine.states, before)

    def test_step_count_advances(self, device):
        cfg = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        assert engine.step_count == 0
        engine.step(10)
        assert engine.step_count == 10
        engine.step(20)
        assert engine.step_count == 30

    def test_result_states_are_cloned(self, device):
        cfg = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        result = engine.step(5)
        # Mutate internal state
        engine.states.zero_()
        # result.states should be unaffected
        assert result.states.abs().sum() > 0


class TestSetInput:
    def test_set_input_1d(self, device):
        cfg = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        currents = torch.ones(50, device=device)
        engine.set_input(currents)
        assert engine._external_currents[:, 0].sum() == 50.0

    def test_set_input_2d(self, device):
        cfg = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="kuramoto", device=str(device))
        engine = DafEngine(cfg)
        currents = torch.ones(50, 8, device=device) * 0.5
        engine.set_input(currents)
        assert torch.allclose(engine._external_currents, currents)

    def test_input_affects_dynamics(self, device):
        cfg = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="kuramoto", device=str(device))
        e1 = DafEngine(cfg)
        e2 = DafEngine(cfg)
        # Same initial state
        e2.states.copy_(e1.states)
        e2.graph = e1.graph

        e1.set_input(torch.ones(50, device=device) * 10.0)
        r1 = e1.step(10)
        r2 = e2.step(10)
        # Different external currents → different final states
        assert not torch.allclose(r1.states, r2.states)


class TestFHNEngine:
    def test_fhn_runs(self, device):
        cfg = DafConfig(num_nodes=100, avg_degree=10, oscillator_model="fhn", device=str(device))
        engine = DafEngine(cfg)
        result = engine.step(n_steps=10)
        assert result.states.shape == (100, 8)
        assert not result.states.isnan().any()
