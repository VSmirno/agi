"""Tests for coupling computation."""

import math
import torch

from snks.daf.graph import SparseDafGraph
from snks.daf.coupling import compute_kuramoto_coupling, compute_fhn_coupling


class TestKuramotoCoupling:
    def test_shape(self, device):
        N, E = 100, 500
        states = torch.randn(N, 8, device=device)
        states[:, 0] = torch.rand(N, device=device) * 2 * math.pi
        g = SparseDafGraph.random_sparse(N, 5, device, seed=42)
        c = compute_kuramoto_coupling(states, g, coupling_strength=0.1)
        assert c.shape == (N,)
        assert c.device.type == device.type

    def test_zero_weights_give_zero(self, device):
        N = 50
        states = torch.randn(N, 8, device=device)
        g = SparseDafGraph.random_sparse(N, 10, device, seed=42)
        g.set_strength(torch.zeros(g.num_edges, device=device))
        c = compute_kuramoto_coupling(states, g, coupling_strength=1.0)
        assert c.abs().max() < 1e-6

    def test_isolated_node_gets_zero(self, device):
        N = 10
        states = torch.randn(N, 8, device=device)
        # All edges point TO node 0, nodes 1..9 have no incoming edges
        edge_index = torch.zeros(2, 5, dtype=torch.int64, device=device)
        edge_index[0] = torch.arange(1, 6, device=device)
        edge_index[1] = 0
        edge_attr = torch.zeros(5, 4, device=device)
        edge_attr[:, 0] = 0.5  # nonzero strength
        g = SparseDafGraph(edge_index, edge_attr, N, device)
        c = compute_kuramoto_coupling(states, g, coupling_strength=1.0)
        assert c[1:].abs().max() < 1e-6
        # Node 0 should have nonzero coupling (unless sin happens to be 0)

    def test_zero_coupling_strength(self, device):
        N = 50
        states = torch.randn(N, 8, device=device)
        g = SparseDafGraph.random_sparse(N, 10, device, seed=42)
        c = compute_kuramoto_coupling(states, g, coupling_strength=0.0)
        assert c.abs().max() < 1e-6

    def test_inhibitory_reverses_sign(self, device):
        """Two nodes: excitatory vs inhibitory edge should give opposite coupling."""
        N = 2
        states = torch.zeros(N, 8, device=device)
        states[0, 0] = 1.0  # src phase
        states[1, 0] = 0.0  # dst phase

        # Excitatory edge 0→1
        ei_exc = torch.tensor([[0], [1]], dtype=torch.int64, device=device)
        ea_exc = torch.tensor([[0.5, 0.0, 0.0, 0.0]], device=device)
        g_exc = SparseDafGraph(ei_exc, ea_exc, N, device)
        c_exc = compute_kuramoto_coupling(states, g_exc, coupling_strength=1.0)

        # Inhibitory edge 0→1
        ea_inh = torch.tensor([[0.5, 0.0, 0.0, 1.0]], device=device)
        g_inh = SparseDafGraph(ei_exc.clone(), ea_inh, N, device)
        c_inh = compute_kuramoto_coupling(states, g_inh, coupling_strength=1.0)

        # Should be opposite sign
        assert c_exc[1] * c_inh[1] < 0


class TestFHNCoupling:
    def test_shape(self, device):
        N = 100
        states = torch.randn(N, 8, device=device)
        g = SparseDafGraph.random_sparse(N, 5, device, seed=42)
        c = compute_fhn_coupling(states, g, coupling_strength=0.1)
        assert c.shape == (N,)

    def test_zero_weights_give_zero(self, device):
        N = 50
        states = torch.randn(N, 8, device=device)
        g = SparseDafGraph.random_sparse(N, 10, device, seed=42)
        g.set_strength(torch.zeros(g.num_edges, device=device))
        c = compute_fhn_coupling(states, g, coupling_strength=1.0)
        assert c.abs().max() < 1e-6

    def test_identical_states_give_zero(self, device):
        """If all v are identical, delta=0 → coupling=0."""
        N = 50
        states = torch.ones(N, 8, device=device) * 0.5
        g = SparseDafGraph.random_sparse(N, 10, device, seed=42)
        c = compute_fhn_coupling(states, g, coupling_strength=1.0)
        assert c.abs().max() < 1e-6
