"""Tests for SparseDafGraph."""

import torch
import pytest

from snks.daf.graph import SparseDafGraph


class TestRandomSparse:
    def test_shapes(self, device):
        g = SparseDafGraph.random_sparse(100, 10, device)
        assert g.edge_index.shape[0] == 2
        assert g.edge_attr.shape[1] == 4
        assert g.edge_index.shape[1] == g.edge_attr.shape[0]
        assert g.edge_index.device.type == device.type

    def test_no_self_loops(self, device):
        g = SparseDafGraph.random_sparse(1000, 20, device, seed=42)
        src, dst = g.edge_index
        assert (src == dst).sum() == 0

    def test_approximate_edge_count(self, device):
        g = SparseDafGraph.random_sparse(1000, 20, device, seed=42)
        assert abs(g.num_edges - 20000) < 200

    def test_seed_reproducibility(self, device):
        g1 = SparseDafGraph.random_sparse(100, 10, device, seed=123)
        g2 = SparseDafGraph.random_sparse(100, 10, device, seed=123)
        assert torch.equal(g1.edge_index, g2.edge_index)
        assert torch.equal(g1.edge_attr, g2.edge_attr)

    def test_excitatory_inhibitory_ratio(self, device):
        g = SparseDafGraph.random_sparse(1000, 50, device, seed=42)
        inhibitory_ratio = g.edge_attr[:, 3].mean().item()
        assert 0.15 < inhibitory_ratio < 0.25

    def test_initial_strengths_small(self, device):
        g = SparseDafGraph.random_sparse(100, 10, device, seed=42)
        strengths = g.get_strength()
        assert strengths.min() >= 0.0
        assert strengths.max() <= 0.5


class TestEdgeOps:
    def test_add_edges(self, device):
        g = SparseDafGraph.random_sparse(100, 5, device, seed=42)
        e_before = g.num_edges
        src = torch.tensor([0, 1], dtype=torch.int64)
        dst = torch.tensor([2, 3], dtype=torch.int64)
        attr = torch.zeros(2, 4)
        g.add_edges(src, dst, attr)
        assert g.num_edges == e_before + 2

    def test_remove_edges(self, device):
        g = SparseDafGraph.random_sparse(100, 5, device, seed=42)
        e_before = g.num_edges
        mask = torch.zeros(e_before, dtype=torch.bool, device=device)
        mask[:5] = True
        g.remove_edges(mask)
        assert g.num_edges == e_before - 5

    def test_add_then_remove(self, device):
        g = SparseDafGraph.random_sparse(100, 5, device, seed=42)
        e_before = g.num_edges
        src = torch.tensor([0, 1], dtype=torch.int64)
        dst = torch.tensor([2, 3], dtype=torch.int64)
        attr = torch.zeros(2, 4)
        g.add_edges(src, dst, attr)
        mask = torch.zeros(g.num_edges, dtype=torch.bool, device=device)
        mask[-2:] = True
        g.remove_edges(mask)
        assert g.num_edges == e_before


class TestStrength:
    def test_get_set_strength(self, device):
        g = SparseDafGraph.random_sparse(50, 5, device, seed=42)
        new_w = torch.ones(g.num_edges, device=device)
        g.set_strength(new_w)
        assert torch.allclose(g.get_strength(), new_w)

    def test_set_strength_is_inplace(self, device):
        g = SparseDafGraph.random_sparse(50, 5, device, seed=42)
        new_w = torch.ones(g.num_edges, device=device) * 0.5
        g.set_strength(new_w)
        assert torch.allclose(g.edge_attr[:, 0], new_w)


class TestDeviceAgnostic:
    def test_to_device(self, device):
        g = SparseDafGraph.random_sparse(50, 5, torch.device("cpu"), seed=42)
        g2 = g.to(device)
        assert g2.edge_index.device.type == device.type
        assert g2.edge_attr.device.type == device.type
