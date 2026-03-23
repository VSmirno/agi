"""Tests for GPU optimization: correctness of inplace functions + performance benchmark."""

import pytest
import torch

from snks.daf.coupling import compute_fhn_coupling, compute_fhn_coupling_inplace
from snks.daf.graph import SparseDafGraph
from snks.daf.oscillator import fhn_derivatives, fhn_derivatives_inplace
from snks.daf.types import DafConfig


# --- Correctness tests ---


class TestFhnDerivativesInplace:
    """Verify inplace FHN derivatives match original implementation."""

    def test_matches_original(self):
        N = 500
        config = DafConfig(num_nodes=N, fhn_I_base=0.5, fhn_a=0.7, fhn_b=0.8, fhn_tau=12.5)
        states = torch.randn(N, 8)

        expected = fhn_derivatives(states, config)
        out = torch.zeros(N, 8)
        fhn_derivatives_inplace(states, config, out)

        assert torch.allclose(expected, out, atol=1e-6), \
            f"Max diff: {(expected - out).abs().max()}"

    def test_unused_columns_zero(self):
        N = 100
        config = DafConfig(num_nodes=N)
        states = torch.randn(N, 8)
        out = torch.zeros(N, 8)
        fhn_derivatives_inplace(states, config, out)

        for col in [1, 2, 3, 5, 6, 7]:
            assert (out[:, col] == 0).all(), f"Column {col} should be zero"

    def test_no_new_allocations_on_gpu(self):
        """Verify no CUDA memory allocations during inplace call (GPU only)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        N = 10_000
        config = DafConfig(num_nodes=N)
        device = torch.device("cuda")
        states = torch.randn(N, 8, device=device)
        out = torch.zeros(N, 8, device=device)

        # Warmup
        fhn_derivatives_inplace(states, config, out)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        mem_before = torch.cuda.memory_allocated()
        fhn_derivatives_inplace(states, config, out)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        assert mem_after == mem_before, \
            f"Memory grew by {mem_after - mem_before} bytes — allocation detected"


class TestFhnCouplingInplace:
    """Verify inplace FHN coupling matches original implementation."""

    def test_matches_original(self):
        N = 500
        graph = SparseDafGraph.random_sparse(N, 30, torch.device("cpu"), seed=42)
        states = torch.randn(N, 8)
        K = 0.1

        expected = compute_fhn_coupling(states, graph, K)

        E = graph.num_edges
        out = torch.zeros(N)
        contrib = torch.empty(E)
        src_v = torch.empty(E)
        dst_v = torch.empty(E)
        compute_fhn_coupling_inplace(states, graph, K, out, contrib, src_v, dst_v)

        assert torch.allclose(expected, out, atol=1e-6), \
            f"Max diff: {(expected - out).abs().max()}"


class TestGraphSortByDst:
    """Verify edge sorting preserves graph semantics."""

    def test_sorted_dst_is_nondecreasing(self):
        graph = SparseDafGraph.random_sparse(1000, 30, torch.device("cpu"), seed=42)
        dst = graph.edge_index[1]
        diffs = dst[1:] - dst[:-1]
        assert (diffs >= 0).all(), "dst indices should be non-decreasing after sort"

    def test_coupling_unchanged_after_sort(self):
        """Coupling results should be identical regardless of edge order."""
        N = 500
        device = torch.device("cpu")
        states = torch.randn(N, 8)

        # Build graph WITHOUT sorting
        graph_unsorted = SparseDafGraph.random_sparse(N, 30, device, seed=42)
        # random_sparse now sorts by default, so we need to unsort
        perm = torch.randperm(graph_unsorted.num_edges)
        graph_unsorted.edge_index = graph_unsorted.edge_index[:, perm].contiguous()
        graph_unsorted.edge_attr = graph_unsorted.edge_attr[perm].contiguous()

        coupling_unsorted = compute_fhn_coupling(states, graph_unsorted, 0.1)

        # Sort it
        graph_unsorted.sort_by_dst()
        coupling_sorted = compute_fhn_coupling(states, graph_unsorted, 0.1)

        assert torch.allclose(coupling_unsorted, coupling_sorted, atol=1e-6)


class TestEngineOptimizedPath:
    """Verify optimized engine produces same dynamics as would the original functions."""

    def test_step_produces_valid_result(self):
        from snks.daf.engine import DafEngine
        config = DafConfig(num_nodes=1000, avg_degree=20, oscillator_model="fhn")
        engine = DafEngine(config, enable_learning=True)

        stim = torch.zeros(1000, device=engine.device)
        stim[:100] = 1.0
        engine.set_input(stim)

        result = engine.step(100)

        assert result.states.shape == (1000, 8)
        assert result.fired_history.shape == (100, 1000)
        assert not torch.isnan(result.states).any()
        assert not torch.isinf(result.states).any()

    def test_buffers_reallocated_after_prune(self):
        from snks.daf.engine import DafEngine
        config = DafConfig(num_nodes=200, avg_degree=10, structural_prune_threshold=0.5)
        engine = DafEngine(config, enable_learning=True)

        E_before = engine.graph.num_edges
        engine._structural_prune()
        E_after = engine.graph.num_edges

        assert engine._contrib_buf.shape[0] == E_after
        assert engine._src_v_buf.shape[0] == E_after
        assert engine._dst_v_buf.shape[0] == E_after


# --- Performance benchmark ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for perf test")
class TestPerformance:
    """GPU performance benchmarks."""

    def test_steps_per_sec_small(self):
        """Small config (10K nodes): sanity check GPU throughput."""
        from snks.daf.bench import run_bench
        config = DafConfig(num_nodes=10_000, avg_degree=30, device="cuda")
        result = run_bench(config, n_steps=200, n_cycles=5, enable_learning=False)
        print(f"\nSmall: {result['steps_sec']:,.0f} steps/sec")
        assert result["steps_sec"] >= 2_000, \
            f"Expected >=2K steps/sec, got {result['steps_sec']:,.0f}"

    def test_steps_per_sec_default(self):
        """Default config (50K nodes): target >= 10K steps/sec on GPU."""
        from snks.daf.bench import run_bench
        config = DafConfig(num_nodes=50_000, avg_degree=50, device="cuda")
        result = run_bench(config, n_steps=200, n_cycles=5, enable_learning=False)
        print(f"\nDefault: {result['steps_sec']:,.0f} steps/sec")
        # Soft gate: 5K on laptop GPU (RTX 3070 Ti), 10K on desktop (RTX 3090)
        assert result["steps_sec"] >= 5_000, \
            f"Expected >=5K steps/sec, got {result['steps_sec']:,.0f}"
