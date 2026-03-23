"""Tests for DCAM (Dual-Code Associative Memory) — Stage 4."""

from __future__ import annotations

import torch
import pytest

from snks.dcam.hac import HACEngine
from snks.dcam.lsh import LSHIndex
from snks.dcam.ssg import StructuredSparseGraph
from snks.dcam.episodic import EpisodicBuffer, Episode
from snks.dcam.consolidation import Consolidation, ConsolidationReport
from snks.dcam.world_model import DcamWorldModel
from snks.daf.types import DcamConfig


# ---------------------------------------------------------------------------
# TestHACEngine
# ---------------------------------------------------------------------------

class TestHACEngine:
    """HACEngine: bind/unbind, bundle, permute, encode_scalar, batch ops."""

    @pytest.fixture
    def hac(self) -> HACEngine:
        return HACEngine(dim=256, device=torch.device("cpu"))

    def test_bind_unbind_roundtrip(self, hac: HACEngine) -> None:
        """Gate 1: unbind(a, bind(a, b)) ≈ b with cosine > 0.9."""
        a = hac.random_vector()
        b = hac.random_vector()
        bound = hac.bind(a, b)
        recovered = hac.unbind(a, bound)
        sim = hac.similarity(recovered, b)
        assert sim > 0.9, f"HAC fidelity {sim:.4f} <= 0.9"

    def test_bundle_similar_to_components(self, hac: HACEngine) -> None:
        """Bundle of {a, b} should be similar to both a and b."""
        a = hac.random_vector()
        b = hac.random_vector()
        bundled = hac.bundle([a, b])
        assert hac.similarity(bundled, a) > 0.3
        assert hac.similarity(bundled, b) > 0.3

    def test_permute_shifts(self, hac: HACEngine) -> None:
        """permute(v, k) differs from v but permute(permute(v, k), -k) == v."""
        v = hac.random_vector()
        shifted = hac.permute(v, 3)
        assert hac.similarity(v, shifted) < 0.9  # shifted is different
        restored = hac.permute(shifted, -3)
        assert torch.allclose(v, restored, atol=1e-6)

    def test_encode_scalar_monotonic(self, hac: HACEngine) -> None:
        """sim(encode(0.0), encode(x)) decreases as x grows."""
        e0 = hac.encode_scalar(0.0)
        e1 = hac.encode_scalar(0.1)
        e5 = hac.encode_scalar(0.5)
        e9 = hac.encode_scalar(0.9)
        s1 = hac.similarity(e0, e1)
        s5 = hac.similarity(e0, e5)
        s9 = hac.similarity(e0, e9)
        assert s1 > s5 > s9, f"Not monotonic: {s1:.3f}, {s5:.3f}, {s9:.3f}"

    def test_batch_bind_shape(self, hac: HACEngine) -> None:
        A = torch.randn(8, hac.dim)
        B = torch.randn(8, hac.dim)
        result = hac.batch_bind(A, B)
        assert result.shape == (8, hac.dim)

    def test_batch_similarity_shape(self, hac: HACEngine) -> None:
        query = hac.random_vector()
        keys = torch.randn(20, hac.dim)
        result = hac.batch_similarity(query, keys)
        assert result.shape == (20,)

    def test_random_vectors_orthogonal(self, hac: HACEngine) -> None:
        """Two random vectors should have near-zero similarity."""
        a = hac.random_vector()
        b = hac.random_vector()
        assert abs(hac.similarity(a, b)) < 0.3


# ---------------------------------------------------------------------------
# TestLSHIndex
# ---------------------------------------------------------------------------

class TestLSHIndex:
    """LSHIndex: SimHash insert/query/remove, fallback."""

    @pytest.fixture
    def lsh(self) -> LSHIndex:
        return LSHIndex(dim=64, n_tables=8, n_bits=8, device=torch.device("cpu"))

    def test_insert_and_query_finds_self(self, lsh: LSHIndex) -> None:
        v = torch.randn(64)
        v = v / v.norm()
        lsh.insert(v, value=0)
        results = lsh.query(v, top_k=1)
        assert len(results) >= 1
        assert results[0][0] == 0
        assert results[0][1] > 0.99

    def test_top_k_ordering(self, lsh: LSHIndex) -> None:
        """Most similar vectors should rank higher."""
        base = torch.randn(64)
        base = base / base.norm()
        # Insert base and a similar vector
        lsh.insert(base, value=0)
        similar = base + 0.1 * torch.randn(64)
        similar = similar / similar.norm()
        lsh.insert(similar, value=1)
        # Insert a random vector
        rand_v = torch.randn(64)
        rand_v = rand_v / rand_v.norm()
        lsh.insert(rand_v, value=2)
        results = lsh.query(base, top_k=3)
        ids = [r[0] for r in results]
        # base should be first, similar second
        assert ids[0] == 0
        if len(ids) > 1 and 1 in ids and 2 in ids:
            assert ids.index(1) < ids.index(2)

    def test_remove(self, lsh: LSHIndex) -> None:
        v = torch.randn(64)
        v = v / v.norm()
        lsh.insert(v, value=42)
        lsh.remove(42)
        results = lsh.query(v, top_k=1)
        ids = [r[0] for r in results]
        assert 42 not in ids

    def test_fallback_empty_bucket(self, lsh: LSHIndex) -> None:
        """When query doesn't match any bucket, fallback scans all vectors."""
        v1 = torch.randn(64)
        v1 = v1 / v1.norm()
        lsh.insert(v1, value=0)
        # Query with very different vector — may not share any bucket
        v2 = -v1  # opposite direction
        results = lsh.query(v2, top_k=1)
        # Fallback should still find v1 as the only stored vector
        assert len(results) == 1
        assert results[0][0] == 0


# ---------------------------------------------------------------------------
# TestStructuredSparseGraph
# ---------------------------------------------------------------------------

class TestStructuredSparseGraph:
    """SSG: multi-layer graph CRUD + prune."""

    @pytest.fixture
    def graph(self) -> StructuredSparseGraph:
        return StructuredSparseGraph()

    def test_add_and_get_neighbors(self, graph: StructuredSparseGraph) -> None:
        graph.add_edge(0, 1, "structural", weight=0.5)
        graph.add_edge(0, 2, "structural", weight=0.8)
        neighbors = graph.get_neighbors(0, "structural")
        assert len(neighbors) == 2
        nmap = dict(neighbors)
        assert nmap[1] == pytest.approx(0.5)
        assert nmap[2] == pytest.approx(0.8)

    def test_update_edge(self, graph: StructuredSparseGraph) -> None:
        graph.add_edge(0, 1, "causal", weight=0.3)
        graph.update_edge(0, 1, "causal", delta=0.2)
        neighbors = dict(graph.get_neighbors(0, "causal"))
        assert neighbors[1] == pytest.approx(0.5)

    def test_prune_removes_weak_edges(self, graph: StructuredSparseGraph) -> None:
        graph.add_edge(0, 1, "temporal", weight=0.001)
        graph.add_edge(0, 2, "temporal", weight=0.5)
        removed = graph.prune(threshold=0.01, layer="temporal")
        assert removed == 1
        neighbors = graph.get_neighbors(0, "temporal")
        assert len(neighbors) == 1
        assert neighbors[0][0] == 2

    def test_layers_independent(self, graph: StructuredSparseGraph) -> None:
        """Edges in different layers don't interfere."""
        graph.add_edge(0, 1, "structural", weight=1.0)
        graph.add_edge(0, 1, "causal", weight=2.0)
        s = dict(graph.get_neighbors(0, "structural"))
        c = dict(graph.get_neighbors(0, "causal"))
        assert s[1] == pytest.approx(1.0)
        assert c[1] == pytest.approx(2.0)

    def test_unknown_layer_raises(self, graph: StructuredSparseGraph) -> None:
        with pytest.raises(ValueError, match="Unknown layer"):
            graph.add_edge(0, 1, "invalid")


# ---------------------------------------------------------------------------
# TestEpisodicBuffer
# ---------------------------------------------------------------------------

class TestEpisodicBuffer:
    """EpisodicBuffer: store, eviction, retrieval."""

    @pytest.fixture
    def buf(self) -> EpisodicBuffer:
        config = DcamConfig(episodic_capacity=5)
        return EpisodicBuffer(config, device=torch.device("cpu"))

    def test_store_returns_id(self, buf: EpisodicBuffer) -> None:
        ctx = torch.randn(64)
        eid = buf.store(active_nodes={0: (0.1, 0.5)}, context_hac=ctx, importance=0.5)
        assert isinstance(eid, int)
        assert len(buf) == 1

    def test_capacity_eviction(self, buf: EpisodicBuffer) -> None:
        """Least important episode is evicted when capacity reached."""
        # Fill buffer with importance = id (so id=0 has lowest)
        for i in range(5):
            buf.store({}, torch.randn(64), importance=float(i))
        assert len(buf) == 5
        # Add one more with high importance
        buf.store({}, torch.randn(64), importance=10.0)
        assert len(buf) == 5
        # Episode 0 (importance=0.0) should have been evicted
        assert buf.get_episode(0) is None
        # Episode 4 (importance=4.0) should still be there
        assert buf.get_episode(4) is not None

    def test_get_episode_by_id(self, buf: EpisodicBuffer) -> None:
        ctx = torch.randn(64)
        eid = buf.store({1: (0.0, 1.0)}, ctx, importance=0.8)
        ep = buf.get_episode(eid)
        assert ep is not None
        assert ep.importance == 0.8
        assert torch.allclose(ep.context_hac, ctx)

    def test_len(self, buf: EpisodicBuffer) -> None:
        assert len(buf) == 0
        buf.store({}, torch.randn(64), importance=0.5)
        buf.store({}, torch.randn(64), importance=0.5)
        assert len(buf) == 2


# ---------------------------------------------------------------------------
# TestConsolidation
# ---------------------------------------------------------------------------

class TestConsolidation:
    """Consolidation: STC, co-activation, causal extraction."""

    @pytest.fixture
    def config(self) -> DcamConfig:
        return DcamConfig(
            episodic_capacity=100,
            consolidation_stc_threshold=0.5,
            consolidation_coact_min=3,
        )

    def test_consolidate_returns_report(self, config: DcamConfig) -> None:
        buf = EpisodicBuffer(config, torch.device("cpu"))
        graph = StructuredSparseGraph()
        cons = Consolidation(config)
        buf.store({0: (0.1, 1.0), 1: (0.2, 1.0)}, torch.randn(64), importance=0.8)
        report = cons.consolidate(buf, graph)
        assert isinstance(report, ConsolidationReport)
        assert report.n_stc_processed >= 0
        assert report.n_coactivation_pairs >= 0
        assert report.n_causal_edges >= 0
        assert report.n_pruned >= 0

    def test_coactivation_creates_edge(self, config: DcamConfig) -> None:
        """3+ co-activations of (0, 1) → structural edge."""
        buf = EpisodicBuffer(config, torch.device("cpu"))
        graph = StructuredSparseGraph()
        cons = Consolidation(config)
        for _ in range(4):
            buf.store({0: (0.0, 1.0), 1: (0.0, 1.0)}, torch.randn(64), importance=0.3)
        report = cons.consolidate(buf, graph)
        assert report.n_coactivation_pairs >= 1
        neighbors = dict(graph.get_neighbors(0, "structural"))
        assert 1 in neighbors

    def test_stc_tags_high_importance(self, config: DcamConfig) -> None:
        """Episodes with importance >= threshold → STC strengthening."""
        buf = EpisodicBuffer(config, torch.device("cpu"))
        graph = StructuredSparseGraph()
        cons = Consolidation(config)
        buf.store({0: (0.0, 1.0), 1: (0.0, 1.0)}, torch.randn(64), importance=0.9)
        report = cons.consolidate(buf, graph)
        assert report.n_stc_processed >= 1

    def test_report_fields_nonneg(self, config: DcamConfig) -> None:
        """All report fields are non-negative."""
        buf = EpisodicBuffer(config, torch.device("cpu"))
        graph = StructuredSparseGraph()
        cons = Consolidation(config)
        report = cons.consolidate(buf, graph)
        assert report.n_stc_processed >= 0
        assert report.n_coactivation_pairs >= 0
        assert report.n_causal_edges >= 0
        assert report.n_pruned >= 0


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """Save/load roundtrip via safetensors + JSON."""

    @pytest.fixture
    def model(self) -> DcamWorldModel:
        config = DcamConfig(
            hac_dim=64, lsh_n_tables=4, lsh_n_bits=4, episodic_capacity=50
        )
        return DcamWorldModel(config, device=torch.device("cpu"))

    def test_save_creates_files(self, model: DcamWorldModel, tmp_path) -> None:
        model.store_episode({0: (0.1, 1.0)}, torch.randn(64), importance=0.5)
        path = str(tmp_path / "ckpt")
        model.save(path)
        assert (tmp_path / "ckpt.safetensors").exists()
        assert (tmp_path / "ckpt.json").exists()

    def test_load_restores_scalar_base(self, model: DcamWorldModel, tmp_path) -> None:
        original_base = model.hac._scalar_base.clone()
        path = str(tmp_path / "ckpt")
        model.save(path)
        model.hac._scalar_base = torch.randn(64)
        model.load(path)
        assert torch.allclose(model.hac._scalar_base, original_base, atol=1e-6)

    def test_load_restores_projections(self, model: DcamWorldModel, tmp_path) -> None:
        original_proj = model.lsh.projections.clone()
        path = str(tmp_path / "ckpt")
        model.save(path)
        model.lsh.projections = torch.randn_like(model.lsh.projections)
        model.load(path)
        assert torch.allclose(model.lsh.projections, original_proj, atol=1e-6)

    def test_roundtrip_query_accuracy(self, model: DcamWorldModel, tmp_path) -> None:
        """Gate 2: save → load → query accuracy Δ ≤ 1%."""
        N = 30
        vectors = [torch.randn(64) for _ in range(N)]
        vectors = [v / v.norm() for v in vectors]
        ids = []
        for i, v in enumerate(vectors):
            eid = model.store_episode({i: (0.0, 1.0)}, v, importance=0.5)
            ids.append(eid)

        def measure_acc(m: DcamWorldModel) -> float:
            hits = 0
            for i, v in enumerate(vectors):
                results = m.query_similar(v, top_k=1)
                if results and results[0][0] == ids[i]:
                    hits += 1
            return hits / N

        acc_before = measure_acc(model)
        path = str(tmp_path / "ckpt")
        model.save(path)
        model2 = DcamWorldModel(model.config, torch.device("cpu"))
        model2.load(path)
        acc_after = measure_acc(model2)
        delta = abs(acc_before - acc_after)
        assert delta <= 0.01, f"Persistence Δ = {delta:.4f} > 0.01"

    def test_load_restores_episode_count(self, model: DcamWorldModel, tmp_path) -> None:
        for i in range(5):
            model.store_episode({i: (0.0, 1.0)}, torch.randn(64), importance=0.5)
        path = str(tmp_path / "ckpt")
        model.save(path)
        model2 = DcamWorldModel(model.config, torch.device("cpu"))
        model2.load(path)
        assert len(model2.buffer) == 5


# ---------------------------------------------------------------------------
# TestDcamWorldModel
# ---------------------------------------------------------------------------

class TestDcamWorldModel:
    """DcamWorldModel facade integration."""

    @pytest.fixture
    def model(self) -> DcamWorldModel:
        config = DcamConfig(
            hac_dim=64, lsh_n_tables=4, lsh_n_bits=4, episodic_capacity=50
        )
        return DcamWorldModel(config, device=torch.device("cpu"))

    def test_init(self, model: DcamWorldModel) -> None:
        assert model.hac.dim == 64
        assert len(model.buffer) == 0

    def test_store_and_query(self, model: DcamWorldModel) -> None:
        v = torch.randn(64)
        v = v / v.norm()
        eid = model.store_episode({0: (0.1, 1.0)}, v, importance=0.8)
        results = model.query_similar(v, top_k=1)
        assert len(results) >= 1
        assert results[0][0] == eid

    def test_consolidate_returns_report(self, model: DcamWorldModel) -> None:
        model.store_episode({0: (0.0, 1.0), 1: (0.0, 1.0)}, torch.randn(64), importance=0.9)
        report = model.consolidate()
        assert isinstance(report, ConsolidationReport)

    def test_save_load_roundtrip(self, model: DcamWorldModel, tmp_path) -> None:
        """Integration: store → save → load → verify."""
        v = torch.randn(64)
        v = v / v.norm()
        eid = model.store_episode({0: (0.0, 1.0)}, v, importance=0.7)
        path = str(tmp_path / "test_ckpt")
        model.save(path)
        model2 = DcamWorldModel(model.config, torch.device("cpu"))
        model2.load(path)
        assert len(model2.buffer) == 1
        results = model2.query_similar(v, top_k=1)
        assert results[0][0] == eid
