"""Tests for Stage 19: Zonal DAF architecture."""

import pytest
import torch

from snks.daf.engine import DafEngine
from snks.daf.graph import SparseDafGraph
from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, ZoneConfig
from snks.encoder.encoder import VisualEncoder
from snks.encoder.text_encoder import TextEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zones_config(with_convergence: bool = False) -> dict[str, ZoneConfig]:
    if with_convergence:
        return {
            "visual": ZoneConfig(start=0, size=200),
            "linguistic": ZoneConfig(start=200, size=150),
            "convergence": ZoneConfig(start=350, size=50),
        }
    return {
        "visual": ZoneConfig(start=0, size=250),
        "linguistic": ZoneConfig(start=250, size=150),
    }


def _make_daf_config(zones: dict[str, ZoneConfig] | None = None) -> DafConfig:
    total = sum(z.size for z in zones.values()) if zones else 400
    return DafConfig(
        num_nodes=total,
        avg_degree=10,
        inter_zone_avg_degree=3,
        zones=zones,
        device="cpu",
        disable_csr=True,
        oscillator_model="fhn",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestZoneConfig:

    def test_backward_compat_no_zones(self):
        """DafConfig without zones works as before."""
        cfg = _make_daf_config(zones=None)
        engine = DafEngine(cfg)
        assert engine.zones is None
        assert engine.states.shape == (400, 8)
        # set_input still works
        currents = torch.zeros(400, 8)
        engine.set_input(currents)
        result = engine.step(10)
        assert result.states.shape == (400, 8)

    def test_zones_stored_on_engine(self):
        zones = _make_zones_config()
        cfg = _make_daf_config(zones)
        engine = DafEngine(cfg)
        assert engine.zones is not None
        assert "visual" in engine.zones
        assert "linguistic" in engine.zones


class TestZonalGraph:

    def test_graph_has_edges(self):
        zones = _make_zones_config()
        total = sum(z.size for z in zones.values())
        graph = SparseDafGraph.random_sparse_zonal(
            total, zones, intra_degree=10, inter_degree=3,
            device=torch.device("cpu"), seed=42,
        )
        assert graph.num_edges > 0
        assert graph.edge_index.shape[0] == 2

    def test_intra_zone_edges_within_bounds(self):
        zones = _make_zones_config()
        total = sum(z.size for z in zones.values())
        graph = SparseDafGraph.random_sparse_zonal(
            total, zones, intra_degree=10, inter_degree=0,
            device=torch.device("cpu"), seed=42,
        )
        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        # With inter_degree=0, all edges should be intra-zone
        for z in zones.values():
            mask_src = (src >= z.start) & (src < z.start + z.size)
            mask_dst = (dst >= z.start) & (dst < z.start + z.size)
            # Edges originating from this zone should also end in this zone
            intra = mask_src & mask_dst
            from_zone = mask_src.sum().item()
            in_zone = intra.sum().item()
            assert from_zone == in_zone, f"Zone {z}: {from_zone} edges from zone but {in_zone} intra"

    def test_inter_zone_edges_exist(self):
        zones = _make_zones_config()
        total = sum(z.size for z in zones.values())
        graph = SparseDafGraph.random_sparse_zonal(
            total, zones, intra_degree=10, inter_degree=5,
            device=torch.device("cpu"), seed=42,
        )
        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        # Find edges between visual and linguistic zones
        vis = zones["visual"]
        ling = zones["linguistic"]
        vis_src = (src >= vis.start) & (src < vis.start + vis.size)
        ling_dst = (dst >= ling.start) & (dst < ling.start + ling.size)
        cross = (vis_src & ling_dst).sum().item()
        assert cross > 0, "No inter-zone edges found between visual and linguistic"

    def test_convergence_zone(self):
        zones = _make_zones_config(with_convergence=True)
        total = sum(z.size for z in zones.values())
        graph = SparseDafGraph.random_sparse_zonal(
            total, zones, intra_degree=10, inter_degree=3,
            device=torch.device("cpu"), seed=42,
        )
        assert graph.num_nodes == total
        assert graph.num_edges > 0


class TestSetInputZone:

    def test_zone_injection_only_affects_zone(self):
        zones = _make_zones_config()
        cfg = _make_daf_config(zones)
        engine = DafEngine(cfg)
        engine._external_currents.zero_()

        vis_zone = zones["visual"]
        currents = torch.ones(vis_zone.size, 8)
        engine.set_input_zone(currents, "visual")

        # Visual zone should have currents
        assert engine._external_currents[:vis_zone.size, 0].sum() > 0
        # Linguistic zone should be zero
        ling = zones["linguistic"]
        assert engine._external_currents[ling.start:ling.start + ling.size].sum() == 0

    def test_zone_injection_raises_without_zones(self):
        cfg = _make_daf_config(zones=None)
        engine = DafEngine(cfg)
        with pytest.raises(ValueError, match="zonal"):
            engine.set_input_zone(torch.zeros(100), "visual")

    def test_multiple_zones_independent(self):
        zones = _make_zones_config()
        cfg = _make_daf_config(zones)
        engine = DafEngine(cfg)
        engine._external_currents.zero_()

        vis = zones["visual"]
        ling = zones["linguistic"]

        engine.set_input_zone(torch.ones(vis.size) * 2.0, "visual")
        engine.set_input_zone(torch.ones(ling.size) * 3.0, "linguistic")

        # Visual zone has strength 2
        assert engine._external_currents[vis.start, 0].item() == pytest.approx(2.0)
        # Linguistic zone has strength 3
        assert engine._external_currents[ling.start, 0].item() == pytest.approx(3.0)


class TestSdrToCurrentsZone:

    def test_visual_encoder_zone_shape(self):
        enc_cfg = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
        encoder = VisualEncoder(enc_cfg)
        sdr = torch.zeros(512)
        sdr[:20] = 1.0

        zone = ZoneConfig(start=0, size=200)
        currents = encoder.sdr_to_currents(sdr, 1000, zone=zone)
        assert currents.shape == (200, 8)

    def test_visual_encoder_no_zone_backward_compat(self):
        enc_cfg = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
        encoder = VisualEncoder(enc_cfg)
        sdr = torch.zeros(512)
        sdr[:20] = 1.0

        currents = encoder.sdr_to_currents(sdr, 1000)
        assert currents.shape == (1000, 8)

    def test_text_encoder_zone_shape(self):
        enc_cfg = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
        # TextEncoder requires sentence-transformers; test sdr_to_currents directly
        from snks.encoder.text_encoder import TextEncoder
        te = TextEncoder(enc_cfg, device="cpu")
        sdr = torch.zeros(512)
        sdr[:20] = 1.0

        zone = ZoneConfig(start=200, size=150)
        currents = te.sdr_to_currents(sdr, 1000, zone=zone)
        assert currents.shape == (150, 8)


class TestPipelineZonal:

    def test_pipeline_no_zones_unchanged(self):
        """Pipeline without zones works exactly as before."""
        cfg = PipelineConfig(
            daf=DafConfig(num_nodes=500, avg_degree=10, device="cpu", disable_csr=True),
            encoder=EncoderConfig(sdr_size=512, sdr_sparsity=0.04),
            steps_per_cycle=10,
            device="cpu",
        )
        from snks.pipeline.runner import Pipeline
        p = Pipeline(cfg)
        img = torch.rand(64, 64)
        result = p.perception_cycle(image=img)
        assert result.n_sks >= 0

    def test_pipeline_zonal_image_only(self):
        """Pipeline with zones handles image-only input."""
        zones = {
            "visual": ZoneConfig(start=0, size=300),
            "linguistic": ZoneConfig(start=300, size=200),
        }
        cfg = PipelineConfig(
            daf=DafConfig(
                num_nodes=500, avg_degree=10, inter_zone_avg_degree=3,
                zones=zones, device="cpu", disable_csr=True,
            ),
            encoder=EncoderConfig(sdr_size=512, sdr_sparsity=0.04),
            steps_per_cycle=10,
            device="cpu",
        )
        from snks.pipeline.runner import Pipeline
        p = Pipeline(cfg)
        img = torch.rand(64, 64)
        result = p.perception_cycle(image=img)
        assert result.n_sks >= 0

    def test_pipeline_zone_no_averaging(self):
        """When zones are configured, currents are NOT averaged."""
        zones = {
            "visual": ZoneConfig(start=0, size=300),
            "linguistic": ZoneConfig(start=300, size=200),
        }
        cfg = PipelineConfig(
            daf=DafConfig(
                num_nodes=500, avg_degree=10, inter_zone_avg_degree=3,
                zones=zones, device="cpu", disable_csr=True,
            ),
            encoder=EncoderConfig(sdr_size=512, sdr_sparsity=0.04, sdr_current_strength=1.0),
            steps_per_cycle=10,
            device="cpu",
        )
        from snks.pipeline.runner import Pipeline
        p = Pipeline(cfg)

        # Verify that in zonal mode, sdr_current_strength is not halved
        # (in old averaging mode, (img + txt) / 2 would halve to 0.5)
        sdr = torch.zeros(512)
        sdr[:50] = 1.0  # plenty of active bits
        vis_zone = zones["visual"]
        currents = p.encoder.sdr_to_currents(sdr, 500, zone=vis_zone)
        max_val = currents[:, 0].max().item()
        assert max_val == pytest.approx(1.0), f"Expected full strength 1.0, got {max_val}"


class TestPipelineCheckpoint:

    def test_save_load_roundtrip(self, tmp_path):
        """Pipeline checkpoint preserves DAF state and GroundingMap."""
        zones = {
            "visual": ZoneConfig(start=0, size=300),
            "linguistic": ZoneConfig(start=300, size=200),
        }
        cfg = PipelineConfig(
            daf=DafConfig(
                num_nodes=500, avg_degree=10, zones=zones,
                device="cpu", disable_csr=True,
            ),
            encoder=EncoderConfig(sdr_size=512, sdr_sparsity=0.04, image_size=32),
            steps_per_cycle=10,
            device="cpu",
            priming_strength=0.3,
        )
        from snks.pipeline.runner import Pipeline

        p = Pipeline(cfg)
        # Train a bit
        img = torch.rand(32, 32)
        p.perception_cycle(image=img, text="test object")
        p.perception_cycle(image=img, text="test object")

        # Verify grounding map populated
        assert len(p.grounding_map._word_to_visual_sdr) > 0

        # Save
        ckpt = str(tmp_path / "ckpt")
        p.save_checkpoint(ckpt)

        # Load into fresh pipeline
        p2 = Pipeline(cfg)
        assert len(p2.grounding_map._word_to_visual_sdr) == 0
        p2.load_checkpoint(ckpt)

        # DAF state restored
        assert p2.engine.step_count == p.engine.step_count
        assert torch.allclose(
            p2.engine.graph.edge_attr, p.engine.graph.edge_attr
        )

        # GroundingMap restored
        vis_orig = p.grounding_map.word_to_visual_sdr("test object")
        vis_loaded = p2.grounding_map.word_to_visual_sdr("test object")
        assert vis_loaded is not None
        assert torch.allclose(vis_loaded, vis_orig)
