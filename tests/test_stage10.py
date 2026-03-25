"""Unit tests for Stage 10: MetaEmbedder + HierarchicalConfig + monitor meta_pe.

All tests are fast (< 5 sec total) and CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn.functional as F

from snks.dcam.hac import HACEngine
from snks.daf.hac_prediction import HACPredictionEngine
from snks.daf.types import (
    HierarchicalConfig,
    HACPredictionConfig,
    MetacogConfig,
)
from snks.metacog.monitor import MetacogMonitor
from snks.sks.meta_embedder import MetaEmbedder


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

HAC_DIM = 64


def make_hac() -> HACEngine:
    return HACEngine(dim=HAC_DIM, device=torch.device("cpu"))


def make_meta_embedder(decay: float = 0.8) -> MetaEmbedder:
    hac = make_hac()
    cfg = HierarchicalConfig(meta_decay=decay)
    return MetaEmbedder(hac=hac, config=cfg)


def rand_embed(dim: int = HAC_DIM) -> torch.Tensor:
    v = torch.randn(dim)
    return v / v.norm().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Minimal GWSState mock (no real DAF needed)
# ---------------------------------------------------------------------------

@dataclass
class _GWS:
    winner_id: int = 0
    winner_nodes: list = field(default_factory=lambda: [0, 1, 2])
    winner_size: int = 3
    dominance: float = 0.8
    n_clusters: int = 1


# ---------------------------------------------------------------------------
# Minimal CycleResult mock
# ---------------------------------------------------------------------------

@dataclass
class _CycleResult:
    mean_prediction_error: float = 0.1
    winner_pe: float = 0.0
    meta_pe: float = 0.0


# ---------------------------------------------------------------------------
# TestMetaEmbedder
# ---------------------------------------------------------------------------

class TestMetaEmbedder:

    def test_update_returns_unit_norm(self) -> None:
        """update() with random embeddings → result has norm ≈ 1.0."""
        me = make_meta_embedder()
        embeddings = {0: rand_embed(), 1: rand_embed(), 2: rand_embed()}
        result = me.update(embeddings)
        assert result is not None
        norm = result.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_update_empty_returns_none_then_previous(self) -> None:
        """Empty dict on first call → None; after real update, empty dict → returns previous meta_embed."""
        me = make_meta_embedder()

        # First call with empty dict must return None
        result_empty = me.update({})
        assert result_empty is None

        # Now do a real update
        embeddings = {0: rand_embed()}
        result_real = me.update(embeddings)
        assert result_real is not None

        # Another empty dict call should return the previously computed embedding
        result_empty2 = me.update({})
        assert result_empty2 is not None
        assert torch.allclose(result_empty2, result_real)

    def test_ewa_decay_towards_new(self) -> None:
        """After many updates with a constant input, meta_embed converges to that input (cosine_sim > 0.99)."""
        # Use low decay so convergence is fast
        me = make_meta_embedder(decay=0.5)
        target = rand_embed()
        embeddings = {0: target.clone()}

        for _ in range(40):
            me.update(embeddings)

        meta = me.get_meta_embed()
        assert meta is not None
        cos_sim = F.cosine_similarity(meta.unsqueeze(0), target.unsqueeze(0)).item()
        assert cos_sim > 0.99, f"Expected cosine_sim > 0.99, got {cos_sim}"

    def test_reset_clears_state(self) -> None:
        """After update, reset(), get_meta_embed() → None."""
        me = make_meta_embedder()
        me.update({0: rand_embed()})
        assert me.get_meta_embed() is not None

        me.reset()
        assert me.get_meta_embed() is None

    def test_different_classes_diverge(self) -> None:
        """Meta-embed after group A vs group B of embeddings should have cosine_sim < 0.7."""
        # Group A: fixed random vectors from one distribution
        torch.manual_seed(42)
        group_a = {i: rand_embed() for i in range(5)}

        # Group B: fixed random vectors in a very different direction
        torch.manual_seed(99)
        group_b_raw = {i: rand_embed() for i in range(5)}
        # Negate to push further apart
        group_b = {k: -v for k, v in group_b_raw.items()}

        me_a = make_meta_embedder(decay=0.5)
        me_b = make_meta_embedder(decay=0.5)

        for _ in range(10):
            me_a.update(group_a)
            me_b.update(group_b)

        meta_a = me_a.get_meta_embed()
        meta_b = me_b.get_meta_embed()
        assert meta_a is not None
        assert meta_b is not None

        cos_sim = F.cosine_similarity(meta_a.unsqueeze(0), meta_b.unsqueeze(0)).item()
        assert cos_sim < 0.7, f"Expected divergent embeddings (cosine_sim < 0.7), got {cos_sim}"


# ---------------------------------------------------------------------------
# TestMonitorMetaPe
# ---------------------------------------------------------------------------

class TestMonitorMetaPe:

    def _make_monitor(self, delta: float = 0.0) -> MetacogMonitor:
        cfg = MetacogConfig(
            alpha=1 / 3,
            beta=1 / 3,
            gamma=1 / 3,
            delta=delta,
        )
        return MetacogMonitor(config=cfg)

    def _run_two_cycles(self, monitor: MetacogMonitor, meta_pe: float) -> float:
        """Run two monitor updates (to build prev_winner) and return confidence."""
        gws = _GWS()
        # First cycle: establishes prev_winner (stability=0)
        cr1 = _CycleResult(mean_prediction_error=0.1, winner_pe=0.0, meta_pe=0.0)
        monitor.update(gws, cr1)
        # Second cycle: uses meta_pe
        cr2 = _CycleResult(mean_prediction_error=0.1, winner_pe=0.0, meta_pe=meta_pe)
        state = monitor.update(gws, cr2)
        return state.confidence

    def test_meta_pe_zero_backward_compatible(self) -> None:
        """When meta_pe=0.0, confidence is same as Stage 9 formula (delta=0 normalises by alpha+beta+gamma)."""
        # Monitor with delta > 0 but meta_pe=0 → warmup guard kicks in, norm = alpha+beta+gamma
        monitor_d = self._make_monitor(delta=0.25)
        conf_with_delta = self._run_two_cycles(monitor_d, meta_pe=0.0)

        # Monitor with delta=0 (pure Stage 9 formula)
        monitor_0 = self._make_monitor(delta=0.0)
        conf_without_delta = self._run_two_cycles(monitor_0, meta_pe=0.0)

        assert abs(conf_with_delta - conf_without_delta) < 1e-6, (
            f"Warmup guard failed: delta monitor confidence {conf_with_delta} "
            f"!= delta=0 confidence {conf_without_delta}"
        )

    def test_meta_pe_included_when_positive(self) -> None:
        """MetacogConfig with delta=0.25, meta_pe=0.5 → confidence differs from meta_pe=0.0."""
        monitor_with_pe = self._make_monitor(delta=0.25)
        conf_with_pe = self._run_two_cycles(monitor_with_pe, meta_pe=0.5)

        monitor_no_pe = self._make_monitor(delta=0.25)
        conf_no_pe = self._run_two_cycles(monitor_no_pe, meta_pe=0.0)

        assert conf_with_pe != conf_no_pe, (
            "Expected confidence to change when meta_pe > 0.0 with delta=0.25, "
            f"got conf_with_pe={conf_with_pe}, conf_no_pe={conf_no_pe}"
        )

    def test_meta_pe_warmup_guard(self) -> None:
        """If meta_pe=0.0, pe_L2_term should be 0 regardless of delta."""
        # High delta=1.0 but meta_pe=0 → should NOT affect confidence vs delta=0
        monitor_high_delta = self._make_monitor(delta=1.0)
        conf_high_delta = self._run_two_cycles(monitor_high_delta, meta_pe=0.0)

        monitor_zero_delta = self._make_monitor(delta=0.0)
        conf_zero_delta = self._run_two_cycles(monitor_zero_delta, meta_pe=0.0)

        assert abs(conf_high_delta - conf_zero_delta) < 1e-6, (
            f"Warmup guard broken: delta=1.0 changed confidence even with meta_pe=0.0 "
            f"(got {conf_high_delta} vs {conf_zero_delta})"
        )


# ---------------------------------------------------------------------------
# TestHACPredictionReset
# ---------------------------------------------------------------------------

class TestHACPredictionReset:

    def test_reset_clears_memory(self) -> None:
        """After observe() a few times, reset() → predict_next() returns None."""
        hac = make_hac()
        cfg = HACPredictionConfig(memory_decay=0.95, enabled=True)
        engine = HACPredictionEngine(hac=hac, config=cfg)

        # Observe a few cycles to build memory
        for i in range(4):
            embeddings = {0: rand_embed(), 1: rand_embed()}
            engine.observe(embeddings)

        # Memory should be built; predict_next should return something
        test_embeddings = {0: rand_embed(), 1: rand_embed()}
        # (We don't assert non-None here because first obs builds prev but no pairs yet)

        engine.reset()

        # After reset, predict_next must return None (memory cleared)
        result = engine.predict_next(test_embeddings)
        assert result is None, f"Expected None after reset, got {result}"

    def test_reset_clears_prev_embeddings(self) -> None:
        """After reset, the first observe() should not produce any pairs (no prev state)."""
        hac = make_hac()
        cfg = HACPredictionConfig(memory_decay=0.95, enabled=True)
        engine = HACPredictionEngine(hac=hac, config=cfg)

        # Build memory
        for _ in range(3):
            engine.observe({0: rand_embed()})

        engine.reset()

        # First observe after reset: _prev_embeddings is None → no pairs → memory stays None
        engine.observe({0: rand_embed()})
        result = engine.predict_next({0: rand_embed()})
        assert result is None, (
            "After reset + single observe, memory should still be None "
            f"(no pairs to bind), got {result}"
        )
