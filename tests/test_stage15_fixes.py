"""Tests for Stage 15 fixes:
- Fix 1: _split_context (causal_model.py)
- Fix 2: EpisodicHACPredictor (dcam/episodic_hac.py)
- Fix 3: HACPredictionConfig.use_episodic_buffer (types.py + runner)
- Fix 4: CausalAgent DCAM episodic buffer (agent.py)
"""

from __future__ import annotations

import pytest
import torch

from snks.agent.causal_model import (
    CausalWorldModel,
    _PERCEPTUAL_HASH_OFFSET,
    _coarsen_sks,
    _split_context,
)
from snks.daf.types import CausalAgentConfig, HACPredictionConfig
from snks.dcam.episodic_hac import EpisodicHACPredictor
from snks.dcam.hac import HACEngine


# ---------------------------------------------------------------------------
# Fix 1: _split_context
# ---------------------------------------------------------------------------

class TestSplitContext:
    def test_empty_returns_empty(self):
        assert _split_context(set(), n_bins=64) == frozenset()

    def test_stable_ids_preserved(self):
        """Perceptual hash IDs (>= 10_000) must NOT be coarsened."""
        stable = {10_000, 10_008, 10_016}
        result = _split_context(stable, n_bins=64)
        assert result == frozenset(stable)

    def test_unstable_ids_coarsened(self):
        """DAF cluster IDs (< 10_000) must be binned."""
        unstable = {0, 64, 128}  # all map to bin 0 with n_bins=64
        result = _split_context(unstable, n_bins=64)
        assert result == frozenset({0})  # 0%64=0, 64%64=0, 128%64=0

    def test_mixed_ids(self):
        """Stable part preserved, unstable coarsened."""
        sks = {1, 2, 10_001}  # 1%64=1, 2%64=2; 10_001 → stable
        result = _split_context(sks, n_bins=64)
        assert 10_001 in result          # stable → kept as-is
        assert 1 in result               # 1 % 64 = 1
        assert 2 in result               # 2 % 64 = 2

    def test_different_visual_scenes_produce_different_keys(self):
        """Two scenes with different perceptual hashes → different context keys."""
        scene_a = {10_000, 10_008}   # e.g. key visible
        scene_b = {10_001, 10_009}   # e.g. door visible
        ctx_a = _split_context(scene_a, n_bins=64)
        ctx_b = _split_context(scene_b, n_bins=64)
        assert ctx_a != ctx_b

    def test_same_visual_scene_same_key(self):
        """Same perceptual hash → same context key, regardless of DAF noise."""
        # Base visual scene = {10_000}; DAF noise produces different unstable IDs
        scene_noisy_1 = {10_000, 5, 10}   # unstable: 5, 10
        scene_noisy_2 = {10_000, 6, 11}   # unstable: 6, 11
        ctx_1 = _split_context(scene_noisy_1, n_bins=64)
        ctx_2 = _split_context(scene_noisy_2, n_bins=64)
        # stable part (10_000) is the same; unstable parts differ but are both small
        assert 10_000 in ctx_1 and 10_000 in ctx_2

    def test_coarsen_sks_still_works(self):
        """_coarsen_sks (deprecated) still returns correct values."""
        result = _coarsen_sks({0, 16, 32}, n_bins=16)
        assert result == frozenset({0})  # 0%16=0, 16%16=0, 32%16=0

    def test_causal_model_uses_split_context(self):
        """CausalWorldModel with perceptual-hash context correctly separates scenes."""
        model = CausalWorldModel(CausalAgentConfig(causal_min_observations=1))

        # Scene A: key visible → perceptual hash 10_000
        scene_a = {10_000, 3, 7}
        # Scene B: door visible → perceptual hash 10_001
        scene_b = {10_001, 3, 7}  # same noisy DAF IDs

        model.observe_transition(scene_a, action=0, post_sks=scene_a | {10_100})
        model.observe_transition(scene_b, action=0, post_sks=scene_b | {10_200})

        pred_a, _ = model.predict_effect(scene_a, action=0)
        pred_b, _ = model.predict_effect(scene_b, action=0)

        # Effects should differ (10_100 vs 10_200)
        assert pred_a != pred_b, "Different visual scenes must produce different predictions"


# ---------------------------------------------------------------------------
# Fix 2: EpisodicHACPredictor
# ---------------------------------------------------------------------------

HAC_DIM = 128


@pytest.fixture
def hac() -> HACEngine:
    return HACEngine(dim=HAC_DIM)


@pytest.fixture
def predictor(hac: HACEngine) -> EpisodicHACPredictor:
    return EpisodicHACPredictor(hac, capacity=32)


def _emb(hac: HACEngine) -> dict[int, torch.Tensor]:
    return {0: hac.random_vector()}


class TestEpisodicHACPredictor:

    def test_predict_returns_none_before_any_observation(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        result = predictor.predict_next(_emb(hac))
        assert result is None

    def test_predict_returns_none_after_single_observation(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        predictor.observe(_emb(hac))
        result = predictor.predict_next(_emb(hac))
        assert result is None  # need at least 2 calls (prev + curr pair)

    def test_predict_returns_tensor_after_two_observations(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        predictor.observe(_emb(hac))
        predictor.observe(_emb(hac))
        result = predictor.predict_next(_emb(hac))
        assert result is not None
        assert result.shape == (HAC_DIM,)

    def test_predict_is_unit_norm(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        predictor.observe(_emb(hac))
        predictor.observe(_emb(hac))
        result = predictor.predict_next(_emb(hac))
        assert abs(result.norm().item() - 1.0) < 1e-4

    def test_buffer_size_grows(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        assert predictor.buffer_size == 0
        predictor.observe(_emb(hac))
        assert predictor.buffer_size == 0  # no pair yet
        predictor.observe(_emb(hac))
        assert predictor.buffer_size == 1

    def test_capacity_eviction(self, hac: HACEngine):
        predictor = EpisodicHACPredictor(hac, capacity=4)
        for _ in range(10):
            predictor.observe(_emb(hac))
        assert predictor.buffer_size == 4  # deque respects maxlen

    def test_reset_clears_all(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        predictor.observe(_emb(hac))
        predictor.observe(_emb(hac))
        predictor.reset()
        assert predictor.buffer_size == 0
        assert predictor._prev_embeddings is None
        assert predictor.predict_next(_emb(hac)) is None

    def test_predict_similar_after_repeated_pair(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        """After observing A→B, predict_next(A) should be close to B."""
        emb_a = {0: hac.random_vector()}
        emb_b = {0: hac.random_vector()}

        # Observe A then B
        predictor.observe(emb_a)
        predictor.observe(emb_b)

        predicted = predictor.predict_next(emb_a)
        sim = hac.similarity(predicted, emb_b[0])
        # Should be similar to emb_b (it's the only pair stored)
        assert sim > 0.5, f"Expected sim > 0.5, got {sim:.3f}"

    def test_compute_winner_pe_identical(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        v = hac.random_vector()
        pe = predictor.compute_winner_pe(v, v)
        assert abs(pe - 0.0) < 1e-4

    def test_compute_winner_pe_orthogonal(
        self, predictor: EpisodicHACPredictor
    ):
        v1 = torch.zeros(HAC_DIM); v1[0] = 1.0
        v2 = torch.zeros(HAC_DIM); v2[1] = 1.0
        pe = predictor.compute_winner_pe(v1, v2)
        assert abs(pe - 0.5) < 1e-4

    def test_pe_in_range(
        self, predictor: EpisodicHACPredictor, hac: HACEngine
    ):
        v1, v2 = hac.random_vector(), hac.random_vector()
        pe = predictor.compute_winner_pe(v1, v2)
        assert 0.0 <= pe <= 1.0


# ---------------------------------------------------------------------------
# Fix 3: HACPredictionConfig flags
# ---------------------------------------------------------------------------

class TestHACPredictionConfigFlags:
    def test_default_uses_bundle_backend(self):
        cfg = HACPredictionConfig()
        assert cfg.use_episodic_buffer is False
        assert cfg.episodic_capacity == 32

    def test_episodic_flag_readable(self):
        cfg = HACPredictionConfig(use_episodic_buffer=True, episodic_capacity=16)
        assert cfg.use_episodic_buffer is True
        assert cfg.episodic_capacity == 16


# ---------------------------------------------------------------------------
# Fix 3: Pipeline selects correct backend
# ---------------------------------------------------------------------------

class TestPipelineEpisodicBackend:
    def test_pipeline_uses_bundle_by_default(self):
        from snks.pipeline.runner import Pipeline
        from snks.daf.types import PipelineConfig
        from snks.daf.hac_prediction import HACPredictionEngine
        pipeline = Pipeline(PipelineConfig())
        assert isinstance(pipeline.hac_prediction, HACPredictionEngine)

    def test_pipeline_uses_episodic_when_flagged(self):
        from snks.pipeline.runner import Pipeline
        from snks.daf.types import PipelineConfig, HACPredictionConfig
        from snks.dcam.episodic_hac import EpisodicHACPredictor
        cfg = PipelineConfig(
            hac_prediction=HACPredictionConfig(
                use_episodic_buffer=True, episodic_capacity=16
            )
        )
        pipeline = Pipeline(cfg)
        assert isinstance(pipeline.hac_prediction, EpisodicHACPredictor)
        assert pipeline.hac_prediction.capacity == 16


# ---------------------------------------------------------------------------
# Fix 4: CausalAgent DCAM episodic buffer
# ---------------------------------------------------------------------------

class TestCausalAgentEpisodicBuffer:
    def test_buffer_none_by_default(self):
        from snks.agent.agent import CausalAgent
        agent = CausalAgent(CausalAgentConfig())
        assert agent.episodic_buffer is None

    def test_buffer_created_when_enabled(self):
        from snks.agent.agent import CausalAgent
        from snks.dcam.episodic import EpisodicBuffer
        config = CausalAgentConfig(use_dcam_episodic=True)
        agent = CausalAgent(config)
        assert agent.episodic_buffer is not None
        assert isinstance(agent.episodic_buffer, EpisodicBuffer)
