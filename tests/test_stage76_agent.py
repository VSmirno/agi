"""Stage 76 Phase 4: Continuous Agent tests.

Uses a mock segmenter + mock env for fast unit-level coverage, and a real
CrafterPixelEnv + real segmenter checkpoint for a smoke test (skipped if
checkpoint missing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.decode_head import NEAR_CLASSES
from snks.agent.perception import HomeostaticTracker
from snks.agent.continuous_agent import (
    perceive_tile_field,
    run_continuous_episode,
    _bootstrap_action,
)
from snks.memory.episodic_sdm import EpisodicSDM
from snks.memory.state_encoder import StateEncoder


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockSegmenter:
    """Returns a deterministic 7×9 tile map with a tree directly north of the player."""

    def classify_tiles(self, pixels):
        # 7 rows × 9 cols
        class_ids = torch.zeros((7, 9), dtype=torch.long)
        confidences = torch.ones((7, 9)) * 0.9
        # "empty" by default
        empty_idx = NEAR_CLASSES.index("empty")
        class_ids.fill_(empty_idx)
        # Put tree at (2, 4) — one step above center (3, 4)
        class_ids[2, 4] = NEAR_CLASSES.index("tree")
        # Put water in a few spots for variety
        class_ids[5, 4] = NEAR_CLASSES.index("water")
        class_ids[4, 6] = NEAR_CLASSES.index("stone")
        return class_ids, confidences


class MockEnv:
    """Minimal env for decision-loop testing. Player never moves, inventory static."""

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        pixels = np.zeros((3, 64, 64), dtype=np.float32)
        info = {
            "inventory": {
                "health": 9, "food": 9, "drink": 9, "energy": 9,
                "wood": 0,
            },
            "player_pos": [32, 32],
            "semantic": np.zeros((64, 64), dtype=np.uint8),
        }
        return pixels, info

    def step(self, action):
        self.step_count += 1
        pixels = np.zeros((3, 64, 64), dtype=np.float32)
        # Gradually drain food/drink so tracker has something to see
        inv = {
            "health": max(0, 9 - self.step_count // 20),
            "food": max(0, 9 - self.step_count // 10),
            "drink": max(0, 9 - self.step_count // 10),
            "energy": max(0, 9 - self.step_count // 15),
            "wood": 0,
        }
        info = {
            "inventory": inv,
            "player_pos": [32, 32],
            "semantic": np.zeros((64, 64), dtype=np.uint8),
        }
        done = inv["health"] <= 0  # only die from health, not max_steps
        return pixels, 0.0, done, info


# ---------------------------------------------------------------------------
# Perception helper
# ---------------------------------------------------------------------------


class TestPerception:
    def test_perceive_tile_field_basic(self):
        seg = MockSegmenter()
        pixels = np.zeros((3, 64, 64), dtype=np.float32)
        vf = perceive_tile_field(pixels, seg)
        # tree detected at (2, 4)
        assert any(d[0] == "tree" for d in vf.detections)
        # water and stone also detected
        assert any(d[0] == "water" for d in vf.detections)
        assert any(d[0] == "stone" for d in vf.detections)
        # Empty excluded
        assert not any(d[0] == "empty" for d in vf.detections)


# ---------------------------------------------------------------------------
# Bootstrap action
# ---------------------------------------------------------------------------


class TestBootstrapAction:
    def test_returns_valid_action_when_no_plan(self):
        from snks.agent.perception import VisualField
        from snks.agent.crafter_spatial_map import CrafterSpatialMap, MOVE_ACTIONS

        store = ConceptStore()
        tracker = HomeostaticTracker()
        vf = VisualField()
        sm = CrafterSpatialMap()
        rng = np.random.RandomState(0)
        # No plan available (empty store) → random move
        action = _bootstrap_action(
            inventory={},
            tracker=tracker,
            vf=vf,
            spatial_map=sm,
            store=store,
            player_pos=(10, 10),
            rng=rng,
            center_r=3,
            center_c=4,
        )
        assert action in MOVE_ACTIONS


# ---------------------------------------------------------------------------
# Full episode (mock env)
# ---------------------------------------------------------------------------


class TestContinuousEpisode:
    def _make_components(self):
        store = ConceptStore()
        tracker = HomeostaticTracker()
        encoder = StateEncoder()
        sdm = EpisodicSDM(capacity=1000)
        return store, tracker, encoder, sdm

    def test_episode_runs_and_fills_sdm(self):
        store, tracker, encoder, sdm = self._make_components()
        env = MockEnv(max_steps=30)
        seg = MockSegmenter()
        rng = np.random.RandomState(0)

        result = run_continuous_episode(
            env=env,
            segmenter=seg,
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=30,
            temperature=1.0,
            bootstrap_k=5,
        )

        # Episode must run to completion
        assert result["length"] == 30
        # Memory must grow by the episode length
        assert len(sdm) == 30
        assert result["sdm_size"] == 30
        # Cause of death should be alive (mock env respects max_steps)
        assert result["cause_of_death"] == "alive"
        # Action entropy ≥ 0
        assert result["action_entropy"] >= 0.0
        # Some action was taken
        assert sum(result["action_counts"].values()) == 30

    def test_cold_start_is_bootstrap_only(self):
        """With bootstrap_k larger than episode length, SDM path never triggers."""
        store, tracker, encoder, sdm = self._make_components()
        env = MockEnv(max_steps=20)
        rng = np.random.RandomState(1)
        result = run_continuous_episode(
            env=env,
            segmenter=MockSegmenter(),
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=20,
            bootstrap_k=1000,  # never reached
        )
        assert result["bootstrap_ratio"] == 1.0
        assert result["sdm_ratio"] == 0.0

    def test_sdm_path_activates_with_warm_memory(self):
        """With low bootstrap_k + min_sdm_size, SDM path activates within episode."""
        store, tracker, encoder, sdm = self._make_components()
        env = MockEnv(max_steps=30)
        rng = np.random.RandomState(1)
        result = run_continuous_episode(
            env=env,
            segmenter=MockSegmenter(),
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=30,
            bootstrap_k=3,
            min_sdm_size=5,  # allow SDM path after 5 episodes in buffer
        )
        # First few steps must be bootstrap; afterwards SDM should kick in
        assert result["bootstrap_ratio"] < 1.0
        assert result["sdm_ratio"] > 0.0

    def test_min_sdm_size_gates_transition(self):
        """With min_sdm_size larger than episode length, SDM path never triggers."""
        store, tracker, encoder, sdm = self._make_components()
        env = MockEnv(max_steps=20)
        rng = np.random.RandomState(4)
        result = run_continuous_episode(
            env=env,
            segmenter=MockSegmenter(),
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=20,
            bootstrap_k=1,
            min_sdm_size=10_000,  # never reached
        )
        assert result["bootstrap_ratio"] == 1.0
        assert result["sdm_ratio"] == 0.0

    def test_body_delta_recorded(self):
        """Body deltas should track observed_max variables."""
        store, tracker, encoder, sdm = self._make_components()
        env = MockEnv(max_steps=30)
        rng = np.random.RandomState(2)
        run_continuous_episode(
            env=env,
            segmenter=MockSegmenter(),
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=30,
        )
        # Find an episode with non-zero health delta (env drains HP at step 20)
        non_zero_delta_eps = [
            ep for ep in sdm._buffer
            if any(v != 0 for v in ep.body_delta.values())
        ]
        assert len(non_zero_delta_eps) > 0
        # body_delta should use tracked vars (not hardcoded)
        for ep in non_zero_delta_eps[:5]:
            for var in ep.body_delta:
                assert var in tracker.observed_max or var in {
                    "health", "food", "drink", "energy", "wood",
                }

    def test_memory_persists_across_episodes(self):
        """Shared SDM accumulates across episodes."""
        store, tracker, encoder, sdm = self._make_components()
        rng = np.random.RandomState(3)

        # Episode 1
        run_continuous_episode(
            env=MockEnv(max_steps=20),
            segmenter=MockSegmenter(),
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=20,
        )
        size_after_1 = len(sdm)
        assert size_after_1 == 20

        # Episode 2 — SDM should keep episode 1 and add 20 more
        run_continuous_episode(
            env=MockEnv(max_steps=20),
            segmenter=MockSegmenter(),
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=20,
        )
        assert len(sdm) == 40


# ---------------------------------------------------------------------------
# Real segmenter smoke test (optional)
# ---------------------------------------------------------------------------


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")


@pytest.mark.skipif(
    not STAGE75_CHECKPOINT.exists(),
    reason="Stage 75 segmenter checkpoint not available",
)
class TestRealSegmenterSmoke:
    def test_one_real_episode(self):
        """Full pipeline smoke test with real Crafter env + Stage 75 segmenter.

        Uses no enemies, limited steps. Just verifies the loop doesn't crash.
        """
        from snks.agent.crafter_pixel_env import CrafterPixelEnv
        from snks.encoder.cnn_encoder import disable_rocm_conv
        from snks.encoder.tile_segmenter import load_tile_segmenter

        disable_rocm_conv()

        # Load Stage 75 segmenter
        segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT))

        env = CrafterPixelEnv(seed=1)
        try:
            env._env._balance_chunk = lambda *a, **kw: None  # disable enemies
        except Exception:
            pass

        store = ConceptStore()
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        tb.load_into(store)
        tracker = HomeostaticTracker()
        tracker.init_from_body_rules(tb.body_rules)
        encoder = StateEncoder()
        sdm = EpisodicSDM(capacity=500)
        rng = np.random.RandomState(0)

        result = run_continuous_episode(
            env=env,
            segmenter=segmenter,
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=50,
            temperature=1.0,
            bootstrap_k=5,
        )

        assert 1 <= result["length"] <= 50
        assert len(sdm) == result["length"]
        assert "final_inv" in result
