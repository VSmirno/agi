"""Integration tests for Stage 68: find_target_with_map.

Tests navigation WITHOUT info["semantic"] — only NearDetector + player_pos.
Requires crafter to be installed; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

try:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.crafter_spatial_map import CrafterSpatialMap, find_target_with_map
    HAS_CRAFTER = True
except ImportError:
    HAS_CRAFTER = False


# ---------------------------------------------------------------------------
# Minimal fake detector for offline testing
# ---------------------------------------------------------------------------

class _ConstantDetector:
    """Always returns the same near_str — for testing without a real encoder."""
    def __init__(self, near: str) -> None:
        self._near = near

    def detect(self, pixels: torch.Tensor) -> str:
        return self._near


class _CyclingDetector:
    """Returns near_str from a cycle: [empty, empty, ..., target, ...]."""
    def __init__(self, target: str, every: int = 5) -> None:
        self._target = target
        self._every = every
        self._count = 0

    def detect(self, pixels: torch.Tensor) -> str:
        self._count += 1
        if self._count % self._every == 0:
            return self._target
        return "empty"


# ---------------------------------------------------------------------------
# Unit-level: find_target_with_map with fake env + fake detector
# ---------------------------------------------------------------------------

class _FakeEnv:
    """Minimal env stub: always returns same pixels + info."""

    def __init__(self, player_pos: tuple[int, int] = (10, 10)) -> None:
        self._pos = list(player_pos)
        self._step_count = 0
        self.n_actions = 17

    def observe(self) -> tuple[np.ndarray, dict]:
        return self._pixels(), {"player_pos": tuple(self._pos), "inventory": {}}

    def reset(self) -> tuple[np.ndarray, dict]:
        self._pos = [10, 10]
        return self.observe()

    def step(self, action: str | int) -> tuple[np.ndarray, float, bool, dict]:
        if isinstance(action, str):
            if action == "move_down":
                self._pos[0] = min(63, self._pos[0] + 1)
            elif action == "move_up":
                self._pos[0] = max(0, self._pos[0] - 1)
            elif action == "move_right":
                self._pos[1] = min(63, self._pos[1] + 1)
            elif action == "move_left":
                self._pos[1] = max(0, self._pos[1] - 1)
        self._step_count += 1
        return self._pixels(), 0.0, False, {"player_pos": tuple(self._pos), "inventory": {}}

    def _pixels(self) -> np.ndarray:
        return np.zeros((3, 64, 64), dtype=np.float32)


class TestFindTargetWithMap:
    def test_found_immediately(self):
        """Detector always returns target — found on first step."""
        env = _FakeEnv()
        detector = _ConstantDetector("tree")
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(0)

        pixels, info, found = find_target_with_map(
            env, detector, smap, "tree", max_steps=10, rng=rng
        )
        assert found is True
        assert isinstance(pixels, torch.Tensor)
        assert pixels.shape == (3, 64, 64)
        assert isinstance(info, dict)

    def test_found_after_some_steps(self):
        """Cycling detector: target appears every 5 steps."""
        env = _FakeEnv()
        detector = _CyclingDetector("stone", every=5)
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(1)

        _, _, found = find_target_with_map(
            env, detector, smap, "stone", max_steps=50, rng=rng
        )
        assert found is True

    def test_not_found_within_max_steps(self):
        """Detector never returns target — found=False."""
        env = _FakeEnv()
        detector = _ConstantDetector("empty")
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(2)

        _, _, found = find_target_with_map(
            env, detector, smap, "tree", max_steps=10, rng=rng
        )
        assert found is False

    def test_spatial_map_grows(self):
        """n_visited increases during navigation."""
        env = _FakeEnv()
        detector = _ConstantDetector("empty")
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(3)

        find_target_with_map(
            env, detector, smap, "tree", max_steps=20, rng=rng
        )
        assert smap.n_visited > 0

    def test_no_info_semantic_used(self):
        """Env returns info WITHOUT semantic — must not raise."""
        env = _FakeEnv()  # _FakeEnv never puts "semantic" in info
        detector = _CyclingDetector("water", every=3)
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(4)

        # Should work fine — no KeyError on "semantic"
        _, _, found = find_target_with_map(
            env, detector, smap, "water", max_steps=20, rng=rng
        )
        assert found is True

    def test_returns_tensor_and_dict(self):
        """Return types are correct."""
        env = _FakeEnv()
        detector = _ConstantDetector("tree")
        smap = CrafterSpatialMap()
        pixels, info, found = find_target_with_map(
            env, detector, smap, "tree", max_steps=5
        )
        assert isinstance(pixels, torch.Tensor)
        assert isinstance(info, dict)
        assert isinstance(found, bool)


# ---------------------------------------------------------------------------
# Integration with real Crafter (skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CRAFTER, reason="crafter not installed")
class TestFindTargetRealEnv:
    def test_spatial_map_updates_player_pos(self):
        """Real env: spatial map accumulates player positions."""
        env = CrafterPixelEnv(seed=42)
        env.reset()
        detector = _ConstantDetector("empty")
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(42)

        find_target_with_map(
            env, detector, smap, "tree", max_steps=30, rng=rng
        )
        # Map should have accumulated positions
        assert smap.n_visited >= 1

    def test_no_semantic_in_situation(self):
        """Verify info["semantic"] is never accessed by find_target_with_map."""
        class _NoSemanticEnv:
            """Wraps CrafterPixelEnv but removes info["semantic"] to catch cheating."""
            def __init__(self, seed):
                self._env = CrafterPixelEnv(seed=seed)

            def observe(self):
                p, info = self._env.observe()
                info.pop("semantic", None)
                return p, info

            def reset(self):
                p, info = self._env.reset()
                info.pop("semantic", None)
                return p, info

            def step(self, action):
                p, r, d, info = self._env.step(action)
                info.pop("semantic", None)
                return p, r, d, info

        env = _NoSemanticEnv(seed=99)
        env.reset()
        detector = _CyclingDetector("tree", every=10)
        smap = CrafterSpatialMap()

        # Must not raise KeyError
        pixels, info, _ = find_target_with_map(
            env, detector, smap, "tree", max_steps=50
        )
        assert isinstance(pixels, torch.Tensor)
