"""Environment-agnostic adapter interface for SNKS agents.

Decouples agent from specific environment implementations (MiniGrid, etc.).
Agent receives only raw observations and produces discrete actions.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EnvAdapter(Protocol):
    """Protocol for environment adapters.

    Any environment that implements this protocol can be used with PureDafAgent.
    """

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment and return initial observation (H, W, 3) uint8."""
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute action. Returns (obs, reward, terminated, truncated, info)."""
        ...

    @property
    def n_actions(self) -> int:
        """Number of discrete actions available."""
        ...

    @property
    def name(self) -> str:
        """Environment name for logging."""
        ...


class MiniGridAdapter:
    """Wraps any MiniGrid gymnasium environment into EnvAdapter.

    Uses RGBImgPartialObsWrapper to get actual RGB pixel observations
    instead of MiniGrid's symbolic encoding (object_type, color, state).
    Agent sees only its partial view rendered as pixels.
    """

    def __init__(self, env_name: str, tile_size: int = 8) -> None:
        import gymnasium

        try:
            import minigrid  # noqa: F401 — registers MiniGrid envs
            from minigrid.wrappers import RGBImgPartialObsWrapper
        except ImportError:
            raise ImportError("minigrid package required for MiniGridAdapter")

        base_env = gymnasium.make(env_name, render_mode="rgb_array")
        self._env = RGBImgPartialObsWrapper(base_env, tile_size=tile_size)
        self._env_name = env_name

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self._env.reset(seed=seed)
        return self._extract_image(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._extract_image(obs), float(reward), terminated, truncated, info

    @property
    def n_actions(self) -> int:
        return int(self._env.action_space.n)

    @property
    def name(self) -> str:
        return self._env_name

    @property
    def env(self):
        """Access underlying gymnasium env (for compatibility)."""
        return self._env

    def get_symbolic_obs(self) -> np.ndarray:
        """Get symbolic 7×7×3 observation from MiniGrid.

        Returns:
            (7, 7, 3) int array — [object_type, color, state] per cell.
        """
        obs = self._env.unwrapped.gen_obs()
        if isinstance(obs, dict):
            return obs["image"]
        return obs

    @staticmethod
    def _extract_image(obs) -> np.ndarray:
        """Extract RGB image from observation."""
        if isinstance(obs, dict):
            return obs.get("image", obs.get("observation", obs))
        return obs


class ArrayEnvAdapter:
    """Adapter for simple array-based environments (no rendering).

    Converts flat state vectors to pseudo-images for DAF processing.
    """

    def __init__(
        self,
        env,
        n_actions: int,
        name: str = "array_env",
        image_size: int = 8,
    ) -> None:
        self._env = env
        self._n_actions = n_actions
        self._name = name
        self._image_size = image_size

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            state = self._env.reset(seed=seed)
        else:
            state = self._env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return self._state_to_image(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self._env.step(action)
        state, reward, *rest = result
        terminated = rest[0] if rest else False
        truncated = rest[1] if len(rest) > 1 else False
        info = rest[2] if len(rest) > 2 else {}
        return self._state_to_image(state), float(reward), terminated, truncated, info

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def name(self) -> str:
        return self._name

    def _state_to_image(self, state) -> np.ndarray:
        """Convert flat state to pseudo-image."""
        arr = np.asarray(state, dtype=np.float32).ravel()
        size = self._image_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i, val in enumerate(arr[: size * size]):
            row, col = divmod(i, size)
            v = int(np.clip(val * 255, 0, 255))
            img[row, col] = [v, v, v]
        return img
