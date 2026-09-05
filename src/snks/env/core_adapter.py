"""Crafter adapter for the small transferable learning-core experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from snks.env.core_types import ActionSpec, Observation, Transition


CRAFTER_ACTIONS = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
)


def project_crafter_observation(
    rgb: np.ndarray,
    info: Mapping[str, Any],
    names: Sequence[str],
    step: int,
) -> Observation:
    """Project native pixels and allowlisted inventory into the agent schema."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.shape != (64, 64, 3):
        raise ValueError(f"expected Crafter RGB shape (64, 64, 3), got {image.shape}")

    inventory = info.get("inventory", {})
    if not isinstance(inventory, Mapping):
        inventory = {}
    sensors = np.zeros(len(names), dtype=np.float32)
    mask = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        value = inventory.get(name)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            sensors[index] = float(value)
            mask[index] = True

    return Observation(
        rgb=image.transpose(2, 0, 1),
        sensors=sensors,
        sensor_mask=mask,
        schema="crafter-v1",
        step=step,
    )


class CrafterCoreAdapter:
    """Expose a fresh native Crafter episode through core data types.

    ``initial_inventory`` and ``local_source`` are controlled development
    fixtures used only by the two first Crafter task families. They modify the
    native world before the inventory-populating noop and are included in
    ``reset_transitions``.
    """

    def __init__(
        self,
        sensor_names: Sequence[str] = (
            "health",
            "food",
            "drink",
            "energy",
            "wood",
        ),
        max_steps: int = 256,
        *,
        initial_inventory: Mapping[str, float] | None = None,
        local_source: str | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.sensor_names = tuple(sensor_names)
        self.max_steps = int(max_steps)
        self.initial_inventory = dict(initial_inventory or {})
        self.local_source = local_source
        self.actions = ActionSpec("crafter-v1", CRAFTER_ACTIONS)
        self.reset_transitions = 0
        self._env: Any | None = None
        self._last_observation: Observation | None = None
        self._last_info: dict[str, Any] = {}
        self._step = 0
        self._raw_reward = 0.0
        self._native_done = False
        self._timed_out = False

    def reset(self, seed: int | None = None) -> Observation:
        """Create a fresh native environment and return its first observation."""
        self.close()
        try:
            import crafter
        except ImportError as exc:
            raise ImportError("CrafterCoreAdapter requires the crafter package") from exc

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        self._env = crafter.Env(**kwargs)
        self._env.reset()
        self._apply_controlled_fixture()

        rgb, reward, done, info = self._env.step(0)
        self.reset_transitions = 1
        self._step = 0
        self._raw_reward = float(reward)
        self._native_done = bool(done)
        self._timed_out = False
        self._last_info = dict(info)
        self._last_observation = project_crafter_observation(
            rgb,
            self._last_info,
            self.sensor_names,
            step=0,
        )
        return self._last_observation

    def step(self, action: int) -> Transition:
        """Execute one native action, keeping death and timeout distinct."""
        if self._env is None or self._last_observation is None:
            raise RuntimeError("reset must be called before step")
        if not 0 <= int(action) < len(self.actions.names):
            raise ValueError(f"invalid Crafter action: {action}")

        before = self._last_observation
        rgb, reward, done, info = self._env.step(int(action))
        self._step += 1
        self._raw_reward += float(reward)
        self._native_done = bool(done)
        self._timed_out = self._step >= self.max_steps and not self._native_done
        self._last_info = dict(info)
        after = project_crafter_observation(
            rgb,
            self._last_info,
            self.sensor_names,
            step=self._step,
        )
        self._last_observation = after
        return Transition(
            before=before,
            action=int(action),
            after=after,
            terminated=self._native_done,
            truncated=self._timed_out,
        )

    def diagnostic_snapshot(self) -> dict[str, Any]:
        """Return evaluator-only native outcomes, never agent observations."""
        return {
            "success": bool(self._last_info.get("success", False)),
            "raw_reward": self._raw_reward,
            "native_done": self._native_done,
            "timed_out": self._timed_out,
            "reset_transitions": self.reset_transitions,
        }

    def close(self) -> None:
        """Release the current native environment, if it supports closing."""
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
        self._env = None
        self._last_observation = None

    def _apply_controlled_fixture(self) -> None:
        if self._env is None:
            return
        player = getattr(self._env, "_player", None)
        if player is None:
            raise RuntimeError("unsupported Crafter version: player state unavailable")
        for name, value in self.initial_inventory.items():
            player.inventory[name] = value

        if self.local_source is None:
            return
        world = getattr(self._env, "_world", None)
        if world is None:
            raise RuntimeError("unsupported Crafter version: world state unavailable")
        row, column = int(player.pos[0]), int(player.pos[1])
        for d_row, d_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            world[row + d_row, column + d_column] = self.local_source
