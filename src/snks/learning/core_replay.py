"""Small, episode-level replay store for real completed experience only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from snks.env.core_types import Episode, Mode, Observation, Transition


class SequenceReplay:
    """Keep a recent ring and a bounded reservoir of completed episodes."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._recent_capacity = max(1, capacity // 2)
        self._reservoir_capacity = capacity - self._recent_capacity
        self._recent: list[Episode] = []
        self._reservoir: list[Episode] = []
        self._evicted = 0

    def append(self, episode: Episode, mode: Mode) -> None:
        """Add one finished training/adaptation episode, exactly once."""
        if mode is Mode.EVALUATE:
            raise PermissionError("evaluation experience must not enter replay")
        if mode not in (Mode.TRAIN, Mode.ADAPT):
            raise ValueError(f"unsupported replay mode: {mode}")
        if episode.split != mode.value:
            raise ValueError(f"{mode.value} replay requires a {mode.value} episode")
        self._validate_episode(episode)
        if episode.uid in {item.uid for item in self._episodes()}:
            return
        self._recent.append(episode)
        if len(self._recent) > self._recent_capacity:
            self._retain_evicted(self._recent.pop(0))

    def sample(
        self,
        batch_size: int,
        length: int,
        burn_in: int,
        recent_fraction: float,
        schema: str | None = None,
        salient_fraction: float = 0.0,
    ) -> list[Episode]:
        """Sample ordered, within-episode windows from one observation schema."""
        if batch_size < 1 or length < 1 or burn_in < 0:
            raise ValueError("batch_size/length must be positive and burn_in non-negative")
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")
        if not 0.0 <= salient_fraction <= 1.0:
            raise ValueError("salient_fraction must be in [0, 1]")
        available = [item for item in self._episodes() if schema is None or self._schema(item) == schema]
        if not available:
            return []
        selected_schema = schema or str(self._rng.choice(sorted({self._schema(item) for item in available})))
        recent = [item for item in self._recent if self._schema(item) == selected_schema]
        reservoir = [item for item in self._reservoir if self._schema(item) == selected_schema]
        selected: list[Episode] = []
        n_recent = round(batch_size * recent_fraction)
        for count, pool, fallback in ((n_recent, recent, reservoir), (batch_size - n_recent, reservoir, recent)):
            choices = pool or fallback
            if choices:
                selected.extend(choices[int(self._rng.integers(len(choices)))] for _ in range(count))
        if not selected:
            return []
        width = length + burn_in
        windows = [self._window(item, width) for item in selected]
        salient = self._salient_transitions(available, selected_schema, burn_in)
        for index in range(min(round(batch_size * salient_fraction), len(windows))):
            if not salient:
                break
            episode, transition_index = salient[int(self._rng.integers(len(salient)))]
            windows[index] = self._window_ending_at(episode, width, transition_index)
        return windows

    def manifest(self) -> dict[str, object]:
        """Expose enough state to audit replay without exposing experience arrays."""
        episodes = self._episodes()
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "episode_count": len(episodes),
            "episodes": len(episodes),
            "recent": len(self._recent),
            "reservoir": len(self._reservoir),
            "hash": self._payload_hash(self._arrays_and_metadata()[0]),
        }

    def save(self, path: str | Path) -> None:
        """Persist arrays plus JSON metadata without pickle serialization."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        arrays, metadata = self._arrays_and_metadata()
        metadata["payload_hash"] = self._payload_hash(arrays)
        arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
        with target.open("xb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SequenceReplay":
        """Load a replay snapshot created by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        expected = metadata.pop("payload_hash")
        if cls._payload_hash(arrays) != expected:
            raise ValueError("replay snapshot hash mismatch")
        replay = cls(int(metadata["capacity"]), int(metadata["seed"]))
        replay._rng.bit_generator.state = metadata["rng_state"]
        replay._evicted = int(metadata["evicted"])
        episodes = [cls._episode_from_arrays(item, arrays) for item in metadata["episodes"]]
        replay._recent = episodes[: int(metadata["recent_count"])]
        replay._reservoir = episodes[int(metadata["recent_count"]):]
        return replay

    def _restore_from(self, source: "SequenceReplay") -> None:
        self.__dict__.update(source.__dict__)

    def _retain_evicted(self, episode: Episode) -> None:
        if self._reservoir_capacity == 0:
            return
        self._evicted += 1
        if len(self._reservoir) < self._reservoir_capacity:
            self._reservoir.append(episode)
        else:
            index = int(self._rng.integers(self._evicted))
            if index < self._reservoir_capacity:
                self._reservoir[index] = episode

    @staticmethod
    def _validate_episode(episode: Episode) -> None:
        if not episode.transitions:
            raise ValueError("cannot replay an empty episode")
        if not (episode.transitions[-1].terminated or episode.transitions[-1].truncated):
            raise ValueError("cannot replay an unfinished episode")
        schemas = {transition.before.schema for transition in episode.transitions}
        schemas.update(transition.after.schema for transition in episode.transitions)
        if len(schemas) != 1:
            raise ValueError("episode transitions must share one schema")

    @staticmethod
    def _schema(episode: Episode) -> str:
        return episode.transitions[0].before.schema

    def _window(self, episode: Episode, width: int) -> Episode:
        count = len(episode.transitions)
        start = 0 if count <= width else int(self._rng.integers(count - width + 1))
        transitions = episode.transitions[start:start + width]
        return Episode(episode.uid, episode.split, episode.family, episode.ruleset, transitions)

    @staticmethod
    def _window_ending_at(episode: Episode, width: int, index: int) -> Episode:
        start = max(0, index - width + 1)
        transitions = episode.transitions[start:index + 1]
        return Episode(episode.uid, episode.split, episode.family, episode.ruleset, transitions)

    @staticmethod
    def _salient_transitions(
        episodes: list[Episode], schema: str, burn_in: int
    ) -> list[tuple[Episode, int]]:
        salient: list[tuple[Episode, int]] = []
        for episode in episodes:
            if SequenceReplay._schema(episode) != schema:
                continue
            for index, transition in enumerate(episode.transitions):
                if index < burn_in:
                    continue
                mask = transition.before.sensor_mask & transition.after.sensor_mask
                sensor_changed = bool(
                    np.any(transition.before.sensors[mask] != transition.after.sensors[mask])
                )
                if transition.terminated or sensor_changed:
                    salient.append((episode, index))
        return salient

    def _episodes(self) -> list[Episode]:
        return [*self._recent, *self._reservoir]

    def _arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        arrays: dict[str, np.ndarray] = {}
        records: list[dict[str, object]] = []
        for index, episode in enumerate(self._episodes()):
            prefix = f"episode_{index}"
            transitions = episode.transitions
            for side in ("before", "after"):
                observations = [getattr(item, side) for item in transitions]
                arrays[f"{prefix}_{side}_rgb"] = np.stack([item.rgb for item in observations])
                arrays[f"{prefix}_{side}_sensors"] = np.stack([item.sensors for item in observations])
                arrays[f"{prefix}_{side}_mask"] = np.stack([item.sensor_mask for item in observations])
                arrays[f"{prefix}_{side}_step"] = np.array([item.step for item in observations], dtype=np.int64)
            arrays[f"{prefix}_action"] = np.array([item.action for item in transitions], dtype=np.int64)
            arrays[f"{prefix}_terminated"] = np.array([item.terminated for item in transitions], dtype=bool)
            arrays[f"{prefix}_truncated"] = np.array([item.truncated for item in transitions], dtype=bool)
            records.append({"uid": episode.uid, "split": episode.split, "family": episode.family,
                            "ruleset": episode.ruleset, "schema": self._schema(episode), "prefix": prefix})
        return arrays, {"capacity": self.capacity, "seed": self.seed, "evicted": self._evicted,
                         "recent_count": len(self._recent), "rng_state": self._rng.bit_generator.state,
                         "episodes": records}

    @staticmethod
    def _episode_from_arrays(record: dict[str, object], arrays: dict[str, np.ndarray]) -> Episode:
        prefix, schema = str(record["prefix"]), str(record["schema"])
        transitions: list[Transition] = []
        for index, action in enumerate(arrays[f"{prefix}_action"]):
            observations = []
            for side in ("before", "after"):
                observations.append(Observation(arrays[f"{prefix}_{side}_rgb"][index],
                                                arrays[f"{prefix}_{side}_sensors"][index],
                                                arrays[f"{prefix}_{side}_mask"][index], schema,
                                                int(arrays[f"{prefix}_{side}_step"][index])))
            transitions.append(Transition(observations[0], int(action), observations[1],
                                          bool(arrays[f"{prefix}_terminated"][index]),
                                          bool(arrays[f"{prefix}_truncated"][index])))
        return Episode(str(record["uid"]), str(record["split"]), str(record["family"]),
                       str(record["ruleset"]), tuple(transitions))

    @staticmethod
    def _payload_hash(arrays: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for key in sorted(arrays):
            value = arrays[key]
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()
