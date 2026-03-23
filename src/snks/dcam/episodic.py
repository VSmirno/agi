"""Episodic buffer — short-term memory for recent episodes.

Stores episodes with importance-weighted eviction when capacity is reached.
LSH indexing is handled externally by DcamWorldModel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from snks.daf.types import DcamConfig


@dataclass
class Episode:
    """A single stored episode."""

    episode_id: int
    active_nodes: dict  # {node_id: (phase, amplitude)} or similar
    context_hac: Tensor  # (D,) HAC vector
    importance: float
    timestamp: int
    consolidated: bool = False


class EpisodicBuffer:
    """Fixed-capacity episodic memory with importance-weighted eviction."""

    def __init__(self, config: DcamConfig, device: torch.device | None = None) -> None:
        self.capacity = config.episodic_capacity
        self.device = device or torch.device("cpu")
        self._episodes: dict[int, Episode] = {}
        self._next_id: int = 0
        self._step: int = 0

    def store(
        self,
        active_nodes: dict,
        context_hac: Tensor,
        importance: float,
    ) -> int:
        """Store an episode. Evicts least important if at capacity."""
        if len(self._episodes) >= self.capacity:
            self._evict()

        eid = self._next_id
        self._next_id += 1
        self._step += 1

        self._episodes[eid] = Episode(
            episode_id=eid,
            active_nodes=active_nodes,
            context_hac=context_hac.detach().to(self.device),
            importance=importance,
            timestamp=self._step,
        )
        return eid

    def _evict(self) -> int:
        """Remove episode with lowest importance. Returns evicted id."""
        victim = min(self._episodes.values(), key=lambda e: e.importance)
        vid = victim.episode_id
        del self._episodes[vid]
        return vid

    def get_episode(self, episode_id: int) -> Episode | None:
        return self._episodes.get(episode_id)

    def get_all_episodes(self) -> list[Episode]:
        return list(self._episodes.values())

    def __len__(self) -> int:
        return len(self._episodes)
