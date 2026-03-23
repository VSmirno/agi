"""DcamWorldModel — facade for the DCAM subsystem.

Orchestrates: HACEngine, LSHIndex, SSG, EpisodicBuffer, Consolidation, Persistence.
"""

from __future__ import annotations

import torch
from torch import Tensor

from snks.daf.types import DcamConfig
from snks.dcam.hac import HACEngine
from snks.dcam.lsh import LSHIndex
from snks.dcam.ssg import StructuredSparseGraph
from snks.dcam.episodic import EpisodicBuffer
from snks.dcam.consolidation import Consolidation, ConsolidationReport
from snks.dcam import persistence


class DcamWorldModel:
    """Top-level facade for Dual-Code Associative Memory."""

    def __init__(self, config: DcamConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.hac = HACEngine(dim=config.hac_dim, device=self.device)
        self.lsh = LSHIndex(
            dim=config.hac_dim,
            n_tables=config.lsh_n_tables,
            n_bits=config.lsh_n_bits,
            device=self.device,
        )
        self.graph = StructuredSparseGraph()
        self.buffer = EpisodicBuffer(config, device=self.device)
        self._consolidation = Consolidation(config)
        self._cycle_count: int = 0

    def store_episode(
        self,
        active_nodes: dict,
        context: Tensor,
        importance: float,
    ) -> int:
        """Store an episode in buffer and index its context in LSH."""
        eid = self.buffer.store(active_nodes, context, importance)
        self.lsh.insert(context.detach(), eid)
        self._cycle_count += 1
        return eid

    def query_similar(
        self,
        query_hac: Tensor,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find episodes with similar context HAC vectors."""
        return self.lsh.query(query_hac, top_k=top_k)

    def consolidate(self) -> ConsolidationReport:
        """Run consolidation if interval reached."""
        return self._consolidation.consolidate(self.buffer, self.graph)

    def save(self, path: str) -> None:
        """Persist full state to disk."""
        persistence.save(self, path)

    def load(self, path: str) -> None:
        """Restore state from disk."""
        persistence.load(self, path)
