"""Consolidation — offline memory consolidation (sleep analogue).

Three passes:
1. STC (Synaptic Tagging and Capture) — strengthen edges for important episodes
2. Co-activation analysis — create edges for frequently co-active node pairs
3. Causal extraction — create causal edges from temporal sequences
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from snks.daf.types import DcamConfig
from snks.dcam.episodic import EpisodicBuffer
from snks.dcam.ssg import StructuredSparseGraph


@dataclass
class ConsolidationReport:
    """Results of a consolidation cycle."""

    n_stc_processed: int
    n_coactivation_pairs: int
    n_causal_edges: int
    n_pruned: int


class Consolidation:
    """Offline consolidation engine."""

    def __init__(self, config: DcamConfig) -> None:
        self.stc_threshold = config.consolidation_stc_threshold
        self.coact_min = config.consolidation_coact_min

    def consolidate(
        self,
        buffer: EpisodicBuffer,
        graph: StructuredSparseGraph,
    ) -> ConsolidationReport:
        """Run full consolidation cycle over unconsolidated episodes."""
        episodes = [
            e for e in buffer.get_all_episodes() if not e.consolidated
        ]

        n_stc = self._stc_pass(episodes, graph)
        n_coact = self._coactivation_pass(episodes, graph)
        n_causal = self._causal_pass(episodes, graph)
        n_pruned = graph.prune(threshold=0.01)

        # Mark as consolidated
        for ep in episodes:
            ep.consolidated = True

        return ConsolidationReport(
            n_stc_processed=n_stc,
            n_coactivation_pairs=n_coact,
            n_causal_edges=n_causal,
            n_pruned=n_pruned,
        )

    def _stc_pass(
        self,
        episodes: list,
        graph: StructuredSparseGraph,
    ) -> int:
        """Synaptic Tagging: strengthen structural edges for important episodes."""
        count = 0
        for ep in episodes:
            if ep.importance < self.stc_threshold:
                continue
            nodes = list(ep.active_nodes.keys())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    graph.update_edge(
                        nodes[i], nodes[j], "structural",
                        delta=ep.importance * 0.1,
                    )
                    count += 1
        return count

    def _coactivation_pass(
        self,
        episodes: list,
        graph: StructuredSparseGraph,
    ) -> int:
        """Count co-activations and strengthen edges above threshold."""
        coact: Counter[tuple[int, int]] = Counter()
        for ep in episodes:
            nodes = sorted(ep.active_nodes.keys())
            for pair in combinations(nodes, 2):
                coact[pair] += 1

        count = 0
        for (n1, n2), freq in coact.items():
            if freq >= self.coact_min:
                graph.update_edge(n1, n2, "structural", delta=freq * 0.1)
                count += 1
        return count

    def _causal_pass(
        self,
        episodes: list,
        graph: StructuredSparseGraph,
    ) -> int:
        """Extract causal edges from temporal episode sequences."""
        sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
        count = 0
        for i in range(len(sorted_eps) - 1):
            ep_before = sorted_eps[i]
            ep_after = sorted_eps[i + 1]
            dt = ep_after.timestamp - ep_before.timestamp
            if dt <= 0 or dt > 100:
                continue

            nodes_before = set(ep_before.active_nodes.keys())
            nodes_after = set(ep_after.active_nodes.keys())
            causes = nodes_before - nodes_after
            effects = nodes_after - nodes_before

            for c in causes:
                for e in effects:
                    graph.update_edge(c, e, "causal", delta=0.5)
                    count += 1
        return count
