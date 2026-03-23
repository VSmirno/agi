"""Prediction engine: causal graph between SKS clusters, prediction error."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from snks.daf.types import PredictionConfig


@dataclass
class CausalEdge:
    """Directed causal link between two SKS clusters."""
    strength: float = 0.0
    confidence: float = 0.0
    count: int = 0


class PredictionEngine:
    """Tracks causal relationships between SKS activations.

    Builds a causal graph: if SKS A regularly precedes SKS B,
    then a directed edge (A→B) is created with increasing confidence.
    """

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self._causal_graph: dict[int, dict[int, CausalEdge]] = {}
        self._history: list[set[int]] = []

    def observe(self, active_sks_ids: set[int]) -> None:
        """Record which SKS are active this cycle. Update causal graph."""
        if self._history:
            prev = self._history[-1]
            for src in prev:
                for dst in active_sks_ids:
                    if src == dst:
                        continue
                    if src not in self._causal_graph:
                        self._causal_graph[src] = {}
                    if dst not in self._causal_graph[src]:
                        self._causal_graph[src][dst] = CausalEdge()
                    edge = self._causal_graph[src][dst]
                    edge.count += 1
                    edge.confidence = min(1.0, edge.count / 10.0)
                    edge.strength = edge.confidence

        # Decay edges not observed
        for src in self._causal_graph:
            for dst, edge in self._causal_graph[src].items():
                src_active = src in active_sks_ids
                dst_in_prev = self._history and dst in self._history[-1]
                if not (src_active or dst_in_prev):
                    edge.confidence *= self.config.causal_decay

        self._history.append(set(active_sks_ids))
        if len(self._history) > self.config.causal_window:
            self._history = self._history[-self.config.causal_window:]

    def predict(self) -> set[int]:
        """Predict which SKS will be active next based on causal graph."""
        if not self._history:
            return set()

        current = self._history[-1]
        predicted: set[int] = set()

        for src in current:
            if src in self._causal_graph:
                for dst, edge in self._causal_graph[src].items():
                    if edge.confidence >= self.config.causal_min_confidence:
                        predicted.add(dst)

        return predicted

    def compute_prediction_error(
        self,
        predicted: set[int],
        actual: set[int],
        n_nodes: int,
        sks_clusters: dict[int, set[int]],
    ) -> torch.Tensor:
        """Compute per-node prediction error.

        Nodes in unexpected (actual but not predicted) clusters get PE = 1.
        Nodes in missed (predicted but not actual) clusters get PE = 1.

        Returns:
            (n_nodes,) float32 prediction error per node.
        """
        pe = torch.zeros(n_nodes)

        # Surprise: actual SKS that weren't predicted
        surprise_ids = actual - predicted
        for sks_id in surprise_ids:
            if sks_id in sks_clusters:
                for node in sks_clusters[sks_id]:
                    if node < n_nodes:
                        pe[node] = 1.0

        # Missing: predicted SKS that didn't appear
        missing_ids = predicted - actual
        for sks_id in missing_ids:
            if sks_id in sks_clusters:
                for node in sks_clusters[sks_id]:
                    if node < n_nodes:
                        pe[node] = 1.0

        return pe

    def get_lr_modulation(self, pe: torch.Tensor, alpha: float) -> torch.Tensor:
        """Convert prediction error to learning rate modulation.

        Returns (N,) = 1 + alpha * pe. Higher PE → faster learning.
        """
        return 1.0 + alpha * pe

    def get_causal_graph(self) -> dict[int, dict[int, CausalEdge]]:
        """Return the causal graph for inspection."""
        return self._causal_graph
