"""Global Workspace — selects the dominant SKS cluster (winner)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snks.daf.types import GWSConfig


@dataclass
class GWSState:
    winner_id: int            # id кластера-победителя (ключ в sks_clusters)
    winner_nodes: set[int]    # узлы победителя
    winner_size: int          # len(winner_nodes)
    winner_score: float       # взвешенный score выбора
    dominance: float          # winner_size / total_active_nodes ∈ [0, 1]
                              # total_active_nodes = число уникальных узлов во всех кластерах


class GlobalWorkspace:
    """Selects the dominant SKS via weighted score.

    score_k = w_size * size_k
             + w_coherence * coherence_k      # reserved, currently 0
             + w_pred * (1 - pred_error_k)    # reserved, currently 0

    Default: w_size=1.0, w_coherence=0.0, w_pred=0.0 (pure size-based).

    fired_history: passed for future coherence computation.
    When w_coherence=0.0, ignored. If fired_history=None, coherence_k=0.0.
    """

    def __init__(self, config: GWSConfig | None = None) -> None:
        if config is None:
            config = GWSConfig()
        self.config = config

    def select_winner(
        self,
        sks_clusters: dict[int, set[int]],
        fired_history: torch.Tensor | None,
    ) -> GWSState | None:
        """Return None if sks_clusters is empty."""
        if not sks_clusters:
            return None

        # Compute total active nodes (union of all clusters)
        all_nodes: set[int] = set()
        for nodes in sks_clusters.values():
            all_nodes |= nodes
        total_active = len(all_nodes)

        cfg = self.config
        best_id: int | None = None
        best_score = -float("inf")

        for cluster_id, nodes in sks_clusters.items():
            size_k = len(nodes)
            # coherence_k and pred_error_k reserved; currently 0
            coherence_k = 0.0
            pred_error_k = 0.0

            score = (
                cfg.w_size * size_k
                + cfg.w_coherence * coherence_k
                + cfg.w_pred * (1.0 - pred_error_k)
            )

            if score > best_score:
                best_score = score
                best_id = cluster_id

        winner_id = best_id  # type: ignore[assignment]
        winner_nodes = sks_clusters[winner_id]
        winner_size = len(winner_nodes)
        dominance = winner_size / total_active if total_active > 0 else 0.0

        return GWSState(
            winner_id=winner_id,
            winner_nodes=winner_nodes,
            winner_size=winner_size,
            winner_score=best_score,
            dominance=dominance,
        )
