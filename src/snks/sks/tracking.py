"""SKS tracking: stable cluster IDs via Hungarian matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class SKSTracker:
    """Assigns stable IDs to SKS clusters across detection cycles.

    Uses Jaccard similarity + Hungarian algorithm to match new clusters
    to previously tracked ones.
    """

    def __init__(self) -> None:
        self._next_id: int = 0
        self._prev_clusters: dict[int, set[int]] = {}

    def update(self, clusters: list[set[int]]) -> dict[int, set[int]]:
        """Match new clusters to previous, assign stable IDs.

        Args:
            clusters: list of node-index sets from detect_sks.

        Returns:
            {sks_id: node_set} with stable IDs.
        """
        if not clusters:
            self._prev_clusters = {}
            return {}

        if not self._prev_clusters:
            result = {}
            for c in clusters:
                result[self._next_id] = c
                self._next_id += 1
            self._prev_clusters = dict(result)
            return result

        prev_ids = list(self._prev_clusters.keys())
        prev_sets = [self._prev_clusters[k] for k in prev_ids]
        n_prev = len(prev_sets)
        n_new = len(clusters)

        # Cost matrix: 1 - Jaccard similarity
        cost = np.ones((n_new, n_prev), dtype=np.float64)
        for i, cn in enumerate(clusters):
            for j, cp in enumerate(prev_sets):
                inter = len(cn & cp)
                union = len(cn | cp)
                if union > 0:
                    cost[i, j] = 1.0 - inter / union

        row_ind, col_ind = linear_sum_assignment(cost)

        result: dict[int, set[int]] = {}
        matched_new: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            jaccard = 1.0 - cost[r, c]
            if jaccard > 0.1:  # minimum overlap threshold
                result[prev_ids[c]] = clusters[r]
                matched_new.add(r)

        # Unmatched new clusters get new IDs
        for i, c in enumerate(clusters):
            if i not in matched_new:
                result[self._next_id] = c
                self._next_id += 1

        self._prev_clusters = dict(result)
        return result
