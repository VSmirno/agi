"""SKS quality metrics: NMI, stability, separability."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score


def compute_nmi(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Normalized Mutual Information between predicted and true cluster labels."""
    return float(normalized_mutual_info_score(true_labels, predicted_labels))


def sks_stability(
    sks_t1: list[set[int]], sks_t2: list[set[int]]
) -> float:
    """Jaccard-based stability between two SKS snapshots.

    For each cluster in t1, find best Jaccard match in t2.
    Returns mean of best matches (averaged over both directions).
    """
    if not sks_t1 or not sks_t2:
        return 0.0

    def _mean_max_jaccard(a: list[set[int]], b: list[set[int]]) -> float:
        scores = []
        for ca in a:
            best = 0.0
            for cb in b:
                inter = len(ca & cb)
                union = len(ca | cb)
                if union > 0:
                    best = max(best, inter / union)
            scores.append(best)
        return sum(scores) / len(scores)

    forward = _mean_max_jaccard(sks_t1, sks_t2)
    backward = _mean_max_jaccard(sks_t2, sks_t1)
    return (forward + backward) / 2.0


def sks_separability(
    clusters: list[set[int]], states: torch.Tensor
) -> float:
    """Inter-cluster / intra-cluster phase distance ratio.

    Higher = better separated clusters.
    Returns 0 if fewer than 2 clusters.
    """
    if len(clusters) < 2:
        return 0.0

    phases = states[:, 0]

    # Intra-cluster: mean pairwise |phase diff| within each cluster
    intra_dists = []
    for c in clusters:
        nodes = sorted(c)
        if len(nodes) < 2:
            continue
        p = phases[nodes]
        diffs = (p.unsqueeze(0) - p.unsqueeze(1)).abs()
        # Exclude diagonal
        n = len(nodes)
        mask = ~torch.eye(n, dtype=torch.bool, device=diffs.device)
        intra_dists.append(diffs[mask].mean().item())

    if not intra_dists:
        return 0.0
    mean_intra = sum(intra_dists) / len(intra_dists)

    # Inter-cluster: mean pairwise |phase diff| between cluster centroids
    centroids = []
    for c in clusters:
        nodes = sorted(c)
        centroids.append(phases[nodes].mean().item())

    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(abs(centroids[i] - centroids[j]))

    mean_inter = sum(inter_dists) / len(inter_dists) if inter_dists else 0.0

    if mean_intra < 1e-8:
        return float(mean_inter) if mean_inter > 0 else 0.0

    return mean_inter / mean_intra
