"""SKS detection: phase coherence matrix + DBSCAN clustering."""

from __future__ import annotations

import torch
import numpy as np
from sklearn.cluster import DBSCAN


def phase_coherence_matrix(
    states: torch.Tensor,
    top_k: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise phase coherence for top-K most active nodes.

    Activity = amplitude (states[:, 1]). If all amplitudes are zero,
    falls back to abs(states[:, 0]).

    Args:
        states: (N, 8) oscillator states. Column 0 = phase.
        top_k: max nodes to include.

    Returns:
        coherence: (K, K) float32, cos(φ_i - φ_j).
        active_indices: (K,) int64, global node indices.
    """
    N = states.shape[0]
    K = min(top_k, N)

    # Activity: prefer amplitude (col 1), fallback to abs(phase)
    activity = states[:, 1].abs()
    if activity.sum() == 0:
        activity = states[:, 0].abs()

    _, active_idx = torch.topk(activity, K, sorted=False)
    phases = states[active_idx, 0]  # (K,)

    # cos(φ_i - φ_j) = cos(φ_i)cos(φ_j) + sin(φ_i)sin(φ_j)
    cos_p = torch.cos(phases)
    sin_p = torch.sin(phases)
    coherence = cos_p.unsqueeze(1) * cos_p.unsqueeze(0) + sin_p.unsqueeze(1) * sin_p.unsqueeze(0)

    return coherence, active_idx


def cofiring_coherence_matrix(
    fired_history: torch.Tensor,
    top_k: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute co-firing correlation matrix for FHN model.

    Selects top-K nodes by total spike count.

    Args:
        fired_history: (T, N) bool.
        top_k: max nodes to include.

    Returns:
        coherence: (K, K) float32, Pearson correlation of firing vectors.
        active_indices: (K,) int64.
    """
    T, N = fired_history.shape
    K = min(top_k, N)

    spike_counts = fired_history.float().sum(dim=0)  # (N,)
    _, active_idx = torch.topk(spike_counts, K, sorted=False)

    # Firing vectors for selected nodes: (T, K)
    vectors = fired_history[:, active_idx].float()

    # Pearson correlation
    mean = vectors.mean(dim=0, keepdim=True)  # (1, K)
    centered = vectors - mean  # (T, K)
    # std per node (L2 norm of centered vector)
    norms = centered.norm(dim=0, keepdim=True).clamp(min=1e-8)  # (1, K)
    normed = centered / norms  # (T, K)

    coherence = normed.T @ normed  # (K, K) — already normalized
    coherence = coherence.clamp(-1.0, 1.0)

    return coherence, active_idx


def detect_sks(
    coherence: torch.Tensor,
    method: str = "dbscan",
    eps: float = 0.3,
    min_samples: int = 10,
    min_size: int = 10,
) -> list[set[int]]:
    """Cluster nodes using DBSCAN on distance = 1 - |coherence|.

    Args:
        coherence: (K, K) coherence matrix.
        method: clustering method (only "dbscan" supported).
        eps: DBSCAN distance threshold.
        min_samples: DBSCAN min neighbors.
        min_size: discard clusters smaller than this.

    Returns:
        List of sets of LOCAL node indices (0..K-1).
    """
    K = coherence.shape[0]
    distance = 1.0 - coherence.abs().cpu().numpy()
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(distance)

    clusters: list[set[int]] = []
    for label in set(labels):
        if label == -1:
            continue
        members = set(int(i) for i in np.where(labels == label)[0])
        if len(members) >= min_size:
            clusters.append(members)

    return clusters
