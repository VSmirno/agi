"""SKSEmbedder: computes HAC-vector representations for SKS clusters (Stage 9).

Each node gets a fixed random unit vector (item memory).
SKS embedding = bundle (sum + normalize) of its member node vectors.
Diversity is guaranteed by HAC geometry: random unit vectors in high-D space
are nearly orthogonal, so different clusters produce distinct embeddings.
"""

from __future__ import annotations

import torch
from torch import Tensor

from snks.device import get_device


class SKSEmbedder:
    """Maps each active SKS cluster to a unit vector in HAC space.

    Item memory is initialized once (random unit vectors) and never updated.
    Embedding for cluster C = normalize(sum(item_memory[nodes in C])).
    """

    def __init__(self, n_nodes: int, hac_dim: int, device: str = "auto") -> None:
        dev = get_device(device)
        # Fixed random unit vectors — never updated
        vecs = torch.randn(n_nodes, hac_dim, device=dev)
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._item_memory: Tensor = vecs / norms  # (n_nodes, hac_dim)
        self._hac_dim = hac_dim

    def embed(self, sks_clusters: dict[int, set[int]]) -> dict[int, Tensor]:
        """Compute HAC embedding for each active SKS cluster.

        Args:
            sks_clusters: dict mapping SKS ID → set of node indices.

        Returns:
            dict mapping SKS ID → unit vector of shape (hac_dim,).
        """
        result: dict[int, Tensor] = {}
        for sks_id, nodes in sks_clusters.items():
            if not nodes:
                continue
            idx = torch.tensor(sorted(nodes), dtype=torch.long,
                               device=self._item_memory.device)
            vecs = self._item_memory[idx]          # (k, hac_dim)
            bundle = vecs.sum(dim=0)               # (hac_dim,)
            norm = bundle.norm().clamp(min=1e-8)
            result[sks_id] = bundle / norm
        return result
