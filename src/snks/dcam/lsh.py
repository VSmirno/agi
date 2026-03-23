"""LSH (Locality-Sensitive Hashing) index using SimHash.

Provides approximate nearest neighbor search for HAC vectors.
Uses L hash tables with K-bit SimHash for O(L*K) query time.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


class LSHIndex:
    """SimHash-based LSH index for cosine similarity search."""

    def __init__(
        self,
        dim: int,
        n_tables: int = 32,
        n_bits: int = 16,
        device: torch.device | None = None,
    ) -> None:
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.device = device or torch.device("cpu")

        # Random hyperplanes for hashing: (n_tables, n_bits, dim)
        self.projections = torch.randn(
            n_tables, n_bits, dim, device=self.device, dtype=torch.float32
        )

        # Hash tables: table_idx → hash_code → [value_ids]
        self._tables: list[dict[int, list[int]]] = [
            defaultdict(list) for _ in range(n_tables)
        ]
        # Stored vectors for fallback linear search and re-ranking
        self._vectors: dict[int, Tensor] = {}
        # For O(1) removal: value_id → list of hash codes per table
        self._id_to_hashes: dict[int, list[int]] = {}

    def _hash_vector(self, v: Tensor) -> list[int]:
        """Compute SimHash codes for all tables. Returns list of L int codes."""
        # (n_tables, n_bits, dim) @ (dim,) → (n_tables, n_bits)
        dots = torch.matmul(self.projections, v.float())
        bits = (dots > 0).int()  # (n_tables, n_bits)
        # Pack bits into integers
        powers = (2 ** torch.arange(self.n_bits, device=self.device)).int()
        codes = (bits * powers).sum(dim=1)  # (n_tables,)
        return codes.tolist()

    def insert(self, key: Tensor, value: int) -> None:
        """Insert a vector with associated value_id."""
        codes = self._hash_vector(key)
        self._id_to_hashes[value] = codes
        self._vectors[value] = key.detach()
        for t, code in enumerate(codes):
            self._tables[t][code].append(value)

    def query(self, key: Tensor, top_k: int = 10) -> list[tuple[int, float]]:
        """Find top_k most similar vectors. Returns [(value_id, similarity)]."""
        if not self._vectors:
            return []

        codes = self._hash_vector(key)

        # Gather candidates from all tables
        candidates: set[int] = set()
        for t, code in enumerate(codes):
            candidates.update(self._tables[t].get(code, []))

        # Fallback: if no candidates found, scan all vectors
        if not candidates:
            candidates = set(self._vectors.keys())

        # Re-rank by cosine similarity
        key_f = key.float()
        results: list[tuple[int, float]] = []
        for vid in candidates:
            v = self._vectors[vid]
            sim = torch.nn.functional.cosine_similarity(
                key_f.unsqueeze(0), v.float().unsqueeze(0)
            ).item()
            results.append((vid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def remove(self, value: int) -> None:
        """Remove a value_id from the index."""
        if value not in self._id_to_hashes:
            return
        codes = self._id_to_hashes.pop(value)
        self._vectors.pop(value, None)
        for t, code in enumerate(codes):
            bucket = self._tables[t].get(code, [])
            if value in bucket:
                bucket.remove(value)
