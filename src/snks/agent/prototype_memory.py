"""Stage 66 v2: Prototype Memory — k-NN in continuous latent space.

Stores (z_real, action, outcome) prototypes.
Query: find k nearest z with matching action, majority vote on outcome.
Replaces symbolic decode → neocortex lookup for pixel path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
    """k-NN world model in continuous z_real space.

    Stores prototypes as (z_real, action, outcome) tuples.
    Query by cosine similarity with action filtering.
    """

    def __init__(self, dim: int = 2048, k: int = 5, min_similarity: float = 0.3):
        self.dim = dim
        self.k = k
        self.min_similarity = min_similarity

        self.z_store: list[torch.Tensor] = []  # each (dim,) L2-normalized
        self.actions: list[str] = []
        self.outcomes: list[dict] = []

    def add(self, z_real: torch.Tensor, action: str, outcome: dict) -> None:
        """Add a prototype to memory."""
        z = z_real.detach().float()
        if z.dim() > 1:
            z = z.squeeze(0)
        z = F.normalize(z, dim=0)
        self.z_store.append(z)
        self.actions.append(action)
        self.outcomes.append(outcome)

    def query(self, z_query: torch.Tensor, action: str) -> tuple[dict, float]:
        """Find k nearest prototypes with matching action, vote on outcome.

        Returns:
            (outcome_dict, confidence). If no match: ({"result": "unknown"}, 0.0).
        """
        if not self.z_store:
            return {"result": "unknown"}, 0.0

        z_q = z_query.detach().float()
        if z_q.dim() > 1:
            z_q = z_q.squeeze(0)
        z_q = F.normalize(z_q, dim=0)

        # Filter by action
        indices = [i for i, a in enumerate(self.actions) if a == action]
        if not indices:
            return {"result": "unknown"}, 0.0

        # Cosine similarity (z_store already L2-normalized)
        z_filtered = torch.stack([self.z_store[i] for i in indices])
        sims = z_filtered @ z_q  # (n_filtered,)

        # Top-k
        k = min(self.k, len(indices))
        top_sims, top_idx = sims.topk(k)

        # Check minimum similarity
        if top_sims[0].item() < self.min_similarity:
            return {"result": "unknown"}, 0.0

        # Majority vote on outcome["result"]
        votes: dict[str, list[float]] = {}
        for sim_val, idx in zip(top_sims, top_idx):
            outcome = self.outcomes[indices[idx.item()]]
            result = outcome.get("result", "unknown")
            votes.setdefault(result, []).append(sim_val.item())

        # Find majority
        best_result = max(votes, key=lambda r: len(votes[r]))
        vote_fraction = len(votes[best_result]) / k
        mean_sim = sum(votes[best_result]) / len(votes[best_result])
        confidence = vote_fraction * mean_sim

        # Return full outcome from the nearest prototype with majority result
        for sim_val, idx in zip(top_sims, top_idx):
            outcome = self.outcomes[indices[idx.item()]]
            if outcome.get("result") == best_result:
                return outcome, confidence

        return {"result": best_result}, confidence

    def __len__(self) -> int:
        return len(self.z_store)

    def stats(self) -> dict[str, int]:
        """Per-action prototype counts."""
        counts: dict[str, int] = {}
        for a in self.actions:
            counts[a] = counts.get(a, 0) + 1
        return counts
