"""Stage 83: VectorWorldModel — embedding-based world model via binary HDC.

Replaces symbolic ConceptStore with vector associations in a shared
binary hyperdimensional space. Concepts, actions, and effects are all
binary vectors. Causal knowledge stored in SDM (Sparse Distributed Memory).
Prediction = SDM read. Learning = SDM write. Generalization = similar
vectors → similar predictions, free from vector algebra.

Design spec: docs/superpowers/specs/2026-04-12-stage83-vector-world-model-design.md
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# BitVector operations (binary XOR algebra)
# ---------------------------------------------------------------------------

def random_bitvector(dim: int, device: torch.device | None = None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Random binary vector {0, 1}^dim."""
    return torch.randint(0, 2, (dim,), dtype=torch.float32,
                         generator=generator, device="cpu").to(device or torch.device("cpu"))


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XOR binding — self-inverse: bind(bind(a, b), b) == a."""
    return (a + b) % 2


def bundle(vecs: list[torch.Tensor],
           weights: list[float] | None = None) -> torch.Tensor:
    """Majority-vote bundling with optional weights.

    For weighted bundle: multiply each vector by its weight before summing,
    then threshold at half the total weight.
    """
    if not vecs:
        raise ValueError("Cannot bundle empty list")
    if len(vecs) == 1:
        return vecs[0].clone()
    stacked = torch.stack(vecs)
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=stacked.device)
        summed = (stacked * w.unsqueeze(1)).sum(dim=0)
        threshold = w.sum().item() / 2.0
    else:
        summed = stacked.sum(dim=0)
        threshold = len(vecs) / 2.0
    return (summed > threshold).float()


def hamming_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized Hamming similarity: fraction of matching bits."""
    return (a == b).float().mean().item()


def encode_scalar(value: int, dim: int, max_val: int = 10) -> torch.Tensor:
    """Thermometer encoding for small non-negative integers.

    Value K → first K * (dim // max_val) bits = 1, rest = 0.
    Invertible via popcount. Crafter values are 0-9.
    """
    bits_per_unit = dim // max_val
    n_ones = min(value, max_val) * bits_per_unit
    vec = torch.zeros(dim, dtype=torch.float32)
    if n_ones > 0:
        vec[:n_ones] = 1.0
    return vec


def decode_scalar(vec: torch.Tensor, max_val: int = 10) -> int:
    """Inverse thermometer: popcount / bits_per_unit, rounded."""
    dim = vec.shape[0]
    bits_per_unit = dim // max_val
    if bits_per_unit == 0:
        return 0
    n_ones = (vec > 0.5).sum().item()
    return min(round(n_ones / bits_per_unit), max_val)


# ---------------------------------------------------------------------------
# CausalSDM — associative memory for (concept, action) → effect
# ---------------------------------------------------------------------------

class CausalSDM:
    """SDM storing causal associations: bind(concept, action) → effect.

    Adapted from vsa_world_model.SDMMemory. Single content store for
    effect vectors instead of separate next-state and reward stores.
    """

    def __init__(self, n_locations: int = 5000, dim: int = 65536,
                 seed: int = 42, device: torch.device | str | None = None):
        self.n_locations = n_locations
        self.dim = dim
        self.n_writes = 0
        self.device = torch.device(device) if device else torch.device("cpu")

        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        # Generate addresses on CPU (randint generator must be CPU),
        # then move to device for all subsequent operations
        addresses_cpu = torch.randint(
            0, 2, (n_locations, dim), dtype=torch.float32, generator=rng,
        )
        self.addresses = addresses_cpu.to(self.device)

        # Calibrate on device (GPU) — vectorized, no loops
        self.activation_radius = self._calibrate_radius()

        # Content: ±1 accumulated counters
        self.content = torch.zeros(
            n_locations, dim, dtype=torch.float32, device=self.device,
        )

    def _calibrate_radius(self) -> int:
        """Find radius so 1-10% of locations activate.

        Fully vectorized on device (GPU). For high-dim binary vectors,
        distances concentrate around dim/2. We find the 5th percentile
        of pairwise distances in one batched operation.
        """
        # Sample subset for distance computation
        n_sample = min(self.n_locations, 300)
        sample = self.addresses[:n_sample]  # already on device

        # Batch pairwise distances: (n_probes, n_sample)
        n_probes = min(50, n_sample)
        probes = sample[:n_probes]  # (n_probes, dim)
        # Vectorized: each probe vs all samples
        dists = (probes.unsqueeze(1) != sample.unsqueeze(0)).sum(dim=2)  # (n_probes, n_sample)
        dists_flat = dists.reshape(-1)

        # 5th percentile → ~5% of locations activate
        target_pct_idx = max(1, int(dists_flat.numel() * 0.05))
        radius = int(dists_flat.kthvalue(target_pct_idx).values.item())

        # Verify with a single probe
        query = self.addresses[0]
        n_act = self._count_activated(query, radius)
        pct = n_act / self.n_locations

        # If way off, nudge ±5% until in range
        for _ in range(10):
            if 0.005 <= pct <= 0.15:
                break
            if pct < 0.005:
                radius = int(radius * 1.02)
            else:
                radius = int(radius * 0.98)
            n_act = self._count_activated(query, radius)
            pct = n_act / self.n_locations

        return radius

    def _count_activated(self, query: torch.Tensor, radius: int) -> int:
        dists = (self.addresses != query.unsqueeze(0)).sum(dim=1)
        return int((dists <= radius).sum().item())

    def _get_activated_mask(self, address: torch.Tensor) -> torch.Tensor:
        address_dev = address.to(self.device)
        dists = (self.addresses != address_dev.unsqueeze(0)).sum(dim=1)
        return dists <= self.activation_radius

    def write(self, address: torch.Tensor, data: torch.Tensor) -> None:
        """Write data at address. Accumulates via ±1 updates."""
        mask = self._get_activated_mask(address)
        update = (2 * data - 1).to(self.device)  # {0,1} → {-1,+1}
        self.content[mask] += update.unsqueeze(0)
        self.n_writes += 1

    def read(self, address: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Read from address. Returns (binary vector, confidence).

        Confidence = mean absolute magnitude of counters (higher = more writes).
        """
        mask = self._get_activated_mask(address)
        n_activated = mask.sum().item()

        if n_activated == 0:
            return torch.zeros(self.dim, dtype=torch.float32,
                               device=self.device), 0.0

        summed = self.content[mask].sum(dim=0)
        predicted = (summed > 0).float()
        magnitude = summed.abs().mean().item()
        confidence = min(magnitude / 5.0, 1.0)
        return predicted, confidence

    def state_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "addresses": self.addresses.cpu(),
            "content": self.content.cpu(),
            "activation_radius": self.activation_radius,
            "n_writes": self.n_writes,
            "n_locations": self.n_locations,
            "dim": self.dim,
        }

    def load_state_dict(self, d: dict) -> None:
        """Load state, replacing addresses and merging content additively."""
        if d["dim"] != self.dim:
            raise ValueError(f"Dimension mismatch: {d['dim']} vs {self.dim}")
        if d["n_locations"] != self.n_locations:
            raise ValueError(f"Location count mismatch: {d['n_locations']} vs {self.n_locations}")
        # Replace addresses (must match for content to be meaningful)
        self.addresses = d["addresses"].to(self.device)
        self.activation_radius = d["activation_radius"]
        # Additive merge of content counters
        loaded_content = d["content"].to(self.device)
        self.content += loaded_content
        self.n_writes += d["n_writes"]


# ---------------------------------------------------------------------------
# VectorWorldModel
# ---------------------------------------------------------------------------

class VectorWorldModel:
    """Embedding-based world model using binary HDC + SDM.

    Concepts, actions, and roles are binary vectors. Causal knowledge
    stored as associations in CausalSDM. Prediction via SDM read,
    learning via SDM write. Generalization through vector similarity.
    """

    def __init__(self, dim: int = 65536, n_locations: int = 5000,
                 seed: int = 42, device: torch.device | str | None = None):
        self.dim = dim
        self.device = torch.device(device) if device else torch.device("cpu")
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

        # Concept embeddings — evolve through experience
        self.concepts: dict[str, torch.Tensor] = {}
        # Action embeddings
        self.actions: dict[str, torch.Tensor] = {}
        # Role vectors for effect encoding/decoding
        self.roles: dict[str, torch.Tensor] = {}

        # Associative memory
        self.memory = CausalSDM(
            n_locations=n_locations, dim=dim, seed=seed, device=self.device,
        )

        # Scalar encoding params
        self.max_scalar = 10

    def _ensure_concept(self, concept_id: str) -> torch.Tensor:
        if concept_id not in self.concepts:
            self.concepts[concept_id] = random_bitvector(
                self.dim, self.device, self._rng,
            )
        return self.concepts[concept_id]

    def _ensure_action(self, action_id: str) -> torch.Tensor:
        if action_id not in self.actions:
            self.actions[action_id] = random_bitvector(
                self.dim, self.device, self._rng,
            )
        return self.actions[action_id]

    def _ensure_role(self, role_name: str) -> torch.Tensor:
        if role_name not in self.roles:
            self.roles[role_name] = random_bitvector(
                self.dim, self.device, self._rng,
            )
        return self.roles[role_name]

    def encode_effect(self, deltas: dict[str, float]) -> torch.Tensor:
        """Encode effect dict as single binary vector.

        {wood: +1, health: -3} → bundle([bind(v_wood, enc(1)), bind(v_health, enc(-3))])

        Negative values use bind with a special NEG role before encoding
        the absolute value, so decode can distinguish +3 from -3.
        """
        if not deltas:
            return torch.zeros(self.dim, dtype=torch.float32, device=self.device)

        parts = []
        neg_role = self._ensure_role("__NEG__")
        for var, val in deltas.items():
            role_vec = self._ensure_role(var)
            if val < 0:
                scalar_vec = encode_scalar(
                    abs(int(val)), self.dim, self.max_scalar,
                ).to(self.device)
                parts.append(bind(bind(role_vec, neg_role), scalar_vec))
            else:
                scalar_vec = encode_scalar(
                    int(val), self.dim, self.max_scalar,
                ).to(self.device)
                parts.append(bind(role_vec, scalar_vec))
        return bundle(parts)

    def decode_effect(self, effect_vector: torch.Tensor) -> dict[str, int]:
        """Decode effect vector by unbinding each known role.

        Returns dict of {role_name: value} for roles with decoded
        value != 0 and reasonable similarity (> 0.55).
        """
        result: dict[str, int] = {}
        neg_role = self._ensure_role("__NEG__")

        for role_name, role_vec in self.roles.items():
            if role_name == "__NEG__":
                continue

            # Try positive
            unbound = bind(effect_vector, role_vec)
            val = decode_scalar(unbound, self.max_scalar)
            # Check similarity to confirm this role is actually present
            reconstructed = encode_scalar(val, self.dim, self.max_scalar).to(self.device)
            sim = hamming_similarity(unbound, reconstructed)

            # Try negative
            unbound_neg = bind(effect_vector, bind(role_vec, neg_role))
            val_neg = decode_scalar(unbound_neg, self.max_scalar)
            reconstructed_neg = encode_scalar(val_neg, self.dim, self.max_scalar).to(self.device)
            sim_neg = hamming_similarity(unbound_neg, reconstructed_neg)

            # Pick whichever has higher similarity
            if sim > sim_neg and sim > 0.55 and val != 0:
                result[role_name] = val
            elif sim_neg > sim and sim_neg > 0.55 and val_neg != 0:
                result[role_name] = -val_neg

        return result

    def predict(self, concept_id: str, action: str) -> tuple[torch.Tensor, float]:
        """Predict effect of action on concept.

        Returns (effect_vector, confidence). Confidence 0 = no knowledge.
        """
        v_concept = self._ensure_concept(concept_id)
        v_action = self._ensure_action(action)
        address = bind(v_concept, v_action)
        return self.memory.read(address)

    def learn(self, concept_id: str, action: str,
              observed_effect: dict[str, float],
              context_vectors: list[torch.Tensor] | None = None) -> float:
        """Learn from observation. Returns surprise (0..1).

        1. Predict before learning
        2. Encode observed effect
        3. Write to SDM
        4. Update concept embedding with context
        5. Return surprise
        """
        # Predict before writing
        predicted, confidence = self.predict(concept_id, action)

        # Encode observed
        observed_vec = self.encode_effect(observed_effect)

        # Write association
        v_concept = self._ensure_concept(concept_id)
        v_action = self._ensure_action(action)
        address = bind(v_concept, v_action)
        self.memory.write(address, observed_vec)

        # Compute surprise
        if confidence < 0.01:
            surprise = 1.0  # No prior knowledge = max surprise
        else:
            surprise = 1.0 - hamming_similarity(predicted, observed_vec)

        # Update concept embedding with context
        if context_vectors and surprise > 0.1:
            ctx = bundle(context_vectors)
            weight = min(surprise, 0.3)  # Cap context influence
            self.concepts[concept_id] = bundle(
                [v_concept, ctx], weights=[1.0 - weight, weight],
            )

        return surprise

    def query_similar(self, concept_id: str, top_k: int = 5
                      ) -> list[tuple[str, float]]:
        """Find concepts with most similar embeddings."""
        if concept_id not in self.concepts:
            return []
        query = self.concepts[concept_id]
        results = []
        for cid, vec in self.concepts.items():
            if cid == concept_id:
                continue
            sim = hamming_similarity(query, vec)
            results.append((cid, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # --- Persistence (knowledge flow) ---

    def save(self, path: str | Path) -> None:
        """Save full model state for knowledge transfer."""
        path = Path(path)
        torch.save({
            "dim": self.dim,
            "max_scalar": self.max_scalar,
            "concepts": {k: v.cpu() for k, v in self.concepts.items()},
            "actions": {k: v.cpu() for k, v in self.actions.items()},
            "roles": {k: v.cpu() for k, v in self.roles.items()},
            "memory": self.memory.state_dict(),
        }, path)

    def load(self, path: str | Path) -> bool:
        """Load and merge experience from a previous generation.

        Returns False if file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            return False

        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["dim"] != self.dim:
            raise ValueError(f"Dimension mismatch: {data['dim']} vs {self.dim}")

        # Merge concepts: bundle existing with loaded (50/50)
        for cid, vec in data["concepts"].items():
            vec_dev = vec.to(self.device)
            if cid in self.concepts:
                self.concepts[cid] = bundle(
                    [self.concepts[cid], vec_dev], weights=[0.5, 0.5],
                )
            else:
                self.concepts[cid] = vec_dev

        # Merge actions/roles: loaded takes precedence if not present
        for aid, vec in data["actions"].items():
            if aid not in self.actions:
                self.actions[aid] = vec.to(self.device)
        for rid, vec in data["roles"].items():
            if rid not in self.roles:
                self.roles[rid] = vec.to(self.device)

        # Merge SDM additively
        self.memory.load_state_dict(data["memory"])
        return True
