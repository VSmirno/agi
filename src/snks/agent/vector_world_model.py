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
import yaml
from pathlib import Path


EFFECT_ADDRESS_VERSION = 1
EFFECT_INTEGRITY_VERSION = 1
EFFECT_ROLE_NAME = "__EFFECT__"
NON_EFFECT_DECODE_ROLES = {
    "__NEG__",
    EFFECT_ROLE_NAME,
    "__OUTCOME_H__",
    "__OPTION_OUTCOME_H__",
    "survived_h",
    "damage_h",
    "died_to",
}
DEFAULT_TEXTBOOK_PATH = Path(__file__).resolve().parents[3] / "configs" / "crafter_textbook.yaml"
CORE_EFFECT_REPAIR_PAIRS = {
    ("table", "place"),
    ("wood_sword", "make"),
    ("wood_pickaxe", "make"),
}


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

    def __init__(self, n_locations: int = 50000, dim: int = 16384,
                 seed: int = 42, device: torch.device | str | None = None):
        self.n_locations = n_locations
        self.dim = dim
        self.seed = int(seed)
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

        On-device (GPU) but memory-efficient: computes distances one
        probe at a time to avoid OOM on large dim×n_locations tensors.
        """
        n_probes = min(30, self.n_locations)
        all_dists: list[torch.Tensor] = []

        for i in range(n_probes):
            # (n_locations,) — one probe vs all addresses
            d = (self.addresses[i].unsqueeze(0) != self.addresses).sum(dim=1)
            all_dists.append(d)

        dists_flat = torch.cat(all_dists)  # (n_probes * n_locations,)

        # 0.5th percentile → ~0.5% of locations activate (SNR ~15)
        target_pct_idx = max(1, int(dists_flat.numel() * 0.005))
        # kthvalue lacks a CUDA-deterministic implementation in torch 2.5.1+cu121,
        # so it blocks torch.use_deterministic_algorithms(True). Offload to CPU:
        # one-shot init call, perf cost negligible, numerics identical.
        radius = int(dists_flat.cpu().kthvalue(target_pct_idx).values.item())

        # Verify and nudge toward 0.3-1.5% activation band
        query = self.addresses[0]
        for _ in range(20):
            n_act = self._count_activated(query, radius)
            pct = n_act / self.n_locations
            if 0.003 <= pct <= 0.015:
                break
            if pct < 0.003:
                radius = int(radius * 1.01)
            else:
                radius = int(radius * 0.99)

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

        Confidence = per-location mean magnitude of counters (signal ≫ noise).
        """
        mask = self._get_activated_mask(address)
        n_activated = mask.sum().item()

        if n_activated == 0:
            return torch.zeros(self.dim, dtype=torch.float32,
                               device=self.device), 0.0

        summed = self.content[mask].sum(dim=0)
        predicted = (summed > 0).float()
        mean_content = summed / n_activated
        confidence = min(mean_content.abs().mean().item(), 1.0)
        return predicted, confidence

    def batch_read(
        self, addresses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched read for K addresses in one GPU op.

        Uses hamming-via-matmul: dist(q,a) = |q| + |a| - 2·(q·a)

        Args:
            addresses: (K, dim) binary queries

        Returns:
            predictions: (K, dim) binary predictions
            confidences: (K,) float in [0, 1]
        """
        addresses = addresses.to(self.device)
        K = addresses.shape[0]

        # Hamming distances via matmul: (K, N)
        q_norm = addresses.sum(dim=1, keepdim=True)  # (K, 1)
        a_norm = self.addresses.sum(dim=1, keepdim=True).T  # (1, N)
        dot = addresses @ self.addresses.T  # (K, N)
        dists = q_norm + a_norm - 2.0 * dot  # (K, N)

        # Activation masks
        masks = (dists <= self.activation_radius).float()  # (K, N)

        # Batched content sum via matmul: (K, dim)
        summed = masks @ self.content

        # Predictions and confidences
        n_activated = masks.sum(dim=1)  # (K,)
        safe_n = n_activated.clamp(min=1.0).unsqueeze(1)  # (K, 1)
        mean_content = summed / safe_n  # (K, dim)

        predictions = (summed > 0).float()  # (K, dim)
        confidences = mean_content.abs().mean(dim=1).clamp(max=1.0)  # (K,)

        # Zero confidence where no locations activated
        zero_mask = n_activated == 0
        confidences = torch.where(
            zero_mask, torch.zeros_like(confidences), confidences,
        )

        return predictions, confidences

    def state_dict(self) -> dict:
        """Serialize for persistence.

        Addresses are deterministic from `seed` (see __init__), so we
        store the seed instead of the (n_locations × dim × 4) addresses
        tensor — cuts the on-disk size of a VectorWorldModel snapshot in
        half (~3.3 GB → 0.0 saved by addresses, content stays the same).
        Legacy snapshots that still contain `addresses` are loaded
        unchanged by load_state_dict().
        """
        return {
            "seed": self.seed,
            "content": self.content.cpu(),
            "activation_radius": self.activation_radius,
            "n_writes": self.n_writes,
            "n_locations": self.n_locations,
            "dim": self.dim,
        }

    def load_state_dict(self, d: dict) -> None:
        """Load state, restoring addresses and merging content additively.

        Two supported snapshot layouts:
          - new (preferred): contains `seed`; addresses are regenerated
            with `torch.Generator().manual_seed(seed)` on load.
          - legacy: contains an `addresses` tensor; used verbatim.

        Content counters are merged additively into whatever the current
        instance holds (so a fresh-init zero content becomes the loaded
        content; an already-trained instance accumulates the loaded
        history on top of its own writes).
        """
        if d["dim"] != self.dim:
            raise ValueError(f"Dimension mismatch: {d['dim']} vs {self.dim}")
        if d["n_locations"] != self.n_locations:
            raise ValueError(f"Location count mismatch: {d['n_locations']} vs {self.n_locations}")
        if "addresses" in d:
            self.addresses = d["addresses"].to(self.device)
        elif "seed" in d:
            self.seed = int(d["seed"])
            rng = torch.Generator(device="cpu")
            rng.manual_seed(self.seed)
            addresses_cpu = torch.randint(
                0, 2, (self.n_locations, self.dim),
                dtype=torch.float32, generator=rng,
            )
            self.addresses = addresses_cpu.to(self.device)
        else:
            raise ValueError("CausalSDM snapshot has neither 'addresses' nor 'seed'")
        self.activation_radius = d["activation_radius"]
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

    def __init__(self, dim: int = 16384, n_locations: int = 50000,
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
        self.effect_address_version = EFFECT_ADDRESS_VERSION
        self.effect_integrity_version = 0
        self.effect_repair_verified = False
        self.last_effect_repair_stats: dict | None = None

        # Associative memory
        self.memory = CausalSDM(
            n_locations=n_locations, dim=dim, seed=seed, device=self.device,
        )

        # Scalar encoding params
        self.max_scalar = 10

        # Action requirements — facts from textbook (category 1).
        # Dict: (concept_id, action) → {required_item: min_count}
        # E.g., ("iron", "do") → {"stone_pickaxe": 1}
        # Used by planner to filter plans whose requirements aren't met.
        self.action_requirements: dict[tuple[str, str], dict[str, int]] = {}
        # Adjacency-requirement facts from textbook: which concept must occupy
        # one of the four cardinal-adjacent tiles for an action to succeed
        # (e.g., make_wood_pickaxe needs `table` adjacent). The sentinel
        # "empty" means "any non-blocking adjacent tile" and is treated as
        # always satisfied at plan-generation time. Used by the planner to
        # chain place_X before make_Y when the required tile isn't adjacent.
        self.near_requirements: dict[tuple[str, str], str] = {}
        # Passive spatial reach facts from textbook.
        # Dict: concept_id -> manhattan range for applying `concept -> proximity`.
        self.proximity_ranges: dict[str, int] = {}
        # Passive movement behavior facts from textbook.
        # Dict: concept_id -> behavior string, e.g. "chase_player".
        self.movement_behaviors: dict[str, str] = {}
        self._option_failure_write_cache: set[
            tuple[str, tuple[tuple[str, str], ...], str, str, str]
        ] = set()

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

    def _advance_rng_past_loaded_vectors(self, seed: int, n_vectors: int) -> None:
        """Avoid reusing loaded vector slots for roles added during repair."""
        self._rng.manual_seed(int(seed))
        for _ in range(max(0, int(n_vectors))):
            torch.randint(
                0, 2, (self.dim,), dtype=torch.float32,
                generator=self._rng, device="cpu",
            )

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
            if role_name in NON_EFFECT_DECODE_ROLES or role_name.startswith("option_ctx:"):
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
        address = self._effect_address(concept_id, action)
        return self.memory.read(address)

    def _effect_address(self, concept_id: str, action: str) -> torch.Tensor:
        """Role-isolated physics-effect address for a (concept, action) pair."""
        v_concept = self._ensure_concept(concept_id)
        v_action = self._ensure_action(action)
        v_role = self._ensure_role(EFFECT_ROLE_NAME)
        return bind(bind(v_concept, v_action), v_role)

    # ------------------------------------------------------------------ #
    # Outcome-role: cross-episode trajectory outcomes per (concept, action)
    # ------------------------------------------------------------------ #
    #
    # Stored in the SAME `self.memory` SDM as physics-effect predictions
    # but at a parallel address `bind(bind(concept, action), role_outcome_h)`.
    # Physics effects use their own `role__EFFECT__` address, so both payload
    # types share one substrate while remaining role-isolated.
    #
    # outcome_dict shape: {"survived_h": bool, "damage_h": int, "died_to": str | None}

    def _outcome_address(self, concept_id: str, action: str) -> torch.Tensor:
        v_concept = self._ensure_concept(concept_id)
        v_action = self._ensure_action(action)
        v_role = self._ensure_role("__OUTCOME_H__")
        return bind(bind(v_concept, v_action), v_role)

    def encode_outcome(self, outcome: dict) -> torch.Tensor:
        """Encode a {survived_h, damage_h, died_to} dict as a single HDC bundle."""
        parts: list[torch.Tensor] = []
        survived_concept = self._ensure_concept(
            "alive" if outcome.get("survived_h", True) else "dead"
        )
        parts.append(bind(self._ensure_role("survived_h"), survived_concept))

        damage = max(0, min(int(outcome.get("damage_h", 0)), self.max_scalar))
        damage_vec = encode_scalar(damage, self.dim, self.max_scalar).to(self.device)
        parts.append(bind(self._ensure_role("damage_h"), damage_vec))

        died_to = outcome.get("died_to") or "none"
        parts.append(bind(self._ensure_role("died_to"), self._ensure_concept(died_to)))
        return bundle(parts)

    def decode_outcome(self, outcome_vec: torch.Tensor) -> dict:
        """Recover {survived_h, damage_h, died_to} from an HDC outcome bundle."""
        survived = self._argmax_outcome_concept(
            outcome_vec, "survived_h", ("alive", "dead"), sim_floor=0.5,
        )
        survived_bool = survived == "alive" if survived is not None else True

        damage_unbound = bind(outcome_vec, self._ensure_role("damage_h"))
        damage = max(0, min(decode_scalar(damage_unbound, self.max_scalar), self.max_scalar))

        # died_to is checked against any concept that has been seen so far;
        # restrict to a small candidate set of known death causes plus "none"
        # to keep the comparison bounded.
        death_candidates = tuple(
            name for name in self.concepts
            if name in (
                "none",
                "zombie",
                "skeleton",
                "arrow",
                "lava",
                "health",
                "done",
                "drink",
                "food",
                "energy",
                "health_critical",
                "food_critical",
                "drink_critical",
                "energy_critical",
                "interrupted",
                "failed",
                "target_lost",
                "max_attempts_exceeded",
            )
        )
        if not death_candidates:
            death_candidates = ("none",)
        died = self._argmax_outcome_concept(
            outcome_vec, "died_to", death_candidates, sim_floor=0.55,
        )
        if died is None or died == "none":
            died = None

        return {"survived_h": survived_bool, "damage_h": damage, "died_to": died}

    def _argmax_outcome_concept(
        self, bundled: torch.Tensor, role_name: str,
        candidates: tuple[str, ...], sim_floor: float = 0.55,
    ) -> str | None:
        if not candidates:
            return None
        role = self._ensure_role(role_name)
        unbound = bind(bundled, role)
        best_name: str | None = None
        best_sim = sim_floor
        for cand in candidates:
            sim = hamming_similarity(unbound, self._ensure_concept(cand))
            if sim > best_sim:
                best_sim = sim
                best_name = cand
        return best_name

    def learn_outcome(self, concept_id: str, action: str, outcome: dict) -> None:
        """Record an observed trajectory outcome for the (concept, action) pair.

        The write goes to a role-distinguished address so physics-effect
        predictions for the same pair are unaffected.
        """
        address = self._outcome_address(concept_id, action)
        outcome_vec = self.encode_outcome(outcome)
        self.memory.write(address, outcome_vec)

    def predict_outcome(self, concept_id: str, action: str) -> tuple[dict | None, float]:
        """Retrieve the expected outcome for a (concept, action) pair.

        Returns (decoded_outcome_dict, confidence). When confidence is below
        the noise floor the dict is None — caller treats that as "no recall".
        """
        address = self._outcome_address(concept_id, action)
        outcome_vec, confidence = self.memory.read(address)
        if confidence < 0.2:
            return None, confidence
        return self.decode_outcome(outcome_vec), confidence

    # ------------------------------------------------------------------ #
    # Option-outcome role: cross-episode outcomes per (context, strategy option)
    # ------------------------------------------------------------------ #
    #
    # This is deliberately separate from the primitive outcome role above:
    # primitive outcome answers "what happened after (near_concept, action)?",
    # option outcome answers "what happened after choosing this strategy in
    # this compact conflict context?" Both live in the same SDM under distinct
    # role-bound addresses.

    def encode_option_context(self, context: dict[str, str]) -> torch.Tensor:
        """Encode a compact option context as an HDC bundle.

        Context values are symbolic buckets, e.g. {"health_bucket": "low"}.
        Deterministic key ordering keeps equivalent dicts address-identical.
        """
        if not context:
            return self._ensure_concept("__EMPTY_OPTION_CONTEXT__")
        parts: list[torch.Tensor] = []
        signature = "|".join(
            f"{key}={context[key]}"
            for key in sorted(context)
        )
        parts.append(
            bind(
                self._ensure_role("option_ctx:signature"),
                self._ensure_concept(f"option_ctx:signature:{signature}"),
            )
        )
        for key in sorted(context):
            role = self._ensure_role(f"option_ctx:{key}")
            value = self._ensure_concept(f"option_ctx:{key}={context[key]}")
            parts.append(bind(role, value))
        return bundle(parts)

    _OPTION_CONTEXT_FIELDS: tuple[str, ...] = (
        "health_bucket",
        "food_bucket",
        "drink_bucket",
        "energy_bucket",
        "threat_pressure",
        "local_restore",
        "capability_state",
        "intent_state",
        "progress_state",
        "goal_family",
    )
    _OPTION_CONTEXT_LEVEL_FIELDS: tuple[tuple[str, tuple[str, ...], float], ...] = (
        ("full", _OPTION_CONTEXT_FIELDS, 1.0),
        (
            "drop_progress",
            tuple(field for field in _OPTION_CONTEXT_FIELDS if field != "progress_state"),
            0.85,
        ),
        (
            "drop_intent",
            tuple(
                field for field in _OPTION_CONTEXT_FIELDS
                if field not in {"intent_state", "progress_state"}
            ),
            0.70,
        ),
        (
            "need_threat_capability",
            (
                "health_bucket",
                "food_bucket",
                "drink_bucket",
                "energy_bucket",
                "threat_pressure",
                "local_restore",
                "capability_state",
            ),
            0.55,
        ),
        (
            "need_threat",
            (
                "health_bucket",
                "food_bucket",
                "drink_bucket",
                "energy_bucket",
                "threat_pressure",
            ),
            0.40,
        ),
        (
            "vitals_only",
            (
                "health_bucket",
                "food_bucket",
                "drink_bucket",
                "energy_bucket",
            ),
            0.25,
        ),
    )

    def abstract_option_contexts(
        self,
        context: dict[str, str],
    ) -> list[tuple[str, dict[str, str], float]]:
        """Return deterministic context abstractions for option-failure recall."""
        levels: list[tuple[str, dict[str, str], float]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for name, fields, weight in self._OPTION_CONTEXT_LEVEL_FIELDS:
            abstracted = {
                field: str(context[field])
                for field in fields
                if field in context
            }
            key = tuple(sorted(abstracted.items()))
            if key in seen:
                continue
            seen.add(key)
            levels.append((name, abstracted, float(weight)))
        return levels

    def abstract_strategy_options(
        self,
        option_id: str,
    ) -> list[tuple[str, str, float]]:
        """Return deterministic option abstractions for option-failure recall."""
        exact = str(option_id)
        kind = exact.split(":", 1)[0] if ":" in exact else exact
        levels = [("exact", exact, 1.0)]
        if kind != exact:
            levels.append(("kind", kind, 0.70))
        return levels

    def _option_failure_write_keys(
        self,
        context: dict[str, str],
        option_id: str,
    ) -> list[tuple[str, dict[str, str], str, str]]:
        """Bounded initial fan-out for sparse option-failure writes.

        Full six-level context × option-level writes created excessive SDM
        crosstalk in smoke profiles. B3 starts with the two abstraction axes
        needed by the observed failures: neighboring intent/progress context
        and same-kind option transfer.
        """
        context_by_level = {
            level: abstracted
            for level, abstracted, _weight in self.abstract_option_contexts(context)
        }
        keys: list[tuple[str, dict[str, str], str, str]] = [
            ("full", context_by_level["full"], "exact", str(option_id)),
        ]
        if "drop_intent" in context_by_level:
            keys.append(("drop_intent", context_by_level["drop_intent"], "exact", str(option_id)))
        option_levels = self.abstract_strategy_options(option_id)
        if len(option_levels) > 1:
            _option_level, option_key, _option_weight = option_levels[1]
            keys.append(("full", context_by_level["full"], "kind", option_key))
        return keys

    def _option_abstraction_vec(self, option_level: str, option_key: str) -> torch.Tensor:
        if option_level == "exact":
            return self._ensure_concept(f"strategy_option:{option_key}")
        return self._ensure_concept(f"strategy_option_{option_level}:{option_key}")

    def _option_outcome_address(
        self,
        context: dict[str, str],
        option_id: str,
    ) -> torch.Tensor:
        context_vec = self.encode_option_context(context)
        option_vec = self._ensure_concept(f"strategy_option:{option_id}")
        role_vec = self._ensure_role("__OPTION_OUTCOME_H__")
        return bind(bind(context_vec, option_vec), role_vec)

    def _option_failure_address(
        self,
        context: dict[str, str],
        option_key: str,
        *,
        context_level: str = "full",
        option_level: str = "exact",
    ) -> torch.Tensor:
        """Sparse negative-outcome address for a strategy option in context.

        The normal option-outcome role stores the aggregate observed horizon
        outcome. That aggregate is intentionally useful for reconstructing
        "what usually happened", but it can wash out rare failures when many
        survived writes share the same coarse context/option key. The planner's
        read-side stimulus is a death-warning, not a value function, so sparse
        failures need a role-isolated hazard channel in the same SDM substrate.
        """
        context_vec = self.encode_option_context(context)
        option_vec = self._option_abstraction_vec(option_level, option_key)
        role_vec = self._ensure_role(
            f"__OPTION_FAILURE_H__:{context_level}:{option_level}"
        )
        return bind(bind(context_vec, option_vec), role_vec)

    @staticmethod
    def _with_option_retrieval_metadata(
        decoded: dict,
        *,
        context_level: str,
        option_level: str,
        context_weight: float,
        option_weight: float,
        role: str,
    ) -> dict:
        out = dict(decoded)
        out["_retrieval"] = {
            "context_level": str(context_level),
            "option_level": str(option_level),
            "abstraction_weight": float(context_weight) * float(option_weight),
            "role": str(role),
        }
        return out

    def learn_option_outcome(
        self,
        context: dict[str, str],
        option_id: str,
        outcome: dict,
    ) -> None:
        """Record an observed horizon outcome for a strategy option in context."""
        address = self._option_outcome_address(context, option_id)
        outcome_vec = self.encode_outcome(outcome)
        self.memory.write(address, outcome_vec)
        if not bool(outcome.get("survived_h", True)):
            for context_level, context_key, option_level, option_key in self._option_failure_write_keys(
                context, option_id
            ):
                cache_key = (
                    str(context_level),
                    tuple(sorted((str(k), str(v)) for k, v in context_key.items())),
                    str(option_level),
                    str(option_key),
                    str(outcome.get("died_to") or "unknown"),
                )
                if cache_key in self._option_failure_write_cache:
                    continue
                self._option_failure_write_cache.add(cache_key)
                failure_address = self._option_failure_address(
                    context_key,
                    option_key,
                    context_level=context_level,
                    option_level=option_level,
                )
                self.memory.write(failure_address, outcome_vec)

    def predict_option_outcome(
        self,
        context: dict[str, str],
        option_id: str,
    ) -> tuple[dict | None, float]:
        """Retrieve learned outcome for a strategy option in compact context."""
        exact_failure_address = self._option_failure_address(
            context,
            str(option_id),
            context_level="full",
            option_level="exact",
        )
        exact_failure_vec, exact_failure_confidence = self.memory.read(exact_failure_address)
        if exact_failure_confidence >= 0.9:
            exact_failure_decoded = self.decode_outcome(exact_failure_vec)
            if not bool(exact_failure_decoded.get("survived_h", True)):
                return self._with_option_retrieval_metadata(
                    exact_failure_decoded,
                    context_level="full",
                    option_level="exact",
                    context_weight=1.0,
                    option_weight=1.0,
                    role="__OPTION_FAILURE_H__",
                ), exact_failure_confidence

        address = self._option_outcome_address(context, option_id)
        outcome_vec, confidence = self.memory.read(address)
        if confidence >= 0.2:
            aggregate_decoded = self.decode_outcome(outcome_vec)
            # Exact aggregate evidence for this context/option is stronger than
            # broad abstract failure recall. This prevents one neighboring
            # failure from suppressing a context with directly observed safe
            # outcome, while exact failures above still remain dominant.
            if bool(aggregate_decoded.get("survived_h", True)):
                return self._with_option_retrieval_metadata(
                    aggregate_decoded,
                    context_level="full",
                    option_level="exact",
                    context_weight=1.0,
                    option_weight=1.0,
                    role="__OPTION_OUTCOME_H__",
                ), confidence

        context_levels = self.abstract_option_contexts(context)
        option_levels = self.abstract_strategy_options(option_id)
        for context_level, context_key, context_weight in context_levels:
            for option_level, option_key, option_weight in option_levels:
                if context_level == "full" and option_level == "exact":
                    continue
                failure_address = self._option_failure_address(
                    context_key,
                    option_key,
                    context_level=context_level,
                    option_level=option_level,
                )
                failure_vec, failure_confidence = self.memory.read(failure_address)
                if failure_confidence >= 0.9:
                    failure_decoded = self.decode_outcome(failure_vec)
                    if not bool(failure_decoded.get("survived_h", True)):
                        return self._with_option_retrieval_metadata(
                            failure_decoded,
                            context_level=context_level,
                            option_level=option_level,
                            context_weight=context_weight,
                            option_weight=option_weight,
                            role="__OPTION_FAILURE_H__",
                        ), failure_confidence

        if confidence < 0.2:
            return None, confidence
        return self._with_option_retrieval_metadata(
            self.decode_outcome(outcome_vec),
            context_level="full",
            option_level="exact",
            context_weight=1.0,
            option_weight=1.0,
            role="__OPTION_OUTCOME_H__",
        ), confidence

    def requirements_met(
        self, concept_id: str, action: str, inventory: dict[str, int],
    ) -> bool:
        """Check if agent's inventory satisfies action requirements.

        Returns True if no requirements declared or all met.
        """
        reqs = self.action_requirements.get((concept_id, action))
        if not reqs:
            return True
        for item, min_count in reqs.items():
            if inventory.get(item, 0) < min_count:
                return False
        return True

    def batch_predict(
        self, pairs: list[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[torch.Tensor, float]]:
        """Predict effects for many (concept, action) pairs in one GPU op.

        Use this at the start of each planning step to precompute all
        needed predictions. Individual predict() calls that hit the cache
        can then be O(1) dict lookups.
        """
        if not pairs:
            return {}

        addresses = []
        for concept_id, action in pairs:
            addresses.append(self._effect_address(concept_id, action))

        addr_tensor = torch.stack(addresses)
        predictions, confidences = self.memory.batch_read(addr_tensor)

        result: dict[tuple[str, str], tuple[torch.Tensor, float]] = {}
        for i, pair in enumerate(pairs):
            result[pair] = (predictions[i], confidences[i].item())
        return result

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
        address = self._effect_address(concept_id, action)
        self.memory.write(address, observed_vec)

        # Compute surprise
        if confidence < 0.01:
            surprise = 1.0  # No prior knowledge = max surprise
        else:
            surprise = 1.0 - hamming_similarity(predicted, observed_vec)

        # Update concept embedding with context
        if context_vectors and surprise > 0.1:
            v_concept = self._ensure_concept(concept_id)
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
            "effect_address_version": self.effect_address_version,
            "effect_integrity_version": (
                self.effect_integrity_version if self.effect_repair_verified else 0
            ),
            "effect_repair_verified": bool(self.effect_repair_verified),
            "concepts": {k: v.cpu() for k, v in self.concepts.items()},
            "actions": {k: v.cpu() for k, v in self.actions.items()},
            "roles": {k: v.cpu() for k, v in self.roles.items()},
            "memory": self.memory.state_dict(),
            "action_requirements": self.action_requirements,
            "near_requirements": self.near_requirements,
            "proximity_ranges": self.proximity_ranges,
            "movement_behaviors": self.movement_behaviors,
        }, path)

    def load(self, path: str | Path) -> bool:
        """Load experience from a previous generation (warm-start transfer).

        Replaces all vectors and SDM address space with those from the saved
        model so that query addresses remain consistent with stored content.
        Gen2 starts as a copy of gen1's knowledge and then continues learning
        in the same vector space.

        Bug A fix: the old approach bundle-merged concept vectors while keeping
        the new model's action vectors unchanged, producing query addresses
        (bind(merged_concept, new_action)) that miss all stored locations
        → confidence=0 on everything → plan=baseline.

        Returns False if file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            return False

        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["dim"] != self.dim:
            raise ValueError(f"Dimension mismatch: {data['dim']} vs {self.dim}")

        # Replace all vectors with loaded ones — address space must be consistent
        # across concept, action, and SDM to produce matching query addresses.
        self.concepts = {k: v.to(self.device) for k, v in data["concepts"].items()}
        self.actions  = {k: v.to(self.device) for k, v in data["actions"].items()}
        self.roles    = {k: v.to(self.device) for k, v in data["roles"].items()}
        loaded_vector_count = len(self.concepts) + len(self.actions) + len(self.roles)
        self.action_requirements = data.get("action_requirements", {})
        self.near_requirements = data.get("near_requirements", {})
        self.proximity_ranges = data.get("proximity_ranges", {})
        self.movement_behaviors = data.get("movement_behaviors", {})

        # Load SDM: addresses either come verbatim (legacy snapshot) or
        # are regenerated from the stored seed (new compact snapshot).
        # Content is REPLACED here (not merged), preserving the gen1
        # knowledge as the gen2 starting point rather than accumulating
        # gen1 on top of a textbook-bootstrapped gen2 SDM.
        mem = data["memory"]
        if "addresses" in mem:
            self.memory.addresses = mem["addresses"].to(self.device)
        elif "seed" in mem:
            self.memory.seed = int(mem["seed"])
            rng = torch.Generator(device="cpu")
            rng.manual_seed(self.memory.seed)
            addresses_cpu = torch.randint(
                0, 2, (self.memory.n_locations, self.memory.dim),
                dtype=torch.float32, generator=rng,
            )
            self.memory.addresses = addresses_cpu.to(self.device)
        else:
            raise ValueError("VectorWorldModel snapshot memory has neither 'addresses' nor 'seed'")
        self.memory.activation_radius = mem["activation_radius"]
        self.memory.content = mem["content"].to(self.device).clone()
        self.memory.n_writes = mem["n_writes"]
        self._advance_rng_past_loaded_vectors(self.memory.seed, loaded_vector_count)
        self.effect_address_version = int(data.get("effect_address_version", 0) or 0)
        self.effect_integrity_version = int(data.get("effect_integrity_version", 0) or 0)
        self.effect_repair_verified = bool(data.get("effect_repair_verified", False))

        integrity = self.verify_effect_rules_from_textbook()
        needs_repair = (
            self.effect_address_version < EFFECT_ADDRESS_VERSION
            or self.effect_integrity_version < EFFECT_INTEGRITY_VERSION
            or not self.effect_repair_verified
            or not integrity["verified"]
        )
        if needs_repair:
            self.repair_effect_rules_from_textbook()
        return True

    def repair_effect_rules_from_textbook(
        self,
        yaml_path: str | Path | None = None,
        max_batches: int = 10,
    ) -> dict:
        """Seed textbook physics rules into the effect-role channel.

        Legacy snapshots wrote action effects at the raw
        `bind(concept, action)` address. New physics reads ignore that
        address, so legacy models need textbook facts copied into
        `bind(bind(concept, action), __EFFECT__)` on load. Existing SDM
        content is preserved; the repair only adds deterministic textbook
        effect writes and fact dictionaries.

        Repair is verified before the snapshot is trusted. Legacy SDM content
        can create strong crosstalk at the new role-bound addresses, so one
        textbook batch is not always enough to overpower old counters.
        """
        from snks.agent.vector_bootstrap import load_from_textbook

        path = Path(yaml_path) if yaml_path is not None else DEFAULT_TEXTBOOK_PATH
        before = self.verify_effect_rules_from_textbook(path)
        stats = {
            "batches": 0,
            "max_batches": int(max_batches),
            "verified_before": bool(before["verified"]),
            "verified": bool(before["verified"]),
            "failed_pairs": before["failed_pairs"],
            "last_textbook_stats": None,
        }

        current = before
        while not current["verified"] and stats["batches"] < max_batches:
            textbook_stats = load_from_textbook(self, path)
            stats["batches"] += 1
            stats["last_textbook_stats"] = textbook_stats
            current = self.verify_effect_rules_from_textbook(path)
            stats["verified"] = bool(current["verified"])
            stats["failed_pairs"] = current["failed_pairs"]

        if current["verified"]:
            self.effect_address_version = EFFECT_ADDRESS_VERSION
            self.effect_integrity_version = EFFECT_INTEGRITY_VERSION
            self.effect_repair_verified = True
        else:
            self.effect_integrity_version = 0
            self.effect_repair_verified = False

        self.last_effect_repair_stats = stats
        return stats

    def mark_effect_integrity_if_verified(
        self,
        yaml_path: str | Path | None = None,
    ) -> dict:
        """Set the persisted effect-integrity marker only after verification."""
        path = Path(yaml_path) if yaml_path is not None else DEFAULT_TEXTBOOK_PATH
        verification = self.verify_effect_rules_from_textbook(path)
        if verification["verified"]:
            self.effect_address_version = EFFECT_ADDRESS_VERSION
            self.effect_integrity_version = EFFECT_INTEGRITY_VERSION
            self.effect_repair_verified = True
        return verification

    def verify_effect_rules_from_textbook(
        self,
        yaml_path: str | Path | None = None,
    ) -> dict:
        """Verify core textbook physics effects in the role-bound channel."""
        path = Path(yaml_path) if yaml_path is not None else DEFAULT_TEXTBOOK_PATH
        expectations = self._core_effect_expectations(path)
        failed: list[dict] = []
        for target, action in sorted(CORE_EFFECT_REPAIR_PAIRS - set(expectations)):
            failed.append({
                "target": target,
                "action": action,
                "confidence": 0.0,
                "decoded": {},
                "missing": {"__textbook_rule__": {"expected_sign": 1, "actual": None}},
                "unexpected": {},
            })

        for (target, action), expected in expectations.items():
            effect_vec, confidence = self.predict(target, action)
            decoded = self.decode_effect(effect_vec) if confidence >= 0.2 else {}
            missing: dict[str, dict[str, int | None]] = {}
            for role_name, expected_delta in expected.items():
                actual = decoded.get(role_name)
                if not self._effect_delta_sign_matches(actual, expected_delta):
                    missing[role_name] = {
                        "expected_sign": 1 if expected_delta > 0 else -1,
                        "actual": actual,
                    }
            unexpected = {
                role_name: value
                for role_name, value in decoded.items()
                if role_name not in expected
            }
            if missing or unexpected:
                failed.append({
                    "target": target,
                    "action": action,
                    "confidence": round(float(confidence), 6),
                    "decoded": decoded,
                    "missing": missing,
                    "unexpected": unexpected,
                })

        return {
            "verified": not failed,
            "checked": len(expectations),
            "failed_pairs": failed,
        }

    @staticmethod
    def _effect_delta_sign_matches(actual: int | None, expected: int) -> bool:
        if actual is None:
            return False
        if expected > 0:
            return actual > 0
        if expected < 0:
            return actual < 0
        return actual == 0

    @staticmethod
    def _core_effect_expectations(yaml_path: Path) -> dict[tuple[str, str], dict[str, int]]:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        expectations: dict[tuple[str, str], dict[str, int]] = {}
        for rule in data.get("rules", []):
            action = rule.get("action")
            target = rule.get("target") or rule.get("result") or rule.get("item")
            if (target, action) not in CORE_EFFECT_REPAIR_PAIRS:
                continue
            effect: dict[str, int] = {}
            rule_effect = rule.get("effect", {})
            for item, delta in rule_effect.get("inventory", {}).items():
                effect[item] = int(delta)
            for var, delta in rule_effect.get("body", {}).items():
                effect[var] = int(delta)
            if effect:
                expectations[(str(target), str(action))] = effect

        return expectations
