"""Stage 78b — MLP residual over ConceptStore rules.

The residual corrects body-delta predictions the rules get wrong. It sees
a coarse (visible concepts, body buckets, action) fingerprint and outputs a
4-dimensional body delta correction that is *added* to the rules prediction.

The rules stay authoritative; the residual is a correction, not a replacement.
The bottleneck hidden dim (default 64) is intentionally small so the network
cannot memorize the rules and is forced to learn only the gap.

Usage:

    predictor = ResidualBodyPredictor(
        n_visible_concepts=9,
        n_actions=8,
        n_body_vars=4,
        hidden_dim=64,
    )

    # Encode a symbolic state + action as a fingerprint
    fp = predictor.encode(visible={"skeleton", "tree"}, body={"health": 5, ...}, action="sleep")
    # fp: (n_input_dim,) tensor

    # Get the residual correction
    correction = predictor(fp)   # (n_body_vars,) tensor

    # Combine with rules prediction (no gradient flows back through rules_pred)
    final_pred = rules_pred + correction

See `docs/superpowers/specs/2026-04-11-stage78b-mlp-residual-design.md`
for design motivation and the Stage 78a FAIL context.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ResidualConfig:
    """Shape and bucket parameters for the residual predictor.

    Bucket count for body vars controls the 1-hot body encoding resolution.
    Hidden dim is the bottleneck — keep small to prevent memorization.
    """
    n_visible_concepts: int = 9
    n_actions: int = 8
    n_body_vars: int = 4
    body_buckets: int = 10       # quantize each body var to this many buckets
    hidden_dim: int = 64
    concept_hash_active: int = 30  # active bits per visible concept


class ResidualBodyPredictor(nn.Module):
    """Small MLP residual: (state, action) → body-delta correction.

    Architecture:
        Linear(input_dim → hidden) → ReLU → Linear(hidden → n_body_vars)

    Input encoding (concatenated):
        visible_concepts: hashed 1-hot over 1000 bits (fixed bucket width)
        body_buckets: 1-hot per variable × buckets (n_body_vars * body_buckets)
        action: 1-hot (n_actions)

    Output: (n_body_vars,) delta correction — added to rules prediction.
    """

    CONCEPT_BIT_WIDTH: int = 1000  # fixed input-zone size for hashed concepts

    def __init__(self, config: ResidualConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResidualConfig()

        self.input_dim = (
            self.CONCEPT_BIT_WIDTH
            + self.config.n_body_vars * self.config.body_buckets
            + self.config.n_actions
        )
        self.output_dim = self.config.n_body_vars

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.output_dim),
        )

        # Small init so initial residual is ≈0 (don't perturb rules on epoch 0).
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(0.1)
                    layer.bias.zero_()

        # Cache for concept → bit indices (hashed once per distinct concept string)
        self._concept_seeds: dict[str, np.ndarray] = {}

    # ---- Encoding ----------------------------------------------------------

    def _concept_bits(self, concept: str) -> np.ndarray:
        """Deterministic hash of a concept name to a set of bit positions."""
        if concept not in self._concept_seeds:
            seed = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            idx = rng.choice(
                np.arange(0, self.CONCEPT_BIT_WIDTH),
                size=self.config.concept_hash_active,
                replace=False,
            )
            self._concept_seeds[concept] = idx
        return self._concept_seeds[concept]

    def encode(
        self,
        visible: set[str],
        body: dict[str, float],
        action_idx: int,
        body_order: tuple[str, ...] = ("health", "food", "drink", "energy"),
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Symbolic (state, action) → fingerprint tensor (input_dim,).

        Body values are bucketed by clamp(int(value), 0, body_buckets-1).
        This keeps the encoding identical regardless of the continuous
        value's sub-integer precision.
        """
        device = device or next(self.mlp.parameters()).device

        x = torch.zeros(self.input_dim, device=device)

        # 1. Visible concepts — hashed multi-hot over first CONCEPT_BIT_WIDTH bits
        for concept in visible:
            bits = self._concept_bits(concept)
            for b in bits:
                x[int(b)] = 1.0

        # 2. Body buckets — 1-hot per variable
        base = self.CONCEPT_BIT_WIDTH
        for i, var in enumerate(body_order):
            val = body.get(var, 0.0)
            bucket = min(self.config.body_buckets - 1, max(0, int(val)))
            x[base + i * self.config.body_buckets + bucket] = 1.0

        # 3. Action — 1-hot
        action_base = self.CONCEPT_BIT_WIDTH + self.config.n_body_vars * self.config.body_buckets
        if 0 <= action_idx < self.config.n_actions:
            x[action_base + action_idx] = 1.0

        return x

    # ---- Forward / training helpers ---------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual correction for a batch (B, input_dim) or single (input_dim,)."""
        return self.mlp(x)

    def predict(
        self,
        visible: set[str],
        body: dict[str, float],
        action_idx: int,
        rules_delta: dict[str, float],
        body_order: tuple[str, ...] = ("health", "food", "drink", "energy"),
    ) -> dict[str, float]:
        """Convenience: encode → residual → combine with rules → dict."""
        fp = self.encode(visible, body, action_idx, body_order=body_order)
        with torch.no_grad():
            correction = self.forward(fp).cpu().numpy()
        out = {}
        for i, var in enumerate(body_order):
            out[var] = float(rules_delta.get(var, 0.0) + correction[i])
        return out

    def residual_loss(
        self,
        fp: torch.Tensor,
        rules_delta: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between (rules + residual) and target.

        Equivalent to MSE(residual, target - rules_delta), which makes the
        training signal explicit: the residual learns the *gap*.
        """
        residual = self.forward(fp)
        combined = rules_delta + residual
        return torch.nn.functional.mse_loss(combined, target_delta)

    def save_state(self, path: str) -> None:
        """Save weights + config."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "n_visible_concepts": self.config.n_visible_concepts,
                    "n_actions": self.config.n_actions,
                    "n_body_vars": self.config.n_body_vars,
                    "body_buckets": self.config.body_buckets,
                    "hidden_dim": self.config.hidden_dim,
                    "concept_hash_active": self.config.concept_hash_active,
                },
                "concept_seeds": {k: v.tolist() for k, v in self._concept_seeds.items()},
            },
            path,
        )

    def load_state(self, path: str) -> None:
        """Load weights + concept seed table."""
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["state_dict"])
        self._concept_seeds = {
            k: np.asarray(v, dtype=np.int64) for k, v in ckpt.get("concept_seeds", {}).items()
        }
