"""Stage 71: GroundingSession — visual + text grounding via co-activation.

"Parent shows child objects and names them."
Uses CrafterControlledEnv to place objects near the agent, encodes pixels
with CNNEncoder, and binds visual embeddings + text SDRs to concepts
in ConceptStore.

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch


class EncoderProtocol(Protocol):
    """Minimal encoder interface for grounding."""

    def __call__(self, pixels: torch.Tensor) -> Any:
        """Encode pixels, return object with .z_real attribute."""
        ...


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface for grounding."""

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to SDR."""
        ...


class ControlledEnvProtocol(Protocol):
    """Minimal controlled env interface."""

    def reset_near(
        self, target: str, inventory: dict | None = None, no_enemies: bool = True
    ) -> tuple[np.ndarray, dict]:
        ...

    def reset(self) -> tuple[np.ndarray, dict]:
        ...


# Categories that can be visually grounded (have a visible representation)
VISUAL_CATEGORIES = {"resource", "crafted", "terrain", "enemy"}


@dataclass
class GroundingReport:
    """Result of a grounding session."""

    grounded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    visual_sim_matrix: dict[tuple[str, str], float] = field(default_factory=dict)


class GroundingSession:
    """Ground concepts visually and textually via co-activation.

    For each visually groundable concept:
    1. Place object near agent K times (different seeds)
    2. Encode pixels with CNNEncoder → z_real
    3. Average and L2-normalize → stable visual prototype
    4. Encode concept name with tokenizer → text_sdr
    5. Store both in ConceptStore

    Usage:
        session = GroundingSession(env, encoder, tokenizer, store)
        report = session.ground_all(rng)
    """

    def __init__(
        self,
        env: ControlledEnvProtocol,
        encoder: EncoderProtocol,
        tokenizer: TokenizerProtocol,
        store: Any,  # ConceptStore — avoid circular import
        *,
        k_samples: int = 5,
    ) -> None:
        self.env = env
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.store = store
        self.k_samples = k_samples

    def ground_all(self, rng: np.random.RandomState | None = None) -> GroundingReport:
        """Ground all concepts that have a visual representation."""
        report = GroundingReport()

        for concept in self.store.concepts.values():
            category = concept.attributes.get("category", "")
            if category in VISUAL_CATEGORIES:
                self.ground_one(concept.id, rng)
                report.grounded.append(concept.id)
            else:
                # Text-only grounding (inventory items, tools)
                self._ground_text_only(concept.id)
                report.skipped.append(concept.id)

        # Compute visual similarity matrix for diagnostics
        report.visual_sim_matrix = self._compute_sim_matrix(report.grounded)
        return report

    def ground_one(
        self, concept_id: str, rng: np.random.RandomState | None = None
    ) -> None:
        """Ground one concept: K visual samples + text SDR."""
        z_accum: list[torch.Tensor] = []

        for seed in range(self.k_samples):
            if rng is not None:
                actual_seed = int(rng.randint(0, 100000))
            else:
                actual_seed = seed

            # Get pixels with target object nearby
            pixels_np = self._get_pixels(concept_id, actual_seed)
            pixels_t = torch.from_numpy(pixels_np).float()
            if pixels_t.dim() == 3 and pixels_t.shape[0] == 3:
                pixels_t = pixels_t.unsqueeze(0)  # (1, 3, 64, 64)

            # Encode
            with torch.no_grad():
                output = self.encoder(pixels_t)
            z_real = output.z_real if hasattr(output, "z_real") else output
            if z_real.dim() == 2:
                z_real = z_real[0]  # (2048,)
            z_accum.append(z_real)

        # Average and L2-normalize
        z_mean = torch.stack(z_accum).mean(dim=0)
        z_norm = torch.nn.functional.normalize(z_mean.unsqueeze(0), dim=1).squeeze(0)
        self.store.ground_visual(concept_id, z_norm)

        # Text grounding
        self._ground_text_only(concept_id)

    def _ground_text_only(self, concept_id: str) -> None:
        """Ground concept with text SDR only."""
        text_sdr = self.tokenizer.encode(concept_id)
        if text_sdr is not None and text_sdr.sum() > 0:
            self.store.ground_text(concept_id, text_sdr)

    def _get_pixels(self, concept_id: str, seed: int) -> np.ndarray:
        """Get pixel observation with target object nearby."""
        if concept_id == "empty":
            pixels, _ = self.env.reset()
        else:
            pixels, _ = self.env.reset_near(
                concept_id, inventory=None, no_enemies=True
            )
        return pixels

    def _compute_sim_matrix(
        self, grounded_ids: list[str]
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise cosine similarity for visual embeddings."""
        sim_matrix: dict[tuple[str, str], float] = {}
        for i, id_a in enumerate(grounded_ids):
            ca = self.store.query_text(id_a)
            if ca is None or ca.visual is None:
                continue
            for id_b in grounded_ids[i:]:
                cb = self.store.query_text(id_b)
                if cb is None or cb.visual is None:
                    continue
                va = torch.nn.functional.normalize(ca.visual.unsqueeze(0), dim=1)
                vb = torch.nn.functional.normalize(cb.visual.unsqueeze(0), dim=1)
                sim = (va @ vb.T).item()
                sim_matrix[(id_a, id_b)] = sim
        return sim_matrix
