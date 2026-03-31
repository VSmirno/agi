"""GroundedTokenizer: word → SDR via GroundingMap lookup (Stage 23).

Drop-in replacement for TextEncoder. No external model needed —
uses SDRs learned during co-activation (Stage 19).
Unknown words return zero SDR (no activation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from snks.daf.types import EncoderConfig
from snks.language.grounding_map import GroundingMap

if TYPE_CHECKING:
    from snks.daf.types import ZoneConfig


class GroundedTokenizer:
    """Word → SDR via GroundingMap lookup. Replaces sentence-transformers."""

    def __init__(self, grounding_map: GroundingMap, config: EncoderConfig) -> None:
        self._gmap = grounding_map
        self.config = config
        self.k = round(config.sdr_size * config.sdr_sparsity)

    def encode(self, text: str) -> torch.Tensor:
        """Encode word to binary SDR.

        Args:
            text: input word/phrase.

        Returns:
            (sdr_size,) binary SDR, float32. Zero vector if unknown.
        """
        sdr = self._gmap.word_to_sdr(text.lower().strip())
        if sdr is None:
            return torch.zeros(self.config.sdr_size)
        return sdr

    @property
    def vocab(self) -> set[str]:
        """Set of known words."""
        return self._gmap.vocab_words

    def sdr_to_currents(
        self, sdr: torch.Tensor, n_nodes: int, zone: ZoneConfig | None = None,
    ) -> torch.Tensor:
        """Map SDR to DAF external currents.

        Identical hash-mapping as TextEncoder.sdr_to_currents().
        """
        PRIME = 2654435761
        sz = zone.size if zone is not None else n_nodes
        node_sdr_idx = (torch.arange(sz, device=sdr.device) * PRIME) % self.config.sdr_size
        currents = torch.zeros(sz, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.config.sdr_current_strength
        return currents
