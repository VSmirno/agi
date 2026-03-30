"""GroundingMap: bidirectional lookup word <-> SKS concept."""

from __future__ import annotations

import json
import os

import torch


class GroundingMap:
    """Bidirectional mapping between words and SKS concept IDs.

    Populated during co-activation (image + text) in the pipeline.
    Used by Verbalizer (Stage 21) and GroundedTokenizer (Stage 23).
    """

    def __init__(self) -> None:
        self._word_to_sks: dict[str, int] = {}
        self._sks_to_word: dict[int, str] = {}
        self._word_to_sdr: dict[str, torch.Tensor] = {}
        self._word_to_visual_sdr: dict[str, torch.Tensor] = {}

    def register(self, word: str, sks_id: int, sdr: torch.Tensor) -> None:
        """Register a grounding association.

        Args:
            word: text label (e.g. "key").
            sks_id: tracked SKS cluster ID.
            sdr: (sdr_size,) binary SDR used to activate the linguistic zone.
        """
        self._word_to_sks[word] = sks_id
        self._sks_to_word[sks_id] = word
        self._word_to_sdr[word] = sdr.detach().cpu()

    def register_visual(self, word: str, visual_sdr: torch.Tensor) -> None:
        """Register visual SDR for cross-modal priming.

        Called during co-activation (image + text). Stores the visual zone
        currents so that text-only presentation can prime the visual zone.
        """
        self._word_to_visual_sdr[word] = visual_sdr.detach().cpu()

    def word_to_visual_sdr(self, word: str) -> torch.Tensor | None:
        """Look up visual SDR for top-down priming."""
        return self._word_to_visual_sdr.get(word)

    def word_to_sks(self, word: str) -> int | None:
        return self._word_to_sks.get(word)

    def sks_to_word(self, sks_id: int) -> str | None:
        return self._sks_to_word.get(sks_id)

    def word_to_sdr(self, word: str) -> torch.Tensor | None:
        return self._word_to_sdr.get(word)

    @property
    def vocab_size(self) -> int:
        return len(self._word_to_sks)

    def save(self, path: str) -> None:
        """Save grounding map to disk (JSON metadata + safetensors SDRs)."""
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        meta = {
            "word_to_sks": self._word_to_sks,
            "sks_to_word": {str(k): v for k, v in self._sks_to_word.items()},
        }
        with open(path + "_meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if self._word_to_sdr:
            tensors = {word: sdr for word, sdr in self._word_to_sdr.items()}
            save_file(tensors, path + "_sdrs.safetensors")

    def load(self, path: str) -> None:
        """Load grounding map from disk."""
        from safetensors.torch import load_file

        with open(path + "_meta.json") as f:
            meta = json.load(f)

        self._word_to_sks = meta["word_to_sks"]
        self._sks_to_word = {int(k): v for k, v in meta["sks_to_word"].items()}

        sdrs_path = path + "_sdrs.safetensors"
        if os.path.exists(sdrs_path):
            self._word_to_sdr = dict(load_file(sdrs_path))
