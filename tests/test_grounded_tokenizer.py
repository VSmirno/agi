"""Unit tests for Stage 23: GroundedTokenizer."""

from __future__ import annotations

import torch
import pytest

from snks.daf.types import EncoderConfig
from snks.encoder.grounded_tokenizer import GroundedTokenizer
from snks.language.grounding_map import GroundingMap


@pytest.fixture
def config() -> EncoderConfig:
    return EncoderConfig(sdr_size=4096, sdr_sparsity=0.04, sdr_current_strength=1.0)


@pytest.fixture
def grounding_map() -> GroundingMap:
    gmap = GroundingMap()
    # Create reproducible SDRs (k=164 active bits out of 4096)
    g = torch.Generator().manual_seed(0)
    for word, sks_id in [("key", 1), ("door", 2), ("ball", 3)]:
        sdr = torch.zeros(4096)
        indices = torch.randperm(4096, generator=g)[:164]
        sdr[indices] = 1.0
        gmap.register(word, sks_id, sdr)
    return gmap


@pytest.fixture
def tokenizer(grounding_map, config) -> GroundedTokenizer:
    return GroundedTokenizer(grounding_map, config)


class TestEncode:
    def test_known_word_returns_correct_sdr(self, tokenizer, grounding_map):
        sdr = tokenizer.encode("key")
        expected = grounding_map.word_to_sdr("key")
        assert torch.equal(sdr, expected)

    def test_unknown_word_returns_zero(self, tokenizer, config):
        sdr = tokenizer.encode("castle")
        assert sdr.shape == (config.sdr_size,)
        assert sdr.sum() == 0.0

    def test_case_insensitive(self, tokenizer, grounding_map):
        sdr_lower = tokenizer.encode("key")
        sdr_upper = tokenizer.encode("Key")
        sdr_caps = tokenizer.encode("KEY")
        assert torch.equal(sdr_lower, sdr_upper)
        assert torch.equal(sdr_lower, sdr_caps)

    def test_strips_whitespace(self, tokenizer, grounding_map):
        sdr_clean = tokenizer.encode("key")
        sdr_space = tokenizer.encode("  key  ")
        assert torch.equal(sdr_clean, sdr_space)


class TestSdrToCurrents:
    def test_shape(self, tokenizer):
        sdr = tokenizer.encode("key")
        currents = tokenizer.sdr_to_currents(sdr, n_nodes=1000)
        assert currents.shape == (1000, 8)

    def test_zero_sdr_zero_currents(self, tokenizer):
        sdr = torch.zeros(4096)
        currents = tokenizer.sdr_to_currents(sdr, n_nodes=500)
        assert currents.sum() == 0.0

    def test_active_sdr_nonzero_currents(self, tokenizer):
        sdr = tokenizer.encode("key")
        currents = tokenizer.sdr_to_currents(sdr, n_nodes=1000)
        assert currents[:, 0].sum() > 0.0


class TestVocab:
    def test_vocab_returns_known_words(self, tokenizer):
        assert tokenizer.vocab == {"key", "door", "ball"}
