"""Tests for GroundingMap (Stage 19)."""

import os
import tempfile

import pytest
import torch

from snks.language.grounding_map import GroundingMap


class TestGroundingMap:

    def test_register_and_lookup(self):
        gm = GroundingMap()
        sdr = torch.zeros(512)
        sdr[:20] = 1.0
        gm.register("key", 42, sdr)

        assert gm.word_to_sks("key") == 42
        assert gm.sks_to_word(42) == "key"
        assert gm.word_to_sdr("key") is not None
        assert gm.word_to_sdr("key").shape == (512,)

    def test_unknown_word_returns_none(self):
        gm = GroundingMap()
        assert gm.word_to_sks("unknown") is None
        assert gm.sks_to_word(999) is None
        assert gm.word_to_sdr("unknown") is None

    def test_overwrite_same_word(self):
        gm = GroundingMap()
        sdr1 = torch.ones(512)
        sdr2 = torch.zeros(512)
        gm.register("door", 10, sdr1)
        gm.register("door", 20, sdr2)

        assert gm.word_to_sks("door") == 20
        assert gm.word_to_sdr("door").sum().item() == 0.0

    def test_vocab_size(self):
        gm = GroundingMap()
        assert gm.vocab_size == 0
        gm.register("a", 1, torch.zeros(10))
        gm.register("b", 2, torch.zeros(10))
        assert gm.vocab_size == 2

    def test_save_load_roundtrip(self):
        gm = GroundingMap()
        sdr_key = torch.zeros(512)
        sdr_key[:20] = 1.0
        sdr_door = torch.ones(512) * 0.5
        gm.register("key", 42, sdr_key)
        gm.register("door", 99, sdr_door)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grounding")
            gm.save(path)

            gm2 = GroundingMap()
            gm2.load(path)

            assert gm2.word_to_sks("key") == 42
            assert gm2.word_to_sks("door") == 99
            assert gm2.sks_to_word(42) == "key"
            assert gm2.sks_to_word(99) == "door"
            assert torch.allclose(gm2.word_to_sdr("key"), sdr_key)
            assert torch.allclose(gm2.word_to_sdr("door"), sdr_door)

    def test_save_load_empty_map(self):
        gm = GroundingMap()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty")
            gm.save(path)

            gm2 = GroundingMap()
            gm2.load(path)
            assert gm2.vocab_size == 0
