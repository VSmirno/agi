"""Tests for SequenceGenerator (Stage 3)."""

import torch
import pytest

from snks.data.stimuli import GratingGenerator
from snks.data.sequences import SequenceGenerator


@pytest.fixture
def gen() -> SequenceGenerator:
    stim = GratingGenerator(image_size=64, seed=42)
    return SequenceGenerator(stim, seed=42)


class TestDeterministicSequence:
    """SequenceGenerator.deterministic — repeating patterns."""

    def test_output_shapes(self, gen: SequenceGenerator) -> None:
        images, labels, next_labels = gen.deterministic([0, 1, 2], n_repeats=5)
        assert images.shape == (15, 64, 64)
        assert labels.shape == (15,)
        assert next_labels.shape == (15,)

    def test_pattern_repeats(self, gen: SequenceGenerator) -> None:
        """Labels follow A→B→C→A→B→C..."""
        _, labels, _ = gen.deterministic([0, 1, 2], n_repeats=4)
        expected = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        assert labels.tolist() == expected

    def test_next_labels_correct(self, gen: SequenceGenerator) -> None:
        """next_labels[i] = labels[i+1] (cyclic)."""
        _, labels, next_labels = gen.deterministic([0, 1, 2], n_repeats=3)
        for i in range(len(labels) - 1):
            assert next_labels[i].item() == labels[i + 1].item()
        # Last wraps around
        assert next_labels[-1].item() == labels[0].item()

    def test_different_lengths(self, gen: SequenceGenerator) -> None:
        for length in [3, 5, 7]:
            order = list(range(length))
            images, labels, _ = gen.deterministic(order, n_repeats=2)
            assert images.shape[0] == length * 2

    def test_images_are_valid(self, gen: SequenceGenerator) -> None:
        images, _, _ = gen.deterministic([0, 1], n_repeats=2)
        assert images.dtype == torch.float32
        assert images.min() >= 0.0
        assert images.max() <= 1.0


class TestStochasticSequence:
    """SequenceGenerator.stochastic — probabilistic transitions."""

    def test_output_shapes(self, gen: SequenceGenerator) -> None:
        transitions = {0: [(1, 1.0)], 1: [(0, 1.0)]}
        images, labels = gen.stochastic(transitions, start=0, n_steps=10)
        assert images.shape == (10, 64, 64)
        assert labels.shape == (10,)

    def test_deterministic_transitions(self, gen: SequenceGenerator) -> None:
        """100% transition prob → deterministic sequence."""
        transitions = {0: [(1, 1.0)], 1: [(2, 1.0)], 2: [(0, 1.0)]}
        _, labels = gen.stochastic(transitions, start=0, n_steps=9)
        assert labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_all_transitions_valid(self, gen: SequenceGenerator) -> None:
        """All labels are valid class indices."""
        transitions = {0: [(1, 0.7), (2, 0.3)], 1: [(0, 1.0)], 2: [(0, 1.0)]}
        _, labels = gen.stochastic(transitions, start=0, n_steps=20)
        for l in labels.tolist():
            assert l in {0, 1, 2}
