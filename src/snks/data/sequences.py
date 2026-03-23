"""Sequence generators for prediction experiments."""

from __future__ import annotations

from typing import Tuple

import torch
import numpy as np

from snks.data.stimuli import GratingGenerator


class SequenceGenerator:
    """Generates ordered sequences of visual stimuli for prediction experiments."""

    def __init__(self, stimulus_gen: GratingGenerator, seed: int = 42) -> None:
        self.stimulus_gen = stimulus_gen
        self.seed = seed

    def deterministic(
        self, class_order: list[int], n_repeats: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate A→B→C→A→B→C... deterministic sequence.

        Args:
            class_order: list of class indices defining the pattern.
            n_repeats: number of full pattern repetitions.

        Returns:
            images: (L, H, W) where L = len(class_order) * n_repeats.
            labels: (L,) current class labels.
            next_labels: (L,) next class labels (cyclic).
        """
        seq_len = len(class_order)
        total = seq_len * n_repeats

        # Build full label sequence
        full_labels = []
        for _ in range(n_repeats):
            full_labels.extend(class_order)

        # Next labels (cyclic)
        next_lab = full_labels[1:] + [full_labels[0]]

        # Generate one image per step
        rng = np.random.RandomState(self.seed)
        images = []
        for i, cls in enumerate(full_labels):
            # Generate single variation for this class
            variation_seed = self.seed + i * 100
            gen = GratingGenerator(
                image_size=self.stimulus_gen.image_size,
                seed=variation_seed,
            )
            img, _ = gen.generate(class_idx=cls, n_variations=1)
            images.append(img[0])

        return (
            torch.stack(images),
            torch.tensor(full_labels, dtype=torch.int64),
            torch.tensor(next_lab, dtype=torch.int64),
        )

    def stochastic(
        self,
        transitions: dict[int, list[tuple[int, float]]],
        start: int,
        n_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequence based on transition probabilities.

        Args:
            transitions: {state: [(next_state, probability), ...]}.
            start: starting state.
            n_steps: number of steps.

        Returns:
            images: (n_steps, H, W).
            labels: (n_steps,).
        """
        rng = np.random.RandomState(self.seed)
        labels = [start]
        current = start

        for _ in range(n_steps - 1):
            options = transitions[current]
            next_states = [s for s, _ in options]
            probs = [p for _, p in options]
            current = rng.choice(next_states, p=probs)
            labels.append(current)

        images = []
        for i, cls in enumerate(labels):
            gen = GratingGenerator(
                image_size=self.stimulus_gen.image_size,
                seed=self.seed + i * 100,
            )
            img, _ = gen.generate(class_idx=cls, n_variations=1)
            images.append(img[0])

        return (
            torch.stack(images),
            torch.tensor(labels, dtype=torch.int64),
        )
