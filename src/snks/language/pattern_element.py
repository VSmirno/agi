"""Data types for abstract pattern reasoning (Stage 31).

PatternElement represents a single cell in a pattern matrix.
PatternMatrix represents a Raven's-style grid with one missing element.
TransformRule captures a discovered transformation between elements.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class PatternElement:
    """Single cell in a pattern matrix."""

    sks_ids: frozenset[int]
    embedding: Tensor           # HAC vector (D,)
    position: tuple[int, ...]   # (row, col) or (index,)


@dataclass
class PatternMatrix:
    """Raven's-style pattern grid with one missing element.

    Elements are stored in row-major order. The missing position
    contains a placeholder element (embedding is zeros).
    """

    elements: list[PatternElement]
    shape: tuple[int, int]      # (rows, cols)
    missing: int                # index of missing element

    def get(self, row: int, col: int) -> PatternElement:
        """Get element at (row, col)."""
        return self.elements[row * self.shape[1] + col]

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]


@dataclass
class TransformRule:
    """A discovered transformation between consecutive elements."""

    transform_vector: Tensor    # HAC vector encoding the transform
    axis: str                   # "row" or "column"
    consistency: float          # mean cosine similarity across instances
