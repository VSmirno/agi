"""AbstractPatternReasoner: Raven's-style pattern completion on SKS (Stage 31).

Discovers transformation rules in sequences of HAC embeddings using
algebraic operations (bind/unbind) and predicts missing elements.
"""

from __future__ import annotations

import torch
from torch import Tensor

from snks.dcam.hac import HACEngine
from snks.language.pattern_element import (
    PatternElement,
    PatternMatrix,
    TransformRule,
)


class AbstractPatternReasoner:
    """Discovers abstract patterns in SKS embedding sequences and predicts missing elements.

    Uses HAC algebra: unbind to extract transforms, bind to apply them.
    Works on 3x3 matrices (Raven's) and linear sequences (analogy).
    """

    def __init__(self, hac: HACEngine, threshold: float = 0.6) -> None:
        self._hac = hac
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Rule discovery
    # ------------------------------------------------------------------

    def discover_rules(self, matrix: PatternMatrix) -> list[TransformRule]:
        """Find consistent transformation rules along rows and columns.

        For each row/column, compute pairwise transforms via unbind.
        A rule is accepted if its consistency >= threshold.
        """
        rules: list[TransformRule] = []

        # Row rules — skip rows that contain the missing element's row
        # unless we have enough non-missing elements
        row_rules = self._discover_axis_rules(matrix, axis="row")
        col_rules = self._discover_axis_rules(matrix, axis="column")

        rules.extend(r for r in row_rules if r.consistency >= self._threshold)
        rules.extend(r for r in col_rules if r.consistency >= self._threshold)

        rules.sort(key=lambda r: r.consistency, reverse=True)
        return rules

    def _discover_axis_rules(
        self, matrix: PatternMatrix, axis: str,
    ) -> list[TransformRule]:
        """Discover transform rules along one axis."""
        rows, cols = matrix.shape
        missing_row = matrix.missing // cols
        missing_col = matrix.missing % cols

        transforms: list[Tensor] = []

        if axis == "row":
            for r in range(rows):
                if r == missing_row:
                    continue
                row_elems = [matrix.get(r, c) for c in range(cols)]
                row_transforms = self._pairwise_transforms(row_elems)
                transforms.extend(row_transforms)
        else:  # column
            for c in range(cols):
                if c == missing_col:
                    continue
                col_elems = [matrix.get(r, c) for r in range(rows)]
                col_transforms = self._pairwise_transforms(col_elems)
                transforms.extend(col_transforms)

        if not transforms:
            return []

        # Compute mean transform and consistency
        mean_t = self._mean_transform(transforms)
        consistency = self._compute_consistency(transforms, mean_t)

        return [TransformRule(
            transform_vector=mean_t,
            axis=axis,
            consistency=consistency,
        )]

    def _pairwise_transforms(self, elements: list[PatternElement]) -> list[Tensor]:
        """Extract pairwise transforms: T_i = unbind(e_i, e_{i+1})."""
        transforms = []
        for i in range(len(elements) - 1):
            t = self._hac.unbind(elements[i].embedding, elements[i + 1].embedding)
            transforms.append(t)
        return transforms

    def _mean_transform(self, transforms: list[Tensor]) -> Tensor:
        """Average transform vectors (bundle)."""
        return self._hac.bundle(transforms)

    def _compute_consistency(self, transforms: list[Tensor], mean: Tensor) -> float:
        """Mean cosine similarity of each transform to the mean."""
        if len(transforms) <= 1:
            return 1.0
        sims = [self._hac.similarity(t, mean) for t in transforms]
        return sum(sims) / len(sims)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_missing(self, matrix: PatternMatrix) -> tuple[Tensor, float]:
        """Predict the HAC embedding of the missing element.

        Uses discovered row and column rules. If both exist, bundles
        the two predictions. Returns (predicted_embedding, confidence).
        """
        rules = self.discover_rules(matrix)
        if not rules:
            return torch.zeros(self._hac.dim), 0.0

        rows, cols = matrix.shape
        missing_row = matrix.missing // cols
        missing_col = matrix.missing % cols

        predictions: list[Tensor] = []
        confidences: list[float] = []

        for rule in rules:
            pred = self._apply_rule(matrix, rule, missing_row, missing_col)
            if pred is not None:
                predictions.append(pred)
                confidences.append(rule.consistency)

        if not predictions:
            return torch.zeros(self._hac.dim), 0.0

        if len(predictions) == 1:
            return predictions[0], confidences[0]

        # Bundle predictions weighted by consistency
        combined = self._hac.bundle(predictions)
        avg_conf = sum(confidences) / len(confidences)
        return combined, avg_conf

    def _apply_rule(
        self,
        matrix: PatternMatrix,
        rule: TransformRule,
        missing_row: int,
        missing_col: int,
    ) -> Tensor | None:
        """Apply a transform rule to predict the missing element."""
        rows, cols = matrix.shape

        if rule.axis == "row":
            # Use the element before the missing one in the same row
            if missing_col > 0:
                prev = matrix.get(missing_row, missing_col - 1)
                return self._hac.bind(prev.embedding, rule.transform_vector)
            elif missing_col < cols - 1:
                # Missing is first in row — use reverse: unbind next from T
                nxt = matrix.get(missing_row, missing_col + 1)
                return self._hac.unbind(rule.transform_vector, nxt)
        else:  # column
            if missing_row > 0:
                prev = matrix.get(missing_row - 1, missing_col)
                return self._hac.bind(prev.embedding, rule.transform_vector)
            elif missing_row < rows - 1:
                nxt = matrix.get(missing_row + 1, missing_col)
                return self._hac.unbind(rule.transform_vector, nxt)

        return None

    # ------------------------------------------------------------------
    # Answer selection (Raven's multiple choice)
    # ------------------------------------------------------------------

    def select_answer(self, prediction: Tensor, options: list[Tensor]) -> int:
        """Select the option most similar to the prediction."""
        keys = torch.stack(options)
        sims = self._hac.batch_similarity(prediction, keys)
        return int(sims.argmax().item())

    # ------------------------------------------------------------------
    # Analogy (A:B :: C:?)
    # ------------------------------------------------------------------

    def solve_analogy(
        self, a: Tensor, b: Tensor, c: Tensor,
    ) -> tuple[Tensor, float]:
        """Solve A:B :: C:? using HAC algebra.

        T = unbind(A, B), D = bind(C, T).
        Confidence = similarity(T applied to A, B).
        """
        t = self._hac.unbind(a, b)
        d = self._hac.bind(c, t)
        # Verify: bind(A, T) should ≈ B
        reconstructed = self._hac.bind(a, t)
        confidence = self._hac.similarity(reconstructed, b)
        return d, confidence
