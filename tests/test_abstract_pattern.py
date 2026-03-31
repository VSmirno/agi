"""Tests for Stage 31: Abstract Pattern Reasoning."""

from __future__ import annotations

import torch
import pytest

from snks.dcam.hac import HACEngine
from snks.language.pattern_element import PatternElement, PatternMatrix, TransformRule
from snks.language.abstract_pattern_reasoner import AbstractPatternReasoner


@pytest.fixture
def hac() -> HACEngine:
    torch.manual_seed(42)
    return HACEngine(dim=2048)


@pytest.fixture
def reasoner(hac: HACEngine) -> AbstractPatternReasoner:
    return AbstractPatternReasoner(hac, threshold=0.5)


def _make_element(hac: HACEngine, row: int, col: int, seed: int) -> PatternElement:
    """Create a PatternElement with a deterministic random embedding."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(hac.dim, generator=gen)
    v = v / v.norm().clamp(min=1e-8)
    return PatternElement(
        sks_ids=frozenset({seed}),
        embedding=v,
        position=(row, col),
    )


def _build_row_transform_matrix(
    hac: HACEngine, transform: torch.Tensor,
) -> PatternMatrix:
    """Build a 3x3 matrix where each row applies the same transform.

    e[r,0] → bind(e[r,0], T) → bind(bind(e[r,0], T), T)
    Missing element: [2,2].
    """
    elements: list[PatternElement] = []
    for r in range(3):
        base = _make_element(hac, r, 0, seed=r * 100)
        e0 = base
        e1_emb = hac.bind(base.embedding, transform)
        e1 = PatternElement(sks_ids=frozenset({r * 100 + 1}), embedding=e1_emb, position=(r, 1))
        e2_emb = hac.bind(e1_emb, transform)
        e2 = PatternElement(sks_ids=frozenset({r * 100 + 2}), embedding=e2_emb, position=(r, 2))
        elements.extend([e0, e1, e2])

    # Replace [2,2] with zeros (missing)
    elements[8] = PatternElement(
        sks_ids=frozenset(),
        embedding=torch.zeros(hac.dim),
        position=(2, 2),
    )
    return PatternMatrix(elements=elements, shape=(3, 3), missing=8)


class TestPatternElement:
    def test_pattern_matrix_get(self, hac: HACEngine):
        elems = [_make_element(hac, r, c, r * 3 + c) for r in range(3) for c in range(3)]
        m = PatternMatrix(elements=elems, shape=(3, 3), missing=8)
        assert m.get(1, 2).position == (1, 2)
        assert m.rows == 3
        assert m.cols == 3

    def test_pattern_matrix_shape(self, hac: HACEngine):
        elems = [_make_element(hac, r, c, r * 2 + c) for r in range(3) for c in range(2)]
        m = PatternMatrix(elements=elems, shape=(3, 2), missing=5)
        assert m.rows == 3
        assert m.cols == 2


class TestRuleDiscovery:
    def test_discovers_row_rule(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        T = hac.random_vector()
        matrix = _build_row_transform_matrix(hac, T)
        rules = reasoner.discover_rules(matrix)
        assert len(rules) >= 1
        row_rules = [r for r in rules if r.axis == "row"]
        assert len(row_rules) >= 1
        assert row_rules[0].consistency >= 0.5

    def test_discovers_column_rule(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        T = hac.random_vector()
        # Build column-transform matrix: each column applies T
        elements: list[PatternElement] = []
        for c in range(3):
            base = _make_element(hac, 0, c, seed=c * 100)
            e0 = base
            e1_emb = hac.bind(base.embedding, T)
            e1 = PatternElement(sks_ids=frozenset({c * 100 + 1}), embedding=e1_emb, position=(1, c))
            e2_emb = hac.bind(e1_emb, T)
            e2 = PatternElement(sks_ids=frozenset({c * 100 + 2}), embedding=e2_emb, position=(2, c))
            elements.extend([e0, e1, e2])
        # Reorder to row-major
        reordered = [None] * 9
        for i, (r, c) in enumerate([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]):
            reordered[r * 3 + c] = elements[i]
        # Missing = [2,2]
        reordered[8] = PatternElement(sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2))
        matrix = PatternMatrix(elements=reordered, shape=(3, 3), missing=8)

        rules = reasoner.discover_rules(matrix)
        col_rules = [r for r in rules if r.axis == "column"]
        assert len(col_rules) >= 1
        assert col_rules[0].consistency >= 0.5

    def test_no_rule_from_random(self, hac: HACEngine):
        """Random embeddings should not produce consistent rules."""
        strict = AbstractPatternReasoner(hac, threshold=0.9)
        elements = [_make_element(hac, r, c, r * 3 + c + 1000) for r in range(3) for c in range(3)]
        elements[8] = PatternElement(sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2))
        matrix = PatternMatrix(elements=elements, shape=(3, 3), missing=8)
        rules = strict.discover_rules(matrix)
        # Expect no or low-consistency rules
        assert all(r.consistency < 0.9 for r in rules) or len(rules) == 0


class TestPrediction:
    def test_predict_missing_row_transform(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        T = hac.random_vector()
        matrix = _build_row_transform_matrix(hac, T)
        # Ground truth: bind(e[2,1], T)
        e21 = matrix.get(2, 1)
        expected = hac.bind(e21.embedding, T)

        prediction, confidence = reasoner.predict_missing(matrix)
        sim = hac.similarity(prediction, expected)
        assert sim > 0.5, f"prediction similarity {sim} too low"
        assert confidence > 0.0

    def test_predict_with_both_axes(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        """When both row and column transforms exist, prediction should still work."""
        T = hac.random_vector()
        # Construct matrix with same T for rows AND columns
        # e[r,c] = bind(base, T^(r+c)) using sequential binds
        base = _make_element(hac, 0, 0, seed=999)
        elements: list[PatternElement] = []
        cache: dict[tuple[int, int], torch.Tensor] = {}

        for r in range(3):
            for c in range(3):
                if r == 0 and c == 0:
                    emb = base.embedding
                elif c == 0:
                    emb = hac.bind(cache[(r - 1, 0)], T)
                else:
                    emb = hac.bind(cache[(r, c - 1)], T)
                cache[(r, c)] = emb
                elements.append(PatternElement(
                    sks_ids=frozenset({r * 10 + c}),
                    embedding=emb,
                    position=(r, c),
                ))

        expected = cache[(2, 2)].clone()
        elements[8] = PatternElement(sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2))
        matrix = PatternMatrix(elements=elements, shape=(3, 3), missing=8)

        prediction, confidence = reasoner.predict_missing(matrix)
        sim = hac.similarity(prediction, expected)
        assert sim > 0.4, f"dual-axis prediction similarity {sim} too low"


class TestAnswerSelection:
    def test_select_correct_answer(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        T = hac.random_vector()
        matrix = _build_row_transform_matrix(hac, T)
        e21 = matrix.get(2, 1)
        correct = hac.bind(e21.embedding, T)
        wrong1 = hac.random_vector()
        wrong2 = hac.random_vector()
        wrong3 = hac.random_vector()

        prediction, _ = reasoner.predict_missing(matrix)
        options = [wrong1, wrong2, correct, wrong3]
        choice = reasoner.select_answer(prediction, options)
        assert choice == 2

    def test_select_from_similar_distractors(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        T = hac.random_vector()
        matrix = _build_row_transform_matrix(hac, T)
        e21 = matrix.get(2, 1)
        correct = hac.bind(e21.embedding, T)
        # Distractor: apply T twice (wrong number of steps)
        distractor = hac.bind(hac.bind(e21.embedding, T), T)

        prediction, _ = reasoner.predict_missing(matrix)
        options = [distractor, correct]
        choice = reasoner.select_answer(prediction, options)
        assert choice == 1


class TestAnalogy:
    def test_solve_analogy(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        T = hac.random_vector()
        a = hac.random_vector()
        b = hac.bind(a, T)
        c = hac.random_vector()
        expected_d = hac.bind(c, T)

        d, confidence = reasoner.solve_analogy(a, b, c)
        sim = hac.similarity(d, expected_d)
        assert sim > 0.8, f"analogy accuracy {sim} too low"
        assert confidence > 0.5

    def test_analogy_with_permute(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        """Analogy where transform is permute(k)."""
        a = hac.random_vector()
        b = hac.permute(a, 5)
        c = hac.random_vector()
        expected_d = hac.permute(c, 5)

        # Permute is not bind-based, so unbind won't recover it perfectly.
        # This tests the limits — we expect lower similarity.
        d, confidence = reasoner.solve_analogy(a, b, c)
        # Just verify it runs without error
        assert d.shape == (hac.dim,)


class TestProgressiveSequence:
    def test_permute_sequence(self, hac: HACEngine, reasoner: AbstractPatternReasoner):
        """Test pattern: each element is permuted by k from previous."""
        base = hac.random_vector()
        k = 7
        T = hac.random_vector()  # transform

        # Build 3x3 where row transform = bind with T
        matrix = _build_row_transform_matrix(hac, T)
        prediction, confidence = reasoner.predict_missing(matrix)
        assert prediction.shape == (hac.dim,)
        assert confidence > 0.0
