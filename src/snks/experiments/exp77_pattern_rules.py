"""Experiment 77: Pattern Rule Discovery (Stage 31).

Tests that AbstractPatternReasoner discovers consistent transformation
rules in 3x3 Raven's-style matrices constructed from HAC embeddings.

Gates:
    rule_consistency >= 0.7 (mean consistency of discovered rules)
    rule_found_rate >= 0.8 (fraction of matrices where at least one rule found)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.dcam.hac import HACEngine
from snks.language.pattern_element import PatternElement, PatternMatrix
from snks.language.abstract_pattern_reasoner import AbstractPatternReasoner


N_TRIALS = 20
THRESHOLD = 0.6


def _make_element(hac: HACEngine, row: int, col: int, embedding: torch.Tensor) -> PatternElement:
    return PatternElement(
        sks_ids=frozenset({row * 10 + col}),
        embedding=embedding,
        position=(row, col),
    )


def _build_constant_row_matrix(hac: HACEngine, seed: int) -> tuple[PatternMatrix, torch.Tensor]:
    """Build 3x3 matrix: each row applies same transform T. Missing = [2,2]."""
    gen = torch.Generator().manual_seed(seed)
    T = torch.randn(hac.dim, generator=gen)
    T = T / T.norm().clamp(min=1e-8)

    elements: list[PatternElement] = []
    ground_truth = None

    for r in range(3):
        base_v = torch.randn(hac.dim, generator=gen)
        base_v = base_v / base_v.norm().clamp(min=1e-8)
        e0 = base_v
        e1 = hac.bind(e0, T)
        e2 = hac.bind(e1, T)
        elements.append(_make_element(hac, r, 0, e0))
        elements.append(_make_element(hac, r, 1, e1))
        elements.append(_make_element(hac, r, 2, e2))
        if r == 2:
            ground_truth = e2.clone()

    # Replace [2,2] with zeros
    elements[8] = PatternElement(
        sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2),
    )
    return PatternMatrix(elements=elements, shape=(3, 3), missing=8), ground_truth


def _build_column_matrix(hac: HACEngine, seed: int) -> tuple[PatternMatrix, torch.Tensor]:
    """Build 3x3 matrix: each column applies same transform T. Missing = [2,2]."""
    gen = torch.Generator().manual_seed(seed)
    T = torch.randn(hac.dim, generator=gen)
    T = T / T.norm().clamp(min=1e-8)

    cache: dict[tuple[int, int], torch.Tensor] = {}
    for c in range(3):
        base_v = torch.randn(hac.dim, generator=gen)
        base_v = base_v / base_v.norm().clamp(min=1e-8)
        cache[(0, c)] = base_v
        cache[(1, c)] = hac.bind(base_v, T)
        cache[(2, c)] = hac.bind(cache[(1, c)], T)

    ground_truth = cache[(2, 2)].clone()

    elements = []
    for r in range(3):
        for c in range(3):
            elements.append(_make_element(hac, r, c, cache[(r, c)]))

    elements[8] = PatternElement(
        sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2),
    )
    return PatternMatrix(elements=elements, shape=(3, 3), missing=8), ground_truth


def main() -> None:
    print("=" * 60)
    print("Experiment 77: Pattern Rule Discovery")
    print("=" * 60)

    torch.manual_seed(31)
    hac = HACEngine(dim=2048)
    reasoner = AbstractPatternReasoner(hac, threshold=THRESHOLD)

    # --- Test 1: Constant Row Transform ---
    print("\n--- Test 1: Constant Row Transform ---")
    consistencies_row: list[float] = []
    found_row = 0

    for i in range(N_TRIALS):
        matrix, _ = _build_constant_row_matrix(hac, seed=i * 7)
        rules = reasoner.discover_rules(matrix)
        if rules:
            found_row += 1
            best = max(r.consistency for r in rules)
            consistencies_row.append(best)

    row_found_rate = found_row / N_TRIALS
    row_mean_consistency = sum(consistencies_row) / len(consistencies_row) if consistencies_row else 0.0
    print(f"  Row found rate: {row_found_rate:.3f} (gate >= 0.8)")
    print(f"  Row mean consistency: {row_mean_consistency:.3f} (gate >= 0.7)")

    # --- Test 2: Constant Column Transform ---
    print("\n--- Test 2: Constant Column Transform ---")
    consistencies_col: list[float] = []
    found_col = 0

    for i in range(N_TRIALS):
        matrix, _ = _build_column_matrix(hac, seed=i * 13 + 1000)
        rules = reasoner.discover_rules(matrix)
        if rules:
            found_col += 1
            best = max(r.consistency for r in rules)
            consistencies_col.append(best)

    col_found_rate = found_col / N_TRIALS
    col_mean_consistency = sum(consistencies_col) / len(consistencies_col) if consistencies_col else 0.0
    print(f"  Column found rate: {col_found_rate:.3f} (gate >= 0.8)")
    print(f"  Column mean consistency: {col_mean_consistency:.3f} (gate >= 0.7)")

    # --- Aggregate ---
    total_found = found_row + found_col
    total_trials = 2 * N_TRIALS
    all_consistencies = consistencies_row + consistencies_col

    rule_found_rate = total_found / total_trials
    rule_consistency = sum(all_consistencies) / len(all_consistencies) if all_consistencies else 0.0

    print("\n" + "=" * 60)
    print(f"rule_found_rate = {rule_found_rate:.3f} (gate >= 0.8)")
    print(f"rule_consistency = {rule_consistency:.3f} (gate >= 0.7)")

    g1 = rule_consistency >= 0.7
    g2 = rule_found_rate >= 0.8

    print(f"\nGate rule_consistency >= 0.7: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate rule_found_rate >= 0.8: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
