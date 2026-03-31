"""Experiment 79: Multi-Rule Pattern Completion (Stage 31).

Tests AbstractPatternReasoner on matrices with BOTH row and column
transforms active simultaneously — the hardest pattern type.

Gates:
    multi_rule_accuracy >= 0.7 (fraction of correct predictions)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.dcam.hac import HACEngine
from snks.language.pattern_element import PatternElement, PatternMatrix
from snks.language.abstract_pattern_reasoner import AbstractPatternReasoner


N_TRIALS = 20
N_OPTIONS = 4
THRESHOLD = 0.5


def _make_element(hac: HACEngine, row: int, col: int, embedding: torch.Tensor) -> PatternElement:
    return PatternElement(
        sks_ids=frozenset({row * 10 + col}),
        embedding=embedding,
        position=(row, col),
    )


def _build_dual_transform_matrix(
    hac: HACEngine, seed: int,
) -> tuple[PatternMatrix, torch.Tensor, list[torch.Tensor]]:
    """Build 3x3 matrix where e[r,c] = bind(base, T^(r+c)).

    Same transform T applies along both rows and columns.
    e[r,c] = bind(e[r, c-1], T) = bind(e[r-1, c], T)
    """
    gen = torch.Generator().manual_seed(seed)
    T = torch.randn(hac.dim, generator=gen)
    T = T / T.norm().clamp(min=1e-8)

    base_v = torch.randn(hac.dim, generator=gen)
    base_v = base_v / base_v.norm().clamp(min=1e-8)

    # Build grid: e[r,c] = bind^(r+c)(base, T)
    cache: dict[tuple[int, int], torch.Tensor] = {}
    # First compute power-of-T chain
    power_t = [base_v]
    for i in range(1, 5):  # max power = 4 (for [2,2])
        power_t.append(hac.bind(power_t[-1], T))

    for r in range(3):
        for c in range(3):
            cache[(r, c)] = power_t[r + c]

    ground_truth = cache[(2, 2)].clone()

    elements = []
    for r in range(3):
        for c in range(3):
            elements.append(_make_element(hac, r, c, cache[(r, c)]))

    elements[8] = PatternElement(
        sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2),
    )
    matrix = PatternMatrix(elements=elements, shape=(3, 3), missing=8)

    # Build options
    options = [ground_truth]
    for _ in range(N_OPTIONS - 1):
        d = torch.randn(hac.dim, generator=gen)
        d = d / d.norm().clamp(min=1e-8)
        options.append(d)

    perm = torch.randperm(N_OPTIONS, generator=gen).tolist()
    shuffled = [options[p] for p in perm]
    correct_idx = perm.index(0)

    return matrix, ground_truth, shuffled


def _build_independent_dual_matrix(
    hac: HACEngine, seed: int,
) -> tuple[PatternMatrix, torch.Tensor, list[torch.Tensor]]:
    """Build 3x3 with independent row transform T_r and column transform T_c.

    e[r,c] = bind(bind(row_base[r], T_c^c), T_r^r)
    """
    gen = torch.Generator().manual_seed(seed)
    T_r = torch.randn(hac.dim, generator=gen)
    T_r = T_r / T_r.norm().clamp(min=1e-8)
    T_c = torch.randn(hac.dim, generator=gen)
    T_c = T_c / T_c.norm().clamp(min=1e-8)

    # Row bases
    row_bases = []
    for _ in range(3):
        v = torch.randn(hac.dim, generator=gen)
        row_bases.append(v / v.norm().clamp(min=1e-8))

    cache: dict[tuple[int, int], torch.Tensor] = {}
    for r in range(3):
        # Apply row transform to get base for this row
        row_vec = row_bases[r]
        for _ in range(r):
            row_vec = hac.bind(row_vec, T_r)
        # Apply column transform
        col_vec = row_vec
        cache[(r, 0)] = col_vec
        for c in range(1, 3):
            col_vec = hac.bind(col_vec, T_c)
            cache[(r, c)] = col_vec

    ground_truth = cache[(2, 2)].clone()

    elements = []
    for r in range(3):
        for c in range(3):
            elements.append(_make_element(hac, r, c, cache[(r, c)]))

    elements[8] = PatternElement(
        sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2),
    )
    matrix = PatternMatrix(elements=elements, shape=(3, 3), missing=8)

    options = [ground_truth]
    for _ in range(N_OPTIONS - 1):
        d = torch.randn(hac.dim, generator=gen)
        d = d / d.norm().clamp(min=1e-8)
        options.append(d)

    perm = torch.randperm(N_OPTIONS, generator=gen).tolist()
    shuffled = [options[p] for p in perm]

    return matrix, ground_truth, shuffled


def main() -> None:
    print("=" * 60)
    print("Experiment 79: Multi-Rule Pattern Completion")
    print("=" * 60)

    torch.manual_seed(79)
    hac = HACEngine(dim=2048)
    reasoner = AbstractPatternReasoner(hac, threshold=THRESHOLD)

    # --- Test 1: Same transform for row and column ---
    print("\n--- Test 1: Shared Row+Column Transform ---")
    correct_shared = 0

    for i in range(N_TRIALS):
        matrix, gt, options = _build_dual_transform_matrix(hac, seed=i * 23 + 300)
        prediction, confidence = reasoner.predict_missing(matrix)
        choice = reasoner.select_answer(prediction, options)
        sim = hac.similarity(options[choice], gt)
        if sim > 0.99:
            correct_shared += 1

    shared_accuracy = correct_shared / N_TRIALS
    print(f"  Correct: {correct_shared}/{N_TRIALS}")
    print(f"  shared_accuracy = {shared_accuracy:.3f}")

    # --- Test 2: Independent row/column transforms ---
    print("\n--- Test 2: Independent Row+Column Transforms ---")
    correct_indep = 0

    for i in range(N_TRIALS):
        matrix, gt, options = _build_independent_dual_matrix(hac, seed=i * 29 + 500)
        prediction, confidence = reasoner.predict_missing(matrix)
        choice = reasoner.select_answer(prediction, options)
        sim = hac.similarity(options[choice], gt)
        if sim > 0.99:
            correct_indep += 1

    indep_accuracy = correct_indep / N_TRIALS
    print(f"  Correct: {correct_indep}/{N_TRIALS}")
    print(f"  indep_accuracy = {indep_accuracy:.3f}")

    # --- Aggregate ---
    total_correct = correct_shared + correct_indep
    total_trials = 2 * N_TRIALS
    multi_rule_accuracy = total_correct / total_trials

    print("\n" + "=" * 60)
    print(f"multi_rule_accuracy = {multi_rule_accuracy:.3f} (gate >= 0.7)")

    g1 = multi_rule_accuracy >= 0.7
    print(f"\nGate multi_rule_accuracy >= 0.7: {'PASS' if g1 else 'FAIL'}")

    if g1:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
