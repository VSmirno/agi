"""Experiment 78: Pattern Completion Accuracy (Stage 31).

Tests AbstractPatternReasoner on:
1. 3x3 matrix completion (row transform)
2. A:B :: C:? analogy

Gates:
    completion_accuracy >= 0.8 (fraction of correct predictions in 3x3)
    analogy_accuracy >= 0.8 (fraction of correct A:B::C:D)
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
SIM_ACCEPT = 0.4  # minimum similarity to count as correct


def _make_element(hac: HACEngine, row: int, col: int, embedding: torch.Tensor) -> PatternElement:
    return PatternElement(
        sks_ids=frozenset({row * 10 + col}),
        embedding=embedding,
        position=(row, col),
    )


def _build_matrix_with_answer(
    hac: HACEngine, seed: int,
) -> tuple[PatternMatrix, torch.Tensor, list[torch.Tensor]]:
    """Build 3x3 row-transform matrix + correct answer + distractors."""
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

    elements[8] = PatternElement(
        sks_ids=frozenset(), embedding=torch.zeros(hac.dim), position=(2, 2),
    )
    matrix = PatternMatrix(elements=elements, shape=(3, 3), missing=8)

    # Build options: correct + 3 random distractors
    options = [ground_truth]
    for _ in range(N_OPTIONS - 1):
        d = torch.randn(hac.dim, generator=gen)
        d = d / d.norm().clamp(min=1e-8)
        options.append(d)

    # Shuffle deterministically
    perm = torch.randperm(N_OPTIONS, generator=gen).tolist()
    shuffled = [options[p] for p in perm]
    correct_idx = perm.index(0)

    return matrix, shuffled[correct_idx], shuffled


def main() -> None:
    print("=" * 60)
    print("Experiment 78: Pattern Completion Accuracy")
    print("=" * 60)

    torch.manual_seed(78)
    hac = HACEngine(dim=2048)
    reasoner = AbstractPatternReasoner(hac, threshold=THRESHOLD)

    # --- Test 1: 3x3 Matrix Completion ---
    print("\n--- Test 1: 3x3 Matrix Completion ---")
    correct_completions = 0

    for i in range(N_TRIALS):
        matrix, gt, options = _build_matrix_with_answer(hac, seed=i * 11 + 100)
        prediction, confidence = reasoner.predict_missing(matrix)
        choice = reasoner.select_answer(prediction, options)
        # Check if selected option matches ground truth
        sim_to_gt = hac.similarity(options[choice], gt)
        if sim_to_gt > 0.99:
            correct_completions += 1

    completion_accuracy = correct_completions / N_TRIALS
    print(f"  Correct: {correct_completions}/{N_TRIALS}")
    print(f"  completion_accuracy = {completion_accuracy:.3f} (gate >= 0.8)")

    # --- Test 2: Analogy A:B :: C:? ---
    print("\n--- Test 2: Analogy A:B :: C:? ---")
    correct_analogies = 0

    for i in range(N_TRIALS):
        gen = torch.Generator().manual_seed(i * 17 + 200)
        T = torch.randn(hac.dim, generator=gen)
        T = T / T.norm().clamp(min=1e-8)

        a = torch.randn(hac.dim, generator=gen)
        a = a / a.norm().clamp(min=1e-8)
        b = hac.bind(a, T)
        c = torch.randn(hac.dim, generator=gen)
        c = c / c.norm().clamp(min=1e-8)
        d_expected = hac.bind(c, T)

        d_pred, confidence = reasoner.solve_analogy(a, b, c)

        # Build options
        options = [d_expected]
        for _ in range(N_OPTIONS - 1):
            distractor = torch.randn(hac.dim, generator=gen)
            distractor = distractor / distractor.norm().clamp(min=1e-8)
            options.append(distractor)

        perm = torch.randperm(N_OPTIONS, generator=gen).tolist()
        shuffled = [options[p] for p in perm]
        correct_idx = perm.index(0)

        choice = reasoner.select_answer(d_pred, shuffled)
        if choice == correct_idx:
            correct_analogies += 1

    analogy_accuracy = correct_analogies / N_TRIALS
    print(f"  Correct: {correct_analogies}/{N_TRIALS}")
    print(f"  analogy_accuracy = {analogy_accuracy:.3f} (gate >= 0.8)")

    # --- Results ---
    print("\n" + "=" * 60)
    g1 = completion_accuracy >= 0.8
    g2 = analogy_accuracy >= 0.8
    print(f"Gate completion_accuracy >= 0.8: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate analogy_accuracy >= 0.8: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
