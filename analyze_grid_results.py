#!/usr/bin/env python3
"""Analyze grid search results and provide recommendations."""

import csv
import sys
from pathlib import Path
from typing import NamedTuple


class Result(NamedTuple):
    denominator: float
    state_weight: float
    action_weight: float
    epsilon: float
    coverage_ratio: float
    curious_coverage: float
    random_coverage: float
    causal_links: int


def load_results(csv_path: str | Path) -> list[Result]:
    """Load results from CSV."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(Result(
                denominator=float(row['denominator']),
                state_weight=float(row['state_weight']),
                action_weight=float(row['action_weight']),
                epsilon=float(row['epsilon']),
                coverage_ratio=float(row['coverage_ratio']),
                curious_coverage=float(row['curious_coverage']),
                random_coverage=float(row['random_coverage']),
                causal_links=int(row['causal_links']),
            ))
    return results


def analyze(csv_path: str | Path) -> None:
    """Analyze and print results."""
    results = load_results(csv_path)
    results.sort(key=lambda r: r.coverage_ratio, reverse=True)

    print("=" * 100)
    print("EXP 9 GRID SEARCH RESULTS — Analysis")
    print("=" * 100)
    print()

    # Top 10
    print("TOP 10 CONFIGURATIONS (by coverage_ratio):")
    print()
    for i, r in enumerate(results[:10], 1):
        status = "✅ GATE PASSED" if r.coverage_ratio > 1.5 else "❌"
        print(
            f"{i:2d}. denom={r.denominator:4.1f} ε={r.epsilon:.2f} "
            f"w=({r.state_weight:.2f},{r.action_weight:.2f}) "
            f"→ ratio={r.coverage_ratio:.4f} {status}"
        )
        print(f"     curious={r.curious_coverage:.4f} random={r.random_coverage:.4f} links={r.causal_links}")
    print()

    # Stats
    print("=" * 100)
    print("STATISTICS:")
    print("=" * 100)
    best = results[0]
    worst = results[-1]
    avg = sum(r.coverage_ratio for r in results) / len(results)

    print(f"Best:   {best.coverage_ratio:.4f}")
    print(f"Worst:  {worst.coverage_ratio:.4f}")
    print(f"Avg:    {avg:.4f}")
    print(f"Target: > 1.5000")
    print()

    # Hypothesis analysis
    print("=" * 100)
    print("HYPOTHESIS ANALYSIS:")
    print("=" * 100)

    # Denominator impact
    print("\n1. State Novelty Denominator Impact (best ratio per denominator):")
    by_denom = {}
    for r in results:
        key = r.denominator
        if key not in by_denom or r.coverage_ratio > by_denom[key].coverage_ratio:
            by_denom[key] = r

    for denom in sorted(by_denom.keys()):
        r = by_denom[denom]
        print(f"   denom={denom:4.1f} → best ratio={r.coverage_ratio:.4f} (w={r.state_weight}, ε={r.epsilon})")

    # Weight impact
    print("\n2. Weight Distribution Impact (best ratio per weight pair):")
    by_weight = {}
    for r in results:
        key = (r.state_weight, r.action_weight)
        if key not in by_weight or r.coverage_ratio > by_weight[key].coverage_ratio:
            by_weight[key] = r

    for (sw, aw) in sorted(by_weight.keys(), reverse=True):
        r = by_weight[(sw, aw)]
        print(f"   w=({sw:.2f},{aw:.2f}) → best ratio={r.coverage_ratio:.4f} (denom={r.denominator}, ε={r.epsilon})")

    # Epsilon impact
    print("\n3. Epsilon Impact (best ratio per epsilon):")
    by_eps = {}
    for r in results:
        key = r.epsilon
        if key not in by_eps or r.coverage_ratio > by_eps[key].coverage_ratio:
            by_eps[key] = r

    for eps in sorted(by_eps.keys()):
        r = by_eps[eps]
        print(f"   ε={eps:.2f} → best ratio={r.coverage_ratio:.4f} (denom={r.denominator}, w={r.state_weight})")

    print()
    print("=" * 100)
    print("RECOMMENDATION:")
    print("=" * 100)
    if best.coverage_ratio > 1.5:
        print(f"✅ GATE PASSED! Use configuration:")
        print(f"   - state_novelty_denominator = {best.denominator}")
        print(f"   - state_weight = {best.state_weight}, action_weight = {best.action_weight}")
        print(f"   - epsilon = {best.epsilon}")
        print(f"   - Expected coverage_ratio: {best.coverage_ratio:.4f}")
    else:
        improvement = best.coverage_ratio - 1.114  # Previous best
        print(f"❌ Gate not reached. Best: {best.coverage_ratio:.4f}")
        print(f"   Improvement from baseline (1.114): +{improvement:.4f} ({improvement/1.114*100:.1f}%)")
        print()
        print("   Next steps:")
        print("   1. Consider Hypothesis C: more sophisticated state representation")
        print("   2. Try decay epsilon: start high (0.3), decay to low (0.05)")
        print("   3. Increase num_nodes DAF: 5000 → 10000")
        print("   4. Use Go-Explore style archiving for state frontiers")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = Path("src/snks/experiments/grid_search_results.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    analyze(csv_path)
