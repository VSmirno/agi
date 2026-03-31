"""Experiment 81: Adaptive vs Fixed Strategy (Stage 32).

Tests that MetaLearner's adaptation improves performance over
a fixed-strategy baseline across a multi-episode sequence.

Gates:
    adaptation_improves = True (adapted agent outperforms fixed)
    meta_vs_fixed_ratio >= 1.2 (at least 1.2x better)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.language.meta_learner import (
    EpisodeResult,
    MetaLearner,
    TaskProfile,
)


def _simulate_episode(
    strategy: str,
    episode: int,
    epsilon: float,
    knowledge: int,
) -> EpisodeResult:
    """Simulate episode outcome based on strategy appropriateness.

    Models a task where:
    - Early episodes (0-2): curiosity is best (need exploration)
    - Mid episodes (3-5): skill is best (knowledge accumulated)
    - Late episodes (6+): skill with low epsilon is best (exploit)
    """
    # Optimal strategy for each phase
    if episode < 3:
        optimal = "curiosity"
    elif episode < 6:
        optimal = "skill"
    else:
        optimal = "skill"

    # Base success probability
    if strategy == optimal:
        base_p = 0.8
    elif strategy in ("curiosity", "explore"):
        base_p = 0.4
    else:
        base_p = 0.3

    # Epsilon penalty: too much exploration in late episodes hurts
    if episode >= 3:
        base_p -= max(0, epsilon - 0.1) * 0.3

    # Knowledge bonus
    base_p += min(0.15, knowledge * 0.01)

    # Deterministic threshold (for reproducibility)
    success = base_p > 0.55
    steps = 20 if success else 55
    skills_used = 2 if (strategy == "skill" and success) else 0
    new_states = max(0, 5 - episode)

    return EpisodeResult(
        success=success,
        steps=steps,
        skills_used=skills_used,
        new_states_discovered=new_states,
        prediction_error=max(0.1, 0.8 - episode * 0.1),
    )


def _run_meta_learner(n_episodes: int = 10) -> tuple[int, list[str]]:
    """Run MetaLearner over simulated episodes, return (total_successes, strategies)."""
    learner = MetaLearner(adaptation_rate=0.1)
    successes = 0
    strategies: list[str] = []
    knowledge = 0

    for ep in range(n_episodes):
        profile = TaskProfile(
            state_coverage=min(1.0, ep * 0.12),
            known_skills=min(5, ep),
            causal_links=min(20, ep * 3),
            mean_prediction_error=max(0.1, 0.8 - ep * 0.1),
            episodes_completed=ep,
        )

        config = learner.select_strategy(profile)
        strategies.append(config.strategy)
        result = _simulate_episode(config.strategy, ep, config.curiosity_epsilon, knowledge)

        if result.success:
            successes += 1
            knowledge += 2

        learner.adapt(profile, result)

    return successes, strategies


def _run_fixed_strategy(strategy: str, n_episodes: int = 10) -> int:
    """Run a fixed strategy over simulated episodes, return total successes."""
    successes = 0
    knowledge = 0

    for ep in range(n_episodes):
        result = _simulate_episode(strategy, ep, epsilon=0.2, knowledge=knowledge)
        if result.success:
            successes += 1
            knowledge += 2

    return successes


def main() -> None:
    print("=" * 60)
    print("Experiment 81: Adaptive vs Fixed Strategy")
    print("=" * 60)

    N = 10

    # MetaLearner (adaptive)
    meta_successes, meta_strategies = _run_meta_learner(N)
    print(f"\n--- MetaLearner (adaptive) ---")
    for i, s in enumerate(meta_strategies):
        print(f"  Episode {i}: strategy={s}")
    print(f"  Total successes: {meta_successes}/{N}")

    # Fixed strategies
    fixed_results = {}
    for strat in ["curiosity", "skill", "explore"]:
        fixed_results[strat] = _run_fixed_strategy(strat, N)
        print(f"\n--- Fixed: {strat} ---")
        print(f"  Total successes: {fixed_results[strat]}/{N}")

    best_fixed = max(fixed_results.values())
    best_fixed_name = max(fixed_results, key=fixed_results.get)

    print(f"\n{'=' * 60}")
    print(f"MetaLearner successes: {meta_successes}")
    print(f"Best fixed ({best_fixed_name}): {best_fixed}")

    ratio = meta_successes / max(best_fixed, 1)
    improves = meta_successes > best_fixed

    print(f"meta_vs_fixed_ratio = {ratio:.2f} (gate >= 1.2)")
    print(f"adaptation_improves = {improves} (gate = True)")

    g1 = improves
    g2 = ratio >= 1.2

    print(f"\nGate adaptation_improves = True: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate meta_vs_fixed_ratio >= 1.2: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
