"""Experiment 86: Plan Depth & Coherence (Stage 34).

Tests that HierarchicalPlanner can generate plans with 1000+ primitive steps
while maintaining high plan coherence (valid precondition chains).

Gates:
    plan_depth >= 1000
    plan_coherence >= 0.9
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.hierarchical_planner import HierarchicalPlanner
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


def _make_planner(grid_size: int = 8) -> HierarchicalPlanner:
    config = CausalAgentConfig(causal_min_observations=1)
    model = CausalWorldModel(config)
    links = [
        (3, {50}, {50, 51}),
        (5, {51, 52}, {51, 52, 53}),
    ]
    for action, ctx, post in links:
        for _ in range(5):
            model.observe_transition(pre_sks=ctx, action=action, post_sks=post)

    lib = SkillLibrary()
    lib.register(Skill(
        name="pickup_key", preconditions=frozenset({50}),
        effects=frozenset({51}), terminal_action=3,
        target_word="key", success_count=10, attempt_count=10,
    ))
    lib.register(Skill(
        name="toggle_door", preconditions=frozenset({51, 52}),
        effects=frozenset({53}), terminal_action=5,
        target_word="door", success_count=8, attempt_count=10,
    ))
    return HierarchicalPlanner(model, lib, grid_size=grid_size)


def test_plan_depth() -> tuple[int, float, list[str]]:
    """Test plan depth across different grid sizes and room counts."""
    details = []
    max_depth = 0
    coherences = []

    configs = [
        (8, 1, "Small (8x8, 1 room)"),
        (16, 2, "Medium (16x16, 2 rooms)"),
        (30, 4, "Large (30x30, 4 rooms)"),
        (40, 6, "XL (40x40, 6 rooms)"),
        (50, 8, "XXL (50x50, 8 rooms)"),
    ]

    for grid_size, n_rooms, label in configs:
        planner = _make_planner(grid_size)
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
            n_rooms=n_rooms,
        )
        depth = plan.total_steps
        coherence = plan.coherence
        n_nodes = plan.n_nodes

        details.append(f"{label}: {depth} steps, {n_nodes} nodes, coherence={coherence:.3f}")
        max_depth = max(max_depth, depth)
        coherences.append(coherence)

    avg_coherence = sum(coherences) / len(coherences)
    return max_depth, avg_coherence, details


def test_plan_structure() -> tuple[float, list[str]]:
    """Verify 3-level hierarchy is maintained."""
    details = []
    planner = _make_planner(grid_size=20)
    plan = planner.plan(
        goal_sks=frozenset({51, 53, 54}),
        current_sks=frozenset({50, 52, 54}),
        n_rooms=3,
    )

    # Check levels.
    assert plan.root.level == 2
    details.append(f"Root level: {plan.root.level} (expected 2)")

    has_level_1 = any(c.level in (1, 2) for c in plan.root.children)
    details.append(f"Has level 1/2 children: {has_level_1}")

    # Check primitives exist.
    primitives = plan.root.flatten_primitives()
    details.append(f"Total primitives: {len(primitives)}")

    # Check all primitives are valid actions (0-5).
    valid_actions = all(0 <= a <= 5 for a in primitives)
    details.append(f"All valid actions: {valid_actions}")

    score = 1.0 if (has_level_1 and len(primitives) > 0 and valid_actions) else 0.0
    return score, details


def main() -> None:
    print("=" * 60)
    print("Experiment 86: Plan Depth & Coherence")
    print("=" * 60)

    print("\n--- Plan Depth Test ---")
    max_depth, avg_coherence, details = test_plan_depth()
    for d in details:
        print(f"  {d}")

    print(f"\n--- Plan Structure Test ---")
    structure_score, struct_details = test_plan_structure()
    for d in struct_details:
        print(f"  {d}")

    print(f"\n{'=' * 60}")
    print(f"plan_depth = {max_depth} (gate >= 1000)")
    print(f"plan_coherence = {avg_coherence:.3f} (gate >= 0.9)")

    g1 = max_depth >= 1000
    g2 = avg_coherence >= 0.9

    print(f"\nGate plan_depth >= 1000: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate plan_coherence >= 0.9: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
