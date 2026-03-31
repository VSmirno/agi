"""Experiment 89: Capabilities Verification (Stage 35).

Verifies that IntegratedAgent has all 10 capabilities and that
the meta-learning → strategy → execution pipeline works correctly.

Gates:
    capabilities_count = 10
    strategy_pipeline >= 0.9
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.agent.causal_model import CausalLink
from snks.language.curiosity_module import CuriosityModule
from snks.language.integrated_agent import IntegratedAgent
from snks.language.skill import Skill


def _make_knowledge_agent() -> IntegratedAgent:
    agent = IntegratedAgent(agent_id="cap_test", grid_size=8)
    links = [
        CausalLink(action=3, context_sks=frozenset({50}),
                    effect_sks=frozenset({51}), strength=0.9, count=5),
        CausalLink(action=5, context_sks=frozenset({51, 52}),
                    effect_sks=frozenset({53}), strength=0.85, count=4),
        CausalLink(action=1, context_sks=frozenset({53, 54}),
                    effect_sks=frozenset({54}), strength=0.8, count=3),
        CausalLink(action=2, context_sks=frozenset({50, 54}),
                    effect_sks=frozenset({54}), strength=0.7, count=3),
        CausalLink(action=0, context_sks=frozenset({52}),
                    effect_sks=frozenset({52}), strength=0.6, count=3),
    ]
    agent.inject_knowledge(links)
    agent.inject_skill(Skill(
        name="pickup_key", preconditions=frozenset({50}),
        effects=frozenset({51}), terminal_action=3,
        target_word="key", success_count=10, attempt_count=10,
    ))
    agent.inject_skill(Skill(
        name="toggle_door", preconditions=frozenset({51, 52}),
        effects=frozenset({53}), terminal_action=5,
        target_word="door", success_count=8, attempt_count=10,
    ))
    # Build coverage.
    for i in range(50):
        key = CuriosityModule.make_key({50 + i % 10}, (i, 0))
        agent.curiosity.observe(key)
    return agent


def test_capabilities() -> tuple[int, list[str]]:
    """Verify all 10 capabilities are available."""
    details = []
    agent = IntegratedAgent()
    caps = agent.capabilities()
    for c in caps:
        status = "OK" if c.available else "MISSING"
        details.append(f"  {c.name}: {status} — {c.description}")
    return agent.n_capabilities(), details


def test_strategy_pipeline() -> tuple[float, list[str]]:
    """Test MetaLearner → strategy → execution pipeline."""
    details = []
    successes = 0
    total = 0

    scenarios = [
        {
            "name": "Fresh agent → curiosity/explore",
            "setup": lambda: IntegratedAgent(),
            "expected_strategies": {"curiosity", "explore"},
        },
        {
            "name": "Knowledgeable agent → skill",
            "setup": _make_knowledge_agent,
            "expected_strategies": {"skill"},
        },
    ]

    for scenario in scenarios:
        total += 1
        agent = scenario["setup"]()
        strategy = agent.select_strategy()
        matched = strategy.strategy in scenario["expected_strategies"]
        if matched:
            successes += 1
        details.append(
            f"  {scenario['name']}: got={strategy.strategy}, "
            f"expected={scenario['expected_strategies']} "
            f"[{'PASS' if matched else 'FAIL'}]"
        )
        details.append(f"    reason: {strategy.reason}")

    # Test full pipeline: strategy → episode.
    total += 1
    agent = _make_knowledge_agent()
    result = agent.run_integrated_episode(
        current_sks=frozenset({50, 52, 54}),
        goal_sks=frozenset({51, 53, 54}),
    )
    pipeline_ok = result.success and result.strategy_used != ""
    if pipeline_ok:
        successes += 1
    details.append(
        f"  Full pipeline: strategy={result.strategy_used}, "
        f"success={result.success}, steps={result.steps} "
        f"[{'PASS' if pipeline_ok else 'FAIL'}]"
    )

    accuracy = successes / total
    return accuracy, details


def main() -> None:
    print("=" * 60)
    print("Experiment 89: Capabilities Verification")
    print("=" * 60)

    print("\n--- Capabilities Check ---")
    n_caps, cap_details = test_capabilities()
    for d in cap_details:
        print(d)

    print(f"\n--- Strategy Pipeline ---")
    pipeline_acc, pipe_details = test_strategy_pipeline()
    for d in pipe_details:
        print(d)

    print(f"\n{'=' * 60}")
    print(f"capabilities_count = {n_caps} (gate = 10)")
    print(f"strategy_pipeline = {pipeline_acc:.3f} (gate >= 0.9)")

    g1 = n_caps == 10
    g2 = pipeline_acc >= 0.9

    print(f"\nGate capabilities_count = 10: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate strategy_pipeline >= 0.9: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
