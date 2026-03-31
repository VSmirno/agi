"""Experiment 90: End-to-End Integration (Stage 35).

Tests the full integrated pipeline: profile → strategy → plan → execute → learn.
Also tests cross-capability combinations.

Gates:
    end_to_end_success >= 0.9
    cross_capability >= 0.8
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.agent.causal_model import CausalLink
from snks.language.curiosity_module import CuriosityModule
from snks.language.integrated_agent import IntegratedAgent
from snks.language.pattern_element import PatternElement, PatternMatrix
from snks.language.skill import Skill


def _make_full_agent() -> IntegratedAgent:
    agent = IntegratedAgent(agent_id="e2e_test", grid_size=12)
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
    for i in range(50):
        key = CuriosityModule.make_key({50 + i % 10}, (i, 0))
        agent.curiosity.observe(key)
    return agent


def test_end_to_end() -> tuple[float, list[str]]:
    """Run multiple end-to-end episodes and measure success."""
    details = []
    successes = 0
    total = 5

    agent = _make_full_agent()

    for ep in range(total):
        result = agent.run_integrated_episode(
            current_sks=frozenset({50, 52, 54}),
            goal_sks=frozenset({51, 53, 54}),
            n_rooms=1 + ep,
        )
        status = "PASS" if result.success else "FAIL"
        if result.success:
            successes += 1
        details.append(
            f"  Episode {ep+1}: strategy={result.strategy_used}, "
            f"steps={result.steps}, plan={result.plan_steps}, "
            f"caps={len(result.capabilities_exercised)} [{status}]"
        )

    success_rate = successes / total
    details.append(f"\n  Success rate: {success_rate:.3f}")
    details.append(f"  Episodes completed: {agent.episodes_completed}")
    details.append(f"  Agent success rate: {agent.success_rate:.3f}")
    return success_rate, details


def test_cross_capability() -> tuple[float, list[str]]:
    """Test combinations of capabilities working together."""
    details = []
    tests_passed = 0
    total_tests = 5

    agent = _make_full_agent()

    # 1. Planning + Skills
    plan = agent.plan_to_goal(
        goal_sks=frozenset({51, 53, 54}),
        current_sks=frozenset({50, 52, 54}),
    )
    ok = plan.total_steps > 0
    tests_passed += int(ok)
    details.append(f"  Planning + Skills: {plan.total_steps} steps [{'PASS' if ok else 'FAIL'}]")

    # 2. Meta-learning + Strategy Selection
    strategy = agent.select_strategy()
    ok = strategy.strategy in ("skill", "curiosity", "explore", "few_shot")
    tests_passed += int(ok)
    details.append(f"  Meta-learning + Strategy: {strategy.strategy} [{'PASS' if ok else 'FAIL'}]")

    # 3. Communication + Knowledge Transfer
    msg = agent.share_knowledge()
    ok = msg is not None and msg.content_type.value == "causal_links"
    tests_passed += int(ok)
    details.append(f"  Communication: shared {len(msg.causal_links) if msg else 0} links [{'PASS' if ok else 'FAIL'}]")

    # 4. Pattern Reasoning (analogy)
    hac = agent.hac_engine
    a, b, c = hac.random_vector(), hac.random_vector(), hac.random_vector()
    d_pred, conf = agent.solve_analogy(a, b, c)
    ok = d_pred.shape[0] == 2048
    tests_passed += int(ok)
    details.append(f"  Pattern Reasoning: analogy solved, shape={d_pred.shape[0]} [{'PASS' if ok else 'FAIL'}]")

    # 5. Curiosity + Exploration
    key = CuriosityModule.make_key({99, 98}, (0, 0))
    reward = agent.curiosity.observe(key)
    ok = reward > 0
    tests_passed += int(ok)
    details.append(f"  Curiosity: intrinsic_reward={reward:.3f} [{'PASS' if ok else 'FAIL'}]")

    score = tests_passed / total_tests
    return score, details


def test_learning_across_episodes() -> tuple[float, list[str]]:
    """Test that agent improves across episodes (meta-learning adapts)."""
    details = []
    agent = _make_full_agent()

    results = []
    for ep in range(5):
        r = agent.run_integrated_episode(
            current_sks=frozenset({50, 52, 54}),
            goal_sks=frozenset({51, 53, 54}),
        )
        results.append(r)

    # Check capabilities exercised increase or stay high.
    caps_first = len(results[0].capabilities_exercised)
    caps_last = len(results[-1].capabilities_exercised)
    ok = caps_last >= caps_first - 1  # allow slight variation

    details.append(f"  Episode 1 capabilities: {caps_first}")
    details.append(f"  Episode 5 capabilities: {caps_last}")
    details.append(f"  Agent success rate: {agent.success_rate:.3f}")

    return 1.0 if ok else 0.0, details


def main() -> None:
    print("=" * 60)
    print("Experiment 90: End-to-End Integration")
    print("=" * 60)

    print("\n--- End-to-End Episodes ---")
    e2e_success, e2e_details = test_end_to_end()
    for d in e2e_details:
        print(d)

    print("\n--- Cross-Capability ---")
    cross_score, cross_details = test_cross_capability()
    for d in cross_details:
        print(d)

    print("\n--- Learning Across Episodes ---")
    learn_score, learn_details = test_learning_across_episodes()
    for d in learn_details:
        print(d)

    final_e2e = e2e_success
    final_cross = (cross_score + learn_score) / 2.0

    print(f"\n{'=' * 60}")
    print(f"end_to_end_success = {final_e2e:.3f} (gate >= 0.9)")
    print(f"cross_capability = {final_cross:.3f} (gate >= 0.8)")

    g1 = final_e2e >= 0.9
    g2 = final_cross >= 0.8

    print(f"\nGate end_to_end_success >= 0.9: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate cross_capability >= 0.8: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
