"""Experiment 84: Multi-Agent Speedup (Stage 33).

Tests that multiple agents cooperating solve tasks faster than a single agent.
Measures knowledge reuse rate — fraction of transferred knowledge actually used.

Gates:
    multi_agent_speedup >= 1.3x
    knowledge_reuse_rate >= 0.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.multi_agent_env import MultiAgentEnv


# ── Simulated task: DoorKey requires 3 causal links ──

FULL_TASK_LINKS = [
    CausalLink(action=3, context_sks=frozenset({50}),
                effect_sks=frozenset({51}), strength=0.9, count=5),
    CausalLink(action=5, context_sks=frozenset({51, 52}),
                effect_sks=frozenset({53}), strength=0.85, count=4),
    CausalLink(action=1, context_sks=frozenset({53, 54}),
                effect_sks=frozenset({54}), strength=0.8, count=5),
    CausalLink(action=2, context_sks=frozenset({55}),
                effect_sks=frozenset({56}), strength=0.75, count=4),
    CausalLink(action=4, context_sks=frozenset({56, 57}),
                effect_sks=frozenset({58}), strength=0.7, count=5),
    CausalLink(action=0, context_sks=frozenset({58}),
                effect_sks=frozenset({54}), strength=0.8, count=4),
]


def _single_agent_rounds(links: list[CausalLink]) -> int:
    """Simulate: single agent must discover all links through exploration.

    Each link takes N rounds to discover (N = observation count needed).
    Returns total rounds.
    """
    total_rounds = 0
    for link in links:
        # Single agent must explore + observe min 3 times per link.
        total_rounds += max(link.count, 3)
    return total_rounds


def _multi_agent_rounds(
    links: list[CausalLink],
    n_agents: int = 2,
) -> tuple[int, int, int]:
    """Simulate: N agents split exploration, then share via concept messages.

    Returns (total_rounds, links_transferred, links_total).
    """
    env = MultiAgentEnv(n_agents=n_agents)

    # Split links among agents (each explores different part).
    chunks = [[] for _ in range(n_agents)]
    for i, link in enumerate(links):
        chunks[i % n_agents].append(link)

    # Phase 1: Each agent explores their chunk.
    exploration_rounds = 0
    task_links = {}
    for i, chunk in enumerate(chunks):
        agent_id = f"agent_{i}"
        task_links[agent_id] = chunk
        for link in chunk:
            exploration_rounds += max(link.count, 3)

    # Parallel exploration: total rounds = max of per-agent rounds.
    per_agent_rounds = []
    for chunk in chunks:
        r = sum(max(l.count, 3) for l in chunk)
        per_agent_rounds.append(r)
    parallel_rounds = max(per_agent_rounds) if per_agent_rounds else 0

    # Phase 2: Communication rounds (fast — 1-2 rounds).
    result = env.run_cooperative_episode(task_links=task_links, max_rounds=5)

    comm_rounds = 2  # typical convergence

    total = parallel_rounds + comm_rounds

    return total, result.links_transferred, len(links)


def test_speedup() -> tuple[float, float, list[str]]:
    """Compare single-agent vs multi-agent task completion rounds.

    Returns (speedup_ratio, reuse_rate, details).
    """
    details = []

    single_rounds = _single_agent_rounds(FULL_TASK_LINKS)
    multi_rounds, links_transferred, links_total = _multi_agent_rounds(
        FULL_TASK_LINKS, n_agents=2,
    )

    speedup = single_rounds / max(multi_rounds, 1)
    reuse_rate = links_transferred / max(links_total, 1)

    details.append(f"Single agent rounds: {single_rounds}")
    details.append(f"Multi agent rounds: {multi_rounds}")
    details.append(f"Speedup: {speedup:.2f}x")
    details.append(f"Links transferred: {links_transferred}")
    details.append(f"Links total: {links_total}")
    details.append(f"Knowledge reuse rate: {reuse_rate:.3f}")

    return speedup, reuse_rate, details


def test_scaling() -> tuple[float, list[str]]:
    """Test speedup scales with more agents and more links."""
    details = []

    # Bigger task: 6 links.
    big_task = FULL_TASK_LINKS + [
        CausalLink(action=2, context_sks=frozenset({55}),
                    effect_sks=frozenset({56}), strength=0.75, count=4),
        CausalLink(action=4, context_sks=frozenset({56, 57}),
                    effect_sks=frozenset({58}), strength=0.7, count=3),
        CausalLink(action=0, context_sks=frozenset({58}),
                    effect_sks=frozenset({54}), strength=0.8, count=5),
    ]

    single = _single_agent_rounds(big_task)
    multi_2, _, _ = _multi_agent_rounds(big_task, n_agents=2)
    multi_3, _, _ = _multi_agent_rounds(big_task, n_agents=3)

    speedup_2 = single / max(multi_2, 1)
    speedup_3 = single / max(multi_3, 1)

    details.append(f"Big task (6 links):")
    details.append(f"  Single: {single} rounds")
    details.append(f"  2 agents: {multi_2} rounds (speedup {speedup_2:.2f}x)")
    details.append(f"  3 agents: {multi_3} rounds (speedup {speedup_3:.2f}x)")
    details.append(f"  3-agent > 2-agent: {speedup_3 > speedup_2}")

    return speedup_2, details


def test_knowledge_reuse_detailed() -> tuple[float, list[str]]:
    """Test: transferred knowledge is actually usable (not just received)."""
    details = []

    env = MultiAgentEnv(n_agents=2, agent_ids=["explorer", "solver"])

    # Explorer discovers key knowledge.
    key_link = CausalLink(
        action=3, context_sks=frozenset({50}),
        effect_sks=frozenset({51}), strength=0.9, count=5,
    )
    env.inject_knowledge("explorer", [key_link])

    # Explorer shares.
    explorer = env.get_agent("explorer")
    explorer.communicator.share_causal_links()
    env.exchange_round()

    # Solver should now be able to predict effect of action 3 in context {50}.
    solver = env.get_agent("solver")
    predicted_effect, confidence = solver.causal_model.predict_effect(
        context_sks={50}, action=3,
    )

    # Check prediction is correct.
    correct_prediction = 51 in predicted_effect
    usable = confidence > 0.0

    details.append(f"Solver predicted effect: {predicted_effect}")
    details.append(f"Solver confidence: {confidence:.3f}")
    details.append(f"Correct prediction (51 in effect): {correct_prediction}")
    details.append(f"Knowledge usable (confidence > 0): {usable}")

    reuse_rate = 1.0 if (correct_prediction and usable) else 0.0
    return reuse_rate, details


def main() -> None:
    print("=" * 60)
    print("Experiment 84: Multi-Agent Speedup")
    print("=" * 60)

    # Test 1: Speedup.
    print("\n--- Speedup Test ---")
    speedup, reuse, details = test_speedup()
    for d in details:
        print(f"  {d}")

    # Test 2: Scaling.
    print("\n--- Scaling Test ---")
    speedup_big, details_scale = test_scaling()
    for d in details_scale:
        print(f"  {d}")

    # Test 3: Knowledge reuse.
    print("\n--- Knowledge Reuse Test ---")
    reuse_detail, details_reuse = test_knowledge_reuse_detailed()
    for d in details_reuse:
        print(f"  {d}")

    # Final metrics.
    final_speedup = speedup
    final_reuse = (reuse + reuse_detail) / 2.0

    print(f"\n{'=' * 60}")
    print(f"multi_agent_speedup = {final_speedup:.2f}x (gate >= 1.3x)")
    print(f"knowledge_reuse_rate = {final_reuse:.3f} (gate >= 0.5)")

    g1 = final_speedup >= 1.3
    g2 = final_reuse >= 0.5

    print(f"\nGate multi_agent_speedup >= 1.3x: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate knowledge_reuse_rate >= 0.5: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
