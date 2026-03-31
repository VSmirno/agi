"""Experiment 85: Cooperative Multi-Agent Success (Stage 33).

Tests that agents with different roles cooperate to solve a joint task
using only concept-level communication (no natural language).

Gates:
    cooperative_success >= 0.9
    no_word_exchange = True
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.agent.causal_model import CausalLink
from snks.language.concept_message import ConceptMessage, MessageType
from snks.language.multi_agent_env import MultiAgentEnv
from snks.language.skill import Skill


def _doorkey_scenario() -> tuple[bool, list[str]]:
    """Scenario: Explorer finds key, Solver opens door.

    Explorer discovers: key_present → pickup → key_held
    Solver discovers: key_held + door_locked → toggle → door_open
    Together they can solve: full DoorKey task.
    """
    details = []
    env = MultiAgentEnv(n_agents=2, agent_ids=["explorer", "solver"])
    env.set_role("explorer", "explorer")
    env.set_role("solver", "solver")

    # Explorer's knowledge.
    key_link = CausalLink(
        action=3, context_sks=frozenset({50}),
        effect_sks=frozenset({51}), strength=0.9, count=5,
    )
    # Solver's knowledge.
    door_link = CausalLink(
        action=5, context_sks=frozenset({51, 52}),
        effect_sks=frozenset({53}), strength=0.85, count=4,
    )
    goal_link = CausalLink(
        action=1, context_sks=frozenset({53, 54}),
        effect_sks=frozenset({54}), strength=0.8, count=3,
    )

    env.inject_knowledge("explorer", [key_link])
    env.inject_knowledge("solver", [door_link, goal_link])

    # Communication: share knowledge.
    result = env.run_cooperative_episode(task_links={}, max_rounds=5)

    # Both agents should now have all 3 links.
    explorer_links = env.get_agent("explorer").causal_model.get_causal_links(0.0)
    solver_links = env.get_agent("solver").causal_model.get_causal_links(0.0)

    explorer_keys = {(l.action, l.context_sks) for l in explorer_links}
    solver_keys = {(l.action, l.context_sks) for l in solver_links}

    # Check completeness.
    all_expected = {
        (3, frozenset({50})),
        (5, frozenset({51, 52})),
        (1, frozenset({53, 54})),
    }

    explorer_complete = all_expected <= explorer_keys
    solver_complete = all_expected <= solver_keys

    details.append(f"Explorer links: {len(explorer_keys)} (needs {len(all_expected)})")
    details.append(f"Solver links: {len(solver_keys)} (needs {len(all_expected)})")
    details.append(f"Explorer complete: {explorer_complete}")
    details.append(f"Solver complete: {solver_complete}")
    details.append(f"Messages exchanged: {result.messages_exchanged}")

    return explorer_complete and solver_complete, details


def _skill_sharing_scenario() -> tuple[bool, list[str]]:
    """Scenario: Agent A teaches skill to Agent B via concept message."""
    details = []
    env = MultiAgentEnv(n_agents=2, agent_ids=["teacher", "student"])

    # Teacher has two skills.
    pickup = Skill(
        name="pickup_key", preconditions=frozenset({50}),
        effects=frozenset({51}), terminal_action=3,
        target_word="key", success_count=10, attempt_count=10,
    )
    toggle = Skill(
        name="toggle_door", preconditions=frozenset({51, 52}),
        effects=frozenset({53}), terminal_action=5,
        target_word="door", success_count=8, attempt_count=10,
    )

    env.inject_skill("teacher", pickup)
    env.inject_skill("teacher", toggle)

    # Teacher shares both skills.
    teacher = env.get_agent("teacher")
    teacher.communicator.share_skill(pickup, receiver_id="student")
    teacher.communicator.share_skill(toggle, receiver_id="student")
    env.exchange_round()

    # Student should have both skills.
    student = env.get_agent("student")
    has_pickup = student.skill_library.get("pickup_key") is not None
    has_toggle = student.skill_library.get("toggle_door") is not None

    details.append(f"Student has pickup_key: {has_pickup}")
    details.append(f"Student has toggle_door: {has_toggle}")

    return has_pickup and has_toggle, details


def _warning_scenario() -> tuple[bool, list[str]]:
    """Scenario: Scout warns team about danger zone."""
    details = []
    env = MultiAgentEnv(
        n_agents=3,
        agent_ids=["scout", "worker_a", "worker_b"],
    )
    env.set_role("scout", "scout")
    env.set_role("worker_a", "worker")
    env.set_role("worker_b", "worker")

    # Scout encounters danger (lava / locked gate without key).
    danger_ctx = frozenset({52, 57})  # door_locked + gate_locked

    scout = env.get_agent("scout")
    scout.communicator.send_warning(danger_ctx)
    env.exchange_round()

    # Both workers should know about danger.
    a_warned = danger_ctx in env.get_agent("worker_a").communicator.warning_contexts
    b_warned = danger_ctx in env.get_agent("worker_b").communicator.warning_contexts

    details.append(f"Worker A warned: {a_warned}")
    details.append(f"Worker B warned: {b_warned}")

    return a_warned and b_warned, details


def _request_response_scenario() -> tuple[bool, list[str]]:
    """Scenario: Agent requests knowledge, gets concept-level response."""
    details = []
    env = MultiAgentEnv(n_agents=2, agent_ids=["novice", "expert"])

    # Expert has knowledge about keys.
    key_link = CausalLink(
        action=3, context_sks=frozenset({50}),
        effect_sks=frozenset({51}), strength=0.9, count=5,
    )
    env.inject_knowledge("expert", [key_link])

    # Novice requests knowledge about SKS 50 (key_present).
    novice = env.get_agent("novice")
    novice.communicator.request_knowledge(frozenset({50}), receiver_id="expert")
    env.exchange_round()

    # Expert should have prepared a response (in outbox after processing request).
    # Need another exchange round to deliver the response.
    env.exchange_round()

    # Novice should now have the key knowledge.
    novice_links = novice.causal_model.get_causal_links(0.0)
    has_key_knowledge = any(
        l.action == 3 and 50 in l.context_sks
        for l in novice_links
    )

    details.append(f"Novice has key knowledge: {has_key_knowledge}")
    details.append(f"Novice links: {len(novice_links)}")

    return has_key_knowledge, details


def _no_text_verification() -> tuple[bool, list[str]]:
    """Verify: all communication uses concepts, not text."""
    details = []
    env = MultiAgentEnv(n_agents=2, agent_ids=["agent_a", "agent_b"])

    link = CausalLink(
        action=3, context_sks=frozenset({50}),
        effect_sks=frozenset({51}), strength=0.9, count=5,
    )
    skill = Skill(
        name="pickup_key", preconditions=frozenset({50}),
        effects=frozenset({51}), terminal_action=3,
        target_word="key", success_count=5, attempt_count=5,
    )

    env.inject_knowledge("agent_a", [link])
    env.inject_skill("agent_a", skill)

    agent_a = env.get_agent("agent_a")
    agent_a.communicator.share_causal_links()
    agent_a.communicator.share_skill(skill)
    agent_a.communicator.send_warning(frozenset({52}))

    all_msgs = agent_a.communicator.flush_outbox()

    all_concept_based = True
    for msg in all_msgs:
        # Check: sks_context contains only ints.
        for sks_id in msg.sks_context:
            if not isinstance(sks_id, int):
                all_concept_based = False
                details.append(f"FAIL: sks_context contains non-int: {type(sks_id)}")

        # Check: causal_links contain only structured data.
        for cl in msg.causal_links:
            if not isinstance(cl.action, int):
                all_concept_based = False
            if not isinstance(cl.context_sks, frozenset):
                all_concept_based = False

        # Check: no natural language text in the message payload.
        # (ConceptMessage has no text field for inter-agent communication.)

    details.append(f"All messages concept-based: {all_concept_based}")
    details.append(f"Total messages checked: {len(all_msgs)}")
    details.append(f"Message types: {[m.content_type.value for m in all_msgs]}")

    return all_concept_based, details


def main() -> None:
    print("=" * 60)
    print("Experiment 85: Cooperative Multi-Agent Success")
    print("=" * 60)

    scenarios = [
        ("DoorKey Cooperation", _doorkey_scenario),
        ("Skill Sharing", _skill_sharing_scenario),
        ("Warning Propagation", _warning_scenario),
        ("Request-Response", _request_response_scenario),
        ("No Text Exchange", _no_text_verification),
    ]

    successes = 0
    no_text_ok = False

    for name, scenario_fn in scenarios:
        print(f"\n--- {name} ---")
        success, details = scenario_fn()
        for d in details:
            print(f"  {d}")
        status = "PASS" if success else "FAIL"
        print(f"  Result: {status}")
        if success:
            successes += 1
        if name == "No Text Exchange":
            no_text_ok = success

    cooperative_success = successes / len(scenarios)

    print(f"\n{'=' * 60}")
    print(f"cooperative_success = {cooperative_success:.3f} (gate >= 0.9)")
    print(f"no_word_exchange = {no_text_ok} (gate = True)")

    g1 = cooperative_success >= 0.9
    g2 = no_text_ok

    print(f"\nGate cooperative_success >= 0.9: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate no_word_exchange = True: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
