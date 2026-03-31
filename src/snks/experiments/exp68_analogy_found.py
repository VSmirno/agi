"""Experiment 68: Analogy Detection (Stage 28).

Gates:
    analogy found = True (at least one analogy found in card/gate world)
    best similarity >= 0.7
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.analogical_reasoner import AnalogicalReasoner
from snks.language.grid_perception import GridPerception, SKS_CARD_PRESENT, SKS_GATE_LOCKED
from snks.language.grounding_map import GroundingMap
from snks.language.skill_agent import SkillAgent


N_TRAIN = 5
MAX_STEPS_TRAIN = 200


def main():
    print("=" * 60)
    print("Experiment 68: Analogy Detection (Stage 28)")
    print("=" * 60)

    # Phase 1: Train on DoorKey-5x5, extract skill library.
    print(f"\n--- Phase 1: Train on DoorKey-5x5 ({N_TRAIN} episodes) ---")
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    agent = SkillAgent(env)

    for ep in range(N_TRAIN):
        if ep > 0:
            env_new = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env_new.reset(seed=ep)
            agent._env = env_new
            agent._executor._env = env_new

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS_TRAIN)
        print(f"  Ep {ep}: success={result.success} steps={result.steps_taken}")

    lib = agent.library
    print(f"  Skills extracted: {len(lib.skills)}")
    for s in lib.skills:
        print(f"    {s.name}: pre={sorted(s.preconditions)} eff={sorted(s.effects)} composite={s.is_composite}")

    # Phase 2: Perceive CardGateWorld, check analogies.
    print(f"\n--- Phase 2: Analogy Detection in CardGateWorld ---")
    from snks.env.card_gate_world import CardGateWorld

    cg_env = CardGateWorld(size=5)
    obs, _ = cg_env.reset(seed=42)

    gmap = GroundingMap()
    perception = GridPerception(gmap)
    uw = cg_env.unwrapped
    current_sks = perception.perceive(
        uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
        carrying=getattr(uw, "carrying", None),
    )

    print(f"  CardGateWorld predicates: {sorted(current_sks)}")
    has_card = SKS_CARD_PRESENT in current_sks
    has_gate = SKS_GATE_LOCKED in current_sks
    print(f"  SKS_CARD_PRESENT={SKS_CARD_PRESENT}: {has_card}")
    print(f"  SKS_GATE_LOCKED={SKS_GATE_LOCKED}: {has_gate}")

    reasoner = AnalogicalReasoner(threshold=0.7)
    analogies = reasoner.find_analogy(lib, current_sks)

    print(f"\n--- Analogies Found ({len(analogies)}) ---")
    for a in analogies:
        print(f"  {a.source_skill_name} → {a.adapted_skill.name}")
        print(f"    similarity={a.similarity:.3f}")
        print(f"    sks_mapping={a.sks_mapping}")
        print(f"    role_mapping={a.role_mapping}")
        print(f"    adapted: pre={sorted(a.adapted_skill.preconditions)} eff={sorted(a.adapted_skill.effects)} target='{a.adapted_skill.target_word}'")

    best_similarity = max((a.similarity for a in analogies), default=0.0)
    analogy_found = len(analogies) > 0

    print(f"\n--- Results ---")
    print(f"  Analogies found: {len(analogies)}")
    print(f"  Best similarity: {best_similarity:.3f}")

    gate_found = analogy_found
    gate_similarity = best_similarity >= 0.7

    print(f"\n{'=' * 60}")
    print(f"GATE: analogy_found  {'PASS' if gate_found else 'FAIL'} ({len(analogies)} >= 1)")
    print(f"GATE: similarity     {'PASS' if gate_similarity else 'FAIL'} ({best_similarity:.3f} >= 0.700)")
    print(f"{'=' * 60}")

    if gate_found and gate_similarity:
        print(">>> Experiment 68: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 68: GATE FAIL <<<")

    cg_env.close()


if __name__ == "__main__":
    main()
