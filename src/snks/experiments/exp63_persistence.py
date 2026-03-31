"""Experiment 63: Causal Knowledge Persistence — Save/Load (Stage 26).

Verify serialization roundtrip preserves causal knowledge.

Gates:
    success rate difference <= 5% (loaded vs direct)
    n_links preserved exactly
"""

from __future__ import annotations

import sys
import tempfile
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.agent.causal_serializer import CausalModelSerializer
from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap


N_TRAIN = 5
N_TEST = 10
MAX_STEPS_TRAIN = 200
MAX_STEPS_TEST = 300


def train_on_doorkey():
    gmap = GroundingMap()
    agent = None

    for seed in range(N_TRAIN):
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env.reset(seed=seed)

        if agent is None:
            agent = GoalAgent(env, grounding_map=gmap)
        else:
            agent._env = env
            agent._executor._env = env

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS_TRAIN)
        print(f"  Train seed={seed}: success={result.success} links={agent.causal_model.n_links}")
        env.close()

    return agent


def run_6x6_episodes(causal_model, label=""):
    successes = 0
    steps_list = []

    for seed in range(50, 50 + N_TEST):
        env = gym.make("MiniGrid-DoorKey-6x6-v0")
        obs, _ = env.reset(seed=seed)

        agent = GoalAgent(env, grounding_map=GroundingMap(), causal_model=causal_model)
        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS_TEST)

        if result.success:
            successes += 1
            steps_list.append(result.steps_taken)
        print(f"  {label} seed={seed}: success={result.success} steps={result.steps_taken}")
        env.close()

    rate = successes / N_TEST
    avg_steps = sum(steps_list) / max(len(steps_list), 1)
    return rate, avg_steps


def main():
    print("=" * 60)
    print("Experiment 63: Causal Knowledge Persistence (Stage 26)")
    print("=" * 60)

    # Phase 1: Train.
    print(f"\n--- Phase 1: Train on DoorKey-5x5 ({N_TRAIN} episodes) ---")
    trained_agent = train_on_doorkey()
    original_model = trained_agent.causal_model
    original_links = original_model.n_links
    print(f"  Causal links: {original_links}")

    # Phase 2: Serialize and load.
    print(f"\n--- Phase 2: Serialize / Deserialize ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    CausalModelSerializer.save(original_model, path, source_env="DoorKey-5x5")
    loaded_model = CausalModelSerializer.load(path)
    loaded_links = loaded_model.n_links

    print(f"  Original links: {original_links}")
    print(f"  Loaded links:   {loaded_links}")
    print(f"  Match: {original_links == loaded_links}")

    # Phase 3: Test direct transfer.
    print(f"\n--- Phase 3: Direct transfer to DoorKey-6x6 ({N_TEST} episodes) ---")
    direct_rate, direct_steps = run_6x6_episodes(original_model, label="Direct")

    # Phase 4: Test loaded transfer.
    print(f"\n--- Phase 4: Loaded transfer to DoorKey-6x6 ({N_TEST} episodes) ---")
    loaded_rate, loaded_steps = run_6x6_episodes(loaded_model, label="Loaded")

    # Results.
    print(f"\n--- Results ---")
    print(f"  Direct:  rate={direct_rate:.3f} avg_steps={direct_steps:.1f}")
    print(f"  Loaded:  rate={loaded_rate:.3f} avg_steps={loaded_steps:.1f}")
    rate_diff = abs(direct_rate - loaded_rate)
    print(f"  Rate difference: {rate_diff:.3f}")

    gate_rate = rate_diff <= 0.05
    gate_links = original_links == loaded_links

    print(f"\n{'=' * 60}")
    print(f"GATE: rate_match   {'PASS' if gate_rate else 'FAIL'} (diff={rate_diff:.3f} <= 0.050)")
    print(f"GATE: link_count   {'PASS' if gate_links else 'FAIL'} ({loaded_links} == {original_links})")
    print(f"{'=' * 60}")

    if gate_rate and gate_links:
        print(">>> Experiment 63: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 63: GATE FAIL <<<")


if __name__ == "__main__":
    main()
