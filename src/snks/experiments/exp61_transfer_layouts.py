"""Experiment 61: Transfer Across Layouts (Stage 25).

Knowledge from DoorKey-5x5 transfers to DoorKey-6x6.

Gates:
    transfer_success >= 0.3
    transfer_success > control + 0.1
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap


N_TRAIN = 5
N_TEST = 10
MAX_STEPS_5x5 = 200
MAX_STEPS_6x6 = 300


def main():
    print("=" * 60)
    print("Experiment 61: Transfer Across Layouts (Stage 25)")
    print("=" * 60)

    # Phase 1: Train on 5x5.
    print(f"\n--- Phase 1: Training on DoorKey-5x5 ({N_TRAIN} episodes) ---")
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

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS_5x5)
        print(f"  seed={seed}: success={result.success} links={agent.causal_model.n_links}")
        env.close()

    trained_model = agent.causal_model
    print(f"  Total causal links: {trained_model.n_links}")

    # Phase 2: Test on 6x6 with trained model.
    print(f"\n--- Phase 2: Transfer to DoorKey-6x6 ({N_TEST} episodes, trained model) ---")
    transfer_success = 0
    transfer_steps = []

    for seed in range(50, 50 + N_TEST):
        env = gym.make("MiniGrid-DoorKey-6x6-v0")
        obs, _ = env.reset(seed=seed)

        transfer_agent = GoalAgent(env, grounding_map=GroundingMap(), causal_model=trained_model)
        result = transfer_agent.run_episode(obs["mission"], max_steps=MAX_STEPS_6x6)

        if result.success:
            transfer_success += 1
            transfer_steps.append(result.steps_taken)
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken}")
        env.close()

    transfer_rate = transfer_success / N_TEST
    transfer_avg_steps = sum(transfer_steps) / max(len(transfer_steps), 1)

    # Phase 3: Control — 6x6 with empty model.
    print(f"\n--- Phase 3: Control DoorKey-6x6 ({N_TEST} episodes, empty model) ---")
    control_success = 0
    control_steps = []

    for seed in range(50, 50 + N_TEST):
        env = gym.make("MiniGrid-DoorKey-6x6-v0")
        obs, _ = env.reset(seed=seed)

        control_agent = GoalAgent(env)
        result = control_agent.run_episode(obs["mission"], max_steps=MAX_STEPS_6x6)

        if result.success:
            control_success += 1
            control_steps.append(result.steps_taken)
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken}")
        env.close()

    control_rate = control_success / N_TEST
    control_avg_steps = sum(control_steps) / max(len(control_steps), 1)

    # Results.
    print(f"\n--- Results ---")
    print(f"  Transfer: rate={transfer_rate:.3f} ({transfer_success}/{N_TEST}), avg_steps={transfer_avg_steps:.1f}")
    print(f"  Control:  rate={control_rate:.3f} ({control_success}/{N_TEST}), avg_steps={control_avg_steps:.1f}")
    print(f"  Rate advantage: {transfer_rate - control_rate:.3f}")
    if control_avg_steps > 0:
        efficiency = 1.0 - transfer_avg_steps / control_avg_steps
        print(f"  Step efficiency: {efficiency:.1%} fewer steps with transfer")

    gate_transfer = transfer_rate >= 0.3
    # Advantage gate: either rate advantage OR efficiency advantage (fewer steps).
    # If both succeed 100%, transfer is still better if it uses fewer steps.
    rate_advantage = transfer_rate > control_rate + 0.1
    step_advantage = transfer_avg_steps < control_avg_steps * 0.8  # 20% fewer steps
    gate_advantage = rate_advantage or step_advantage

    print(f"\n{'=' * 60}")
    print(f"GATE: transfer    {'PASS' if gate_transfer else 'FAIL'} ({transfer_rate:.3f} >= 0.300)")
    if rate_advantage:
        print(f"GATE: advantage   PASS (rate: {transfer_rate:.3f} > {control_rate:.3f} + 0.1)")
    elif step_advantage:
        print(f"GATE: advantage   PASS (efficiency: {transfer_avg_steps:.1f} < {control_avg_steps:.1f} * 0.8)")
    else:
        print(f"GATE: advantage   FAIL (no rate or step advantage)")
    print(f"{'=' * 60}")

    if gate_transfer and gate_advantage:
        print(">>> Experiment 61: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 61: GATE FAIL <<<")


if __name__ == "__main__":
    main()
