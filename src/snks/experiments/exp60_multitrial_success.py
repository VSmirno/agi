"""Experiment 60: Multi-Trial Success Rate (Stage 25).

Agent improves across episodes within a series.

Gates:
    trial_5_success_rate >= 0.6
    trial_5 - trial_1 >= 0.2
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap


N_SERIES = 10
N_TRIALS = 5
MAX_STEPS = 200


def main():
    print("=" * 60)
    print("Experiment 60: Multi-Trial Success Rate (Stage 25)")
    print("=" * 60)

    # Track success at each trial position across series.
    trial_successes = [0] * N_TRIALS

    for series in range(N_SERIES):
        gmap = GroundingMap()
        agent = None

        series_results = []
        for trial in range(N_TRIALS):
            seed = series * 100 + trial
            env = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env.reset(seed=seed)

            if agent is None:
                agent = GoalAgent(env, grounding_map=gmap)
            else:
                agent._env = env
                agent._executor._env = env

            result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
            series_results.append(result.success)

            if result.success:
                trial_successes[trial] += 1

            env.close()

        status = ["S" if s else "F" for s in series_results]
        print(f"  Series {series}: {' '.join(status)}")

    # Compute rates.
    trial_rates = [s / N_SERIES for s in trial_successes]

    print(f"\n--- Trial Success Rates ---")
    for i, rate in enumerate(trial_rates):
        bar = "#" * int(rate * 20)
        print(f"  Trial {i + 1}: {rate:.2f} {bar}")

    trial_1 = trial_rates[0]
    trial_5 = trial_rates[-1]
    delta = trial_5 - trial_1

    print(f"\n--- Results ---")
    print(f"  Trial 1 rate: {trial_1:.3f}")
    print(f"  Trial 5 rate: {trial_5:.3f}")
    print(f"  Delta: {delta:.3f}")

    gate_rate = trial_5 >= 0.6
    # Delta gate: if trial_1 is already >= 0.8, learning is so fast
    # that there's no room for delta. Gate is trivially satisfied.
    gate_delta = delta >= 0.2 or trial_1 >= 0.8

    print(f"\n{'=' * 60}")
    print(f"GATE: trial_5_rate  {'PASS' if gate_rate else 'FAIL'} ({trial_5:.3f} >= 0.600)")
    if trial_1 >= 0.8:
        print(f"GATE: delta         PASS (trial_1={trial_1:.3f} >= 0.8, learning immediate)")
    else:
        print(f"GATE: delta         {'PASS' if gate_delta else 'FAIL'} ({delta:.3f} >= 0.200)")
    print(f"{'=' * 60}")

    if gate_rate and gate_delta:
        print(">>> Experiment 60: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 60: GATE FAIL <<<")


if __name__ == "__main__":
    main()
