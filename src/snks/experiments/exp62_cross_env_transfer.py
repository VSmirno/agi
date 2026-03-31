"""Experiment 62: Cross-Environment Transfer (Stage 26).

Knowledge from DoorKey-5x5 transfers to MultiRoomDoorKey.

Gates:
    transfer_success >= 0.7
    speedup >= 2x (exploration episodes: transfer < control)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.env.multi_room import MultiRoomDoorKey
from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap
from snks.language.transfer_agent import TransferAgent


N_TRAIN = 5
N_TEST = 10
MAX_STEPS_TRAIN = 200
MAX_STEPS_TEST = 300


def train_on_doorkey() -> GoalAgent:
    """Train GoalAgent on DoorKey-5x5 and return it."""
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


def run_multiroom_episodes(causal_model=None, label=""):
    """Run N_TEST episodes in MultiRoomDoorKey. Returns TransferAgent with stats."""
    transfer_agent = TransferAgent(causal_model=causal_model)

    for seed in range(50, 50 + N_TEST):
        env = MultiRoomDoorKey(size=10, seed=seed)
        obs, _ = env.reset()
        instruction = "use the keys to open the doors and get to the goal"

        result = transfer_agent.run_episode(env, instruction, max_steps=MAX_STEPS_TEST)
        print(f"  {label} seed={seed}: success={result.success} steps={result.steps_taken} explored={result.explored}")
        env.close()

    return transfer_agent


def main():
    print("=" * 60)
    print("Experiment 62: Cross-Environment Transfer (Stage 26)")
    print("=" * 60)

    # Phase 1: Train on DoorKey-5x5.
    print(f"\n--- Phase 1: Training on DoorKey-5x5 ({N_TRAIN} episodes) ---")
    trained_agent = train_on_doorkey()
    trained_model = trained_agent.causal_model
    print(f"  Total causal links: {trained_model.n_links}")

    # Phase 2: Transfer to MultiRoom.
    print(f"\n--- Phase 2: Transfer to MultiRoomDoorKey ({N_TEST} episodes) ---")
    transfer = run_multiroom_episodes(causal_model=trained_model, label="Transfer")
    t_stats = transfer.get_stats()

    # Phase 3: Control (no pre-loaded knowledge).
    print(f"\n--- Phase 3: Control MultiRoomDoorKey ({N_TEST} episodes) ---")
    control = run_multiroom_episodes(causal_model=None, label="Control")
    c_stats = control.get_stats()

    # Results.
    print(f"\n--- Results ---")
    print(f"  Transfer: rate={t_stats.success_rate:.3f} ({t_stats.successes}/{N_TEST})")
    print(f"    mean_steps={t_stats.mean_steps:.1f} exploration_episodes={t_stats.exploration_episodes}")
    print(f"  Control:  rate={c_stats.success_rate:.3f} ({c_stats.successes}/{N_TEST})")
    print(f"    mean_steps={c_stats.mean_steps:.1f} exploration_episodes={c_stats.exploration_episodes}")

    # Speedup: how many fewer exploration episodes with transfer
    if c_stats.exploration_episodes > 0:
        speedup = c_stats.exploration_episodes / max(t_stats.exploration_episodes, 0.5)
    else:
        speedup = float("inf") if t_stats.exploration_episodes == 0 else 0.0

    print(f"  Speedup: {speedup:.1f}x fewer exploration episodes")

    gate_success = t_stats.success_rate >= 0.7
    gate_speedup = speedup >= 2.0

    print(f"\n{'=' * 60}")
    print(f"GATE: transfer_success  {'PASS' if gate_success else 'FAIL'} ({t_stats.success_rate:.3f} >= 0.700)")
    print(f"GATE: speedup           {'PASS' if gate_speedup else 'FAIL'} ({speedup:.1f}x >= 2.0x)")
    print(f"{'=' * 60}")

    if gate_success and gate_speedup:
        print(">>> Experiment 62: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 62: GATE FAIL <<<")


if __name__ == "__main__":
    main()
