"""Experiment 95: Curriculum Speedup (Stage 36).

Compares curriculum-trained agent vs from-scratch agent on 8x8.
Curriculum should provide >= 1.5x speedup (fewer episodes to reach 50% success).

Gates:
    curriculum_speedup >= 1.5   (curriculum agent reaches 50% in fewer episodes)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap
from snks.language.autonomous_agent import AutonomousAgent


def _episodes_to_threshold(
    causal_model: CausalWorldModel,
    grid_size: int,
    threshold: float = 0.5,
    max_episodes: int = 200,
    window: int = 20,
) -> tuple[int, float]:
    """Run episodes until success_rate >= threshold over last `window` episodes.

    Returns (episodes_needed, final_success_rate).
    """
    from collections import deque
    from snks.language.autonomous_agent import _env_name

    env_name = _env_name(grid_size)
    results: deque[bool] = deque(maxlen=window)
    max_steps = grid_size * grid_size * 3

    for ep in range(max_episodes):
        env = gym.make(env_name)
        obs, _ = env.reset(seed=ep + 1000)  # different seeds from curriculum
        gmap = GroundingMap()
        agent = GoalAgent(env, grounding_map=gmap, causal_model=causal_model)
        instruction = obs.get("mission", "go to the goal") if isinstance(obs, dict) else "go to the goal"
        result = agent.run_episode(instruction, max_steps=max_steps)
        results.append(result.success)
        env.close()

        if len(results) >= window:
            rate = sum(results) / len(results)
            if rate >= threshold:
                return ep + 1, rate

    final_rate = sum(results) / len(results) if results else 0.0
    return max_episodes, final_rate


def main():
    print("=" * 60)
    print("Experiment 95: Curriculum Speedup (Stage 36)")
    print("=" * 60)

    threshold = 0.5

    # --- Curriculum agent: train on 5x5 first, then test on 8x8 ---
    print("\n--- Curriculum: warmup on 5x5 ---")
    curr_agent = AutonomousAgent(levels=[5], advance_threshold=0.3)
    curr_agent.run_curriculum(total_episodes=50)
    stats5 = curr_agent.curriculum.get_stats(5)
    print(f"  5x5 warmup: success={stats5.success_rate:.2f} "
          f"links={curr_agent.causal_model.n_links}")

    print(f"\n--- Curriculum: testing on 8x8 ---")
    curr_eps, curr_rate = _episodes_to_threshold(
        curr_agent.causal_model, grid_size=8, threshold=threshold,
    )
    print(f"  Curriculum: {curr_eps} episodes to reach {threshold} "
          f"(final rate={curr_rate:.3f})")

    # --- From-scratch agent: directly on 8x8 ---
    print(f"\n--- From-scratch: testing on 8x8 ---")
    scratch_model = CausalWorldModel(CausalAgentConfig(causal_min_observations=1))
    scratch_eps, scratch_rate = _episodes_to_threshold(
        scratch_model, grid_size=8, threshold=threshold,
    )
    print(f"  From-scratch: {scratch_eps} episodes to reach {threshold} "
          f"(final rate={scratch_rate:.3f})")

    # Speedup
    speedup = scratch_eps / max(curr_eps, 1)

    print(f"\n--- Results ---")
    print(f"  Curriculum:   {curr_eps} episodes (rate={curr_rate:.3f})")
    print(f"  From-scratch: {scratch_eps} episodes (rate={scratch_rate:.3f})")
    print(f"  Speedup: {speedup:.2f}x")

    gate_speedup = speedup >= 1.5

    print(f"\n{'=' * 60}")
    print(f"GATE: speedup {'PASS' if gate_speedup else 'FAIL'} ({speedup:.2f}x >= 1.50x)")
    print(f"{'=' * 60}")

    if gate_speedup:
        print("*** ALL GATES PASS ***")
    else:
        print("*** SOME GATES FAIL ***")


if __name__ == "__main__":
    main()
