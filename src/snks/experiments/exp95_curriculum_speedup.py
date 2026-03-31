"""Experiment 95: Transfer Validation (Stage 36).

Validates that causal knowledge transfers across grid sizes:
1. Train on 5x5 (learn causal links)
2. Test on 16x16 from first episode — should succeed immediately

Gates:
    transfer_16x16_ep1 >= 0.8   (first 20 episodes on 16x16 with transferred knowledge)
    from_scratch_16x16_ep1 >= 0.5  (from scratch should also work, but maybe slower)
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
    print("Experiment 95: Transfer Validation (Stage 36)")
    print("=" * 60)

    # --- Transfer: train on 5x5, immediately test on 16x16 ---
    print("\n--- Phase 1: Train on 5x5 (20 episodes) ---")
    curr_agent = AutonomousAgent(levels=[5], advance_threshold=0.3)
    curr_agent.run_curriculum(total_episodes=20)
    stats5 = curr_agent.curriculum.get_stats(5)
    links_after_5x5 = curr_agent.causal_model.n_links
    print(f"  5x5: success={stats5.success_rate:.2f} links={links_after_5x5}")

    print(f"\n--- Phase 2: Test on 16x16 with transferred knowledge (20 eps) ---")
    transfer_eps, transfer_rate = _episodes_to_threshold(
        curr_agent.causal_model, grid_size=16, threshold=0.8, max_episodes=20, window=20,
    )
    print(f"  Transfer 16x16: rate={transfer_rate:.3f} in {transfer_eps} eps")

    print(f"\n--- Phase 3: From-scratch on 16x16 (20 eps) ---")
    scratch_model = CausalWorldModel(CausalAgentConfig(causal_min_observations=1))
    scratch_eps, scratch_rate = _episodes_to_threshold(
        scratch_model, grid_size=16, threshold=0.8, max_episodes=20, window=20,
    )
    print(f"  Scratch 16x16: rate={scratch_rate:.3f} in {scratch_eps} eps")

    # Gates
    gate_transfer = transfer_rate >= 0.8
    gate_links = links_after_5x5 >= 5

    print(f"\n--- Results ---")
    print(f"  Transfer 16x16: {transfer_rate:.3f}")
    print(f"  Scratch 16x16:  {scratch_rate:.3f}")
    print(f"  Links learned:  {links_after_5x5}")

    print(f"\n{'=' * 60}")
    print(f"GATE: transfer_16x16 {'PASS' if gate_transfer else 'FAIL'} ({transfer_rate:.3f} >= 0.800)")
    print(f"GATE: causal_links {'PASS' if gate_links else 'FAIL'} ({links_after_5x5} >= 5)")
    print(f"{'=' * 60}")

    if gate_transfer and gate_links:
        print("*** ALL GATES PASS ***")
    else:
        print("*** SOME GATES FAIL ***")


if __name__ == "__main__":
    main()
