"""Experiment 96: Multi-Room Navigation (Stage 37).

GoalAgent must navigate through rooms with closed doors.
Tests progressive difficulty: UnlockPickup → MultiRoom-N2 → MultiRoom-N4.

Gates:
    unlock_success >= 0.8       (simple unlock + pickup)
    multiroom_n2_success >= 0.5 (2 rooms)
    multiroom_n4_success >= 0.2 (4 rooms, stretch)
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


def _run_env(env_name: str, model: CausalWorldModel, n_episodes: int, max_steps: int) -> dict:
    """Run episodes in an environment, return stats."""
    successes = 0
    total_steps = 0
    explored_count = 0

    for seed in range(n_episodes):
        env = gym.make(env_name)
        obs, _ = env.reset(seed=seed)
        gmap = GroundingMap()
        agent = GoalAgent(env, grounding_map=gmap, causal_model=model)
        mission = obs.get("mission", "go to the goal") if isinstance(obs, dict) else "go to the goal"
        result = agent.run_episode(mission, max_steps=max_steps)
        if result.success:
            successes += 1
        total_steps += result.steps_taken
        if result.explored:
            explored_count += 1
        env.close()

        if (seed + 1) % 10 == 0:
            print(f"    ep={seed+1}: success={successes}/{seed+1} "
                  f"({successes/(seed+1):.2f}) links={model.n_links}")

    return {
        "env": env_name,
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": round(successes / n_episodes, 3),
        "total_steps": total_steps,
        "explored": explored_count,
        "causal_links": model.n_links,
    }


def main():
    print("=" * 60)
    print("Experiment 96: Multi-Room Navigation (Stage 37)")
    print("=" * 60)

    # Shared causal model (curriculum transfer)
    model = CausalWorldModel(CausalAgentConfig(causal_min_observations=1))

    # Phase 1: Warmup on DoorKey-5x5
    print("\n--- Phase 1: DoorKey-5x5 warmup (20 eps) ---")
    r0 = _run_env("MiniGrid-DoorKey-5x5-v0", model, 20, 200)
    print(f"  DoorKey-5x5: {r0['success_rate']} links={r0['causal_links']}")

    # Phase 2: UnlockPickup (simple: unlock door + pick up box)
    print("\n--- Phase 2: UnlockPickup (30 eps) ---")
    r1 = _run_env("MiniGrid-UnlockPickup-v0", model, 30, 300)
    print(f"  UnlockPickup: {r1['success_rate']} links={r1['causal_links']}")

    # Phase 3: MultiRoom-N2-S4 (2 rooms)
    print("\n--- Phase 3: MultiRoom-N2-S4 (50 eps) ---")
    r2 = _run_env("MiniGrid-MultiRoom-N2-S4-v0", model, 50, 500)
    print(f"  MultiRoom-N2: {r2['success_rate']} links={r2['causal_links']}")

    # Phase 4: MultiRoom-N4-S5 (4 rooms)
    print("\n--- Phase 4: MultiRoom-N4-S5 (50 eps) ---")
    r3 = _run_env("MiniGrid-MultiRoom-N4-S5-v0", model, 50, 1000)
    print(f"  MultiRoom-N4: {r3['success_rate']} links={r3['causal_links']}")

    # Gates
    gate_unlock = r1["success_rate"] >= 0.8
    gate_n2 = r2["success_rate"] >= 0.5
    gate_n4 = r3["success_rate"] >= 0.2

    print(f"\n{'=' * 60}")
    print(f"GATE: unlock {'PASS' if gate_unlock else 'FAIL'} ({r1['success_rate']} >= 0.800)")
    print(f"GATE: multiroom_n2 {'PASS' if gate_n2 else 'FAIL'} ({r2['success_rate']} >= 0.500)")
    print(f"GATE: multiroom_n4 {'PASS' if gate_n4 else 'FAIL'} ({r3['success_rate']} >= 0.200)")
    print(f"{'=' * 60}")

    if gate_unlock and gate_n2:
        print("*** ALL GATES PASS ***")
    else:
        print("*** SOME GATES FAIL ***")


if __name__ == "__main__":
    main()
