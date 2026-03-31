"""Experiment 59: Causal Learning Speed (Stage 25).

How many episodes until GoalAgent learns DoorKey rules.

Gate: episodes_to_learn <= 5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.language.goal_agent import GoalAgent
from snks.language.grid_perception import SKS_KEY_HELD, SKS_DOOR_OPEN
from snks.language.grounding_map import GroundingMap


N_EPISODES = 10
MAX_STEPS = 200


def check_links(model) -> dict[str, bool]:
    """Check if key causal links are learned."""
    links = model.get_causal_links(min_confidence=0.0)

    has_pickup_key = False
    has_toggle_door = False

    for link in links:
        # pickup → key_held
        if link.action == 3 and SKS_KEY_HELD in link.effect_sks:
            has_pickup_key = True
        # toggle → door_open
        if link.action == 5 and SKS_DOOR_OPEN in link.effect_sks:
            has_toggle_door = True

    return {"pickup_key": has_pickup_key, "toggle_door": has_toggle_door}


def main():
    print("=" * 60)
    print("Experiment 59: Causal Learning Speed (Stage 25)")
    print("=" * 60)

    gmap = GroundingMap()
    agent = None
    learned_at = None

    for episode in range(N_EPISODES):
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env.reset(seed=episode)

        if agent is None:
            agent = GoalAgent(env, grounding_map=gmap)
        else:
            agent._env = env
            agent._executor._env = env

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)

        status = check_links(agent.causal_model)
        n_links = agent.causal_model.n_links

        print(f"  Episode {episode + 1}: success={result.success} explored={result.explored} "
              f"links={n_links} pickup_key={status['pickup_key']} toggle_door={status['toggle_door']}")

        if status["pickup_key"] and status["toggle_door"] and learned_at is None:
            learned_at = episode + 1

        env.close()

    print(f"\n--- Results ---")
    if learned_at is not None:
        print(f"  All key links learned at episode: {learned_at}")
    else:
        print(f"  Links NOT fully learned after {N_EPISODES} episodes")

    gate_pass = learned_at is not None and learned_at <= 5
    print(f"\n{'=' * 60}")
    print(f"GATE: learning_speed {'PASS' if gate_pass else 'FAIL'} "
          f"({learned_at if learned_at else '>10'} <= 5)")
    print(f"{'=' * 60}")

    if gate_pass:
        print(">>> Experiment 59: GATE PASS <<<")
    else:
        print(">>> Experiment 59: GATE FAIL <<<")


if __name__ == "__main__":
    main()
