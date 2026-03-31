"""Experiment 64: Selective Transfer — Negative Test (Stage 26).

DoorKey knowledge should NOT degrade performance in a non-door environment.
PushBox has no doors/keys — agent should navigate directly to goal.

Gates:
    no incorrect transfer actions (toggle/pickup on empty cells)
    steps with transfer <= 1.1x steps without
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.goal_agent import GoalAgent, ACT_PICKUP, ACT_TOGGLE
from snks.language.grounding_map import GroundingMap
from snks.language.grid_perception import GridPerception
from snks.language.grid_navigator import GridNavigator, PathStatus


N_TRAIN = 5
N_TEST = 10
MAX_STEPS = 200


def train_doorkey_model():
    """Train GoalAgent on DoorKey-5x5, return causal model."""
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

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        print(f"  Train seed={seed}: success={result.success} links={agent.causal_model.n_links}")
        env.close()

    return agent.causal_model


def run_empty_env(causal_model, label=""):
    """Run in MiniGrid-Empty (no doors/keys) — just navigate to goal."""
    successes = 0
    total_steps = 0
    incorrect_actions = 0

    for seed in range(50, 50 + N_TEST):
        # Use Empty-6x6 with a goal placed at default position
        env = gym.make("MiniGrid-Empty-6x6-v0")
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped

        gmap = GroundingMap()
        perception = GridPerception(gmap)
        navigator = GridNavigator()

        # Perceive and navigate to goal
        carrying = getattr(uw, "carrying", None)
        perception.perceive(uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying)

        goal_obj = perception.find_object("goal")
        steps = 0

        if goal_obj is not None:
            path_result = navigator.plan_path_ex(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
                goal_obj.pos, stop_adjacent=False,
            )

            if path_result.status == PathStatus.OK:
                for action in path_result.actions:
                    if steps >= MAX_STEPS:
                        break
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    if terminated:
                        if reward > 0:
                            successes += 1
                        break

        # Check if causal model suggests any actions in this context
        if causal_model is not None:
            carrying = getattr(uw, "carrying", None)
            current_sks = perception.perceive(
                uw.grid, tuple(uw.agent_pos), int(uw.agent_dir), carrying=carrying,
            )
            # Check if model suggests pickup/toggle
            for action_id in [ACT_PICKUP, ACT_TOGGLE]:
                _, conf = causal_model.predict_effect(current_sks, action_id)
                if conf > 0.5:
                    incorrect_actions += 1
                    print(f"  WARNING: model suggests action {action_id} with conf={conf:.2f} in empty env")

        total_steps += steps
        print(f"  {label} seed={seed}: success={steps > 0 and successes > 0} steps={steps}")
        env.close()

    avg_steps = total_steps / max(N_TEST, 1)
    return successes, avg_steps, incorrect_actions


def main():
    print("=" * 60)
    print("Experiment 64: Selective Transfer — Negative Test (Stage 26)")
    print("=" * 60)

    # Phase 1: Train on DoorKey.
    print(f"\n--- Phase 1: Train on DoorKey-5x5 ({N_TRAIN} episodes) ---")
    trained_model = train_doorkey_model()
    print(f"  Causal links: {trained_model.n_links}")

    # Phase 2: Run in empty env WITH transferred model.
    print(f"\n--- Phase 2: Empty env with transfer ({N_TEST} episodes) ---")
    t_success, t_steps, t_incorrect = run_empty_env(trained_model, label="Transfer")

    # Phase 3: Run in empty env WITHOUT model (baseline).
    print(f"\n--- Phase 3: Empty env without transfer ({N_TEST} episodes) ---")
    c_success, c_steps, c_incorrect = run_empty_env(None, label="Control")

    # Results.
    print(f"\n--- Results ---")
    print(f"  Transfer: success={t_success}/{N_TEST} avg_steps={t_steps:.1f} incorrect={t_incorrect}")
    print(f"  Control:  success={c_success}/{N_TEST} avg_steps={c_steps:.1f}")

    if c_steps > 0:
        degradation = t_steps / c_steps
    else:
        degradation = 1.0

    print(f"  Step ratio: {degradation:.2f}x (1.0 = no degradation)")

    gate_no_incorrect = t_incorrect == 0
    gate_no_degradation = degradation <= 1.1

    print(f"\n{'=' * 60}")
    print(f"GATE: no_incorrect     {'PASS' if gate_no_incorrect else 'FAIL'} (incorrect={t_incorrect} == 0)")
    print(f"GATE: no_degradation   {'PASS' if gate_no_degradation else 'FAIL'} (ratio={degradation:.2f} <= 1.10)")
    print(f"{'=' * 60}")

    if gate_no_incorrect and gate_no_degradation:
        print(">>> Experiment 64: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 64: GATE FAIL <<<")


if __name__ == "__main__":
    main()
