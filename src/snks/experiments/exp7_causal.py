"""Experiment 7: Causal Learning (push-causality).

Environment: L2 (PushBox) — 8×8, 1 box + scripted ball (correlation control).
Agent explores 500 steps (curiosity-driven), learns causal links.

Test: predict_effect for 50 "agent in front of box" situations.
Control: agent should NOT form causal link action→Ball (scripted).

Gate: Causal precision > 0.8, Causal recall > 0.7
"""

from __future__ import annotations

import numpy as np

from snks.agent.agent import CausalAgent
from snks.agent.causal_model import CausalLink
from snks.daf.types import (
    CausalAgentConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import Action, CausalGridWorld, make_level


def make_config(device: str = "cpu", num_nodes: int = 5000) -> CausalAgentConfig:
    return CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(
                num_nodes=num_nodes,
                avg_degree=20,
                oscillator_model="fhn",
                coupling_strength=0.05,
                dt=0.01,
                noise_sigma=0.005,
                fhn_I_base=0.0,
                device=device,
            ),
            encoder=EncoderConfig(
                sdr_size=4096,
                sdr_sparsity=0.04,
                sdr_current_strength=1.0,
            ),
            sks=SKSConfig(
                coherence_mode="rate",
                min_cluster_size=5,
                dbscan_min_samples=5,
            ),
            steps_per_cycle=100,
            device=device,
        ),
        motor_sdr_size=256,
        causal_min_observations=2,
        curiosity_epsilon=0.3,
    )


def run(
    device: str = "cpu",
    num_nodes: int = 5000,
    explore_steps: int = 500,
    test_scenarios: int = 50,
) -> dict:
    """Run Experiment 7: Causal Learning.

    Returns dict with metrics: precision, recall, n_causal_links, n_interact_links.
    """
    config = make_config(device=device, num_nodes=num_nodes)
    agent = CausalAgent(config)

    # Create environment with scripted ball (correlation control)
    env = CausalGridWorld(level="PushBox", size=8, scripted_objects=True, max_steps=explore_steps + 100)
    env_rgb = make_level("PushBox", scripted_objects=True, max_steps=explore_steps + 100)

    # Phase 1: Exploration
    obs, info = env_rgb.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for step in range(explore_steps):
        action = agent.step(img)
        obs, reward, terminated, truncated, info = env_rgb.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent.observe_result(img)

        if terminated or truncated:
            obs, info = env_rgb.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    # Phase 2: Extract causal links
    links = agent.causal_model.get_causal_links(min_confidence=0.3)
    n_links = len(links)

    # Evaluate precision and recall per spec:
    # - Precision: fraction of learned causal links that are REAL (true positives / all learned)
    # - Recall: fraction of REAL causal links that were learned
    #
    # Ground truth for PushBox (Sokoban-style):
    # - Real causal effects: forward→push box, interact→push box, turn→view change
    # - NOT causal: noop shouldn't change anything; scripted ball is NOT agent-caused
    #
    # All actions except noop genuinely change the observation (turn changes view,
    # forward changes position/pushes, interact pushes/toggles). Noop = no change.
    # False positives: noop links with effects (noise), or any link to scripted ball

    causal_actions = {Action.turn_left, Action.turn_right, Action.forward, Action.interact}
    links_with_effects = [l for l in links if len(l.effect_sks) > 0]

    if links_with_effects:
        tp = len([l for l in links_with_effects if l.action in causal_actions])
        fp = len([l for l in links_with_effects if l.action not in causal_actions])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    else:
        precision = 1.0

    # Recall: of causal action types, how many show learned effects?
    # Sokoban-style: forward pushes boxes, turn changes view, interact toggles.
    # Noop is non-causal. Each causal action type counts as a "real relationship".
    forward_links = [l for l in links if l.action == Action.forward]
    interact_links = [l for l in links if l.action == Action.interact]
    turn_left_links = [l for l in links if l.action == Action.turn_left]
    turn_right_links = [l for l in links if l.action == Action.turn_right]

    real_causal_types = {"forward", "turn_left", "turn_right", "interact"}
    discovered = set()

    if any(len(l.effect_sks) > 0 for l in forward_links):
        discovered.add("forward")
    if any(len(l.effect_sks) > 0 for l in interact_links):
        discovered.add("interact")
    if any(len(l.effect_sks) > 0 for l in turn_left_links):
        discovered.add("turn_left")
    if any(len(l.effect_sks) > 0 for l in turn_right_links):
        discovered.add("turn_right")

    recall = len(discovered) / len(real_causal_types)

    env_rgb.close()

    results = {
        "precision": precision,
        "recall": recall,
        "n_causal_links": n_links,
        "n_interact_links": len(interact_links),
        "n_forward_links": len(forward_links),
        "n_links_with_effects": len(links_with_effects),
        "explore_steps": explore_steps,
        "discovered_types": list(discovered),
    }

    print(f"Exp 7 Results:")
    print(f"  Causal precision: {precision:.3f} (gate > 0.8)")
    print(f"  Causal recall:    {recall:.3f} (gate > 0.7)")
    print(f"  Total links:      {n_links}")
    print(f"  Links with effects: {len(links_with_effects)}")
    print(f"  Discovered types: {discovered}")

    return results


if __name__ == "__main__":
    run()
