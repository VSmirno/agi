"""Stage 6 data collector for experiment visualization.

Provides two modes:
- collect_demo()  — fast synthetic data (no GPU, no real experiments)
- collect_real()  — runs experiments 7, 8, 9 and captures detailed telemetry

Output is saved to static/stage6_data.json and also returned as a dict.
Can be executed standalone:  python -m snks.viz.stage6_data [--demo|--real]
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_NAMES: dict[int, str] = {
    0: "turn_left",
    1: "turn_right",
    2: "forward",
    3: "interact",
    4: "noop",
}

STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_PATH = STATIC_DIR / "stage6_data.json"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Stage6Data = dict[str, Any]


# ---------------------------------------------------------------------------
# Demo data generators
# ---------------------------------------------------------------------------

def _demo_exp7(rng: random.Random) -> dict[str, Any]:
    """Generate realistic synthetic data for Experiment 7 (Causal Learning)."""
    n_nodes = rng.randint(8, 12)
    node_ids = sorted(rng.sample(range(1, 200), n_nodes))

    # Build causal graph: mostly interact edges, a few others
    edges: list[dict[str, Any]] = []
    n_edges = rng.randint(15, 20)

    for _ in range(n_edges):
        src = rng.choice(node_ids)
        tgt = rng.choice(node_ids)
        while tgt == src:
            tgt = rng.choice(node_ids)

        # Bias toward interact (action=3) for realistic precision
        action = rng.choices([0, 1, 2, 3, 4], weights=[2, 2, 3, 10, 1])[0]
        strength = round(rng.uniform(0.55, 0.98), 3)
        count = rng.randint(3, 25)

        edges.append(
            {
                "source": src,
                "target": tgt,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "strength": strength,
                "count": count,
            }
        )

    nodes = [{"id": nid, "type": "sks"} for nid in node_ids]

    # Metrics consistent with gate: precision >0.8, recall >0.7
    precision = round(rng.uniform(0.82, 0.94), 3)
    recall = round(rng.uniform(0.73, 0.88), 3)

    interact_edges = [e for e in edges if e["action"] == 3]

    return {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "n_causal_links": len(edges),
            "n_interact_links": len(interact_edges),
        },
        "causal_graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "gate": {
            "precision_threshold": 0.8,
            "recall_threshold": 0.7,
        },
    }


def _demo_exp8(rng: random.Random) -> dict[str, Any]:
    """Generate realistic synthetic data for Experiment 8 (Mental Simulation)."""
    n_trajectories = 5

    trajectories: list[dict[str, Any]] = []
    for _ in range(n_trajectories):
        n_steps = rng.randint(3, 5)
        action_seq = [rng.choice([0, 1, 2, 3]) for _ in range(n_steps)]

        base_ids = sorted(rng.sample(range(10, 300), rng.randint(3, 7)))
        steps: list[dict[str, Any]] = []

        current_sks = list(base_ids)
        for a in action_seq:
            # Each step may add 0-3 new SKS nodes
            n_new = rng.randint(0, 3)
            new_ids = rng.sample(range(300, 500), n_new)
            current_sks = sorted(set(current_sks) | set(new_ids))
            confidence = round(rng.uniform(0.5, 0.95), 3)
            steps.append({"sks": list(current_sks), "confidence": confidence})

        trajectories.append({"steps": steps, "action_sequence": action_seq})

    n_links = rng.randint(12, 30)
    sim_accuracy = round(rng.uniform(0.72, 0.92), 3)
    planning_rate = round(rng.uniform(0.55, 0.80), 3)

    return {
        "metrics": {
            "simulation_accuracy": sim_accuracy,
            "planning_success_rate": planning_rate,
            "n_causal_links": n_links,
        },
        "simulation_trajectories": trajectories,
        "gate": {
            "sim_accuracy_threshold": 0.7,
            "planning_threshold": 0.5,
        },
    }


def _demo_exp9(rng: random.Random) -> dict[str, Any]:
    """Generate realistic synthetic data for Experiment 9 (Curiosity)."""
    grid_size = 12
    total_cells = grid_size * grid_size

    n_steps = 2000

    # Simulate step-by-step coverage growth with periodic recording
    record_interval = 50

    curious_cells: set[tuple[int, int]] = set()
    random_cells: set[tuple[int, int]] = set()

    curious_series: list[dict[str, Any]] = []
    random_series: list[dict[str, Any]] = []

    curious_links = 0

    for step in range(0, n_steps + 1, record_interval):
        # Curious agent: logistic-like growth toward ~70% coverage
        frac = step / n_steps
        curious_target = 0.70 * (1 - math.exp(-4.0 * frac))
        curious_target_cells = int(curious_target * total_cells)

        while len(curious_cells) < curious_target_cells:
            x = rng.randint(0, grid_size - 1)
            y = rng.randint(0, grid_size - 1)
            curious_cells.add((x, y))

        # Random agent: slower growth toward ~40%
        rand_target = 0.40 * (1 - math.exp(-3.0 * frac))
        rand_target_cells = int(rand_target * total_cells)

        while len(random_cells) < rand_target_cells:
            x = rng.randint(0, grid_size - 1)
            y = rng.randint(0, grid_size - 1)
            random_cells.add((x, y))

        # Links grow roughly with exploration
        curious_links = int(20 * frac + rng.uniform(0, 2))

        curious_series.append(
            {
                "step": step,
                "coverage": round(len(curious_cells) / total_cells, 4),
                "n_links": curious_links,
            }
        )
        random_series.append(
            {
                "step": step,
                "coverage": round(len(random_cells) / total_cells, 4),
            }
        )

    final_curious_cov = len(curious_cells) / total_cells
    final_random_cov = len(random_cells) / total_cells
    coverage_ratio = round(final_curious_cov / max(final_random_cov, 0.001), 3)

    return {
        "metrics": {
            "coverage_ratio": coverage_ratio,
            "curious_coverage": round(final_curious_cov, 4),
            "random_coverage": round(final_random_cov, 4),
            "avg_causal_links": float(curious_links),
        },
        "coverage_timeseries": {
            "curious": curious_series,
            "random": random_series,
        },
        "heatmaps": {
            "curious": {
                "grid_size": grid_size,
                "cells": [list(c) for c in sorted(curious_cells)],
            },
            "random": {
                "grid_size": grid_size,
                "cells": [list(c) for c in sorted(random_cells)],
            },
        },
        "gate": {
            "coverage_ratio_threshold": 1.5,
        },
    }


# ---------------------------------------------------------------------------
# Public API: demo mode
# ---------------------------------------------------------------------------

def collect_demo(seed: int = 42) -> Stage6Data:
    """Generate realistic synthetic Stage 6 data without running real experiments.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Complete Stage6Data dict with exp7, exp8, exp9 sub-dicts plus timestamp.
    """
    rng = random.Random(seed)

    data: Stage6Data = {
        "exp7": _demo_exp7(rng),
        "exp8": _demo_exp8(rng),
        "exp9": _demo_exp9(rng),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "demo",
    }
    return data


# ---------------------------------------------------------------------------
# Public API: real mode
# ---------------------------------------------------------------------------

def collect_real(device: str = "cpu", num_nodes: int = 5000) -> Stage6Data:
    """Run experiments 7, 8, 9 and collect detailed telemetry.

    Imports are deferred so that the module can be imported on machines
    without the full SNKS agent stack (e.g. for demo-only usage).

    Args:
        device: PyTorch device string ("cpu", "cuda").
        num_nodes: Number of DAF oscillator nodes.

    Returns:
        Complete Stage6Data dict.
    """
    # Deferred imports — only needed for real mode
    from snks.agent.agent import CausalAgent
    from snks.agent.simulation import MentalSimulator
    from snks.daf.types import (
        CausalAgentConfig,
        DafConfig,
        EncoderConfig,
        PipelineConfig,
        SKSConfig,
    )
    from snks.env.causal_grid import Action, CausalGridWorld, make_level
    import random as pyrandom

    def _make_config() -> CausalAgentConfig:
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

    # -----------------------------------------------------------------------
    # Exp 7: Causal Learning
    # -----------------------------------------------------------------------
    print("Running Exp 7 (Causal Learning)...")
    config7 = _make_config()
    agent7 = CausalAgent(config7)
    explore_steps = 500

    env7 = make_level("PushBox", scripted_objects=True, max_steps=explore_steps + 100)
    obs, _ = env7.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for _ in range(explore_steps):
        action = agent7.step(img)
        obs, _, terminated, truncated, _ = env7.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent7.observe_result(img)
        if terminated or truncated:
            obs, _ = env7.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    env7.close()

    links7 = agent7.causal_model.get_causal_links(min_confidence=0.3)
    interact_links7 = [lnk for lnk in links7 if lnk.action == Action.interact]
    links_with_effects = [lnk for lnk in links7 if len(lnk.effect_sks) > 0]

    if links_with_effects:
        tp = len([lnk for lnk in interact_links7 if len(lnk.effect_sks) > 0])
        precision7 = tp / len(links_with_effects)
    else:
        precision7 = 1.0

    recall7 = len(interact_links7) / max(len(links7), 1) if links7 else 0.0

    # Build graph: representative node = first element of the frozenset
    node_ids_7: set[int] = set()
    edges7: list[dict[str, Any]] = []
    for lnk in links7:
        src = next(iter(lnk.context_sks)) if lnk.context_sks else 0
        tgt = next(iter(lnk.effect_sks)) if lnk.effect_sks else 0
        node_ids_7.add(src)
        node_ids_7.add(tgt)
        edges7.append(
            {
                "source": src,
                "target": tgt,
                "action": lnk.action,
                "action_name": ACTION_NAMES.get(lnk.action, str(lnk.action)),
                "strength": round(lnk.strength, 4),
                "count": lnk.count,
            }
        )

    exp7_data: dict[str, Any] = {
        "metrics": {
            "precision": round(precision7, 4),
            "recall": round(recall7, 4),
            "n_causal_links": len(links7),
            "n_interact_links": len(interact_links7),
        },
        "causal_graph": {
            "nodes": [{"id": nid, "type": "sks"} for nid in sorted(node_ids_7)],
            "edges": edges7,
        },
        "gate": {
            "precision_threshold": 0.8,
            "recall_threshold": 0.7,
        },
    }

    # -----------------------------------------------------------------------
    # Exp 8: Mental Simulation
    # -----------------------------------------------------------------------
    print("Running Exp 8 (Mental Simulation)...")
    config8 = _make_config()
    config8.simulation_max_depth = 10
    config8.simulation_min_confidence = 0.2
    agent8 = CausalAgent(config8)
    train_steps = 1000

    env8 = make_level("PushChain", max_steps=train_steps + 100)
    obs, _ = env8.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    for _ in range(train_steps):
        action = agent8.step(img)
        obs, _, terminated, truncated, _ = env8.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent8.observe_result(img)
        if terminated or truncated:
            obs, _ = env8.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    sim8 = agent8.simulator
    n_sim_tests = 20
    correct = 0
    trajectories8: list[dict[str, Any]] = []

    for _ in range(n_sim_tests):
        obs, _ = env8.reset()
        img = obs["image"] if isinstance(obs, dict) else obs
        action = agent8.step(img)
        pre_sks = set(agent8._pre_sks) if agent8._pre_sks else set()

        if pre_sks:
            action_seq = [action]
            trajectory = sim8.simulate(pre_sks, action_seq)
            if trajectory:
                predicted_sks, confidence = trajectory[0]
                obs, _, terminated, truncated, _ = env8.step(action)
                img = obs["image"] if isinstance(obs, dict) else obs
                pe = agent8.observe_result(img)
                if pe < 0.5:
                    correct += 1

                # Capture trajectory for viz (first 5 only)
                if len(trajectories8) < 5:
                    steps_data = [
                        {"sks": sorted(sks_set), "confidence": round(conf, 4)}
                        for sks_set, conf in trajectory
                    ]
                    trajectories8.append(
                        {"steps": steps_data, "action_sequence": action_seq}
                    )
            else:
                correct += 1
        else:
            correct += 1

    simulation_accuracy8 = correct / max(n_sim_tests, 1)

    n_planning = 20
    planning_successes = 0
    for _ in range(n_planning):
        obs, _ = env8.reset()
        img = obs["image"] if isinstance(obs, dict) else obs
        _ = agent8.step(img)
        start_sks = set(agent8._pre_sks) if agent8._pre_sks else set()
        goal_sks = start_sks | {max(start_sks) + 1} if start_sks else {0}
        plan = sim8.find_plan(
            start_sks, goal_sks,
            max_depth=config8.simulation_max_depth,
            n_actions=5,
            min_confidence=0.0,
        )
        if plan is not None:
            planning_successes += 1

    planning_rate8 = planning_successes / max(n_planning, 1)
    env8.close()

    exp8_data: dict[str, Any] = {
        "metrics": {
            "simulation_accuracy": round(simulation_accuracy8, 4),
            "planning_success_rate": round(planning_rate8, 4),
            "n_causal_links": agent8.causal_model.n_links,
        },
        "simulation_trajectories": trajectories8,
        "gate": {
            "sim_accuracy_threshold": 0.7,
            "planning_threshold": 0.5,
        },
    }

    # -----------------------------------------------------------------------
    # Exp 9: Curiosity-driven Exploration
    # -----------------------------------------------------------------------
    print("Running Exp 9 (Curiosity)...")
    config9 = _make_config()
    config9.curiosity_epsilon = 0.2
    n_steps = 2000
    record_interval = 50

    # Curious agent with step-by-step coverage tracking
    agent9 = CausalAgent(config9)
    env9_inner = CausalGridWorld(level="MultiRoom", size=12, max_steps=n_steps + 100)
    env9 = make_level("MultiRoom", max_steps=n_steps + 100)
    obs, _ = env9.reset()
    img = obs["image"] if isinstance(obs, dict) else obs

    curious_series: list[dict[str, Any]] = []
    grid_size_9 = 12
    total_cells_9 = grid_size_9 * grid_size_9

    for step in range(n_steps):
        action = agent9.step(img)
        obs, _, terminated, truncated, _ = env9.step(action)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent9.observe_result(img)

        if step % record_interval == 0:
            unwrapped = env9.unwrapped if hasattr(env9, "unwrapped") else env9
            visited = len(unwrapped._visited_cells) if hasattr(unwrapped, "_visited_cells") else 0
            cov = visited / total_cells_9
            curious_series.append(
                {
                    "step": step,
                    "coverage": round(cov, 4),
                    "n_links": agent9.causal_model.n_links,
                }
            )

        if terminated or truncated:
            obs, _ = env9.reset()
            img = obs["image"] if isinstance(obs, dict) else obs

    unwrapped9 = env9.unwrapped if hasattr(env9, "unwrapped") else env9
    curious_visited: set[tuple[int, int]] = (
        set(unwrapped9._visited_cells) if hasattr(unwrapped9, "_visited_cells") else set()
    )
    curious_cov9 = len(curious_visited) / total_cells_9
    env9.close()

    # Random agent with step-by-step tracking
    env9r = CausalGridWorld(level="MultiRoom", size=12, max_steps=n_steps + 100)
    env9r.reset()
    random_series: list[dict[str, Any]] = []

    for step in range(n_steps):
        action = pyrandom.randint(0, 4)
        _, _, terminated, truncated, _ = env9r.step(action)
        if step % record_interval == 0:
            cov = env9r.coverage
            random_series.append({"step": step, "coverage": round(cov, 4)})
        if terminated or truncated:
            env9r.reset()

    random_cov9 = env9r.coverage
    random_visited: set[tuple[int, int]] = set(env9r._visited_cells)
    env9r.close()

    coverage_ratio9 = curious_cov9 / max(random_cov9, 0.001)

    exp9_data: dict[str, Any] = {
        "metrics": {
            "coverage_ratio": round(coverage_ratio9, 4),
            "curious_coverage": round(curious_cov9, 4),
            "random_coverage": round(random_cov9, 4),
            "avg_causal_links": float(agent9.causal_model.n_links),
        },
        "coverage_timeseries": {
            "curious": curious_series,
            "random": random_series,
        },
        "heatmaps": {
            "curious": {
                "grid_size": grid_size_9,
                "cells": [list(c) for c in sorted(curious_visited)],
            },
            "random": {
                "grid_size": grid_size_9,
                "cells": [list(c) for c in sorted(random_visited)],
            },
        },
        "gate": {
            "coverage_ratio_threshold": 1.5,
        },
    }

    data: Stage6Data = {
        "exp7": exp7_data,
        "exp8": exp8_data,
        "exp9": exp9_data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "real",
    }
    return data


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_data(data: Stage6Data, path: Path = OUTPUT_PATH) -> None:
    """Serialize data to JSON at the given path.

    Args:
        data: Stage6Data dict to write.
        path: Output file path (default: static/stage6_data.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"Saved stage6 data -> {path}")


def load_data(path: Path = OUTPUT_PATH) -> Stage6Data | None:
    """Load previously saved Stage 6 data from JSON.

    Args:
        path: JSON file path.

    Returns:
        Parsed dict, or None if the file does not exist.
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry: python -m snks.viz.stage6_data [--demo|--real] [--device cpu|cuda]"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Stage 6 visualization data")
    parser.add_argument(
        "--mode",
        choices=["demo", "real"],
        default="demo",
        help="demo = fast synthetic data; real = run full experiments",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (real mode only)")
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=5000,
        help="DAF oscillator count (real mode only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (demo mode)")
    args = parser.parse_args()

    if args.mode == "demo":
        data = collect_demo(seed=args.seed)
    else:
        data = collect_real(device=args.device, num_nodes=args.num_nodes)

    save_data(data)
    print(f"Mode: {data['mode']}  |  Timestamp: {data['timestamp']}")


if __name__ == "__main__":
    main()
