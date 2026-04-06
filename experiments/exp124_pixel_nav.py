"""Stage 68: Pixel Navigation — cognitive map replaces info["semantic"].

Phases:
0. Load encoder (Stage 67 checkpoint) + create NearDetector.
1. Navigation smoke: find_target_with_map vs random walk — success rate + avg steps.
2. QA gate: Crafter QA L1-L4 using spatial map navigation (no info["semantic"]).
3. Regression: exp123 pipeline (Stage 67), gate ≥90%.

Run on minipc. CNN trains on GPU via disable_rocm_conv() (AMD ROCm fallback kernel).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.encoder.near_detector import NearDetector
from snks.agent.decode_head import NEAR_CLASSES
from snks.agent.crafter_pixel_env import CrafterPixelEnv, NEAR_OBJECTS
from snks.agent.crafter_spatial_map import CrafterSpatialMap, find_target_with_map
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_RULES

# Import training pipeline from exp122 (unchanged)
from exp122_pixels import (
    phase1_collect,
    phase2_train_encoder,
    make_situation_label,
    NEAR_TO_IDX,
    _situation_from_info,
)

# Import regression pipeline from exp123
from exp123_pixel_agent import (
    phase1_smoke_test,
    phase2_qa_gate,
    phase3_regression,
)


# ---------------------------------------------------------------------------
# Phase 0: Load encoder
# ---------------------------------------------------------------------------

def phase0_load_encoder(
    n_trajectories: int = 50,
    steps_per_traj: int = 200,
    epochs: int = 100,
) -> tuple[CNNEncoder, NearDetector]:
    """Train encoder (same as exp123) and wrap in NearDetector."""
    print("Phase 0: Training encoder (JEPA + SupCon)...")
    dataset = phase1_collect(n_trajectories=n_trajectories, steps_per_traj=steps_per_traj)
    encoder, _, _ = phase2_train_encoder(dataset, epochs=epochs)
    detector = NearDetector(encoder)
    print(f"Phase 0 done: NearDetector ready, {len(NEAR_CLASSES)} classes")
    return encoder, detector


# ---------------------------------------------------------------------------
# Phase 1: Navigation smoke — spatial map vs random walk
# ---------------------------------------------------------------------------

# Targets that appear regularly in Crafter worlds
_NAV_TARGETS = ["tree", "stone", "water", "grass"]


def phase1_nav_smoke(
    detector: NearDetector,
    n_seeds: int = 20,
    max_steps: int = 300,
) -> dict:
    """Compare find_target_with_map (spatial map) vs random walk (no map).

    For each target × seed:
    - Spatial map: find_target_with_map (NearDetector + CrafterSpatialMap)
    - Random walk: random moves + NearDetector check (no map, same detector)

    Both use NearDetector — the variable is navigation strategy, not perception.
    Gate (smoke): spatial map success rate ≥60%.
    """
    print(f"\nPhase 1: Navigation smoke ({len(_NAV_TARGETS)} targets × {n_seeds} seeds)...")
    t0 = time.time()

    map_results: list[dict] = []
    rnd_results: list[dict] = []

    for target in _NAV_TARGETS:
        map_found = 0
        rnd_found = 0
        map_steps_list: list[int] = []
        rnd_steps_list: list[int] = []

        for seed_idx in range(n_seeds):
            seed = 3000 + seed_idx * 7

            # --- Spatial map navigation ---
            env_map = CrafterPixelEnv(seed=seed)
            env_map.reset()
            smap = CrafterSpatialMap()
            rng_map = np.random.RandomState(seed)
            steps_map = 0

            pixels_map, info_map = env_map.observe()
            for s in range(max_steps):
                pix_t = torch.from_numpy(pixels_map)
                near = detector.detect(pix_t)
                smap.update(info_map["player_pos"], near)
                if near == target:
                    map_found += 1
                    steps_map = s + 1
                    map_steps_list.append(steps_map)
                    break
                known_pos = smap.find_nearest(target, info_map["player_pos"])
                if known_pos is not None:
                    from snks.agent.crafter_spatial_map import _step_toward
                    action = _step_toward(info_map["player_pos"], known_pos, rng_map)
                else:
                    unvisited = smap.unvisited_neighbors(info_map["player_pos"], radius=5)
                    if unvisited:
                        from snks.agent.crafter_spatial_map import _step_toward
                        goal = unvisited[int(rng_map.randint(len(unvisited)))]
                        action = _step_toward(info_map["player_pos"], goal, rng_map)
                    else:
                        action = str(rng_map.choice(
                            ["move_left", "move_right", "move_up", "move_down"]
                        ))
                pixels_map, _, done, info_map = env_map.step(action)
                if done:
                    pixels_map, info_map = env_map.reset()
                    smap.reset()

            # --- Random walk (same NearDetector, no map) ---
            env_rnd = CrafterPixelEnv(seed=seed)
            env_rnd.reset()
            rng_rnd = np.random.RandomState(seed)
            pixels_rnd, info_rnd = env_rnd.observe()
            steps_rnd = 0

            for s in range(max_steps):
                near_rnd = detector.detect(torch.from_numpy(pixels_rnd))
                if near_rnd == target:
                    rnd_found += 1
                    steps_rnd = s + 1
                    rnd_steps_list.append(steps_rnd)
                    break
                action_rnd = str(rng_rnd.choice(
                    ["move_left", "move_right", "move_up", "move_down"]
                ))
                pixels_rnd, _, done_rnd, info_rnd = env_rnd.step(action_rnd)
                if done_rnd:
                    pixels_rnd, info_rnd = env_rnd.reset()

        map_rate = map_found / n_seeds
        rnd_rate = rnd_found / n_seeds
        map_avg = sum(map_steps_list) / max(len(map_steps_list), 1)
        rnd_avg = sum(rnd_steps_list) / max(len(rnd_steps_list), 1)

        map_results.append({"target": target, "rate": map_rate, "avg_steps": map_avg})
        rnd_results.append({"target": target, "rate": rnd_rate, "avg_steps": rnd_avg})

        print(
            f"  {target}: map={map_rate:.0%} ({map_avg:.0f} steps avg) | "
            f"random={rnd_rate:.0%} ({rnd_avg:.0f} steps avg)"
        )

    overall_map = sum(r["rate"] for r in map_results) / len(map_results)
    overall_rnd = sum(r["rate"] for r in rnd_results) / len(rnd_results)
    passed = overall_map >= 0.60

    print(f"\n{'='*50}")
    print(f"Nav smoke: map={overall_map:.0%} | random={overall_rnd:.0%}")
    print(f"{'PASS' if passed else 'FAIL'} (threshold: 60%, {time.time()-t0:.0f}s)")
    print(f"{'='*50}")

    return {
        "overall_map": overall_map,
        "overall_random": overall_rnd,
        "map_results": map_results,
        "rnd_results": rnd_results,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Phase 2: QA gate — prototype collection + QA via spatial map
# ---------------------------------------------------------------------------

def _collect_prototypes_with_map(
    encoder: CNNEncoder,
    detector: NearDetector,
    cls: CLSWorldModel,
    n_seeds: int = 50,
) -> dict:
    """Collect prototypes using spatial map navigation. No info["semantic"]."""
    print(f"  Collecting prototypes ({n_seeds} seeds × {len(CRAFTER_RULES)} rules)...")
    t0 = time.time()
    n_added = 0
    n_skipped = 0

    for rule in CRAFTER_RULES:
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 4000 + seed_idx * 11

            env = CrafterPixelEnv(seed=seed)
            env.reset()
            smap = CrafterSpatialMap()
            rng = np.random.RandomState(seed)

            pixels_t, info, found = find_target_with_map(
                env, detector, smap, rule["near"],
                max_steps=300, rng=rng,
            )

            if not found:
                n_skipped += 1
                continue

            with torch.no_grad():
                out = encoder(pixels_t)

            outcome = {"result": rule["result"], "gives": rule.get("gives", "")}
            cls.prototype_memory.add(out.z_real, rule["action"], outcome)
            rule_added += 1
            n_added += 1

        print(f"    {rule['action']} near {rule['near']}: {rule_added}/{n_seeds}")

    from snks.agent.crafter_trainer import generate_taught_transitions
    cls.train(generate_taught_transitions())

    print(f"  Prototypes: {n_added} added, {n_skipped} skipped ({time.time()-t0:.0f}s)")
    return {"n_added": n_added, "n_skipped": n_skipped}


def phase2_qa_gate(
    encoder: CNNEncoder,
    detector: NearDetector,
    n_seeds: int = 50,
) -> dict:
    """QA gate using spatial map navigation. Gate: ≥90% accuracy."""
    print("\nPhase 2: QA gate with spatial map navigation...")

    cls = CLSWorldModel(dim=2048, device="cpu")
    _collect_prototypes_with_map(encoder, detector, cls, n_seeds=n_seeds)

    print("  Running QA gate test...")
    correct = 0
    total = 0
    results = []

    for rule in CRAFTER_RULES:
        env = CrafterPixelEnv(seed=55555)
        env.reset()
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(55555)

        pixels_t, info, found = find_target_with_map(
            env, detector, smap, rule["near"],
            max_steps=300, rng=rng,
        )

        if not found:
            continue

        with torch.no_grad():
            outcome, conf, source = cls.query_from_pixels(pixels_t, rule["action"], encoder)

        expected = rule["result"]
        got = outcome.get("result", "unknown")
        is_correct = got == expected

        results.append({
            "rule": f"{rule['action']} near {rule['near']}",
            "expected": expected,
            "got": got,
            "conf": conf,
            "source": source,
            "correct": is_correct,
        })

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    passed = accuracy >= 0.90

    print(f"\n{'='*50}")
    print(f"GATE: {correct}/{total} = {accuracy:.0%}")
    print(f"{'PASS' if passed else 'FAIL'} (threshold: 90%)")
    print(f"{'='*50}")

    for r in results:
        mark = "+" if r["correct"] else "-"
        print(f"  [{mark}] {r['rule']}: expected={r['expected']} got={r['got']} "
              f"conf={r['conf']:.2f} src={r['source']}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed": passed,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Stage 68: Pixel Navigation — cognitive map replaces info[\"semantic\"]")
    print("=" * 60)

    # Phase 0: Train encoder
    encoder, detector = phase0_load_encoder(
        n_trajectories=50, steps_per_traj=200, epochs=100
    )

    # Phase 1: Navigation smoke
    nav = phase1_nav_smoke(detector, n_seeds=20, max_steps=300)

    # Phase 2: QA gate with spatial map
    qa = phase2_qa_gate(encoder, detector, n_seeds=50)

    # Phase 3: Regression (Stage 67 pipeline)
    reg = phase3_regression(encoder)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 68 SUMMARY")
    print(f"  Phase 1 nav smoke:  map={nav['overall_map']:.0%} "
          f"({'PASS' if nav['passed'] else 'FAIL'}, threshold 60%)")
    print(f"  Phase 2 QA gate:    {qa['accuracy']:.0%} "
          f"({'PASS' if qa['passed'] else 'FAIL'}, threshold 90%)")
    print(f"  Phase 3 regression: {reg['accuracy']:.0%} "
          f"({'PASS' if reg['passed'] else 'FAIL'}, threshold 90%)")

    all_passed = nav["passed"] and qa["passed"] and reg["passed"]
    print(f"\n{'STAGE 68 COMPLETE' if all_passed else 'STAGE 68 INCOMPLETE'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
