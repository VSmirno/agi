#!/usr/bin/env python3
"""Exp121: Stage 65 gate — calibrated uncertainty.

Gate:
- Brier score < 0.15 on held-out transitions
- Confidence ~ accuracy correlation ρ > 0.7
- Crafter QA ≥80% (regression)
- MiniGrid QA ≥90% (regression)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import torch

from snks.agent.calibration import CalibrationTracker
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import generate_taught_transitions
from snks.agent.crafter_env_symbolic import CrafterSymbolicEnv
from snks.agent.minigrid_env_symbolic import MiniGridSymbolicEnv
from snks.agent.curiosity_explorer import DirectedCrafterExplorer, DirectedMiniGridExplorer
from snks.agent.world_model_trainer import (
    extract_demo_transitions, Transition, TRAIN_COLORS,
)


def main():
    print("=" * 60)
    print("EXP121: Stage 65 — Calibrated Uncertainty")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Phase 1: Train from demos (same as Stage 64)
    taught = generate_taught_transitions()
    with open("_docs/demo_episodes_bosslevel.json") as f:
        demos = json.load(f)
    mg_demo = extract_demo_transitions(demos)

    model = CLSWorldModel(dim=2048, n_locations=5000, device=device)
    model.train(taught + mg_demo)
    print(f"Initial neocortex: {len(model.neocortex)}")

    # Phase 2: Exploration (populates WM)
    print(f"\n{'='*60}")
    print("EXPLORATION")
    mg_env = MiniGridSymbolicEnv(colors=TRAIN_COLORS, seed=42)
    mg_explorer = DirectedMiniGridExplorer(model, explore_threshold=0.3, seed=42)
    mg_explorer.explore_directed(mg_env, n_episodes=30, steps_per_episode=30)

    cr_env = CrafterSymbolicEnv(seed=42)
    cr_explorer = DirectedCrafterExplorer(model, explore_threshold=0.3, seed=42)
    for batch in range(5):
        cr_explorer.explore(cr_env, n_episodes=20, steps_per_episode=80)
        print(f"  Crafter batch {batch+1}: neocortex={len(model.neocortex)}")

    model.abstraction.discover_categories(model.neocortex)
    model.abstraction.build_abstract_sdm()
    print(f"Final neocortex: {len(model.neocortex)}, categories: {len(model.abstraction.categories)}")

    # Phase 3: Calibration measurement
    print(f"\n{'='*60}")
    print("CALIBRATION MEASUREMENT")
    tracker = CalibrationTracker(n_buckets=5)

    # Test on MiniGrid scenarios
    mg_test_env = MiniGridSymbolicEnv(colors=TRAIN_COLORS, seed=999)
    for _ in range(200):
        mg_test_env.reset()
        situation = mg_test_env.observe()
        for action in mg_test_env.available_actions():
            predicted, conf, source = model.query(situation, action)
            actual_outcome, _ = mg_test_env.step(action)
            tracker.record(conf, predicted.get("result", "unknown"),
                          actual_outcome.get("result", "unknown"))
            # Reset env to same state for next action
            mg_test_env.set_scenario(**situation)

    # Test on Crafter scenarios
    cr_test_env = CrafterSymbolicEnv(seed=999)
    for ep in range(50):
        cr_test_env.reset()
        for step in range(20):
            situation = cr_test_env.observe()
            action = cr_test_env.available_actions()[step % len(cr_test_env.available_actions())]
            predicted, conf, source = model.query(situation, action)
            actual_outcome, _ = cr_test_env.step(action)
            tracker.record(conf, predicted.get("result", "unknown"),
                          actual_outcome.get("result", "unknown"))
            cr_test_env.next_target()

    summary = tracker.summary()
    print(f"Predictions: {summary['n_predictions']}")
    print(f"Brier score: {summary['brier_score']}  [gate < 0.15]")
    print(f"Correlation: {summary['correlation']}  [gate > 0.7]")
    print("Calibration curve:")
    for bucket in summary["calibration_curve"]:
        bar = "█" * int(bucket["acc"] * 20)
        print(f"  conf={bucket['conf']:.2f}  acc={bucket['acc']:.2f}  "
              f"n={bucket['n']:4d}  {bar}")

    # Phase 4: QA regression
    print(f"\n{'='*60}")
    print("CRAFTER QA (regression)")
    from exp119_abstraction import test_crafter_qa
    c_ok, c_total, c_fails = test_crafter_qa(model)
    for f in c_fails:
        print(f)
    crafter_pct = c_ok * 100 // c_total
    print(f"Crafter QA: {c_ok}/{c_total} = {crafter_pct}%  [gate ≥80%]")

    print(f"\n{'='*60}")
    print("MINIGRID QA (regression)")
    from exp118_world_model_qa import (
        level1_object_identity, level2_preconditions,
        level3_consequences, level4_planning,
    )
    c3, t3, f3 = level1_object_identity(model)
    c4, t4, f4 = level2_preconditions(model)
    c5, t5, f5 = level3_consequences(model)
    c6, t6, f6 = level4_planning(model)
    for f in f3 + f4 + f5 + f6:
        print(f)
    mg_total = c3 + c4 + c5 + c6
    mg_max = t3 + t4 + t5 + t6
    mg_pct = mg_total * 100 // mg_max
    print(f"MiniGrid: {mg_total}/{mg_max} = {mg_pct}%")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    brier = summary["brier_score"]
    rho = summary["correlation"]
    print(f"Brier score: {brier}  (gate < 0.15)")
    print(f"Correlation: {rho}  (gate > 0.7)")
    print(f"Crafter QA: {crafter_pct}%  (gate ≥80%)")
    print(f"MiniGrid QA: {mg_pct}%  (gate ≥90%)")

    brier_pass = brier < 0.15
    rho_pass = rho > 0.7
    crafter_pass = crafter_pct >= 80
    mg_pass = mg_pct >= 90

    print(f"\nBrier < 0.15: {'PASS' if brier_pass else 'FAIL'}")
    print(f"ρ > 0.7: {'PASS' if rho_pass else 'FAIL'}")
    print(f"Crafter ≥80%: {'PASS' if crafter_pass else 'FAIL'}")
    print(f"MiniGrid ≥90%: {'PASS' if mg_pass else 'FAIL'}")
    print(f"ALL GATES: {'PASS' if all([brier_pass, rho_pass, crafter_pass, mg_pass]) else 'FAIL'}")


if __name__ == "__main__":
    main()
