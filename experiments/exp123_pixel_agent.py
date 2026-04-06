"""Stage 67: Pixel Agent — NearDetector replaces symbolic near.

Phases:
0. Train encoder (same pipeline as exp122 Stage 66)
1. Smoke test: NearDetector vs ground truth near (≥70%)
2. QA gate: Crafter QA L1-L4 with CNN near (≥90%)
3. Regression: exp122 gate test (≥90%, Stage 66 achieved 100%)

Run on minipc. CNN training on GPU (torch.backends.cudnn.enabled=False for ROCm).
"""

from __future__ import annotations

import time

import torch
import numpy as np

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.encoder.near_detector import NearDetector
from snks.agent.decode_head import NEAR_CLASSES
from snks.agent.crafter_pixel_env import (
    CrafterPixelEnv, ACTION_NAMES, NEAR_OBJECTS, SEMANTIC_NAMES, INVENTORY_ITEMS,
)
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_TAUGHT, CRAFTER_RULES

# Import helpers from exp122 to avoid duplication
from exp122_pixels import (
    _detect_near_from_info,
    _situation_from_info,
    make_situation_label,
    phase1_collect,
    phase2_train_encoder,
    phase3_collect_prototypes,
    phase4_gate_test,
)


def _build_situation(pixels: torch.Tensor, info: dict, detector: NearDetector) -> dict[str, str]:
    """Build situation dict from pixels + info using NearDetector.

    Args:
        pixels: (3, 64, 64) float32 [0, 1].
        info: native Crafter info dict from CrafterPixelEnv.step().
        detector: NearDetector instance.

    Returns:
        situation dict compatible with CLSWorldModel.query().
    """
    near_str = detector.detect(pixels)
    situation: dict[str, str] = {"domain": "crafter", "near": near_str}
    for item, count in info.get("inventory", {}).items():
        if count > 0:
            situation[f"has_{item}"] = str(count)
    return situation


def phase0_train_encoder(
    n_trajectories: int = 50,
    steps_per_traj: int = 200,
    epochs: int = 100,
) -> tuple[CNNEncoder, NearDetector]:
    """Phase 0: Train encoder and create NearDetector.

    Uses same pipeline as exp122 (JEPA + SupCon + VICReg).
    Runs on CPU (Conv2d incompatible with ROCm).
    """
    print("Phase 0: Training encoder (JEPA + SupCon)...")
    dataset = phase1_collect(n_trajectories=n_trajectories, steps_per_traj=steps_per_traj)
    encoder, _, _ = phase2_train_encoder(dataset, epochs=epochs)
    detector = NearDetector(encoder)
    print(f"Phase 0 done: NearDetector ready, {len(NEAR_CLASSES)} classes")
    return encoder, detector


def phase1_smoke_test(
    detector: NearDetector,
    n_frames: int = 500,
    seed: int = 77777,
) -> dict:
    """Phase 1: Smoke test — NearDetector vs ground truth near.

    Collects n_frames from random trajectories and compares CNN near
    with ground truth near from semantic map.

    Gate (smoke, not hard): ≥70% agreement.
    """
    print(f"\nPhase 1: Smoke test — NearDetector vs ground truth ({n_frames} frames)...")

    correct = 0
    total = 0
    confusion: dict[str, dict[str, int]] = {}

    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)

    for i in range(n_frames):
        gt_near = _detect_near_from_info(info)
        pix_tensor = torch.from_numpy(pixels)
        cnn_near = detector.detect(pix_tensor)

        if gt_near not in confusion:
            confusion[gt_near] = {}
        confusion[gt_near][cnn_near] = confusion[gt_near].get(cnn_near, 0) + 1

        if cnn_near == gt_near:
            correct += 1
        total += 1

        action_idx = rng.randint(0, 17)
        pixels, _, done, info = env.step(action_idx)
        if done:
            pixels, info = env.reset()

    accuracy = correct / max(total, 1)
    passed = accuracy >= 0.70

    print(f"\nSmoke test: {correct}/{total} = {accuracy:.1%} "
          f"({'PASS' if passed else 'FAIL'}, threshold 70%)")

    # Print top confusions
    print("\nTop confusions (gt → cnn: count):")
    for gt, preds in sorted(confusion.items()):
        for cnn_pred, count in sorted(preds.items(), key=lambda x: -x[1]):
            if gt != cnn_pred and count >= 5:
                print(f"  {gt} → {cnn_pred}: {count}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "passed": passed}


def phase2_qa_gate(
    encoder: CNNEncoder,
    detector: NearDetector,
    n_seeds: int = 50,
) -> dict:
    """Phase 2: Crafter QA gate using NearDetector for near detection.

    Builds CLS world model, collects prototypes, runs QA L1-L4.
    Near comes from CNN (NearDetector), inventory from info["inventory"].

    Gate: ≥90% QA accuracy.
    """
    print("\nPhase 2: QA gate with NearDetector...")

    cls = CLSWorldModel(dim=2048, device="cpu")

    # Collect prototypes using NearDetector to find target situations
    print(f"  Collecting prototypes ({n_seeds} seeds × {len(CRAFTER_RULES)} rules)...")
    t0 = time.time()
    n_added = 0
    n_skipped = 0

    for rule in CRAFTER_RULES:
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 2000 + seed_idx * 17  # different from exp122 seeds

            env = CrafterPixelEnv(seed=seed)
            pixels, info = env.reset()

            found = False
            for _ in range(300):
                pixels, _, done, info = env.step(
                    np.random.choice(["move_left", "move_right", "move_up", "move_down"])
                )
                # Use NearDetector (CNN) to find target — this is the key change
                pix_tensor = torch.from_numpy(pixels)
                cnn_near = detector.detect(pix_tensor)
                if cnn_near == rule["near"]:
                    found = True
                    break
                if done:
                    pixels, info = env.reset()

            if not found:
                n_skipped += 1
                continue

            with torch.no_grad():
                out = encoder(torch.from_numpy(pixels))

            outcome = {"result": rule["result"], "gives": rule.get("gives", "")}
            cls.prototype_memory.add(out.z_real, rule["action"], outcome)
            rule_added += 1
            n_added += 1

        print(f"    {rule['action']} near {rule['near']}: {rule_added}/{n_seeds}")

    # Load symbolic rules into neocortex
    from snks.agent.crafter_trainer import generate_taught_transitions
    cls.train(generate_taught_transitions())

    print(f"  Prototypes: {n_added} added, {n_skipped} skipped ({time.time() - t0:.0f}s)")

    # QA gate test — using NearDetector to find situations
    print("  Running QA gate test...")
    correct = 0
    total = 0
    results = []

    for rule in CRAFTER_RULES:
        env = CrafterPixelEnv(seed=88888)
        pixels, info = env.reset()

        found = False
        for _ in range(300):
            pixels, _, done, info = env.step(
                np.random.choice(["move_left", "move_right", "move_up", "move_down"])
            )
            pix_tensor = torch.from_numpy(pixels)
            if detector.detect(pix_tensor) == rule["near"]:
                found = True
                break
            if done:
                pixels, info = env.reset()

        if not found:
            continue

        pix_tensor = torch.from_numpy(pixels)
        with torch.no_grad():
            outcome, conf, source = cls.query_from_pixels(pix_tensor, rule["action"], encoder)

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
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {r['rule']}: expected={r['expected']} got={r['got']} "
              f"conf={r['conf']:.2f} src={r['source']}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "passed": passed,
        "results": results,
        "cls": cls,
    }


def phase3_regression(encoder: CNNEncoder) -> dict:
    """Phase 3: Regression — run exp122 gate test, verify ≥90%.

    Stage 66 achieved 100%. Regression floor = 90%.
    Uses symbolic near (via _detect_near_from_info) to find prototypes,
    then prototype_memory.query() from pixels — same as exp122 Phase 4.
    """
    print("\nPhase 3: Regression (exp122 pipeline)...")

    cls = CLSWorldModel(dim=2048, device="cpu")
    phase3_collect_prototypes(encoder, cls, n_seeds=50)
    gate = phase4_gate_test(cls, encoder)

    passed = gate["accuracy"] >= 0.90
    print(f"\nRegression: {gate['accuracy']:.0%} "
          f"({'PASS' if passed else 'FAIL'}, threshold: 90%)")
    return {**gate, "passed": passed}


def main() -> None:
    print("Stage 67: Pixel Agent — NearDetector replaces symbolic near")
    print("=" * 60)

    # Phase 0: Train encoder + create NearDetector
    encoder, detector = phase0_train_encoder(
        n_trajectories=50, steps_per_traj=200, epochs=100
    )

    # Phase 1: Smoke test
    smoke = phase1_smoke_test(detector, n_frames=500)

    # Phase 2: QA gate with NearDetector
    qa = phase2_qa_gate(encoder, detector, n_seeds=50)

    # Phase 3: Regression (exp122)
    reg = phase3_regression(encoder)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 67 SUMMARY")
    print(f"  Phase 1 smoke:  {smoke['accuracy']:.1%} "
          f"({'PASS' if smoke['passed'] else 'FAIL'}, threshold 70%)")
    print(f"  Phase 2 gate:   {qa['accuracy']:.0%} "
          f"({'PASS' if qa['passed'] else 'FAIL'}, threshold 90%)")
    print(f"  Phase 3 reg:    {reg['accuracy']:.0%} "
          f"({'PASS' if reg['passed'] else 'FAIL'}, threshold 90%)")

    all_passed = smoke["passed"] and qa["passed"] and reg["passed"]
    print(f"\n{'STAGE 67 COMPLETE' if all_passed else 'STAGE 67 INCOMPLETE'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
