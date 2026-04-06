"""Stage 66 v2: Pixel perception — prototype memory pipeline.

Phases:
1. Collect pixel transitions from real Crafter env (with situation labels)
2. Train CNN encoder: JEPA + supervised contrastive + VICReg
3. Collect prototypes: encode pixels → store (z, action, outcome) in memory
4. Gate test: ≥50% Crafter QA from pixels via k-NN

All on CPU (Conv2d incompatible with ROCm on AMD GPU).
"""

from __future__ import annotations

import time

import torch
import numpy as np

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.crafter_pixel_env import (
    CrafterPixelEnv, ACTION_NAMES, NEAR_OBJECTS, SEMANTIC_NAMES, INVENTORY_ITEMS,
)
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_TAUGHT, CRAFTER_RULES


def _detect_near_from_info(info: dict) -> str:
    """Detect nearest relevant object from native Crafter info dict.

    Replicates the logic previously in CrafterPixelEnv._detect_nearby().
    Used only in exp122/exp123 for ground truth labelling and prototype search.
    """
    semantic = info.get("semantic")
    player_pos = info.get("player_pos")
    if semantic is None or player_pos is None:
        return "empty"
    py, px = int(player_pos[0]), int(player_pos[1])
    h, w = semantic.shape
    best_obj = "empty"
    best_dist = float("inf")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w:
                sid = int(semantic[ny, nx])
                name = SEMANTIC_NAMES.get(sid, "unknown")
                if name in NEAR_OBJECTS:
                    dist = abs(dy) + abs(dx)
                    if dist < best_dist:
                        best_dist = dist
                        best_obj = name
    return best_obj


def _situation_from_info(info: dict) -> dict[str, str]:
    """Build old-style situation dict from native Crafter info dict."""
    situation: dict[str, str] = {
        "domain": "crafter",
        "near": _detect_near_from_info(info),
    }
    for item in INVENTORY_ITEMS:
        count = info.get("inventory", {}).get(item, 0)
        if count > 0:
            situation[f"has_{item}"] = str(count)
    return situation


def make_situation_label(sym_obs: dict) -> str:
    """Compound situation label: near + inventory keys."""
    near = sym_obs.get("near", "empty")
    inv_parts = sorted(k for k in sym_obs if k.startswith("has_"))
    inv_key = "_".join(inv_parts) if inv_parts else "noinv"
    return f"{near}_{inv_key}"


def phase1_collect(
    n_trajectories: int = 50,
    steps_per_traj: int = 200,
    seed: int = 42,
) -> dict[str, torch.Tensor | list]:
    """Phase 1: Collect pixel transitions with situation labels.

    Returns dict with tensors + label mapping.
    """
    print(f"Phase 1: Collecting {n_trajectories} × {steps_per_traj} transitions...")
    t0 = time.time()

    all_pt, all_pt1, all_actions = [], [], []
    all_situation_labels = []
    all_near_labels = []
    label_to_idx: dict[str, int] = {}

    for traj in range(n_trajectories):
        env = CrafterPixelEnv(seed=seed + traj * 7)
        pixels, info = env.reset()
        sym = _situation_from_info(info)

        for step in range(steps_per_traj):
            action_idx = np.random.RandomState(seed + traj * 1000 + step).randint(0, 17)
            next_pixels, reward, done, next_info = env.step(action_idx)
            next_sym = _situation_from_info(next_info)

            all_pt.append(torch.from_numpy(pixels))
            all_pt1.append(torch.from_numpy(next_pixels))
            all_actions.append(action_idx)

            # Situation label for contrastive loss
            sit_label = make_situation_label(sym)
            if sit_label not in label_to_idx:
                label_to_idx[sit_label] = len(label_to_idx)
            all_situation_labels.append(label_to_idx[sit_label])

            # Near label for supervised near_head training
            all_near_labels.append(NEAR_TO_IDX.get(sym.get("near", "empty"), 0))

            pixels = next_pixels
            sym = next_sym

            if done:
                pixels, info = env.reset()
                sym = _situation_from_info(info)

        elapsed = time.time() - t0
        eta = elapsed / (traj + 1) * (n_trajectories - traj - 1)
        if (traj + 1) % 10 == 0:
            print(f"  traj {traj + 1}/{n_trajectories} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    dataset = {
        "pixels_t": torch.stack(all_pt),
        "pixels_t1": torch.stack(all_pt1),
        "actions": torch.tensor(all_actions),
        "situation_labels": torch.tensor(all_situation_labels),
        "near_labels": torch.tensor(all_near_labels, dtype=torch.long),
        "label_to_idx": label_to_idx,
    }

    n_labels = len(label_to_idx)
    print(f"Phase 1 done: {len(all_pt)} transitions, {n_labels} unique situations "
          f"in {time.time() - t0:.0f}s")
    return dataset


def phase2_train_encoder(
    dataset: dict,
    epochs: int = 100,
    batch_size: int = 256,
) -> tuple[CNNEncoder, JEPAPredictor, list]:
    """Phase 2: JEPA + supervised contrastive + VICReg training."""
    print(f"\nPhase 2: Training encoder (JEPA + SupCon, {epochs} epochs)...")

    encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    predictor = JEPAPredictor()
    trainer = PredictiveTrainer(
        encoder, predictor,
        contrastive_weight=0.5,
        near_weight=1.0,
        device="cpu",
    )

    history = trainer.train_full(
        dataset["pixels_t"],
        dataset["pixels_t1"],
        dataset["actions"],
        situation_labels=dataset["situation_labels"],
        near_labels=dataset.get("near_labels"),
        epochs=epochs,
        batch_size=batch_size,
        log_every=10,
    )

    final = history[-1]
    print(f"Phase 2 done: pred={final['pred_loss']:.4f} con={final['con_loss']:.4f}")
    return encoder, predictor, history


def phase3_collect_prototypes(
    encoder: CNNEncoder,
    cls: CLSWorldModel,
    n_seeds: int = 50,
) -> dict:
    """Phase 3: Collect prototypes — encode pixels, store in prototype memory.

    For each rule × seed: find the situation, encode, store (z, action, outcome).
    """
    print(f"\nPhase 3: Collecting prototypes ({n_seeds} seeds × {len(CRAFTER_RULES)} rules)...")

    encoder.eval()
    n_added = 0
    n_skipped = 0

    for rule in CRAFTER_RULES:
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 1000 + seed_idx * 13  # different from Phase 1 seeds

            env = CrafterPixelEnv(seed=seed)
            pixels, info = env.reset()

            # Random walk to find target near object
            found = False
            for _ in range(300):
                pixels, _, done, info = env.step(
                    np.random.choice(["move_left", "move_right", "move_up", "move_down"])
                )
                if _detect_near_from_info(info) == rule["near"]:
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

        print(f"  {rule['action']} near {rule['near']}: {rule_added}/{n_seeds} prototypes")

    # Also load symbolic rules into neocortex (backward compat)
    from snks.agent.crafter_trainer import generate_taught_transitions
    cls.train(generate_taught_transitions())

    print(f"Phase 3 done: {n_added} prototypes, {n_skipped} skipped")
    print(f"  Memory stats: {cls.prototype_memory.stats()}")
    return {"n_added": n_added, "n_skipped": n_skipped}


def phase4_gate_test(
    cls: CLSWorldModel,
    encoder: CNNEncoder,
) -> dict:
    """Phase 4: Gate test — ≥50% Crafter QA from pixels via prototype memory."""
    print("\nPhase 4: Gate test...")

    encoder.eval()
    correct = 0
    total = 0
    results = []

    for rule in CRAFTER_RULES:
        # Use a seed not used in Phase 3
        env = CrafterPixelEnv(seed=9999)
        pixels, info = env.reset()

        # Search for target object
        found = False
        for _ in range(300):
            pixels, _, done, info = env.step(
                np.random.choice(["move_left", "move_right", "move_up", "move_down"])
            )
            if _detect_near_from_info(info) == rule["near"]:
                found = True
                break
            if done:
                pixels, info = env.reset()

        if not found:
            continue

        with torch.no_grad():
            pix_tensor = torch.from_numpy(pixels)
            outcome, conf, source = cls.query_from_pixels(
                pix_tensor, rule["action"], encoder,
            )

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

    print(f"\n{'='*50}")
    print(f"GATE: {correct}/{total} = {accuracy:.0%}")
    print(f"{'PASS' if accuracy >= 0.50 else 'FAIL'} (threshold: 50%)")
    print(f"{'='*50}")

    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {r['rule']}: expected={r['expected']} got={r['got']} "
              f"conf={r['conf']:.2f} src={r['source']}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    print("Stage 66 v2: Prototype Memory Pipeline")
    print("All on CPU (Conv2d incompatible with ROCm)")

    # Phase 1: Collect data
    dataset = phase1_collect(n_trajectories=50, steps_per_traj=200)

    # Phase 2: Train encoder (JEPA + SupCon)
    encoder, predictor, enc_hist = phase2_train_encoder(dataset, epochs=100)

    # Phase 3: Collect prototypes
    cls = CLSWorldModel(dim=2048, device="cpu")
    phase3_collect_prototypes(encoder, cls, n_seeds=50)

    # Phase 4: Gate test
    gate = phase4_gate_test(cls, encoder)

    return gate


if __name__ == "__main__":
    main()
