"""Stage 66: Pixel perception — full training pipeline.

Phases:
1. Collect pixel transitions from real Crafter env
2. Train VQ encoder + JEPA predictor (self-supervised)
3. Train decode head (supervised on symbolic ground truth)
4. Train CLS world model from pixel observations
5. Run gate test: ≥50% Crafter QA from pixels

Run on minipc only (GPU required for training).
"""

from __future__ import annotations

import time

import torch
import numpy as np

from snks.device import get_device
from snks.encoder.vq_patch_encoder import VQPatchEncoder
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.agent.decode_head import (
    DecodeHead, symbolic_to_gt_tensors, NEAR_CLASSES, INVENTORY_ITEMS,
)
from snks.agent.crafter_pixel_env import CrafterPixelEnv, ACTION_TO_IDX
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_TAUGHT, CRAFTER_RULES


def phase1_collect(
    n_trajectories: int = 50,
    steps_per_traj: int = 200,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Phase 1: Collect pixel transitions from real Crafter.

    Returns dict with tensors: pixels_t, pixels_t1, actions, gt_near, gt_inv.
    """
    print(f"Phase 1: Collecting {n_trajectories} × {steps_per_traj} transitions...")
    t0 = time.time()

    all_pt, all_pt1, all_actions = [], [], []
    all_near, all_inv = [], []

    for traj in range(n_trajectories):
        # Different seed per trajectory — encoder must generalize across maps
        env = CrafterPixelEnv(seed=seed + traj * 7)
        pixels, sym = env.reset()

        for step in range(steps_per_traj):
            # Random action for exploration
            action_idx = np.random.RandomState(seed + traj * 1000 + step).randint(0, 17)
            next_pixels, next_sym, reward, done = env.step(action_idx)

            # Store transition
            all_pt.append(torch.from_numpy(pixels))
            all_pt1.append(torch.from_numpy(next_pixels))
            all_actions.append(action_idx)

            # Ground truth for decode head supervision
            near_idx, inv_vec = symbolic_to_gt_tensors(sym)
            all_near.append(near_idx)
            all_inv.append(inv_vec)

            pixels = next_pixels
            sym = next_sym

            if done:
                pixels, sym = env.reset()

        elapsed = time.time() - t0
        eta = elapsed / (traj + 1) * (n_trajectories - traj - 1)
        if (traj + 1) % 10 == 0:
            print(f"  traj {traj + 1}/{n_trajectories} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    dataset = {
        "pixels_t": torch.stack(all_pt),          # (N, 3, 64, 64)
        "pixels_t1": torch.stack(all_pt1),         # (N, 3, 64, 64)
        "actions": torch.tensor(all_actions),       # (N,)
        "gt_near": torch.tensor(all_near),          # (N,)
        "gt_inv": torch.tensor(all_inv),            # (N, n_items)
    }

    print(f"Phase 1 done: {len(all_pt)} transitions in {time.time() - t0:.0f}s")
    return dataset


def phase2_train_encoder(
    dataset: dict[str, torch.Tensor],
    epochs: int = 100,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> tuple[VQPatchEncoder, JEPAPredictor, list]:
    """Phase 2: Self-supervised encoder + predictor training."""
    print(f"\nPhase 2: Training encoder + predictor ({epochs} epochs)...")

    encoder = VQPatchEncoder(device=device).to(device)
    predictor = JEPAPredictor().to(device)
    trainer = PredictiveTrainer(encoder, predictor, device=device)

    history = trainer.train_full(
        dataset["pixels_t"],
        dataset["pixels_t1"],
        dataset["actions"],
        epochs=epochs,
        batch_size=batch_size,
        log_every=10,
    )

    final = history[-1]
    print(f"Phase 2 done: pred_loss={final['pred_loss']:.4f} "
          f"codebook_util={final['codebook_util']:.2f}")
    return encoder, predictor, history


def phase3_train_decode_head(
    encoder: VQPatchEncoder,
    dataset: dict[str, torch.Tensor],
    epochs: int = 20,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> tuple[DecodeHead, list]:
    """Phase 3: Supervised decode head training."""
    print(f"\nPhase 3: Training decode head ({epochs} epochs)...")

    head = DecodeHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    # Pre-encode all frames — use z_local (scene-invariant) for decode head
    encoder.eval()
    all_z = []
    with torch.no_grad():
        for i in range(0, len(dataset["pixels_t"]), batch_size):
            batch = dataset["pixels_t"][i:i + batch_size].to(device)
            out = encoder(batch)
            all_z.append(out.z_local)
    all_z = torch.cat(all_z)

    gt_near = dataset["gt_near"].to(device)
    gt_inv = dataset["gt_inv"].to(device)

    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(all_z), device=device)
        epoch_metrics: dict[str, float] = {}
        n_batches = 0

        for i in range(0, len(all_z), batch_size):
            idx = perm[i:i + batch_size]
            metrics = head.train_step(
                all_z[idx], gt_near[idx], gt_inv[idx], optimizer,
            )
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

        epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        history.append(epoch_metrics)

        if epoch % 5 == 0:
            print(f"  epoch {epoch}: near_acc={epoch_metrics['near_acc']:.3f} "
                  f"loss={epoch_metrics['total_loss']:.4f}")

    print(f"Phase 3 done: near_acc={history[-1]['near_acc']:.3f}")
    return head, history


def phase4_train_cls(
    encoder: VQPatchEncoder,
    decode_head: DecodeHead,
    device: torch.device = torch.device("cpu"),
) -> CLSWorldModel:
    """Phase 4: Train CLS world model from pixel demonstrations."""
    print("\nPhase 4: Training CLS from pixel demonstrations...")

    cls = CLSWorldModel(dim=2048, device=device)

    # Teach rules via pixel env
    for rule in CRAFTER_TAUGHT:
        env = CrafterPixelEnv(seed=42)
        pixels, sym = env.reset()

        # Navigate to find the target object
        found = False
        for _ in range(100):
            pixels, sym, _, done = env.step("move_right")
            if sym.get("near") == rule["near"]:
                found = True
                break
            if done:
                pixels, sym = env.reset()

        if not found:
            # Use another seed
            for seed_offset in range(10):
                env = CrafterPixelEnv(seed=42 + seed_offset + 100)
                pixels, sym = env.reset()
                for _ in range(200):
                    pixels, sym, _, done = env.step(
                        np.random.choice(["move_left", "move_right", "move_up", "move_down"])
                    )
                    if sym.get("near") == rule["near"]:
                        found = True
                        break
                    if done:
                        pixels, sym = env.reset()
                if found:
                    break

        if not found:
            print(f"  Warning: could not find '{rule['near']}' for teaching")
            continue

        # Execute the taught action
        pix_tensor = torch.from_numpy(pixels).to(device)
        outcome = {"result": rule["result"], "gives": rule["gives"]}
        transitions = [(pix_tensor, rule["action"], outcome, 1.0)]
        cls.train_from_pixels(transitions, encoder, decode_head)
        print(f"  Taught: {rule['action']} near {rule['near']} → {rule['result']}")

    # Also train from symbolic transitions for neocortex baseline
    from snks.agent.crafter_trainer import generate_taught_transitions
    symbolic_transitions = generate_taught_transitions()
    cls.train(symbolic_transitions)

    print(f"Phase 4 done: neocortex={len(cls.neocortex)} rules")
    print(f"  Keys: {sorted(cls.neocortex.keys())}")
    return cls


def phase5_gate_test(
    cls: CLSWorldModel,
    encoder: VQPatchEncoder,
    decode_head: DecodeHead,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Phase 5: Gate test — ≥50% Crafter QA from pixels."""
    print("\nPhase 5: Gate test...")

    # Build QA scenarios from CRAFTER_RULES
    correct = 0
    total = 0
    results = []

    for rule in CRAFTER_RULES:
        # Try to set up scenario in real env
        env = CrafterPixelEnv(seed=42)
        pixels, sym = env.reset()

        # Search for target object
        found = False
        for _ in range(300):
            pixels, sym, _, done = env.step(
                np.random.choice(["move_left", "move_right", "move_up", "move_down"])
            )
            if sym.get("near") == rule["near"]:
                found = True
                break
            if done:
                pixels, sym = env.reset()

        if not found:
            continue

        pix_tensor = torch.from_numpy(pixels).to(device)
        outcome, conf, source = cls.query_from_pixels(
            pix_tensor, rule["action"], encoder, decode_head,
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

    # Per-source breakdown
    by_source: dict[str, list] = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r["correct"])
    for src, vals in by_source.items():
        print(f"  {src}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.0%}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
        "by_source": {k: sum(v)/len(v) for k, v in by_source.items()},
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    # Phase 1: Collect data
    dataset = phase1_collect(n_trajectories=50, steps_per_traj=200)

    # Phase 2: Train encoder
    encoder, predictor, enc_hist = phase2_train_encoder(
        dataset, epochs=100, device=device,
    )

    # Phase 3: Train decode head
    decode_head, head_hist = phase3_train_decode_head(
        encoder, dataset, epochs=20, device=device,
    )

    # Phase 4: Train CLS
    cls = phase4_train_cls(encoder, decode_head, device=device)

    # Phase 5: Gate test
    gate = phase5_gate_test(cls, encoder, decode_head, device=device)

    return gate


if __name__ == "__main__":
    main()
