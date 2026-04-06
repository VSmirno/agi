#!/usr/bin/env python3
"""Stage 66: headless training — generates demos/checkpoints/stage66.pt.

Run on minipc (no display needed):
    cd /opt/agi
    tmux new -s s66train
    HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src venv/bin/python demos/stage66_train.py

Parameters match exp122_pixels.py: 50 trajs × 200 steps, 100 epochs, 50 seeds.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.agent.prototype_memory import PrototypeMemory
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_trainer import CRAFTER_RULES, generate_taught_transitions
from snks.agent.decode_head import NEAR_CLASSES

CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "stage66.pt")


def make_situation_label(sym_obs: dict) -> str:
    near = sym_obs.get("near", "empty")
    inv_parts = sorted(k for k in sym_obs if k.startswith("has_"))
    inv_key = "_".join(inv_parts) if inv_parts else "noinv"
    return f"{near}_{inv_key}"


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def phase1_collect(n_trajectories: int = 50, steps_per_traj: int = 200,
                   seed: int = 42) -> dict:
    print(f"Phase 1: collecting {n_trajectories}×{steps_per_traj} transitions...")
    t0 = time.time()

    all_pt, all_pt1, all_actions, all_sit_labels = [], [], [], []
    label_to_idx: dict[str, int] = {}

    for traj in range(n_trajectories):
        env = CrafterPixelEnv(seed=seed + traj * 7)
        pixels, sym = env.reset()

        rng = np.random.RandomState(seed + traj * 1000)
        for step in range(steps_per_traj):
            action_idx = rng.randint(0, 17)
            next_pixels, next_sym, _r, done = env.step(action_idx)

            all_pt.append(torch.from_numpy(pixels))
            all_pt1.append(torch.from_numpy(next_pixels))
            all_actions.append(action_idx)

            sit_label = make_situation_label(sym)
            if sit_label not in label_to_idx:
                label_to_idx[sit_label] = len(label_to_idx)
            all_sit_labels.append(label_to_idx[sit_label])

            pixels = next_pixels
            sym = next_sym
            if done:
                pixels, sym = env.reset()

        elapsed = time.time() - t0
        eta = elapsed / (traj + 1) * (n_trajectories - traj - 1)
        if (traj + 1) % 10 == 0:
            print(f"  traj {traj+1}/{n_trajectories}  "
                  f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    dataset = {
        "pixels_t":         torch.stack(all_pt),
        "pixels_t1":        torch.stack(all_pt1),
        "actions":          torch.tensor(all_actions),
        "situation_labels": torch.tensor(all_sit_labels),
        "label_to_idx":     label_to_idx,
    }
    n_trans = len(all_pt)
    n_labels = len(label_to_idx)
    print(f"Phase 1 done: {n_trans} transitions, {n_labels} situations "
          f"in {time.time()-t0:.0f}s")
    return dataset


# ── Phase 2 ──────────────────────────────────────────────────────────────────

def phase2_train(dataset: dict, epochs: int = 100,
                 batch_size: int = 256) -> tuple[CNNEncoder, JEPAPredictor]:
    print(f"\nPhase 2: training encoder ({epochs} epochs, batch={batch_size})...")
    t0 = time.time()

    from snks.encoder.predictive_trainer import supcon_loss

    encoder  = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    predictor = JEPAPredictor()

    ds = TensorDataset(
        dataset["pixels_t"],
        dataset["pixels_t1"],
        dataset["actions"],
        dataset["situation_labels"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    encoder.train()
    predictor.train()

    for epoch in range(epochs):
        total_pred = 0.0
        total_con  = 0.0
        n_batches  = 0

        for pt, pt1, acts, labels in loader:
            out_t  = encoder(pt)
            out_t1 = encoder(pt1)
            z_pred = predictor(out_t.z_real, acts)
            pred_loss = F.mse_loss(z_pred, out_t1.z_real.detach())
            con_loss  = supcon_loss(out_t.z_real, labels)
            loss = pred_loss + 0.5 * con_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_pred += pred_loss.item()
            total_con  += con_loss.item()
            n_batches  += 1

        elapsed = time.time() - t0
        eta     = elapsed / (epoch + 1) * (epochs - epoch - 1)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  "
                  f"pred={total_pred/n_batches:.4f}  "
                  f"con={total_con/n_batches:.4f}  "
                  f"{elapsed:.0f}s  ETA {eta:.0f}s")

    encoder.eval()
    print(f"Phase 2 done in {time.time()-t0:.0f}s")
    return encoder, predictor


# ── Phase 3 ──────────────────────────────────────────────────────────────────

def phase3_prototypes(encoder: CNNEncoder, cls: CLSWorldModel,
                      n_seeds: int = 50) -> None:
    print(f"\nPhase 3: collecting prototypes ({n_seeds} seeds × {len(CRAFTER_RULES)} rules)...")
    t0 = time.time()
    encoder.eval()

    n_added = 0
    n_skipped = 0

    for ri, rule in enumerate(CRAFTER_RULES):
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 1000 + seed_idx * 13 + ri * 100
            env = CrafterPixelEnv(seed=seed)
            pixels, sym = env.reset()

            rng = np.random.RandomState(seed)
            found = False
            for _ in range(300):
                act_name = rng.choice(
                    ["move_left", "move_right", "move_up", "move_down"]
                )
                pixels, sym, _, done = env.step(act_name)
                if sym.get("near") == rule["near"]:
                    found = True
                    break
                if done:
                    pixels, sym = env.reset()

            if not found:
                n_skipped += 1
                continue

            with torch.no_grad():
                out = encoder(torch.from_numpy(pixels))

            outcome = {"result": rule["result"], "gives": rule.get("gives", "")}
            cls.prototype_memory.add(out.z_real, rule["action"], outcome)
            rule_added += 1
            n_added += 1

        print(f"  [{ri+1:2d}/{len(CRAFTER_RULES)}] {rule['action']} near {rule['near']}: "
              f"{rule_added}/{n_seeds}")

    # Symbolic rules → neocortex
    cls.train(generate_taught_transitions())

    print(f"Phase 3 done: {n_added} prototypes, {n_skipped} skipped "
          f"in {time.time()-t0:.0f}s")
    print(f"  Stats: {cls.prototype_memory.stats()}")


# ── Gate test ─────────────────────────────────────────────────────────────────

def gate_test(encoder: CNNEncoder, cls: CLSWorldModel) -> float:
    print("\nGate test...")
    encoder.eval()
    correct = 0
    total   = 0

    for rule in CRAFTER_RULES:
        env = CrafterPixelEnv(seed=9999)
        pixels, sym = env.reset()
        rng = np.random.RandomState(42)
        found = False
        for _ in range(300):
            pixels, sym, _, done = env.step(
                rng.choice(["move_left", "move_right", "move_up", "move_down"])
            )
            if sym.get("near") == rule["near"]:
                found = True
                break
            if done:
                pixels, sym = env.reset()

        if not found:
            continue

        with torch.no_grad():
            pix_t = torch.from_numpy(pixels)
            outcome, conf, source = cls.query_from_pixels(pix_t, rule["action"], encoder)

        expected = rule["result"]
        got      = outcome.get("result", "unknown")
        ok       = got == expected
        mark     = "✓" if ok else "✗"
        print(f"  {mark} {rule['action']} near {rule['near']}: "
              f"expected={expected} got={got} conf={conf:.2f} src={source}")
        if ok:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    print(f"\nGATE: {correct}/{total} = {acc:.0%}  "
          f"{'PASS' if acc >= 0.5 else 'FAIL'} (threshold 50%)")
    return acc


# ── Save ──────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, encoder: CNNEncoder, cls: CLSWorldModel) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "encoder_state":      encoder.state_dict(),
        "prototype_z":        [z.cpu() for z in cls.prototype_memory.z_store],
        "prototype_actions":  cls.prototype_memory.actions,
        "prototype_outcomes": cls.prototype_memory.outcomes,
        "n_near_classes":     len(NEAR_CLASSES),
        "dim":                2048,
    }
    torch.save(state, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nCheckpoint saved → {path}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Stage 66 — headless training")
    print(f"Output: {CKPT_PATH}\n")

    t_start = time.time()

    dataset          = phase1_collect(n_trajectories=50, steps_per_traj=200)
    encoder, _pred   = phase2_train(dataset, epochs=100, batch_size=256)
    cls              = CLSWorldModel(dim=2048, device="cpu")
    phase3_prototypes(encoder, cls, n_seeds=50)
    acc              = gate_test(encoder, cls)
    save_checkpoint(CKPT_PATH, encoder, cls)

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min  gate={acc:.0%}")


if __name__ == "__main__":
    main()
