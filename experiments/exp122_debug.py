"""Stage 66 debug: trace query_from_pixels data flow.

Minimal reproduction: train CLS, then trace one query step by step.
"""

from __future__ import annotations
import torch
import numpy as np

from snks.device import get_device
from snks.encoder.vq_patch_encoder import VQPatchEncoder  # noqa: used for AGENT_PATCHES
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.agent.decode_head import (
    DecodeHead, symbolic_to_gt_tensors, NEAR_CLASSES, INVENTORY_ITEMS,
)
from snks.agent.crafter_pixel_env import CrafterPixelEnv, ACTION_TO_IDX
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_TAUGHT, CRAFTER_RULES
from snks.agent.crafter_encoder import make_crafter_key


def main():
    device = get_device()
    print(f"Device: {device}")

    # === Phase 1: Minimal data collection ===
    print("\n=== Phase 1: Collect 2000 transitions ===")
    env = CrafterPixelEnv(seed=42)
    pts, pts1, acts, syms = [], [], [], []
    pixels, sym = env.reset()
    for i in range(2000):
        action = np.random.RandomState(42 + i).randint(0, 17)
        next_pixels, next_sym, _, done = env.step(action)
        pts.append(torch.from_numpy(pixels))
        pts1.append(torch.from_numpy(next_pixels))
        acts.append(action)
        syms.append(sym)
        pixels, next_sym = next_pixels, next_sym
        if done:
            pixels, sym = env.reset()
    pt = torch.stack(pts)
    pt1 = torch.stack(pts1)
    a = torch.tensor(acts)
    print(f"Collected {len(pts)} transitions")

    # === Phase 2: Train encoder (20 epochs, fast) ===
    print("\n=== Phase 2: Train encoder (20 epochs) ===")
    encoder = VQPatchEncoder(device=device).to(device)
    predictor = JEPAPredictor().to(device)
    trainer = PredictiveTrainer(encoder, predictor, device=device)
    trainer.train_full(pt, pt1, a, epochs=20, batch_size=256, log_every=10)

    # === Phase 3: Train decode head (10 epochs) ===
    print("\n=== Phase 3: Train decode head (10 epochs) ===")
    head = DecodeHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    encoder.eval()
    all_agent_idx, all_z_local, all_near, all_inv = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(pt), 256):
            batch = pt[i:i+256].to(device)
            out = encoder(batch)
            all_agent_idx.append(out.indices[:, VQPatchEncoder.AGENT_PATCHES])
            all_z_local.append(out.z_local)
    all_agent_idx = torch.cat(all_agent_idx)
    all_z_local = torch.cat(all_z_local)
    for sym in syms:
        ni, iv = symbolic_to_gt_tensors(sym)
        all_near.append(ni)
        all_inv.append(iv)
    gt_near = torch.tensor(all_near, device=device)
    gt_inv = torch.tensor(all_inv, device=device)
    for epoch in range(10):
        perm = torch.randperm(len(all_agent_idx), device=device)
        for i in range(0, len(all_agent_idx), 256):
            idx = perm[i:i+256]
            head.train_step(all_agent_idx[idx], gt_near[idx], gt_inv[idx], optimizer,
                            z_local=all_z_local[idx])

    # === Phase 4: Train CLS from symbolic ===
    print("\n=== Phase 4: Train CLS ===")
    cls = CLSWorldModel(dim=2048, device=device)
    from snks.agent.crafter_trainer import generate_taught_transitions
    cls.train(generate_taught_transitions())
    print(f"Neocortex keys: {sorted(cls.neocortex.keys())}")

    # === Phase 5: TRACE one query ===
    print("\n=== Phase 5: TRACE query_from_pixels ===")
    # Find a tree in fresh env
    env2 = CrafterPixelEnv(seed=99)
    pixels, sym = env2.reset()
    for _ in range(200):
        pixels, sym, _, done = env2.step("move_right")
        if sym.get("near") == "tree":
            break
        if done:
            pixels, sym = env2.reset()

    print(f"\nGround truth symbolic: {sym}")
    print(f"Ground truth near: {sym.get('near')}")

    pix_tensor = torch.from_numpy(pixels).to(device)

    # Layer 1: Encoder
    out = encoder(pix_tensor)
    print(f"\nLayer 1 — Encoder:")
    print(f"  z_real norm: {out.z_real.norm().item():.2f}")
    print(f"  z_vsa mean: {out.z_vsa.mean().item():.2f}")
    print(f"  indices[:10]: {out.indices[:10].tolist()}")

    # Layer 2: Decode head (uses agent-adjacent patch indices)
    agent_indices = out.indices[VQPatchEncoder.AGENT_PATCHES]
    key_base, certainty = head.decode_situation_key(agent_indices, out.z_local)
    print(f"\nLayer 2 — Decode head:")
    print(f"  key_base: '{key_base}'")
    print(f"  certainty: {certainty:.3f}")

    # What does decode head predict?
    with torch.no_grad():
        logits = head(agent_indices, out.z_local)
    near_probs = torch.softmax(logits["near_logits"], dim=-1)
    inv_probs = torch.sigmoid(logits["inventory_logits"])
    top5_near = near_probs.topk(5)
    print(f"  near top5: {[(NEAR_CLASSES[i], f'{p:.3f}') for i, p in zip(top5_near.indices.tolist(), top5_near.values.tolist())]}")
    print(f"  inv probs: {[(INVENTORY_ITEMS[i], f'{p:.3f}') for i, p in enumerate(inv_probs.tolist()) if p > 0.1]}")

    # Layer 3: Neocortex lookup
    action = "do"
    key = key_base + action
    print(f"\nLayer 3 — Neocortex lookup:")
    print(f"  primary key: '{key}'")
    print(f"  in neocortex: {key in cls.neocortex}")

    # Fallback
    near_name = key_base.split("_")[1] if "_" in key_base else "empty"
    fallback_key = f"crafter_{near_name}_noinv_{action}"
    print(f"  fallback near_name: '{near_name}'")
    print(f"  fallback key: '{fallback_key}'")
    print(f"  fallback in neocortex: {fallback_key in cls.neocortex}")

    # Layer 4: Hippocampus
    predicted, raw_conf = cls.hippocampus.read_next(out.z_vsa, cls._zeros)
    print(f"\nLayer 4 — Hippocampus:")
    print(f"  raw_conf: {raw_conf:.4f}")
    outcome = cls._decode_outcome(predicted)
    print(f"  decoded outcome: {outcome}")

    # Full query
    print(f"\n=== Full query_from_pixels result ===")
    outcome, conf, source = cls.query_from_pixels(pix_tensor, "do", encoder, head)
    print(f"  outcome: {outcome}")
    print(f"  conf: {conf:.4f}")
    print(f"  source: {source}")
    print(f"  expected: collected")


if __name__ == "__main__":
    main()
