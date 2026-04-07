"""Training worker for Crafter Survival Demo.

Collects data and trains encoder — called from DemoEngine._train_worker().
Reuses Phase 0/1/2 logic from exp128 but with progress callbacks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv, CrafterControlledEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.near_detector import NearDetector
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer

ProgressCB = Callable[[str, int, int, float], None]


def _collect_random_walk(
    n_traj: int = 50,
    steps_per: int = 200,
    seed: int = 42,
    progress_cb: ProgressCB | None = None,
) -> dict:
    """Collect random walk trajectories for JEPA training."""
    all_pt, all_pt1, all_actions = [], [], []

    for i in range(n_traj):
        env = CrafterPixelEnv(seed=seed + i * 7)
        pixels, _ = env.reset()
        rng = np.random.RandomState(seed + i * 1000)

        for _ in range(steps_per):
            action = int(rng.randint(0, 17))
            next_pixels, _, done, _ = env.step(action)
            all_pt.append(torch.from_numpy(pixels))
            all_pt1.append(torch.from_numpy(next_pixels))
            all_actions.append(action)
            pixels = next_pixels
            if done:
                pixels, _ = env.reset()

        if progress_cb and (i + 1) % 10 == 0:
            progress_cb("collecting", i + 1, n_traj, 0.0)

    return {
        "pixels_t": torch.stack(all_pt),
        "pixels_t1": torch.stack(all_pt1),
        "actions": torch.tensor(all_actions),
    }


def _collect_labeled_frames(
    progress_cb: ProgressCB | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect labeled frames for near-detector training via controlled env."""
    labeled: list[tuple[torch.Tensor, int]] = []

    targets = [
        ("tree", {}, 30),
        ("stone", {"wood_pickaxe": 1}, 30),
        ("coal", {"wood_pickaxe": 1}, 20),
        ("iron", {"stone_pickaxe": 1}, 20),
        ("water", {}, 20),
    ]

    for target, inv, n_seeds in targets:
        idx = NEAR_TO_IDX.get(target)
        if idx is None:
            continue
        for s in range(n_seeds):
            try:
                env = CrafterControlledEnv(seed=100 + s * 7)
                pixels, _ = env.reset_near(target, inventory=inv, no_enemies=True)
                labeled.append((torch.from_numpy(pixels), idx))
            except Exception:
                pass

    # Empty frames from walk
    empty_idx = NEAR_TO_IDX.get("empty", 0)
    for s in range(30):
        env = CrafterPixelEnv(seed=500 + s * 7)
        pixels, _ = env.reset()
        labeled.append((torch.from_numpy(pixels), empty_idx))

    if progress_cb:
        progress_cb("labeled", len(labeled), len(labeled), 0.0)

    pixels = torch.stack([p for p, _ in labeled]).float()
    labels = torch.tensor([l for _, l in labeled], dtype=torch.long)
    return pixels, labels


def run_training(
    epochs: int = 150,
    existing_store: ConceptStore | None = None,
    progress_cb: ProgressCB | None = None,
) -> tuple[CNNEncoder, NearDetector, ConceptStore]:
    """Full training pipeline: collect → train encoder → train detector."""

    # Phase 1: Collect random walk data
    if progress_cb:
        progress_cb("collecting trajectories", 0, 50, 0.0)
    dataset = _collect_random_walk(n_traj=50, steps_per=200, progress_cb=progress_cb)

    # Phase 2: Train encoder (JEPA)
    encoder = CNNEncoder()
    predictor = JEPAPredictor(embed_dim=2048)
    trainer = PredictiveTrainer(encoder, predictor)

    for epoch in range(epochs):
        loss = trainer.train_epoch(
            dataset["pixels_t"], dataset["pixels_t1"], dataset["actions"],
            batch_size=128,
        )
        if progress_cb:
            progress_cb("JEPA", epoch + 1, epochs, loss)

    encoder.eval()

    # Phase 3: Train near detector
    if progress_cb:
        progress_cb("detector training", 0, 1, 0.0)

    pixels_labeled, near_labels = _collect_labeled_frames(progress_cb)

    # Fine-tune near_head
    optimizer = torch.optim.Adam(encoder.near_head.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        encoder.train()
        out = encoder(pixels_labeled)
        loss = loss_fn(out.near_logits, near_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if progress_cb and (epoch + 1) % 10 == 0:
            progress_cb("detector", epoch + 1, 50, loss.item())

    encoder.eval()
    detector = NearDetector(encoder)

    # ConceptStore
    store = existing_store or ConceptStore()
    if not store.concepts:
        from pathlib import Path
        tb_path = Path(__file__).parent.parent.parent / "configs" / "crafter_textbook.yaml"
        if tb_path.exists():
            tb = CrafterTextbook(str(tb_path))
            tb.load_into(store)

    # Visual grounding
    if progress_cb:
        progress_cb("grounding", 0, 1, 0.0)

    visual_targets = [c.id for c in store.concepts.values()
                      if c.attributes.get("category") in ("resource", "crafted", "terrain", "enemy")]

    for target in visual_targets:
        z_accum = []
        for k in range(5):
            try:
                ctrl_env = CrafterControlledEnv(seed=42 + k * 100)
                if target == "empty":
                    px, _ = ctrl_env.reset()
                else:
                    px, _ = ctrl_env.reset_near(target, no_enemies=True)
                with torch.no_grad():
                    out = encoder(torch.from_numpy(px).float().unsqueeze(0))
                z_accum.append(out.z_real[0])
            except Exception:
                break
        if z_accum:
            z_mean = torch.stack(z_accum).mean(dim=0)
            z_norm = torch.nn.functional.normalize(z_mean.unsqueeze(0), dim=1).squeeze(0)
            store.ground_visual(target, z_norm)

    if progress_cb:
        progress_cb("done", epochs, epochs, 0.0)

    return encoder, detector, store
