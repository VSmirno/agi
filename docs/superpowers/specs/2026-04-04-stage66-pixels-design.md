# Stage 66: Symbolic Features → Pixels

**Date:** 2026-04-04
**Gate:** ≥50% Crafter QA from pixel input (currently 87% with symbolic)
**Approach:** Dual-path CLS with VQ Patch Codebook encoder + mini-JEPA predictive training

---

## Motivation

Stage 65 completed calibrated uncertainty on symbolic input. Stage 66 removes the last major scaffolding: hand-crafted symbolic features. The agent must learn to perceive the world from raw pixels.

Long-term goal: JEPA-style world model that watches video, learns dynamics, forms world model. Stage 66 is a pragmatic step: pixel encoder with predictive self-supervised training, integrated into the existing CLS architecture via dual path.

Key architectural insight from VideoWorld 2 (CVPR 2026): decouple visual appearance from core dynamics. Our CLS already does this — neocortex stores causal rules, not pixel patterns. We just need an encoder that bridges pixels to latent space.

---

## Architecture

### 1. VQ Patch Codebook Encoder

Designed for memory-heavy, GPU-light operation (96GB unified RAM, weak GPU compute).

```
Input: (3, 64, 64) RGB float32

1. Patchify: 8×8 non-overlapping patches → 64 patches × 192 dim each
2. Codebook: 4096 prototypes (4096 × 192), cosine similarity lookup
3. Embedding table: 4096 × 2048 (32MB in RAM)
4. Per-patch: top-1 cosine match → embedding lookup → e_i (2048-dim)
5. Aggregate: mean(e_1..e_64) → z (2048-dim)
```

- **Codebook updates:** EMA (exponential moving average), no backprop
- **Embedding table:** learned via straight-through estimator
- **Parameters:** ~400K trainable (embedding table) + 800KB codebook (EMA)
- **GPU compute:** near-zero — cosine similarity (64×192 @ 192×4096) + mean pooling
- **Why 8×8 patches:** Crafter sprites render at ~7×7 pixels per object. Each patch ≈ one object.

Replaces: `rgb_conv.py` + `VisualEncoder`. DAF engine does NOT participate in pixel pipeline (R1 verdict: oscillations unnecessary for perception).

### 2. Predictive Training (mini-JEPA)

```
z[t] (2048) + action_embed (64) → MLP predictor → z_pred[t+1] (2048)
                                                        |
                                          Loss = MSE(z_pred, stop_grad(z_target))
```

- **Predictor:** Linear(2112, 1024) → ReLU → Linear(1024, 2048). ~4M params.
- **Action embedding:** Embedding(num_actions, 64). Crafter has ~17 actions.
- **Target:** z_target = encoder(frame[t+1]) with stop_gradient (JEPA-style).
- **Collapse prevention:** VICReg-style variance term — penalize if std(z) across batch < threshold.
- **Data source:** CuriosityExplorer rollouts in pixel Crafter.

### 3. Dual Path CLS Integration

```
z (2048-dim from encoder)
    |
    +--→ SDM Hippocampus: store(z, outcome) / query(z) → prediction + confidence
    |    No changes — SDM already accepts 2048-dim vectors.
    |
    +--→ Decode Head → situation key → Neocortex lookup
         Linear(2048, 256) → ReLU → heads:
           - near_object: Linear(256, N_objects) → softmax
           - inventory: Linear(256, N_items) → sigmoid (multi-label)
           - standing_on: Linear(256, N_objects) → softmax
         → concatenate into situation key string → neocortex dict lookup
```

- **Decode head trained supervised** on (z, ground_truth_situation) pairs from exploration rollouts. Crafter API provides symbolic obs alongside pixels — used for training only, not inference.
- **Confidence merge:** CLS query tries both paths, takes max confidence. Decode head confidence is reduced proportionally to softmax entropy.
- **Unchanged:** consolidation (SDM → neocortex), abstraction engine, calibration tracker (Stage 65).

---

## Training Pipeline

### Phase 1: Data Collection

```
CuriosityExplorer (Stage 64 reuse) + pixel Crafter env
    → 50 trajectories × 200 steps = 10K transition pairs
    → each step: (pixels, action, next_pixels, symbolic_obs)
    → symbolic_obs for decode head supervision only
    → stored on disk
```

### Phase 2: Encoder + Predictor (self-supervised)

```python
for batch in rollout_pairs:
    z_t = encoder(pixels_t)             # codebook lookup, minimal GPU
    z_t1 = encoder(pixels_t1)           # target, stop_grad
    z_pred = predictor(z_t, action)     # MLP forward

    loss = mse(z_pred, z_t1.detach()) + vicreg_variance(z_t)
    loss.backward()                      # straight-through → embeddings
    update_codebook_ema(patches_t)       # EMA, no gradient
```

~100 epochs. Batch size 256. Fast on GPU.

### Phase 3: Decode Head (supervised)

```python
for batch in rollout_pairs:
    z_t = encoder(pixels_t).detach()    # encoder frozen
    pred_near = head_near(z_t)
    pred_inv = head_inventory(z_t)

    loss = CE(pred_near, gt_near) + BCE(pred_inv, gt_inventory)
```

Encoder frozen. ~20 epochs.

### Phase 4: CLS Learning (from pixels)

Same as Stage 64 but through pixel encoder:
- DemoTeacher teaches 5 rules: pixels → z → decode → neocortex stores rule; pixels → z → hippocampus stores (z, outcome)
- CuriosityExplorer discovers new rules: pixels → z → CLS query → curiosity score → explore

---

## Gate Test

```python
def test_crafter_qa_from_pixels():
    encoder = load_trained_encoder()
    cls = load_cls_with_dual_path(encoder)

    correct = 0
    for situation_pixels, action, expected_outcome in CRAFTER_QA:
        outcome, conf, source = cls.query_from_pixels(situation_pixels, action)
        if outcome == expected_outcome:
            correct += 1

    accuracy = correct / len(CRAFTER_QA)
    assert accuracy >= 0.50
```

### Additional Metrics (non-gate)

- Decode head accuracy: symbolic reconstruction quality
- Hippocampus-only accuracy: pure latent path without neocortex
- Predictor MSE: temporal prediction quality
- Codebook utilization: fraction of 4096 entries actively used

---

## File Plan

| File | Purpose |
|------|---------|
| `src/snks/encoder/vq_patch_encoder.py` | VQ Patch Codebook encoder |
| `src/snks/encoder/predictive_trainer.py` | Mini-JEPA training loop |
| `src/snks/agent/decode_head.py` | z → situation key decode heads |
| `src/snks/agent/cls_world_model.py` | Modified: add `query_from_pixels()` dual path |
| `src/snks/agent/crafter_pixel_env.py` | Pixel Crafter env wrapper |
| `experiments/exp122_pixels.py` | Stage 66 gate experiment |
| `tests/test_stage66_pixels.py` | Gate test |

---

## What This Sets Up for Future Stages

- Hippocampus already operates in latent space → removing decode head = full latent CLS
- Codebook entries = discovered visual vocabulary → interpretable without situation dict
- Predictor = embryonic JEPA world model → scale to video with more data
- Encoder architecture is swappable → plug in DINOv2 or larger ViT when GPU allows
