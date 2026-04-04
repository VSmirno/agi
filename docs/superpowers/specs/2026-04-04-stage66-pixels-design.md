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
5. Two outputs:
   a. z_real = mean(e_1..e_64) → 2048-dim float (for predictor + decode head)
   b. z_vsa = bundle(bind(vsa_codebook[idx_i], position_role[i]) for i in 1..64)
              → 2048-dim binary (for SDM hippocampus)
```

- **Codebook updates:** EMA (exponential moving average), no backprop
- **Embedding table:** learned via straight-through estimator
- **Parameters:** ~8.4M trainable (embedding table 4096×2048) + 800KB codebook (EMA)
- **GPU compute:** near-zero — cosine similarity (64×192 @ 192×4096) + mean pooling
- **Why 8×8 patches:** Crafter sprites render at ~7×7 pixels per object. Each patch ≈ one object.

Replaces: `rgb_conv.py` + `VisualEncoder`. DAF engine does NOT participate in pixel pipeline (R1 verdict: oscillations unnecessary for perception).

### 2. SDM Compatibility: Binary VSA Address from Codebook Indices

The existing SDM hippocampus uses binary VSA vectors (XOR bind, Hamming distance). The pixel encoder produces real-valued embeddings. Solution: construct a **binary VSA address** from the quantized codebook indices, not from the real-valued z.

```
64 patches → 64 codebook indices (integers)
Each index i at position p:
    vsa_bind(codebook_vsa_vector[idx_i], position_role[p])   # XOR
Bundle (majority vote) all 64 bindings → binary 2048-dim address
```

- `codebook_vsa_vector`: pre-allocated random binary VSA vector per codebook entry (4096 vectors, allocated once via existing VSACodebook)
- `position_role`: pre-allocated random binary VSA vector per patch position (64 vectors)
- This is native VSA encoding: codebook entries = fillers, positions = roles. Same algebra as `_encode_situation()`.
- **SDM write/read unchanged** — receives binary 2048-dim vector as always.

### 3. Predictive Training (mini-JEPA)

Operates on real-valued z_real (NOT on binary z_vsa).

```
z_real[t] (2048) + action_embed (64) → MLP predictor → z_pred[t+1] (2048)
                                                             |
                                               Loss = MSE(z_pred, stop_grad(z_target))
```

- **Predictor:** Linear(2112, 1024) → ReLU → Linear(1024, 2048). ~4M params.
- **Action embedding:** Embedding(num_actions, 64). Crafter has ~17 actions.
- **Target:** z_target = encoder(frame[t+1]).z_real with stop_gradient (JEPA-style).
- **Collapse prevention:** VICReg-style variance term.
  - Per-dimension std across batch, penalize if std < 1.0
  - Loss weight: 0.1 × mean(max(0, 1.0 - std(z, dim=batch)))
- **Data source:** Exploration rollouts in pixel Crafter env.

### 4. Dual Path CLS Integration

```
pixels → VQ Patch Encoder → z_real (float 2048) + z_vsa (binary 2048)
    |
    +--→ z_vsa → SDM Hippocampus (binary, Hamming distance)
    |        read_next(z_vsa, zeros) → predicted outcome vector + confidence
    |        write(z_vsa, zeros, outcome_vec, reward) — amplified N times
    |        Interface UNCHANGED from current _write_on_surprise / query
    |
    +--→ z_real → Decode Head → situation key → Neocortex lookup
         Linear(2048, 256) → ReLU → heads:
           - near_object: Linear(256, N_objects) → softmax
           - inventory: Linear(256, N_items) → sigmoid (multi-label)
           - standing_on: Linear(256, N_objects) → softmax
         → concatenate into situation key string → neocortex dict lookup
```

**Query logic (`query_from_pixels`):**
1. Decode situation key from z_real → try neocortex (exact match)
2. If neocortex hit: return outcome, conf = 0.95 × decode_certainty, source = "neocortex"
   - decode_certainty = 1.0 - mean_entropy(softmax heads) / log(N_classes)
   - This penalizes neocortex path when decode head is uncertain
3. If neocortex miss: try hippocampus via z_vsa
4. Return best non-zero-confidence answer. If both hit, prefer neocortex if conf_neo > conf_hippo, else hippocampus.
5. Abstraction engine: skipped in pixel path (relies on symbolic object names). Future stage: abstract over latent clusters.

**Unchanged:** consolidation logic, calibration tracker (Stage 65), SDM internals.

---

## Training Pipeline

### Phase 1: Data Collection

```
CuriosityExplorer (Stage 64, adapted for pixel env wrapper)
    + pixel Crafter env (wraps real Crafter, returns both pixels AND symbolic obs)
    → 50 trajectories × 200 steps = 10K transition pairs
    → each step: (pixels_t, action, pixels_t1, symbolic_obs)
    → symbolic_obs for decode head supervision only, NOT used at inference
    → stored on disk as .pt files
```

CuriosityExplorer adaptation: wrap pixel env to expose `observe() → dict` for explorer logic (using ground truth symbolic obs during collection only). Explorer sees symbolic obs for action selection; we record pixels alongside.

### Phase 2: Encoder + Predictor (self-supervised)

```python
for batch in rollout_pairs:
    z_t = encoder(pixels_t).z_real            # codebook lookup, minimal GPU
    z_t1 = encoder(pixels_t1).z_real          # target, stop_grad
    z_pred = predictor(z_t, action)           # MLP forward

    pred_loss = mse(z_pred, z_t1.detach())
    var_loss = 0.1 * mean(relu(1.0 - std(z_t, dim=0)))  # per-dim variance
    loss = pred_loss + var_loss
    loss.backward()                            # straight-through → embeddings
    update_codebook_ema(patches_t)             # EMA, no gradient
```

~100 epochs. Batch size 256.

**Codebook initialization:** k-means on first 5K patches (one pass, ~5 seconds). Avoids dead entries from random init.

**Dead entry reset:** After each epoch, entries with usage count < 2 are re-initialized to random patches from the current batch.

### Phase 3: Decode Head (supervised)

```python
for batch in rollout_pairs:
    z_t = encoder(pixels_t).z_real.detach()   # encoder frozen
    pred_near = head_near(z_t)
    pred_inv = head_inventory(z_t)

    loss = CE(pred_near, gt_near) + BCE(pred_inv, gt_inventory)
```

Encoder frozen. ~20 epochs. Ground truth from symbolic_obs collected in Phase 1.

### Phase 4: CLS Learning (from pixels)

Reuses `generate_crafter_transitions()` from `crafter_trainer.py` but through pixel env:

- **Teaching (CRAFTER_TAUGHT rules):** For each of 5 taught rules, set up the situation in pixel Crafter env, execute action, record (pixels, action, outcome) as Transition. CLS.train() writes to both SDM (via z_vsa) and neocortex (via decode head → situation key).
- **Exploration:** CuriosityExplorer with pixel env discovers additional rules. For each novel transition: SDM stores z_vsa, consolidation may promote to neocortex via decoded key.

---

## Gate Test

Pixel QA scenarios are rendered by the pixel Crafter env:

```python
def test_crafter_qa_from_pixels():
    """Stage 66 gate: ≥50% Crafter QA accuracy from pixels."""
    pixel_env = CrafterPixelEnv()
    encoder = load_trained_encoder()
    cls = load_cls_with_dual_path(encoder)

    correct = 0
    total = 0
    for scenario in CRAFTER_QA_SCENARIOS:
        # Set up env to specific state, get pixel observation
        pixels = pixel_env.setup_scenario(scenario["near"], scenario.get("inventory", {}))
        action = scenario["action"]
        expected = scenario["expected_outcome"]

        outcome, conf, source = cls.query_from_pixels(pixels, action)
        if outcome.get("result") == expected:
            correct += 1
        total += 1

    accuracy = correct / total
    assert accuracy >= 0.50, f"Pixel QA {accuracy:.0%} < 50%"
```

`CRAFTER_QA_SCENARIOS`: derived from existing `CRAFTER_RULES` in `crafter_trainer.py`. Each scenario specifies (near, inventory, action, expected_outcome). The pixel env renders the corresponding visual frame.

### Additional Metrics (non-gate)

- Decode head accuracy: % of correctly reconstructed situation keys
- Hippocampus-only accuracy: QA using only z_vsa → SDM path (no neocortex)
- Predictor MSE: temporal prediction quality in latent space
- Codebook utilization: fraction of 4096 entries with usage > 0 (target: >50%)

---

## File Plan

| File | Purpose |
|------|---------|
| `src/snks/encoder/vq_patch_encoder.py` | VQ Patch Codebook encoder (z_real + z_vsa outputs) |
| `src/snks/encoder/predictive_trainer.py` | Mini-JEPA training loop (Phase 2 + Phase 3) |
| `src/snks/agent/decode_head.py` | z_real → situation key classification heads |
| `src/snks/agent/cls_world_model.py` | Modified: add `query_from_pixels()` dual path |
| `src/snks/agent/crafter_pixel_env.py` | Pixel Crafter env wrapper (returns pixels + symbolic obs) |
| `src/snks/agent/crafter_trainer.py` | Modified: `CRAFTER_QA_SCENARIOS` list for pixel gate test |
| `experiments/exp122_pixels.py` | Stage 66 full pipeline: collect → train → gate |
| `tests/test_stage66_pixels.py` | Gate test + additional metrics |

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Codebook collapse (few entries used) | HIGH | k-means init + dead entry reset + VICReg variance |
| Decode head < 60% accuracy | MEDIUM | Hippocampus path compensates; gate is 50% not 87% |
| 10K transitions insufficient | MEDIUM | Increase to 100 trajectories if codebook utilization < 50% |
| Crafter pixel env rendering issues | LOW | Verify sprite dimensions match 8×8 patch assumption |
| Abstraction engine disabled for pixels | LOW | Acceptable for Stage 66; future stage adds latent abstraction |

---

## What This Sets Up for Future Stages

- Hippocampus already operates in latent space (via z_vsa) → removing decode head = full latent CLS
- Codebook entries = discovered visual vocabulary → interpretable without situation dict
- Predictor = embryonic JEPA world model → scale to video with more data
- Encoder architecture is swappable → plug in DINOv2 or larger ViT when GPU allows
- z_real path → future neocortex can transition from dict to latent nearest-neighbor
