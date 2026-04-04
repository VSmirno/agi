# Stage 66 Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-04-stage66-pixels-design.md`
**Gate:** ≥50% Crafter QA from pixels

---

## Task 1: VQ Patch Codebook Encoder

**File:** `src/snks/encoder/vq_patch_encoder.py` (new)

1. Create `VQPatchEncoder(nn.Module)`:
   - `__init__(self, patch_size=8, codebook_size=4096, embed_dim=2048, vsa_dim=2048)`
   - Allocate codebook `(4096, 192)` as buffer (not parameter)
   - Allocate embedding table `(4096, 2048)` as nn.Embedding
   - Allocate `position_roles` `(64, vsa_dim)` binary VSA vectors as buffer
   - Allocate `codebook_vsa` `(4096, vsa_dim)` binary VSA vectors as buffer
   - `_initialized` flag for k-means init

2. Implement `forward(self, pixels: Tensor) -> namedtuple(z_real, z_vsa, indices)`:
   - Patchify: unfold (B, 3, 64, 64) → (B, 64, 192)
   - Cosine similarity with codebook → (B, 64, 4096)
   - Argmax → indices (B, 64)
   - Lookup embedding table → (B, 64, 2048), mean → z_real (B, 2048)
   - Straight-through: z_real gets gradient as if indices were differentiable
   - Build z_vsa: for each patch, XOR codebook_vsa[idx] with position_roles[pos], majority bundle → z_vsa (B, 2048) binary

3. Implement `update_codebook_ema(self, patches, indices, momentum=0.99)`:
   - EMA update codebook prototypes
   - Track usage counts per entry
   - Reset dead entries (usage < 2 in epoch) to random batch patches

4. Implement `init_codebook_kmeans(self, patches)`:
   - One-pass k-means on first batch of patches
   - Set `_initialized = True`

**Verify:** `pytest -xvs` — unit test with random tensor (B=4, 3, 64, 64) → outputs correct shapes

---

## Task 2: Predictive Trainer (mini-JEPA)

**File:** `src/snks/encoder/predictive_trainer.py` (new)

1. Create `JEPAPredictor(nn.Module)`:
   - `__init__(self, z_dim=2048, action_dim=64, hidden=1024, n_actions=17)`
   - Action embedding: `nn.Embedding(n_actions, action_dim)`
   - MLP: Linear(z_dim + action_dim, hidden) → ReLU → Linear(hidden, z_dim)

2. Create `PredictiveTrainer`:
   - `__init__(self, encoder, predictor, lr=1e-3, vicreg_weight=0.1)`
   - `train_step(self, pixels_t, pixels_t1, actions) -> dict`:
     - z_t = encoder(pixels_t).z_real
     - z_t1 = encoder(pixels_t1).z_real.detach()  # stop gradient
     - z_pred = predictor(z_t, actions)
     - pred_loss = MSE(z_pred, z_t1)
     - var_loss = vicreg_weight * mean(relu(1.0 - std(z_t, dim=0)))
     - total = pred_loss + var_loss
     - backward + step
     - encoder.update_codebook_ema(patches_t, indices_t)
     - return {"pred_loss": ..., "var_loss": ..., "codebook_usage": ...}

   - `train_epoch(self, dataset) -> dict`: iterate batches, aggregate stats
   - `train(self, dataset, epochs=100) -> dict`: full training loop with logging

**Verify:** Unit test — one train_step doesn't crash, losses are finite

---

## Task 3: Decode Head

**File:** `src/snks/agent/decode_head.py` (new)

1. Create `DecodeHead(nn.Module)`:
   - `__init__(self, z_dim=2048, hidden=256, n_objects=15, n_items=10)`
   - Linear(z_dim, hidden) → ReLU → shared backbone
   - `head_near`: Linear(hidden, n_objects)
   - `head_standing`: Linear(hidden, n_objects)
   - `head_inventory`: Linear(hidden, n_items)

2. Implement `forward(self, z_real) -> dict`:
   - Return {"near_logits", "standing_logits", "inventory_logits"}

3. Implement `decode_situation_key(self, z_real) -> tuple[str, float]`:
   - Forward → argmax near, argmax standing, sigmoid inventory
   - Build situation key string (same format as `make_crafter_key`)
   - Compute decode_certainty = 1.0 - mean(normalized_entropy per head)
   - Return (key, certainty)

4. Implement `train_step(self, z_real, gt_near, gt_standing, gt_inventory) -> dict`:
   - CE loss for near + standing, BCE for inventory
   - Return losses

**Verify:** Unit test — random z → decode_situation_key returns valid string + certainty in [0,1]

---

## Task 4: Pixel Crafter Env

**File:** `src/snks/agent/crafter_pixel_env.py` (new)

1. Create `CrafterPixelEnv`:
   - `__init__(self)`: instantiate real Crafter env (`crafter.Env`)
   - `reset() -> tuple[np.ndarray, dict]`: return (pixels_64x64_rgb, symbolic_obs)
   - `step(action) -> tuple[np.ndarray, dict, float, bool]`: return (pixels, symbolic_obs, reward, done)
   - `observe() -> tuple[np.ndarray, dict]`: return current (pixels, symbolic_obs)
   - `setup_scenario(near, inventory) -> np.ndarray`: set env to specific state, return pixels

2. Handle Crafter's native observation format:
   - Crafter returns (64, 64, 3) uint8 → normalize to float32 [0, 1], transpose to (3, 64, 64)
   - Extract symbolic obs from Crafter's internal state for supervision

**Verify:** Instantiate env, reset, take 5 steps — pixels shape is (3, 64, 64), values in [0,1]

---

## Task 5: CLS Dual Path Integration

**File:** `src/snks/agent/cls_world_model.py` (modify)

1. Add `query_from_pixels(self, pixels, action, encoder, decode_head)`:
   - result = encoder(pixels)
   - z_real, z_vsa = result.z_real, result.z_vsa
   - Neocortex path: key, certainty = decode_head.decode_situation_key(z_real)
   - If key in self.neocortex: conf = 0.95 * certainty, outcome from neocortex
   - Hippocampus path: predicted, raw_conf = self.hippocampus.read_next(z_vsa, self._zeros)
   - If raw_conf > 0.01: calibrated conf, outcome decoded
   - Return best (outcome, conf, source)

2. Add `train_from_pixels(self, pixel_transitions, encoder, decode_head)`:
   - For each transition: encode pixels → z_vsa, z_real
   - Write to hippocampus via z_vsa (same amplification)
   - Decode key from z_real → store in neocortex
   - Consolidate as usual

**Verify:** Unit test — query_from_pixels with mock encoder returns valid (outcome, conf, source)

---

## Task 6: Data Collection + Training Pipeline

**File:** `experiments/exp122_pixels.py` (new)

1. Phase 1 — collect data:
   - Instantiate CrafterPixelEnv
   - Run CuriosityExplorer (adapted: uses symbolic obs for action selection, records pixels)
   - 50 trajectories × 200 steps → save as .pt dataset
   - Log: total transitions, unique symbolic states seen

2. Phase 2 — train encoder + predictor:
   - Load dataset
   - Init encoder (k-means codebook from first batch)
   - Train 100 epochs with PredictiveTrainer
   - Log: pred_loss, var_loss, codebook_usage per epoch

3. Phase 3 — train decode head:
   - Freeze encoder
   - Train decode head 20 epochs on (z_real, gt_symbolic_obs) pairs
   - Log: near_accuracy, inventory_accuracy per epoch

4. Phase 4 — CLS learning from pixels:
   - Load trained encoder + decode head
   - Teach CRAFTER_TAUGHT rules via pixel env
   - Run exploration with CuriosityExplorer (pixel mode)
   - Train CLS with pixel transitions

5. Phase 5 — gate test:
   - Run QA scenarios from CRAFTER_QA_SCENARIOS
   - Report accuracy, per-source breakdown

**Verify:** Full pipeline runs end-to-end (may run on minipc only)

---

## Task 7: Gate Test

**File:** `tests/test_stage66_pixels.py` (new)

1. Add `CRAFTER_QA_SCENARIOS` — derived from CRAFTER_RULES in crafter_trainer.py
2. `test_crafter_qa_from_pixels()` — gate: accuracy ≥ 0.50
3. `test_decode_head_accuracy()` — non-gate metric
4. `test_hippocampus_only_accuracy()` — non-gate metric
5. `test_codebook_utilization()` — codebook entries used > 50%

**Verify:** `pytest tests/test_stage66_pixels.py -v` passes

---

## Task 8: ROADMAP Update + Commit

1. Update ROADMAP.md — Stage 66 status
2. Update memory — roadmap progress
3. Final commit
