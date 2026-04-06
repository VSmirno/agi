# Stage 66 v2: Pixel Perception via Prototype Memory

**Date:** 2026-04-06
**Gate:** ≥50% Crafter QA from pixel input
**Supersedes:** 2026-04-04-stage66-pixels-design.md (VQ/CNN + decode head approach — failed at 0%)

---

## Motivation

Stage 66 v1 failed at 0% gate across three encoder architectures (VQ Patch Codebook, lookup table, CNN). Root cause: the decode → symbolic key → neocortex dict pipeline. Every step loses information and compounds errors. The CNN encoder trained with JEPA learns scene-specific features (terrain, layout) that don't generalize across seeds — held-out near classification was 40%.

The fundamental problem: converting latent representations back to symbols defeats the purpose of learning representations. We were building OCR, not perception.

v2 removes the symbolic decoding entirely. The world model operates directly in latent space.

---

## Architecture

### Overview

```
pixels → CNN encoder → z_real (2048 float)
                            ↓
                  PrototypeMemory (k-NN)
                  find k nearest (z_stored, same action)
                  majority vote → outcome + confidence
```

Three knowledge paths in CLS (additive, not replacing):
1. **Neocortex** (symbolic) — dict lookup by situation key. Unchanged. For symbolic envs.
2. **Hippocampus** (binary VSA) — SDM with Hamming distance. Unchanged. For symbolic envs.
3. **Prototype Memory** (continuous latent) — k-NN in z_real space. NEW. For pixel envs.

### CNN Encoder

Same architecture as v1 (already working on CPU):

```
SimpleConv(3→32, 3×3, stride=2)  → (32, 32, 32)
SimpleConv(32→64, 3×3, stride=2) → (64, 16, 16)
SimpleConv(64→128, 3×3, stride=2) → (128, 8, 8)
SimpleConv(128→256, 3×3, stride=2) → (256, 4, 4)
Flatten → Linear(4096, 2048) → LayerNorm → z_real
```

Standard Conv2d (not depthwise separable — ROCm segfaults on grouped convs).
Runs on CPU. ~8.9M parameters.

`near_logits` head remains in `CNNEncoder` but is unused in pixel pipeline (backward compat for tests).

### Encoder Training: JEPA + Contrastive

Two losses, trained jointly:

**JEPA loss** (self-supervised): predict z_real[t+1] from z_real[t] + action embedding.
```
z_pred = predictor(z_t, action)
jepa_loss = MSE(z_pred, stop_grad(z_{t+1}))
```

**Supervised Contrastive loss** (SupCon): frames with same situation should cluster in z_real space.
Labels are compound: `near_label + inventory_summary` (e.g., "table_has_wood", "table_noinv", "tree_noinv").
This prevents collapsing inventory-dependent situations (6/17 crafting rules share near=table but differ by inventory).
```
For each z_i in batch:
    positives = {z_j : situation_label[j] == situation_label[i], j != i}
    negatives = {z_j : situation_label[j] != situation_label[i]}
    contrastive_loss = -log(sum(exp(sim(z_i, z_pos)/tau)) / sum(exp(sim(z_i, z_all)/tau)))
    (z_all excludes the anchor z_i itself)
```
Temperature tau = 0.1. Uses symbolic labels from training data only (not at inference).

Situation label construction:
```python
def make_situation_label(sym_obs: dict) -> str:
    near = sym_obs.get("near", "empty")
    inv_parts = sorted(k for k in sym_obs if k.startswith("has_"))
    inv_key = "_".join(inv_parts) if inv_parts else "noinv"
    return f"{near}_{inv_key}"
```

**VICReg variance** (collapse prevention):
```
var_loss = mean(relu(1.0 - std(z, dim=batch)))
```

**Total:**
```
loss = jepa_loss + 0.5 * contrastive_loss + 0.1 * var_loss
```

Why contrastive and not classification head:
- Classification head trains a linear layer ON TOP of features — doesn't reshape the representation
- Contrastive loss reshapes the ENCODER'S representation space directly — z becomes invariant to seed/background

### Prototype Memory

```python
class PrototypeMemory:
    z_store: Tensor       # (N, 2048) float32, L2-normalized
    actions: list[str]    # N action names
    outcomes: list[dict]  # N outcome dicts {"result": ..., "gives": ...}

    def query(z_query, action, k=5) -> (outcome, confidence):
        # 1. Filter prototypes where action matches
        # 2. Cosine similarity between z_query and filtered z_store
        # 3. Top-k nearest
        # 4. Majority vote on outcome["result"]
        # 5. confidence = vote_fraction * mean_similarity_of_majority
        # 6. If max similarity < 0.3 → return ("unknown", 0.0)

    def add(z_real, action, outcome):
        # L2-normalize z_real, append to stores
```

Memory footprint: 800 prototypes × 2048 × 4 bytes = 6.5MB.
Query cost: O(N × 2048) cosine — trivial on CPU.

### CLS Integration

`CLSWorldModel` gains:
- `self.prototype_memory: PrototypeMemory`
- `query_from_prototypes(pixels, action, encoder) → (outcome, confidence, "prototype")`

The `query_from_pixels` method is rewritten:
```python
def query_from_pixels(self, pixels, action, encoder, decode_head=None):
    # decode_head kept in signature for backward compat, ignored
    z_real = encoder(pixels).z_real
    outcome, confidence = self.prototype_memory.query(z_real, action)
    return outcome, confidence, "prototype"
```

No decode head. No symbolic key. No neocortex lookup in pixel path.
Signature preserved with `decode_head=None` default for backward compat.

---

## Training Pipeline

### Phase 1: Data Collection (~11 seconds)

Unchanged from v1.

```
50 trajectories × 200 steps = 10K transitions
Each: (pixels_t, pixels_t1, action_idx, situation_label)
situation_label = compound of near + inventory (see SupCon section)
Multi-seed for diversity (seed + traj * 7)
```

`near_label` extracted from symbolic ground truth (used for contrastive loss during training only).

### Phase 2: Encoder Training (~15 minutes on CPU)

```
100 epochs, batch 256
Optimizer: Adam(encoder.params + predictor.params, lr=1e-3)
Loss: JEPA + 0.5 * SupCon + 0.1 * VICReg variance
```

Encoder and predictor both on CPU (Conv2d incompatible with ROCm).

### Phase 3: Prototype Collection (~2 minutes)

Encoder frozen. Two prototype sources:

**a) Collection rules** (7 rules: do near tree/stone/coal/iron/diamond/water/cow):
For each rule, for each of 50 seeds:
- Create env, random walk to find target near object
- Encode pixels → z_real, store (z_real, action, outcome)
- Skip if not found after 300 steps
- No inventory requirements for collection rules

**b) Crafting/placing rules** (10 rules: place_table, make_wood_pickaxe, etc.):
These require specific inventory. Instead of trying to achieve the inventory in-game:
- Collect prototypes from Phase 1 replay data: find transitions where symbolic obs matches the rule's preconditions
- Phase 1 has 10K transitions with symbolic ground truth — filter for matching (near, inventory, action) tuples
- If insufficient matches: run additional targeted collection trajectories

Also load symbolic rules into neocortex (backward compatibility).

Expected: ~500-800 prototypes total. Rare situations (diamond, iron) may have fewer prototypes — k=3 fallback when fewer than 5 available.

### Phase 4: Gate Test

For each rule in CRAFTER_RULES:
1. Create env with NEW seed (not used in Phase 3)
2. Find target situation
3. Encode pixels → query PrototypeMemory
4. Compare predicted outcome to expected

Gate: ≥50% correct.

---

## What Changes in Code

| File | Change |
|------|--------|
| `src/snks/agent/prototype_memory.py` | **NEW** — PrototypeMemory class (~80 lines) |
| `src/snks/encoder/predictive_trainer.py` | Add SupCon loss; `train_step`/`train_full` gain `situation_labels` param (~40 lines) |
| `src/snks/agent/cls_world_model.py` | Add `query_from_prototypes()`, store PrototypeMemory (~30 lines) |
| `experiments/exp122_pixels.py` | Rewrite: 4 phases instead of 5, no decode head (~200 lines) |
| `tests/test_stage66_pixels.py` | Update tests for new architecture (~50 lines) |

**Unchanged:** cnn_encoder.py, decode_head.py, crafter_pixel_env.py, crafter_trainer.py, SDM, VSA, neocortex.

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Contrastive loss doesn't separate near classes in z_real | HIGH | Monitor cluster separation (inter/intra class cosine) during training. Increase λ_con if needed. |
| k-NN noisy with small prototype set | MEDIUM | 50 seeds × 16 rules = 800 prototypes should suffice. Increase seeds if accuracy < 50%. |
| Encoder on CPU too slow for 100 epochs | LOW | Already tested: ~15 min. Acceptable. |
| JEPA and contrastive conflict (one wants scene-specific, other wants invariant) | MEDIUM | λ_con = 0.5 balances. JEPA predicts dynamics, contrastive groups situations — complementary, not conflicting. |
| Some CRAFTER_RULES situations hard to find in env | LOW | Skip unfindable rules. Gate threshold is 50%, not 100%. |

---

## Why This Will Work (Where v1 Failed)

v1 failure chain: `z not invariant across seeds → decode head fails → symbolic key wrong → neocortex miss → hippocampus miss (binary z_vsa also not invariant)`

v2 fixes each link:
1. **Contrastive loss** makes z_real cluster by near-object, not by seed/terrain
2. **No decode step** — no information loss from z → symbols
3. **No neocortex lookup** — no exact-match fragility
4. **Continuous similarity** instead of binary Hamming — graceful degradation
5. **k-NN with 50 seeds per rule** — covers visual variation, majority vote is robust

---

## What This Sets Up for Future Stages

- Prototype memory = stepping stone to full latent world model
- Contrastive encoder = foundation for transfer to other visual environments
- k-NN query = can be upgraded to learned metric (Siamese net) when needed
- JEPA predictor = embryonic temporal world model, scales with data
- No symbolic bottleneck = ready for environments without hand-labeled objects
