# Exp129: Iterative Encoder Finetuning — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-07-exp129-encoder-finetuning-design.md`

## Task 1: finetune_encoder() function

**File:** `experiments/exp129_finetune.py`
**Steps:**
1. Create file with imports from exp127/exp128 (CNNEncoder, PredictiveTrainer, JEPAPredictor, NearDetector)
2. Implement `finetune_encoder(encoder, dataset, epochs=50, lr=3e-4, device="cuda")`:
   - Takes existing encoder (warm start — NOT fresh CNNEncoder())
   - Creates JEPAPredictor + PredictiveTrainer with passed encoder and lr
   - Calls `trainer.train_full()` with dataset pixels/labels
   - Returns (encoder, NearDetector(encoder))
3. Implement `_save_checkpoint(encoder, detector, tag)` — saves to `demos/checkpoints/exp129/{tag}/`

**Verify:** Import check on minipc

## Task 2: Controlled data collection (cached)

**File:** `experiments/exp129_finetune.py`
**Steps:**
1. Implement `collect_controlled()` — reuse `_run_controlled_batch`, `_run_controlled_items_batch`, `_collect_empty_walk_frames` from exp127/exp128
2. Same params as exp128 phase 2: stone 80, coal 50, iron 50, empty/table 100, empty walk 100
3. Return `{pixels, near_labels, trained_classes}` — same format as exp128
4. Cache result (collect once, reuse every iteration)

**Verify:** Run standalone, check frame counts match exp128

## Task 3: Natural data collection

**File:** `experiments/exp129_finetune.py`
**Steps:**
1. Implement `collect_natural(detector, iteration)`:
   - TREE_CHAIN × 80 seeds via `_run_chain_batch_generic`
   - STONE_CHAIN × 80 seeds via `_run_chain_batch_generic`
   - seed_base = 40000 + iteration * 1000
   - Return `{frames, stone_rate, tree_rate}`
2. Handle 0-frame case for stone: log warning, return empty

**Verify:** Run with exp128 detector, check tree_rate ~70%, stone_rate reported

## Task 4: Per-class mixing

**File:** `experiments/exp129_finetune.py`
**Steps:**
1. Implement `mix_datasets(controlled, natural, iteration)`:
   - For each class in NEAR_CLASSES:
     - If class has natural frames (tree, stone, empty): blend at ratio `controlled × (1 - i/5) + natural × (i/5)`
     - If class has NO natural frames (coal, iron, zombie, table): 100% controlled
   - Subsample to match ratio per class
   - `_balance_classes(all_labeled, max_ratio=4.0)`
   - Return `{pixels, near_labels}` ready for finetune_encoder

**Verify:** Check class distribution at iteration 1 vs 3 — controlled ratio shifts

## Task 5: Main loop + evaluation

**File:** `experiments/exp129_finetune.py`
**Steps:**
1. Implement `main()`:
   - Load exp128/final checkpoint (encoder, detector)
   - Copy concept_store to `demos/checkpoints/exp129/concept_store/`
   - Collect controlled data (cached)
   - Loop i=1..5:
     - Phase A: `collect_natural(detector, i)`
     - Phase B: `mix_datasets(controlled, natural, i)`
     - Phase C: `finetune_encoder(encoder, mixed, epochs=50, lr=3e-4)`
     - Phase D: Evaluate stone_rate, tree_rate on 50 eval seeds
     - Save checkpoint `iter{i}/`
     - Log iteration metrics
     - Check stopping: delta(stone) < 5% or tree < 60% or i=5
   - Copy best iteration to `final/`
2. Print summary table at end

**Verify:** Full run on minipc, check stone_rate improves across iterations

## Task 6: Run on minipc + copy checkpoint

**Steps:**
1. git push
2. git pull on minipc
3. Launch in tmux: `HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src:experiments python -u experiments/exp129_finetune.py`
4. Monitor progress
5. Copy `demos/checkpoints/exp129/final/` to local
6. Update `server.py` CKPT_DIR

## Dependencies

```
Task 1 ──> Task 5
Task 2 ──> Task 5
Task 3 ──> Task 5
Task 4 ──> Task 5
Task 5 ──> Task 6
```

Tasks 1-4 are independent, can be written together in one file.
Task 5 wires them into main().
Task 6 is deploy.
