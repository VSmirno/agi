# Exp129: Iterative Encoder Finetuning — Design Spec

**Date:** 2026-04-07
**Status:** Draft
**Depends on:** exp128 (Stage 71) checkpoint
**Scope:** Encoder curriculum finetuning only. Navigation, ConceptStore, demo code — unchanged.

## Problem

Exp128 trains the outcome encoder on controlled env data (objects placed adjacent to player, no enemies, grassland context). In standard env:
- Tree chain natural: 74% success (ok)
- Stone chain natural: 4% success (bad)
- Coal/iron natural: 0% (uses controlled env as workaround)

Domain gap: encoder saw coal/iron/stone only in artificial conditions. Standard env has different surroundings, lighting, obstacles.

## Goal

Iterative curriculum finetuning: bootstrap on controlled data, then progressively shift toward natural env data. The encoder sees both domains.

**Success criteria:**
- Stone natural success rate: 4% → ≥30%
- Tree natural success rate: no regression below 60%

## Architecture

```
exp129_finetune.py

Load exp128/final checkpoint (encoder_v0, detector_v0, concept_store)
Collect controlled_data once (cached)

Loop i = 1..5:
  Phase A: Collect natural frames
    TREE_CHAIN × 80 seeds (CrafterPixelEnv, standard, enemies ON)
    STONE_CHAIN × 80 seeds (CrafterPixelEnv, standard, enemies ON)
    Using ScenarioRunner.run_chain() — same as exp128 phase 2
    OutcomeLabeler for labeling (no semantic GT)
    → natural_frames, stone_success_rate, tree_success_rate

  Phase B: Mix datasets
    controlled weight = 1 - i/5
    natural weight = i/5
    i=1: 80% controlled + 20% natural
    i=2: 60% controlled + 40% natural
    i=3: 40% controlled + 60% natural
    i=4: 20% controlled + 80% natural
    i=5: 0% controlled + 100% natural
    Subsample larger set to match ratio, preserve class balance

  Phase C: Finetune encoder
    Warm start: encoder.load_state_dict(prev_iteration)
    LR: initial_lr × 0.3 (prevent catastrophic forgetting)
    Epochs: 50 per iteration (vs 150 for initial training)
    Uses PredictiveTrainer (JEPA + SupCon + near_loss) — same as exp128

  Phase D: Evaluate + checkpoint
    Metrics: stone_natural_rate, tree_natural_rate
    Save checkpoint: demos/checkpoints/exp129/iter{i}/
      encoder.pt, detector.pt, concept_store/
    Log: iteration, stone_rate, tree_rate, delta
    Stop if delta(stone_rate) < 5% between iterations

Save best iteration as demos/checkpoints/exp129/final/
```

## Data Collection Details

**Natural collection (Phase A):**
- Reuses `_run_chain_batch_generic` from exp128
- TREE_CHAIN: `ScenarioStep("tree", "do", "tree", repeat=5, use_semantic_nav=True)`
- STONE_CHAIN: tree→table→pickaxe→stone (4 steps, semantic nav)
- 80 seeds each, seed_base offset per iteration to avoid overlap
- Enemies ON (standard conditions)
- Success = OutcomeLabeler returns expected near_label

**Controlled collection (cached):**
- Same as exp128 phase 2: `_run_controlled_batch` for stone/coal/iron/empty
- `CrafterControlledEnv.reset_near()` with `no_enemies=True`
- Collected once at start, reused every iteration

## Finetuning Details

**Warm start:** `encoder.load_state_dict(checkpoint)` — not from scratch. Each iteration continues from previous.

**Learning rate:** `lr = base_lr * 0.3`. PredictiveTrainer default lr (from exp127/128). Reduced to prevent destroying features learned on controlled data while adapting to natural.

**Epochs:** 50 per iteration. Warm start converges faster than cold start (150 epochs).

**Loss:** Same as exp128 — JEPA prediction + variance + contrastive + near classification. No changes to loss function.

**Class balance:** `_balance_classes(all_labeled, max_ratio=4.0)` — same as exp128.

## Checkpoints

Saved after EVERY phase within each iteration:

```
demos/checkpoints/exp129/
  iter1/
    encoder.pt
    detector.pt
    concept_store/   (copied from exp128, unchanged)
  iter2/
    ...
  iter3/
    ...
  final/             (symlink or copy of best iteration)
    encoder.pt
    detector.pt
    concept_store/
```

`concept_store/` is copied from exp128/final — not retrained in exp129 (out of scope).

## Stopping Criteria

1. `stone_natural_rate` improvement < 5% between consecutive iterations → STOP
2. `tree_natural_rate` drops below 60% → STOP (regression)
3. Max 5 iterations reached → STOP
4. Best iteration = highest `stone_natural_rate` without tree regression

## Integration with Demo

After exp129 completes:
- `server.py` CKPT_DIR → `demos/checkpoints/exp129`
- No changes to agent_loop.py, engine.py, or UI
- Demo uses same `run_chain()` + hardcoded chains
- Expected improvement: stone natural 4% → ≥30% means agent finds stone ~1/3 of time instead of ~never

## Non-Goals

- Pixel-only navigation (Stage 72)
- Coal/iron natural navigation (controlled env still required)
- Changes to ConceptStore, ChainGenerator, ReactiveCheck
- Changes to demo code (only checkpoint path)
- Online learning / replay buffer
