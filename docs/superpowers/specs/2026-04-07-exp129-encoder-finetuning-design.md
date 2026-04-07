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
    TREE_CHAIN × 80 seeds (CrafterPixelEnv, standard, enemies ON by default)
    STONE_CHAIN × 80 seeds (CrafterPixelEnv, standard, enemies ON by default)
    Using _run_chain_batch_generic (calls runner.run_chain() internally)
    OutcomeLabeler for labeling (no semantic GT)
    → natural_frames, stone_success_rate, tree_success_rate
    If stone natural yields 0 frames: skip stone natural for this iteration

  Phase B: Mix datasets (per-class)
    For tree/stone (classes with natural data available):
      controlled_ratio = 1 - i/5
      natural_ratio = i/5
    For coal/iron/empty (no natural collection):
      ALWAYS 100% controlled — these classes never lose representation
    Subsample per-class to match ratio, then _balance_classes(max_ratio=4.0)

  Phase C: Finetune encoder
    NEW function finetune_encoder() — does NOT reuse phase2_train_outcome_encoder
    Accepts: existing encoder (warm start), lr, epochs
    Warm start: encoder.load_state_dict(prev_iteration) before training
    LR: 3e-4 (base 1e-3 × 0.3)
    Epochs: 50 per iteration (vs 150 for initial training)
    Creates PredictiveTrainer(encoder, predictor, lr=3e-4, ...) with the loaded encoder
    Loss: JEPA prediction + variance + contrastive + near (same as exp128)

  Phase D: Evaluate + checkpoint
    Run STONE_CHAIN × 50 seeds (eval only, separate seed range)
    Run TREE_CHAIN × 50 seeds (regression check)
    Metrics: stone_natural_rate, tree_natural_rate
    Save checkpoint: demos/checkpoints/exp129/iter{i}/
      encoder.pt, detector.pt
    concept_store: symlinked from exp128/final (saved once, not per iteration)
    Log: iteration, stone_rate, tree_rate, absolute delta vs previous
    Stop if absolute delta(stone_rate) < 5% (e.g. 30%→34% = 4% = STOP)

Save best iteration (highest stone_rate without tree regression) as:
  demos/checkpoints/exp129/final/
```

## Data Collection Details

**Natural collection (Phase A):**
- Reuses `_run_chain_batch_generic` from exp128 (calls `runner.run_chain()` internally)
- TREE_CHAIN: `ScenarioStep("tree", "do", "tree", repeat=5, use_semantic_nav=True)`
- STONE_CHAIN: tree→table→pickaxe→stone (4 steps, semantic nav)
- 80 seeds each, seed_base = 40000 + i*1000 (avoid overlap with exp128 seeds)
- Enemies ON (default CrafterPixelEnv behavior)
- Success = OutcomeLabeler returns expected near_label
- If Phase A yields 0 stone frames: log warning, use 100% controlled for stone

**Controlled collection (cached at start):**
- Same as exp128 phase 2: `_run_controlled_batch` for stone/coal/iron
- `_run_controlled_items_batch` for empty/table
- `_collect_empty_walk_frames` for empty walk
- `CrafterControlledEnv.reset_near()` with `no_enemies=True`
- Collected ONCE at start, reused every iteration unchanged

## Finetuning Implementation

**New function `finetune_encoder()`** (cannot reuse `phase2_train_outcome_encoder` — it creates fresh encoder, no warm start, no lr param):

```python
def finetune_encoder(
    encoder: CNNEncoder,       # existing encoder (warm start)
    dataset: dict,             # {pixels, near_labels} mixed
    epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cuda",
) -> tuple[CNNEncoder, NearDetector]:
    """Finetune existing encoder on mixed dataset.
    
    Warm start: uses passed encoder weights, not fresh init.
    """
    predictor = JEPAPredictor(embed_dim=encoder.embed_dim)
    trainer = PredictiveTrainer(
        encoder, predictor,
        contrastive_weight=0.3,
        near_weight=3.0,
        lr=lr,
        device=device,
    )
    # trainer uses the PASSED encoder, not a new one
    trainer.train_full(pixels_t, pixels_t1, actions,
                       near_labels=near_labels,
                       epochs=epochs, batch_size=64, log_every=10)
    encoder.eval()
    detector = NearDetector(encoder)
    return encoder, detector
```

Note: `PredictiveTrainer.__init__` accepts `lr` parameter (confirmed in source). The key difference from `phase2_train_outcome_encoder` is that we pass in an existing encoder instead of creating `CNNEncoder()`.

**Warm start detail:** The encoder's `state_dict` persists across iterations via the checkpoint load. Each iteration starts from the previous iteration's weights, not from exp128's initial weights.

## Checkpoints

```
demos/checkpoints/exp129/
  iter1/encoder.pt, detector.pt
  iter2/encoder.pt, detector.pt
  ...
  concept_store/          (symlink or copy from exp128/final, once)
  final/
    encoder.pt            (copy of best iteration)
    detector.pt
    concept_store/        (symlink to above)
```

Saved after Phase D of each iteration. concept_store unchanged — copied once from exp128.

## Stopping Criteria

1. Absolute `stone_natural_rate` improvement < 5% between consecutive iterations → STOP
2. `tree_natural_rate` drops below 60% → STOP (regression)
3. Max 5 iterations → STOP
4. Best iteration = highest `stone_natural_rate` where `tree_natural_rate` ≥ 60%

## Integration with Demo

After exp129:
- `server.py` CKPT_DIR → `demos/checkpoints/exp129`
- No changes to agent_loop.py, engine.py, or UI
- Demo uses same `run_chain()` + hardcoded chains (TREE/STONE/IRON_CHAIN)

## Non-Goals

- Pixel-only navigation (Stage 72)
- Coal/iron natural navigation (controlled env still required)
- Changes to ConceptStore, ChainGenerator, ReactiveCheck
- Changes to demo code (only checkpoint path)
- Online learning / replay buffer
