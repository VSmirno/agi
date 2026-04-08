# Stage 75: Per-Tile Visual Field — Agent Sees Entire Screen

**Date:** 2026-04-08
**Status:** Draft
**Depends on:** Stage 74 (homeostatic drives, spatial map, ConceptStore)

## Problem

Agent perceives only center tile (near_head, 2×2 center of 4×4 grid). Can't see trees 3 tiles away. Step-by-step diagnostic showed: agent walks PAST trees for 90 steps because they're adjacent but not in center detection.

Human sees the entire 9×7 screen. Agent should too.

## Insight: Vision ≠ Knowledge (IDEOLOGY.md)

CNN classifies (V1, innate). Agent names through interaction (world model, learned).

- CNN: "class_7 at position (2,1)" — doesn't know it's a tree
- Agent: does "do" near class_7 → gets wood → "class_7 = tree"
- CNN detects. Agent understands.

## Solution

Train per-tile classification head on single feature map positions. Agent sees full screen through classification, names objects through experience.

### Architecture

```
Frame 64×64 → CNN (4 conv layers) → feature map (256, 4, 4)

Current (near_head, broken for per-tile):
  center 2×2 features → flatten(1024) → Linear → 12 classes
  Only detects CENTER. Trained on 4 concatenated positions.

New (tile_head, per-position):
  EACH position feature → Linear(256→12) → 12 classes
  Detects ALL 16 positions. Trained on single positions.
```

### Training

Semantic map (`info["semantic"]`) as teacher:

```python
for each frame in training_data:
    feature_map = encoder(frame)  # (256, 4, 4)
    
    for gy in range(4):
        for gx in range(4):
            feature = feature_map[:, gy, gx]  # (256,)
            
            # GT: what tiles does this cell cover?
            # Map pixel region to world tiles → majority class
            gt_class = semantic_map_label_for_cell(gy, gx)
            
            loss += cross_entropy(tile_head(feature), gt_class)
```

Semantic map used ONLY during training. NOT during inference. Like a teacher showing flash cards — teaches, then steps back.

### Inference (Runtime)

```python
def perceive_full_screen(pixels, encoder):
    feature_map = encoder(pixels)  # (256, 4, 4)
    
    detections = []
    for gy in range(4):
        for gx in range(4):
            feature = feature_map[:, gy, gx]
            class_id = tile_head(feature).argmax()
            confidence = softmax(tile_head(feature))[class_id]
            detections.append((class_id, confidence, gy, gx))
    
    return detections  # 16 positions with class IDs
```

Agent receives: "class_3 at (0,1), class_7 at (1,2), ..."

Agent's ConceptStore maps class_id → object name through interaction:
- First encounter: class_7 is unknown
- "do" near class_7 → wood → ConceptStore: class_7 = "tree"
- Next time: sees class_7 → knows it's tree → navigates to it

### What Changes

| Before | After |
|--------|-------|
| near_head: center only (2×2) | tile_head: all 16 positions |
| near_head input: 1024 (4 cells concat) | tile_head input: 256 (1 cell) |
| Agent blind to adjacent tiles | Agent sees 9×7 tile area |
| Cosine matching / patch matching | Classification per position |
| Templates / prototypes | class_id → name mapping |

### What Stays

- HomeostaticTracker (body rates, drives)
- ConceptStore (world model, causal rules, planning)
- CrafterTextbook (knowledge from parent)
- Motor babbling / collision learning for NAMING
- Spatial map (now filled from full-screen detections)
- CNN encoder (frozen, same weights)

### Training Data

Use existing exp128 pipeline:
- ScenarioRunner collects labeled frames
- Each frame has semantic map GT
- For each frame × each grid position → (feature, GT_class) pair
- ~10K frames × 16 positions = ~160K training samples

### Class Mapping (Agent's Job)

tile_head outputs class_0...class_11. Agent doesn't know names. Mapping learned through experience:

```python
class_mapping: dict[int, str] = {}  # class_id → concept_name

# When agent interacts:
# "do" near class_7 → got wood → class_mapping[7] = "tree"
# collision with class_3 + damage → class_mapping[3] = "zombie"
# "do" near class_5 → drink restored → class_mapping[5] = "water"
```

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| Detection accuracy (all tiles) | 0% (only center) | ≥60% per tile |
| Wood collection (steps to 3 wood) | ~130 steps | ≤50 steps |
| Survival | 176 steps | ≥200 steps |
| Sword crafted | 0/500 | ≥50/500 |

## Limitations

**Per-tile classification is NOT how humans see.** Humans perceive the entire image holistically — objects emerge from the visual field, not from a grid of classified tiles. This stage uses tile-based classification as a practical step. Future work should explore:

- Object detection (bounding boxes, not grid cells)
- Attention-based perception (saliency maps)
- Holistic scene understanding (full-image features → object list)
- DAF/SKS oscillator-based perception (self-organizing object segmentation)

The tile grid is a simplification that works for Crafter (fixed sprite sizes, no scaling, no perspective). It will NOT generalize to environments with variable object sizes, overlapping objects, or continuous visual spaces.

## Risks

1. **Feature locality** — CNN features at position (0,0) may encode context from neighboring tiles (receptive field > 1 tile). Mitigation: 4 conv layers with stride=2 each → receptive field grows → features not purely local. But should still be dominated by center content.
2. **Class imbalance** — "grass" everywhere, "diamond" rare. Mitigation: balanced sampling during training.
3. **Semantic map as teacher** — introduces GT dependency during training (not inference). This is acceptable per IDEOLOGY: "CNN = V1, trainable module. Retrain in background."

## Non-Goals

- Holistic perception (future stage)
- DAF/SKS integration
- Object detection with bounding boxes
- Variable resolution / multi-scale
