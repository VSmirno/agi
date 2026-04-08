# Self-Supervised Metric Learning — CNN Learns From Agent's Experience

**Date:** 2026-04-08
**Status:** Draft
**Depends on:** Stage 74 (homeostatic agent, spatial perception, experiential grounding)

## Problem

CNN features trained for classification (cross-entropy on near_head) don't work for cosine matching. Inter-class cosine similarity 0.55-0.82 — agent can't distinguish tree from grass.

Exhaustively tested: 256/512 channels, 4×4/8×8 grid, SupCon on z_real, sandbox curriculum, 500 episodes. Nothing helps. Root cause: training objective (classification) ≠ usage (cosine similarity matching).

Features are "sorted by surname" (classification boundaries) but we need them "sorted by address" (metric distances).

## Solution

Agent collects observations through verified interactions (babble → outcome). Periodically fine-tunes CNN features via contrastive loss on its OWN experience. Self-supervised, generalizable to any environment.

### 1. Observation Buffer in ConceptStore

Store K=20 raw center_features per concept from verified interactions:

```python
@dataclass
class Concept:
    id: str
    visual: torch.Tensor | None = None          # EMA prototype (for matching)
    observations: list[torch.Tensor] = field(default_factory=list)  # raw features
    # ... existing fields ...

MAX_OBSERVATIONS = 20

def add_observation(self, feature: torch.Tensor) -> None:
    """Store raw feature from verified interaction."""
    self.observations.append(feature.detach().clone())
    if len(self.observations) > MAX_OBSERVATIONS:
        self.observations.pop(0)  # FIFO
```

Observations added ONLY on verified action outcome (babble → got wood → store center_feature as "tree" observation). NOT from visual perception — from action.

### 2. Background CNN Retrain

Trigger: observation buffer has ≥5 concepts with ≥10 observations each (≥50 total samples).

```python
def retrain_features(encoder, concept_store):
    """Fine-tune CNN for cosine matching using agent's experience.
    
    SupCon loss on per-position features:
    - Positive pairs: two observations of same concept
    - Negative pairs: observations of different concepts
    
    Only fine-tunes last conv layer + proj. Freezes early layers (low-level
    features are general, high-level features need adaptation).
    """
    # Collect training data from observation buffers
    features = []
    labels = []
    label_idx = 0
    for concept in concept_store.concepts.values():
        if len(concept.observations) < 5:
            continue
        for obs in concept.observations:
            features.append(obs)
            labels.append(label_idx)
        label_idx += 1
    
    if label_idx < 3:  # need ≥3 concepts
        return False
    
    features = torch.stack(features)
    labels = torch.tensor(labels)
    
    # SupCon loss fine-tuning
    # Freeze early conv layers, train only last conv + proj
    for param in encoder.conv[:-3].parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=1e-4,
    )
    
    for epoch in range(50):
        loss = supcon_loss(F.normalize(features, dim=1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Unfreeze all
    for param in encoder.parameters():
        param.requires_grad = True
    
    return True
```

### 3. When to Retrain

Not on a timer — on a TRIGGER:

```python
def should_retrain(concept_store) -> bool:
    """Retrain when enough verified observations accumulated."""
    concepts_with_obs = sum(
        1 for c in concept_store.concepts.values()
        if len(c.observations) >= 10
    )
    return concepts_with_obs >= 5
```

Retrain fires once, early in agent's life (~50-100 episodes). After retrain, features are metric — prototypes and matching work correctly. Agent might retrain again later if new concepts are discovered.

### 4. What Changes After Retrain

- Same cosine matching, same prototypes, same threshold
- But features now optimized for cosine distance
- Inter-class similarity should drop from 0.55-0.82 to <0.3
- Intra-class similarity stays ~0.99
- Perception accuracy dramatically improves
- Spatial map entries become reliable
- Agent finds objects in ~20 steps instead of ~80

### 5. Integration With Existing Architecture

```
Episode loop:
  perceive_field() → visual field (unchanged)
  on_action_outcome() → grounding + ADD OBSERVATION (new)
  select_goal() → drives (unchanged)
  plan execution → (unchanged)
  
Background (triggered):
  should_retrain()? → retrain_features() → CNN improved
  
After retrain:
  Existing prototypes still work (same feature space, just reorganized)
  But NEW prototypes from post-retrain observations will be better
  Old prototypes gradually replaced through EMA updates
```

### 6. What This Achieves

The virtuous cycle:
1. Agent babbles → discovers objects → stores observations
2. Enough observations → retrain CNN → features become metric
3. Better features → perception works → agent finds objects faster
4. Faster objects → more observations → even better features

This is V1 adaptation through experience — biologically grounded, self-supervised, generalizable.

## What Stays

- HomeostaticTracker, drives, curiosity, preparation
- ConceptStore world model
- Motor babbling for discovery
- EMA prototypes for matching
- Spatial map for navigation
- Textbook for causal rules

## What Changes

| Component | Before | After |
|-----------|--------|-------|
| Concept.observations | not stored | K=20 raw features per concept |
| on_action_outcome | grounds prototype only | + stores raw observation |
| CNN training | classification only | + self-supervised SupCon from experience |
| Retrain trigger | manual | automatic when buffer full |

## Success Criteria

| Metric | Before (classification CNN) | After (metric CNN) | Gate |
|--------|------|------|------|
| Inter-class cosine sim | 0.55-0.82 | <0.3 | Required |
| Map stale rate | 100% | <20% | Required |
| Tree nav | 54% | ≥70% | ≥50% |
| Survival | 173 | ≥200 | ≥200 |
| Sword crafted | 0/500 | ≥50/500 | ≥10% |

## Risks

1. **Few observations per concept** — early retrain on 50 samples may overfit. Mitigation: freeze early layers, train only last conv + proj. Short training (50 epochs).
2. **Feature space shift breaks old prototypes** — after retrain, existing EMA prototypes are in OLD feature space. Mitigation: prototypes updated through EMA on next encounters. First ~10 episodes after retrain will have lower accuracy while prototypes refresh.
3. **Retrain on biased data** — agent sees "tree" 100 times but "stone" 5 times. Mitigation: balance sampling during SupCon (equal samples per concept).

## Non-Goals

- DAF/SKS oscillator perception
- End-to-end RL
- Online learning (continuous gradient updates)
- Near_head replacement (stays for backward compat)
