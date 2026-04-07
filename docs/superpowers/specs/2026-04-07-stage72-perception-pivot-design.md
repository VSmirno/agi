# Stage 72: Perception Pivot — Reuniting Crafter Agent with SNKS Pipeline

**Date:** 2026-04-07
**Status:** Draft
**Depends on:** Stage 71 (exp128), Pipeline (runner.py), ConceptStore
**Replaces:** exp129 (iterative finetuning — wrong approach, backprop-centric)

## Problem

Stages 66-71 built a parallel system (CNN NearDetector + ConceptStore + ChainGenerator) disconnected from the core SNKS pipeline (DAF/SKS/GWS/Metacog/Hebbian). Navigation relies on ground truth semantic map (`use_semantic_nav=True`). NearDetector uses supervised backprop classification — not self-organization. No continuous learning.

## Goal

Reconnect Crafter agent to the SNKS pipeline. Replace supervised NearDetector with self-organized perception. Replace ground truth navigation with experience-based. Enable continuous learning from experience.

**Success criteria:**
- Tree navigation success ≥50% without semantic GT
- Stone navigation success ≥20% without semantic GT
- Agent visually grounds new concepts from experience (one-shot)
- Survival drives (food/drink/energy) work through planner, not hardcoded rules
- Demo looks like an agent that learns, not a scripted robot

## Architecture

```
Pixels (64×64)
    ↓
CNN encoder (frozen V1, from exp128) → z_real (2048)
    ↓
DAF (oscillators synchronize on z_real features)
    ↓
SKS detect → clusters = "objects in field of view"
    ↓
ConceptStore.query_visual(cluster_embedding) → concept matching
    ↓
GWS winner → "focus of attention" (highest salience)
    ↓
Metacog → prediction error? → surprise → update confidence / trigger grounding
    ↓
IntrinsicCost → survival drives (food/drink/energy as homeostatic cost)
    ↓
ConceptStore.plan() → backward chaining (goal from drives or resource gathering)
    ↓
CrafterSpatialMap → navigation from experience (no semantic GT)
    ↓
Action execution → outcome → verify → learn
```

## Components: What Changes

### 1. Perception: NearDetector → SKS + ConceptStore

**Remove:** `NearDetector.detect(pixels) → str` (supervised near_head classification)

**Replace with:** 
```python
def perceive(pixels, encoder, daf, concept_store):
    z_real = encoder(pixels)               # frozen CNN
    daf.inject(z_real)                      # inject into oscillator field
    daf.step(n_steps)                       # let oscillators synchronize
    clusters = detect_sks(daf)             # find synchronized groups
    
    # Match each cluster to known concepts
    for cluster in clusters:
        embedding = sks_embedder.embed(cluster)
        concept = concept_store.query_visual(embedding)
        # concept is None if nothing matches → unknown object
    
    # GWS selects winner (highest salience)
    winner = gws.compete(clusters)
    return winner_concept, all_concepts
```

SKS clusters self-organize — no labels needed. ConceptStore matching is cosine similarity against prototypes learned from experience.

**DAF scaling for Crafter:** DAF currently works on small grids (MiniGrid). For 2048-dim z_real, we don't run DAF on raw pixels — we run it on CNN features. The oscillator field size = embedding dim or a projection to manageable size (e.g., 256 oscillators on PCA-projected z_real). This keeps DAF tractable.

### 2. Grounding: Controlled Env → Experience

**Remove:** `GroundingSession` (controlled env, 5 samples per concept)

**Replace with:** experiential grounding on first successful interaction:

```python
def on_action_outcome(action, inv_before, inv_after, z_real, concept_store):
    label = outcome_labeler.label(action, inv_before, inv_after)
    if label is not None:
        concept = concept_store.query_text(label)  # "tree" from textbook
        if concept and concept.visual is None:
            # First encounter — one-shot grounding
            concept_store.ground_visual(label, z_real)
        elif concept and concept.visual is not None:
            # Seen before — update prototype (EMA)
            concept.visual = 0.9 * concept.visual + 0.1 * z_real
            concept.visual = F.normalize(concept.visual, dim=0)
```

Textbook provides names + causal rules. Visual prototypes come only from experience.

### 3. Navigation: Semantic GT → CrafterSpatialMap

**Remove:** `use_semantic_nav=True` from all chains. Remove `_find_target_semantic()` usage.

**Replace with:** `find_target_with_map()` using ConceptStore perception (not NearDetector):

```python
# On every step:
concept = perceive(pixels, encoder, daf, concept_store)
spatial_map.update(player_pos, concept.id if concept else "unknown")

# To navigate:
target_pos = spatial_map.find_nearest("tree", player_pos)
action = step_toward(player_pos, target_pos)
```

Agent builds cognitive map from its own perception. Navigation quality improves as perception improves (more accurate concepts in map).

### 4. Survival Drives: Hardcoded → Homeostatic

**Remove:** `ReactiveCheck.check_needs()` as decision-maker

**Replace with:** IntrinsicCost drives → ConceptStore planning:

```python
def select_goal(inventory, concept_store, intrinsic_cost):
    # Homeostatic drives (from IntrinsicCost)
    drives = {
        "food": max(0, 5 - inventory.get("food", 9)),     # urgency
        "drink": max(0, 5 - inventory.get("drink", 9)),
        "energy": max(0, 4 - inventory.get("energy", 9)),
    }
    
    # Resource gathering drive (lower priority)
    drives["resources"] = 2  # constant background drive
    
    # Highest drive wins (GWS-like competition)
    top_drive = max(drives, key=drives.get)
    
    if top_drive == "food":
        return concept_store.plan("restore_food")   # → seek cow → do
    elif top_drive == "drink":
        return concept_store.plan("restore_drink")  # → seek water → do
    elif top_drive == "energy":
        return [PlannedStep(action="sleep", ...)]
    else:
        return concept_store.plan(next_resource_goal)
```

Survival needs compete with resource gathering through drive strength. No hardcoded "if food < 4". The threshold is implicit in the drive function shape.

### 5. Reflexes: ReactiveCheck stays

Zombie detection → flee/attack. This IS a reflex, not planning. ReactiveCheck stays but uses ConceptStore perception instead of NearDetector:

```python
winner_concept = perceive(pixels, ...)
if winner_concept and winner_concept.attributes.get("dangerous"):
    # Reflex: immediate, interrupts everything
    if has_weapon(inventory):
        return "do"  # attack
    else:
        return flee(rng)  # run away
```

### 6. Prediction-Verification Loop

On every action:
```
1. PREDICT: concept_store.predict_before_action(near, action, inventory)
   → "I expect to get wood"

2. ACT: env.step(action)

3. VERIFY: concept_store.verify_after_action(prediction, actual_outcome)
   → confidence += 0.15 (correct) or -= 0.15 (wrong)

4. SURPRISE: if prediction was wrong and no rule explains it
   → concept_store.record_surprise(outcome, action, near)
   → metacog PE spike → potential new rule discovery
```

This already exists in ConceptStore. Just needs to be wired into the main loop.

## Continuous Learning Summary

| What | How | Speed | Trigger |
|------|-----|-------|---------|
| Visual grounding | ground_visual(z_real) | One-shot | First successful interaction |
| Prototype refinement | EMA: 0.9×old + 0.1×new | Every encounter | Recognized concept |
| Causal confidence | verify: ±0.15 | Every action | Action outcome observed |
| Danger learning | confidence "dangerous" | One-shot | Got hurt |
| Cognitive map | spatial_map.update | Every step | Always |
| Surprise → new facts | record_surprise | Instant | Prediction error |
| CNN retrain (V1) | Replay buffer, background | Rare | Perception quality degrades |

## Curriculum (env settings, not training phases)

Curriculum is about what the agent faces, not when we retrain:

| Phase | Enemies | Objects | Duration |
|-------|---------|---------|----------|
| 1 | OFF | tree only (start in forest) | Until tree nav ≥50% |
| 2 | OFF | tree + stone | Until stone nav ≥20% |
| 3 | OFF | tree + stone + coal | Until coal grounded |
| 4 | ON | all | Continuous |

Transitions based on performance metrics, not fixed iteration counts. Agent stays in each phase until competent.

## What Stays From Stage 71

- **CNN encoder** (frozen, as V1 — feature extraction)
- **ConceptStore** (world model — concepts, rules, confidence, visual prototypes)
- **CrafterTextbook** (initial rules from "teacher")
- **ChainGenerator** (backward chaining planner — uses ConceptStore rules)
- **OutcomeLabeler** (detects what happened from inventory delta)
- **CrafterSpatialMap** (cognitive map from experience)

## What Changes

| Old | New |
|-----|-----|
| NearDetector (supervised near_head) | SKS clusters + ConceptStore.query_visual() |
| use_semantic_nav=True (GT map) | CrafterSpatialMap only (from experience) |
| GroundingSession (controlled env) | Experiential grounding (one-shot from action outcome) |
| ReactiveCheck as planner | IntrinsicCost drives → ConceptStore.plan() |
| ReactiveCheck flee/attack | Stays as reflex (but uses new perception) |
| Batch training phases | Continuous learning (verify, ground, surprise) |
| ScenarioRunner blocking chains | Agent loop: perceive → decide → act → learn (tick-based) |

## What Gets Removed

- `NearDetector` class (or repurposed as fallback)
- `near_head` in CNNEncoder (not used for perception)
- `_find_target_semantic()` (no GT navigation)
- `GroundingSession` (no controlled env grounding)
- `use_semantic_nav` flag (always False)
- Batch finetuning (exp129 approach)

## Demo Integration

The demo (FastAPI + WS) wraps this new agent loop. Every `perceive → decide → act → learn` tick updates the snapshot. UI shows:
- What agent sees (SKS clusters → concepts)
- What agent wants (drive strengths)
- What agent plans (ConceptStore.plan chain)
- What agent learned (new groundings, confidence changes, surprises)

## Non-Goals

- Replacing CNN encoder with DAF for raw pixel processing
- Multi-agent
- Language generation
- Transfer to other environments

## Risks

1. **DAF scaling:** 2048-dim oscillator field may be too large. Mitigation: PCA projection to 256 dims.
2. **SKS noise:** CNN features may not produce clean clusters. Mitigation: tune coupling strength, try different coherence measures.
3. **One-shot grounding noise:** First z_real for "tree" may be atypical. Mitigation: EMA update on subsequent encounters.
4. **Navigation cold start:** Empty spatial map = blind exploration. Mitigation: random walk phase at episode start fills map quickly.
