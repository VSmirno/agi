# Stage 72: Perception Pivot — Self-Organized Perception + Continuous Learning

**Date:** 2026-04-07
**Status:** Draft
**Depends on:** Stage 71 (exp128), ConceptStore, CrafterSpatialMap
**Replaces:** exp129 (iterative finetuning — wrong approach, backprop-centric)

## Problem

Stages 66-71 built a system that relies on:
1. **NearDetector** — supervised backprop classification (near_head). Not self-organization.
2. **Semantic GT navigation** — `use_semantic_nav=True` reads ground truth map. Cheating.
3. **Controlled env grounding** — visual prototypes from lab conditions, not real experience.
4. **Batch training** — collect → train → deploy. No continuous learning.

## Goal

Replace supervised perception with self-organized concept matching. Remove GT navigation. Learn from experience continuously.

**Success criteria:**
- Tree navigation success ≥50% without semantic GT
- Stone navigation success ≥20% without semantic GT
- Agent visually grounds new concepts from experience (one-shot)
- Survival drives (food/drink/energy) work through planner, not hardcoded rules
- Demo shows an agent that learns, not a scripted robot

## Architecture

```
Pixels (64×64)
    ↓
CNN encoder (frozen, from exp128) → z_real (2048)
    ↓
ConceptStore.query_visual(z_real) → best matching concept (cosine sim)
    ↓                                 + similarity score
    ↓                                 + None if below threshold (unknown)
    ↓
Drives: select_goal(survival_needs, resource_progression)
    ↓
ConceptStore.plan(goal) → backward chaining
    ↓
CrafterSpatialMap → navigate from experience (no GT)
    ↓
Action → outcome → verify → learn (continuous)
```

No DAF/SKS/GWS in this stage. ConceptStore.query_visual() replaces both NearDetector and SKS — cosine similarity matching against prototypes learned from experience. This IS self-organized: no labels, no backprop, prototypes from agent's own interactions.

DAF/SKS integration deferred — requires CNN→oscillator projection layer (undefined). ConceptStore achieves the same functional goal (concept recognition from features) without the oscillator intermediary.

## Components: What Changes

### 1. Perception: NearDetector → ConceptStore.query_visual()

**Remove:** `NearDetector.detect(pixels) → str` (supervised near_head)

**Replace with:**
```python
def perceive(pixels, encoder, concept_store, min_similarity=0.5):
    with torch.no_grad():
        z_real = encoder(pixels).z_real        # frozen CNN
    concept, similarity = concept_store.query_visual_scored(z_real)
    if similarity < min_similarity:
        return None, z_real  # unknown object
    return concept, z_real
```

New method `query_visual_scored()` — same as existing `query_visual()` but returns similarity score. Enables "I don't recognize this" when score is low.

### 2. Grounding: Controlled Env → Experience

**Remove:** `GroundingSession` (controlled env, 5 samples per concept)

**Replace with:** experiential grounding on first successful interaction:
```python
def on_action_outcome(action, inv_before, inv_after, z_real, concept_store):
    label = outcome_labeler.label(action, inv_before, inv_after)
    if label is None:
        return  # action had no recognizable effect
    
    concept = concept_store.query_text(label)  # "tree" from textbook
    if concept is None:
        return  # unknown label
    
    if concept.visual is None:
        # First encounter — one-shot grounding
        z_norm = F.normalize(z_real.unsqueeze(0), dim=1).squeeze(0)
        concept_store.ground_visual(label, z_norm)
    else:
        # Seen before — update prototype (EMA)
        z_norm = F.normalize(z_real.unsqueeze(0), dim=1).squeeze(0)
        concept.visual = F.normalize(
            (0.9 * concept.visual + 0.1 * z_norm).unsqueeze(0), dim=1
        ).squeeze(0)
```

Textbook provides names + causal rules ("do tree gives wood").
Visual prototypes come only from experience — what the agent saw when it got wood.

### 3. Navigation: Semantic GT → CrafterSpatialMap

**Remove:** `use_semantic_nav=True` everywhere. Remove `_find_target_semantic()` calls.

**Replace with:** `find_target_with_map()` using ConceptStore perception:
```python
# Every step:
concept, z_real = perceive(pixels, encoder, concept_store)
concept_id = concept.id if concept else "unknown"
spatial_map.update(player_pos, concept_id)

# To navigate:
target_pos = spatial_map.find_nearest("tree", player_pos)
# If not in map — explore (random walk fills map)
```

Agent builds cognitive map from its own perception. Map quality improves as prototypes refine with experience.

### 4. Survival Drives: Hardcoded ReactiveCheck → Drive-Based Planning

**Remove:** `ReactiveCheck.check_needs()` as decision-maker

**Replace with:** drive competition → ConceptStore planning:
```python
def select_goal(inventory, concept_store):
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    
    # Drive strengths (higher = more urgent)
    drives = {
        "restore_food":   max(0, 5 - food) * 2,     # urgent when low
        "restore_drink":  max(0, 5 - drink) * 2,
        "restore_energy": max(0, 4 - energy) * 2,
        "wood":           1,                          # background resource drive
        "stone_item":     0.5,
    }
    
    # Winner = highest drive
    goal = max(drives, key=drives.get)
    if drives[goal] <= 0:
        goal = "wood"  # default: gather resources
    
    return concept_store.plan(goal)
```

Textbook already has rules: "do cow restores food", "do water restores drink", "sleep restores energy". ConceptStore.plan() generates chains from these rules. No separate ReactiveCheck.check_needs().

### 5. Reflexes: ReactiveCheck.check() stays for danger only

Zombie nearby → flee/attack. This is a reflex, not planning.

```python
concept, z_real = perceive(pixels, encoder, concept_store)
if concept and concept.attributes.get("dangerous"):
    if has_weapon(inventory):
        return "do"    # attack
    else:
        return flee()  # run away
# Otherwise: continue plan
```

ReactiveCheck.check() (danger only) stays. ReactiveCheck.check_needs() removed (replaced by drive-based planning).

### 6. Prediction-Verification Loop (already exists, just wire in)

```
1. PREDICT: concept_store.predict_before_action(near, action, inventory)
2. ACT: env.step(action)
3. VERIFY: concept_store.verify_after_action(prediction, actual)
   → confidence ±0.15
4. SURPRISE: if no rule predicted this → record_surprise()
```

All methods exist in ConceptStore. Wire into agent loop.

### 7. CNN Retraining (background, by trigger)

CNN encoder frozen by default. If perception quality degrades (average best-match similarity drops below threshold over N steps), trigger background retrain on replay buffer.

Not part of the main loop. Separate background thread. Same as "V1 adapts slowly" — rare, asynchronous.

## Continuous Learning Summary

| What | How | Speed | Trigger |
|------|-----|-------|---------|
| Visual grounding | ground_visual(z_real) | One-shot | First successful interaction |
| Prototype refinement | EMA: 0.9×old + 0.1×new | Every encounter | Recognized concept |
| Causal confidence | verify: ±0.15 | Every action | Action outcome |
| Danger learning | "dangerous" confidence | One-shot | Got hurt |
| Cognitive map | spatial_map.update | Every step | Always |
| Surprise | record_surprise | Instant | Unpredicted outcome |
| CNN retrain | Replay buffer, background | Rare | Perception quality drop |

## Curriculum (env settings)

| Phase | Enemies | Target Objects | Gate to advance |
|-------|---------|---------------|-----------------|
| 1 | OFF | tree (start in forest) | tree nav ≥50% |
| 2 | OFF | tree + stone | stone nav ≥20% |
| 3 | OFF | tree + stone + coal | coal grounded from experience |
| 4 | ON | all | survival episode length ≥200 |

Transitions based on metrics. Agent stays until competent.

## What Stays From Stage 71

- **CNN encoder** (frozen V1 — feature extraction only)
- **ConceptStore** (world model — concepts, rules, confidence, visual prototypes)
- **CrafterTextbook** (initial rules from "teacher")
- **ChainGenerator** (backward chaining from ConceptStore rules)
- **OutcomeLabeler** (what happened from inventory delta)
- **CrafterSpatialMap** (cognitive map)
- **ReactiveCheck.check()** (danger reflex only)

## What Changes

| Old | New |
|-----|-----|
| NearDetector (supervised near_head) | ConceptStore.query_visual() (cosine sim) |
| use_semantic_nav=True (GT map) | CrafterSpatialMap only (from experience) |
| GroundingSession (controlled env) | Experiential grounding (one-shot) |
| ReactiveCheck.check_needs() | Drive-based goal selection → ConceptStore.plan() |
| Batch training phases | Continuous learning (verify, ground, EMA, surprise) |
| ScenarioRunner blocking chains | Agent loop: perceive → decide → act → learn |

## What Gets Removed

- `NearDetector` class usage (keep code for fallback)
- `near_head` usage in perception
- `_find_target_semantic()` usage
- `GroundingSession` usage
- `use_semantic_nav=True` in all chains
- `ReactiveCheck.check_needs()` usage
- Batch finetuning approach (exp129)

## What's Deferred (future stages)

- **DAF/SKS integration** — requires CNN→oscillator projection. When solved: SKS clusters replace ConceptStore.query_visual() for richer multi-object perception.
- **GWS attention** — currently drive competition is max(). When DAF/SKS works: GWS winner-take-all replaces it.
- **IntrinsicCost** — currently drive function is manual. When DAF works: homeostatic oscillator firing rate drives.
- **Metacog surprise** — currently ConceptStore.record_surprise(). When pipeline integrated: MetacogMonitor PE tracking.

## Demo Integration

Demo agent loop becomes: perceive → select_goal → plan → navigate → act → learn. Every tick updates snapshot. UI shows:
- What agent sees (ConceptStore match + similarity)
- What agent wants (drive strengths)
- What agent plans (chain from ConceptStore.plan())
- What agent learned (grounding events, confidence changes, surprises)

## Non-Goals

- DAF/SKS oscillator perception (deferred)
- Replacing CNN with non-backprop encoder
- Multi-agent
- Language generation

## Risks

1. **One-shot grounding noise** — first z_real may be atypical. Mitigation: EMA refinement.
2. **Navigation cold start** — empty spatial map. Mitigation: random walk phase fills map.
3. **Cosine similarity threshold** — too high = never matches, too low = wrong matches. Mitigation: tune on controlled data, start at 0.5.
4. **CNN features insufficient** — frozen CNN from exp128 may not distinguish stone from coal. Mitigation: background retrain trigger.
