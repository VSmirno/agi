# Stage 73: Autonomous Craft — Plan-Driven Crafting Through Experience

**Date:** 2026-04-07
**Status:** Draft
**Depends on:** Stage 72 (exp130 — perception pivot, motor babbling, experiential grounding)

## Problem

Stage 72 proved self-organized perception bootstrap works: motor babbling → outcome → grounding → navigation. Tree nav 60%, 3 concepts grounded (tree, water, cow). But agent cannot progress beyond gathering wood:

1. **No crafting** — agent never place_table or make_pickaxe because it doesn't know what "empty" looks like (can't navigate to placement site) and never tries craft actions.
2. **No zombie survival** — zombie not visually grounded, ReactiveCheck never fires, agent dies on contact.
3. **No verification** — causal confidence never updates because verify only runs on plan execution, which rarely happens.
4. **Stone/coal/iron unreachable** — requires pickaxe, which requires crafting chain.

## Goal

Agent autonomously executes craft chain (wood → table → pickaxe → stone) through textbook knowledge + experiential grounding. No scaffolding, no controlled env.

**Success criteria:**

| Gate | Metric | Threshold |
|------|--------|-----------|
| Tree nav | success rate, no GT | ≥50% (already 60%) |
| Stone nav | success rate, with craft chain | ≥20% |
| Concepts grounded | from pure experience | ≥5 |
| Survival with enemies | mean episode length | ≥200 |
| Verification | rules with confidence >0.5 | ≥3 |

## Architecture Changes

### 1. Bootstrap Perception: Empty Grounding From First Frame

The agent's first visual experience IS empty terrain — the background against which all objects are recognized (figure/ground separation).

```python
# On episode start, before any actions:
if concept_store.query_text("empty").visual is None:
    pixels = env.observe()
    z_real = encoder(pixels).z_real
    z_norm = F.normalize(z_real)
    concept_store.ground_visual("empty", z_norm)
```

One-time grounding per ConceptStore lifetime. After this, agent can:
- Recognize empty tiles via perceive()
- Navigate to empty for place_table
- Distinguish empty from objects

### 2. Zombie Grounding Through Damage

Textbook already registered "zombie" with `dangerous=true`. Agent knows the WORD and the DANGER, but not the FACE. Damage event gives the face.

```python
# Every step, check for unexplained damage:
health_before = inv_before.get("health", 9)
health_after = inv_after.get("health", 9)
food = inv_after.get("food", 0)
drink = inv_after.get("drink", 0)

if health_after < health_before:
    # Not starvation (food=0 or drink=0 causes gradual health loss)
    if food > 0 and drink > 0:
        # Damage from entity — ground as zombie
        concept = concept_store.query_text("zombie")
        if concept is not None and concept.visual is None:
            concept_store.ground_visual("zombie", z_real)
```

After grounding: ReactiveCheck recognizes zombie → flee/attack → survival improves.

### 3. Plan-Driven Crafting

Agent knows rules from textbook. When `select_goal` returns "wood_pickaxe" (wood≥2, no pickaxe), `store.plan("wood_pickaxe")` generates backward chain:

1. `do tree → wood` (already works)
2. `place table on empty requires wood:2` → navigate to "empty" (grounded from first frame) → `place_table` → outcome: lost wood:2 → **ground z_real as "table"**
3. `make wood_pickaxe near table requires wood:1` → navigate to "table" (just grounded) → `make_wood_pickaxe` → gained pickaxe

After pickaxe: `select_goal` shifts to "stone_item" → navigate + babble on stone terrain → grounding "stone" → mine stone.

**Craft babbling:** When inventory allows craft actions and agent is exploring, motor babbling also tries `place_table` / `make_*` (not just "do"). Probability proportional to ungrounded craft targets remaining. But priority is plan-driven execution from textbook — babble is for discovery of things NOT in textbook.

### 4. Universal Verification

Every successful action confirms a causal rule. No separate predict/verify pair needed:

```python
def verify_outcome(action, label, outcome, concept_store):
    """After any action with outcome, confirm the causal rule."""
    if label is None or outcome is None:
        return
    concept_store.verify(label, action, outcome)
```

Called from:
- Motor babble with outcome
- Plan step execution (probe success)
- Reactive action (attacked zombie)

Surprise: if action produces outcome not in textbook → `record_surprise()`. Only logged for now (rule discovery deferred per IDEOLOGY.md).

## What Changes

| Old (Stage 72) | New (Stage 73) |
|----------------|----------------|
| No empty grounding | Empty grounded from first frame |
| Zombie not recognized | Zombie grounded through damage |
| Babble = only "do" | Babble includes place/make when inventory allows |
| Verify only on plan probe | Verify on every successful action |
| select_goal: static wood drive | select_goal: inventory-dependent progression (already in Stage 72) |

## What Stays

- CNN encoder frozen (V1 eye)
- ConceptStore as world model
- Motor babbling with curiosity decay
- CrafterSpatialMap for navigation
- ReactiveCheck for danger reflex
- Textbook as teacher (rules given, vision learned)
- Three learning speeds (instant/gradual/background)
- EMA prototype refinement

## Continuous Learning Flow

```
Episode start:
  ground "empty" (if first time)
  ↓
Every step:
  perceive → concept + z_real
  update spatial map
  check damage → ground zombie (if first time)
  ↓
  reactive check (danger) → flee/attack
  ↓
  select_goal (drives) → plan (backward chain)
  ↓
  execute plan step:
    navigate → act → outcome
    ↓
    verify_outcome → confidence ±0.15
    on_action_outcome → grounding (one-shot/EMA)
  ↓
  OR explore:
    motor babble (do / place / make)
    outcome → verify + ground
```

## Expected Grounding Timeline

| Episode | Event | Concepts grounded |
|---------|-------|-------------------|
| 0 (start) | First frame | empty |
| 1-3 | Babble near tree | tree |
| 3-10 | Babble near water/cow | water, cow |
| 5-15 | place_table (plan-driven) | table |
| 10-20 | Zombie attack | zombie |
| 15-30 | Mine stone (after pickaxe) | stone |
| Total | | ≥7 concepts |

## Non-Goals

- Coal/iron/diamond mining (requires stone_pickaxe — next stage)
- DAF/SKS oscillator perception
- Rule discovery from surprise
- CNN retraining
- Multi-agent

## Risks

1. **Empty grounding noise** — first frame may have tree/water nearby. Mitigation: agent starts in open terrain most seeds; EMA refines.
2. **Zombie grounding timing** — agent may die before grounding. Mitigation: one frame is enough; damage detection runs every step.
3. **Craft chain failure** — place_table might fail (wrong terrain). Mitigation: retry at different empty position.
4. **Stone recognition** — frozen CNN may not distinguish stone from other terrain. Mitigation: EMA from multiple encounters; background CNN retrain deferred.
