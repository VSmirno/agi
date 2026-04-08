# Stage 74: Homeostatic Agent — Goal-Free Self-Organizing Behavior

**Date:** 2026-04-07
**Status:** Implemented
**Depends on:** Stage 73 (spatial perception, experiential grounding, verification)

## Problem

Stages 72-73 proved perception bootstrap and craft chain work, but agent behavior is controlled by hardcoded drives:

- `max(0, 5 - food) * 2.0` — arbitrary multiplier
- `if has_sword == 0: drives["wood_sword"] = 3.0` — hardcoded strategy
- `if took_damage: attack/flee` — hardcoded reflex

Result: agent doesn't learn strategy from experience. 200 episodes — survival doesn't improve. 80% deaths from zombie, but agent doesn't learn to craft sword first.

## Insight

The agent should have NO explicit goal. Its "goal" is its body — homeostatic variables that must stay in range. All behavior emerges from body + world model + experience.

Biology analogy: an organism doesn't receive a "mission". It has DNA-programmed homeostasis. Hunger, pain, thirst are not learned — they're built into the body. What IS learned: how to satisfy them.

## Architecture

### What's hardcoded (the "DNA")

Only the body and the curiosity instinct:

```python
# Body — physical needs
HOMEOSTATIC_VARIABLES = {
    "health": {"min": 0, "max": 9, "death_at": 0},
    "food":   {"min": 0, "max": 9, "affects": "health"},  # food=0 → health drops
    "drink":  {"min": 0, "max": 9, "affects": "health"},  # drink=0 → health drops
    "energy": {"min": 0, "max": 9, "affects": "speed"},   # energy=0 → slower
}

# Curiosity — cognitive need (as biological as hunger)
# When body is fine, curiosity dominates → agent explores
CURIOSITY_SOURCES = {
    "ungrounded_concepts",   # concepts in textbook without visual prototype
    "low_confidence_rules",  # causal rules with confidence < threshold
    "unvisited_area",        # spatial map coverage
}
```

These are the ONLY things that are "programmed" — the body and the drive to understand. Everything else is derived.

The full set of innate drives:

| Drive | Source | Biology | When dominant |
|-------|--------|---------|---------------|
| health | rate of change | pain | zombie attacking |
| food | rate of change | hunger | food depleting |
| drink | rate of change | thirst | drink depleting |
| energy | rate of change | fatigue | energy low |
| **curiosity** | model incompleteness | curiosity | body is fine |

Curiosity IS a biological drive — not a hack for exploration. When all physical needs are met, the organism explores. Babies don't sit still when fed and safe — they crawl, grab, taste, experiment. This is DNA-level programming.

### What's learned (the "experience")

#### 1. Urgency from observation

Agent tracks rate of change of each variable:

```python
class HomeostaticTracker:
    """Tracks how fast each variable changes and what causes it."""
    
    # Running average: delta per step for each variable
    rates: dict[str, float]  # {"health": -0.02, "food": -0.04, ...}
    
    # Conditional rates: delta when specific concept is visible
    conditional_rates: dict[tuple[str, str], float]
    # ("zombie", "health"): -2.0  — "health drops 2/step when zombie visible"
    # ("cow", "food"): +0.5       — "food rises when near cow and do"
    
    def update(self, variable: str, delta: float, visible_concepts: set[str]):
        """Called every step with observed changes."""
        ...
```

No formulas. Just observation: "when zombie is visible, my health drops fast."

#### 2. Drive = urgency × deficit

```python
def compute_drive(variable: str, current_value: float, rate: float) -> float:
    """How urgent is this variable's need?
    
    drive = how fast am I approaching death from this variable.
    """
    if rate >= 0:
        return 0.0  # variable is stable or rising — no urgency
    
    # Time until critical: current_value / abs(rate)
    steps_until_zero = current_value / abs(rate) if rate < 0 else float("inf")
    
    # Urgency = inverse of time until crisis
    return 1.0 / max(1.0, steps_until_zero)
```

Agent with food=8, rate=-0.04: `steps_until_zero = 200`, urgency = 0.005 (low).
Agent with health=3, rate=-2.0 (zombie!): `steps_until_zero = 1.5`, urgency = 0.67 (HIGH).

No magic numbers. Pure physics of the body.

#### 3. Curiosity drive

```python
def compute_curiosity(concept_store, spatial_map) -> float:
    """How incomplete is the agent's world model?
    
    High when lots of unknowns. Low when world is well understood.
    This is biological curiosity — the drive to know.
    """
    total_concepts = len(concept_store.concepts)
    grounded = sum(1 for c in concept_store.concepts.values() if c.visual is not None)
    
    # Fraction of world still unknown
    visual_gap = 1.0 - (grounded / max(1, total_concepts))
    
    # Average rule confidence (low = uncertain about causal model)
    confidences = [
        link.confidence
        for c in concept_store.concepts.values()
        for link in c.causal_links
    ]
    mean_confidence = sum(confidences) / max(1, len(confidences))
    knowledge_gap = 1.0 - mean_confidence
    
    # Spatial coverage
    visited = spatial_map.n_visited
    map_gap = 1.0 / max(1, visited / 50)  # decays as map fills
    
    return (visual_gap + knowledge_gap + map_gap) / 3.0
```

Curiosity naturally decays as the agent learns. A fully grounded, high-confidence, well-explored agent has low curiosity — it shifts to exploitation.

#### 4. Goal selection = highest urgency → world model plan

```python
def select_goal(tracker, inventory, concept_store, visual_field, spatial_map):
    # Compute urgency for each homeostatic variable
    urgencies = {}
    for var in HOMEOSTATIC_VARIABLES:
        value = inventory.get(var, 9)
        rate = tracker.rates.get(var, 0)
        urgencies[var] = compute_drive(var, value, rate)
    
    # Add curiosity as a drive
    urgencies["curiosity"] = compute_curiosity(concept_store, spatial_map)
    
    # Most urgent need
    critical = max(urgencies, key=urgencies.get)
    
    if critical == "curiosity":
        # Body is fine → explore, babble, discover
        return "explore", []
    
    # Ask world model: "what fixes this variable?"
    # Two strategies, tried in order:
    
    # Strategy 1: "what restores this variable?" (direct fix)
    # Textbook has: "do cow restores food", "do water restores drink"
    plan = concept_store.plan(f"restore_{critical}")
    if plan:
        return f"restore_{critical}", plan
    
    # Strategy 2: "what causes this variable to drop? remove that cause."
    # Use conditional_rates: find concept with most negative rate for this variable
    worst_cause = None
    worst_rate = 0.0
    for (concept_id, var), rate in tracker.conditional_rates.items():
        if var == critical and rate < worst_rate:
            worst_rate = rate
            worst_cause = concept_id
    
    if worst_cause:
        # Find how to neutralize the cause
        # e.g. worst_cause="zombie" for health → plan("kill_zombie")
        cause_concept = concept_store.query_text(worst_cause)
        if cause_concept:
            for link in cause_concept.causal_links:
                if link.action == "do":  # "do zombie with sword → kill_zombie"
                    plan = concept_store.plan(link.result)
                    if plan:
                        return link.result, plan
    
    # Nothing works yet → curiosity (explore, discover solutions)
    return "explore", []
```

**"I need a sword"** emerges naturally:
1. Tracker observes: conditional_rate("zombie", "health") = -2.0
2. Urgency for health = very high
3. Strategy 1: plan("restore_health") → no rule → fail
4. Strategy 2: worst_cause for health = "zombie" → zombie has link "do → kill_zombie" → plan("kill_zombie") → backward chain → "need sword" → "need table" → "need wood"
5. Agent crafts sword NOT because programmer said so, but because world model traced the causal chain from "my health is dropping" to "zombie is the cause" to "sword removes zombie"

#### 4. Conditional rate learning (the reflex mechanism)

```python
def update_rates(tracker, inv_before, inv_after, visible_concepts):
    """Every step: observe what changed and what was visible."""
    for var in HOMEOSTATIC_VARIABLES:
        delta = inv_after.get(var, 0) - inv_before.get(var, 0)
        
        # Update unconditional rate (background)
        tracker.rates[var] = ema(tracker.rates[var], delta)
        
        # Update conditional rates (what's nearby affects what?)
        for concept in visible_concepts:
            key = (concept, var)
            tracker.conditional_rates[key] = ema(
                tracker.conditional_rates.get(key, 0), delta
            )
```

After 50 episodes, conditional_rates contains:
- `("zombie", "health"): -1.8` — zombie causes health loss
- `("cow", "food"): +0.3` — cow near + do restores food
- `("water", "drink"): +0.4` — water near + do restores drink
- `("tree", "health"): 0.0` — tree is neutral for health

These are LEARNED associations, not programmed rules.

### What's removed

Everything from select_goal that was hardcoded:
- `drives["wood_sword"] = 3.0` — gone
- `drives["wood_pickaxe"] = 1.5` — gone
- `drives["restore_food"] = max(0, 5 - food) * 2.0` — gone
- `if danger_visible and has_sword == 0` — gone
- All strategy logic — gone

### What stays

- CNN encoder (frozen V1 eye)
- ConceptStore (world model — concepts, rules, confidence)
- CrafterTextbook (knowledge from parent)
- perceive_field() (spatial visual field)
- Motor babbling (curiosity-driven exploration)
- Experiential grounding (one-shot + EMA)
- Verification (confidence from outcomes)
- CrafterSpatialMap (cognitive map)

### Forward planning through world model (deferred)

The urgency + conditional_rates + Strategy 2 (remove cause) already gives the agent the ability to derive sword-crafting from zombie encounters. Full forward simulation ("mental what-if with multiple paths") is a powerful extension but not required for Stage 74.

Deferred to future stage: evaluate multiple plans by simulating homeostatic outcomes. For now, the agent picks the highest urgency and plans for it. This is greedy but sufficient — when health urgency is 0.67 and food urgency is 0.005, the choice is clear without simulation.

### Textbook additions for homeostasis

Current textbook has resource rules. Add body rules:

```yaml
# Body rules (how the body works)
body:
  - "food 0 causes health loss"
  - "drink 0 causes health loss"  
  - "health 0 causes death"
  - "zombie nearby causes health loss fast"
  - "energy 0 causes slow movement"
```

These are "innate knowledge" — how the body works. Like a baby knowing pain = bad. Not learned, but part of the body's "DNA".

## Success criteria

| Gate | Metric | Threshold |
|------|--------|-----------|
| No hardcoded drives | select_goal has zero magic numbers | 0 constants |
| Learned rates | conditional_rates populated from experience | ≥5 associations |
| Survival improvement | mean episode length grows over 200 episodes | positive trend |
| Sword emergence | agent crafts sword without being told to | ≥30% of last 100 episodes |
| Tree nav | maintained from Stage 73 | ≥50% |
| Grounding | maintained | ≥5 concepts |

## Risks

1. **Cold start** — no rates = only curiosity fires = agent explores. This IS correct behavior — a newborn explores. Body rules in textbook provide initial rates for food/drink depletion so agent starts seeking food/water from first episode.
2. **Rate estimation noise** — few samples = noisy rates. Mitigation: EMA with slow decay (alpha ~0.05).
3. **Zombie kills before learning** — agent dies before conditional_rates for zombie stabilize. Mitigation: body rules in textbook give "zombie nearby causes health loss fast" as innate knowledge. Agent knows zombie is dangerous from birth (like innate fear of snakes).
4. **Curiosity vs survival balance** — too curious = ignores threats. Mitigation: urgency from body rates naturally dominates when health is dropping. Curiosity only wins when body is fine.

## Non-goals

- Forward simulation / mental what-if (future stage)
- DAF/SKS oscillator perception
- RL / backprop for policy
- Natural language goal specification
- Multi-agent
