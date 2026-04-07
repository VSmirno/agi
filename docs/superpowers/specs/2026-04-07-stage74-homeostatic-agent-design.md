# Stage 74: Homeostatic Agent — Goal-Free Self-Organizing Behavior

**Date:** 2026-04-07
**Status:** Draft
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

Only the body:

```python
HOMEOSTATIC_VARIABLES = {
    "health": {"min": 0, "max": 9, "death_at": 0},
    "food":   {"min": 0, "max": 9, "affects": "health"},  # food=0 → health drops
    "drink":  {"min": 0, "max": 9, "affects": "health"},  # drink=0 → health drops
    "energy": {"min": 0, "max": 9, "affects": "speed"},   # energy=0 → slower
}
```

This is the ONLY thing that's "programmed". Everything else is derived.

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

#### 3. Goal selection = highest urgency → world model plan

```python
def select_goal(tracker, inventory, concept_store, visual_field):
    # Compute urgency for each homeostatic variable
    urgencies = {}
    for var in HOMEOSTATIC_VARIABLES:
        value = inventory.get(var, 9)
        rate = tracker.rates.get(var, 0)
        urgencies[var] = compute_drive(var, value, rate)
    
    # Most urgent variable
    critical_var = max(urgencies, key=urgencies.get)
    
    if urgencies[critical_var] <= 0:
        # All stable — explore/collect (background activity)
        return "explore", []
    
    # Ask world model: "what restores this variable?"
    # This comes from textbook + experience, NOT hardcoded
    restore_goal = f"restore_{critical_var}"
    plan = concept_store.plan(restore_goal)
    
    if not plan:
        # World model doesn't know how to fix this yet
        # → explore (motor babbling might discover solution)
        return "explore", []
    
    return restore_goal, plan
```

**"I need a sword"** emerges naturally:
1. Tracker observes: health rate = -2.0 when zombie visible
2. Urgency for health = very high
3. World model: "what restores health?" → no direct rule
4. But: "what STOPS health from dropping?" → "remove zombie" → "do zombie with sword" → "make sword" → plan
5. Agent crafts sword NOT because programmer said so, but because world model predicts it solves the health crisis

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

### Forward planning through world model

Current ConceptStore.plan() does backward chaining: "what produces X?" This gives a sequence of steps. But it doesn't answer "what should I do FIRST given multiple threats?"

For that, the agent needs forward simulation:

```
Current state: health=5, food=3, zombie visible
  → health urgency = HIGH (rate -2.0, ~2.5 steps to death)
  → food urgency = LOW (rate -0.04, ~75 steps to crisis)

World model simulation:
  Path A: fight zombie → if win, health rate → 0 → survive
  Path B: run + find food → zombie follows → health keeps dropping → die
  Path C: craft sword → takes ~30 steps → zombie kills in 2.5 → die

  → Path A is best IF have sword
  → If no sword: Path B (flee) buys time → then craft sword while safe
```

This is model-based decision making using causal rules for mental simulation. The agent runs "what if" scenarios through its world model.

Implementation: for each possible plan, estimate outcome by chaining causal rules + homeostatic rates. Pick plan with best predicted homeostatic state.

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
| Sword emergence | agent crafts sword without being told to | ≥30% episodes |
| Tree nav | maintained from Stage 73 | ≥50% |
| Grounding | maintained | ≥5 concepts |

## Risks

1. **Cold start** — no rates = no urgency = no action. Mitigation: initial rates from body rules ("food depletes over time" gives starting rate).
2. **Rate estimation noise** — few samples = noisy rates. Mitigation: EMA with slow decay.
3. **Forward simulation complexity** — evaluating all possible plans is expensive. Mitigation: only evaluate top-2 urgencies, max depth 5.
4. **Zombie kills before learning** — agent dies before conditional_rates stabilize. Mitigation: body rules give "zombie causes health loss" as innate knowledge (from textbook body section).

## Non-goals

- DAF/SKS oscillator perception
- RL / backprop for policy
- Natural language goal specification
- Multi-agent
