# Stage 63: Abstract World Model + Crafter

## Summary

Evolve CLSWorldModel from concrete lookup table to abstract reasoning.
SDM stores ~20-50 abstract rules (not 6000 concrete transitions).
Neocortex stores concrete instances. Abstraction extracted from data patterns.
Test on two domains: MiniGrid + Crafter.

Grid reconnect with curiosity deferred to Stage 64.

**Key hypothesis:** Abstract rules in SDM generalize to unseen objects and
new domains without retraining.

## Gate

- Pure SDM generalization: held-out colors ≥85% WITHOUT neocortex substitution
- Object-type hold-out: train key+door, test ball+box ≥80%
- Crafter QA: ≥90% on tech tree (L1-L4)
- Abstraction: ≥3 auto-discovered categories (carryable, openable, solid, ...)

## Architecture

```
CLSWorldModel v2
├── neocortex: dict[str, Rule]              # concrete instances (unchanged)
├── hippocampus: SDMMemory(dim=2048)        # abstract rules (~20-50)
├── abstraction_engine: AbstractionEngine   # NEW: discovers categories
│   ├── discover_categories(neocortex) → categories
│   ├── encode_abstract_rule(category, action, outcome)
│   └── categories: dict[str, set[str]]
├── codebook: VSACodebook(dim=2048)
└── domain_encoders:                        # NEW: per-domain transition format
    ├── MiniGridEncoder                     # existing situation schema
    └── CrafterEncoder                      # recipe-based schema
```

## Abstraction Engine

### Category Discovery

After neocortex training, scan rules and group objects by (action → outcome):

```python
patterns = {}
for rule in neocortex.values():
    key = (rule.action, rule.outcome["result"])
    patterns.setdefault(key, set()).add(rule.facing_obj)

# Result:
# ("pickup", "picked_up")     → {key, ball, box}     = "carryable"
# ("toggle", "door_opened")   → {door}               = "openable"
# ("forward", "blocked")      → {wall, key, ball...}  = "solid"
# ("forward", "moved")        → {empty, open_door}    = "passable"
```

### Multi-Category Objects

Objects belong to MULTIPLE categories:
- key = carryable + solid + activator
- door = openable + solid + unlockable
- ball = carryable + solid

SDM key = `bind(category, action)`. Each abstract rule is one (category, action) → outcome.
At query time, determine which category applies by checking the action:
- "forward" on key → use "solid" category → blocked
- "pickup" on key → use "carryable" category → picked_up

```python
def query_abstract(self, obj_type, action):
    for category, members in self.categories.items():
        if obj_type in members or self._is_similar(obj_type, members):
            key_vec = bind(category_vec, action_vec)
            outcome, conf = sdm.read(key_vec)
            if conf > threshold:
                return outcome
    return unknown
```

For UNSEEN objects: check if they match any category via SDM similarity.

### Same-Color Encoding

Color relationship preserved separately from categories:
- `bind(color_X, color_X) = zero` → "same color" address
- `bind(color_X, color_Y) ≠ zero` → "different color" address

Abstract rule: `bind(category_unlockable, bind(same_color, action_toggle)) → unlocked`

This separates structural knowledge (category) from color knowledge (VSA identity).

### Abstract Rules in SDM (~20-50 total)

```
bind(carryable, pickup)         → picked_up
bind(carryable, pickup_blocked) → failed_carrying  (when already carrying)
bind(solid, forward)            → blocked
bind(passable, forward)         → moved
bind(openable, toggle)          → door_opened
bind(unlockable, same_color_toggle) → door_unlocked
bind(unlockable, diff_color_toggle) → door_still_locked
bind(droppable, drop_empty)     → dropped
bind(droppable, drop_blocked)   → drop_blocked
...
```

## Crafter Domain

### Transition Format

Crafter transitions have a DIFFERENT schema from MiniGrid:

**MiniGrid:** spatial interaction
```
(facing_obj, obj_color, obj_state, carrying, action) → outcome
```

**Crafter:** recipe-based crafting
```
(ingredients: list, station: str, action: "craft") → product
```

### Shared Transition Interface

```python
@dataclass
class Transition:
    domain: str              # "minigrid" | "crafter"
    situation: dict[str, str]  # domain-specific keys
    action: str
    outcome: dict[str, str]
    reward: float

# MiniGrid situation keys: facing_obj, obj_color, obj_state, carrying, ...
# Crafter situation keys: ingredient, station, tool, ...
```

### Per-Domain Encoders

Each domain has its own encoder that maps situation → VSA vector:

```python
class MiniGridEncoder:
    def encode(self, situation, action) → torch.Tensor:
        # compound filler + color binds (existing)

class CrafterEncoder:
    def encode(self, situation, action) → torch.Tensor:
        # ingredient + station + action → compound filler
```

### Cross-Domain Abstraction

Both domains produce categories that may share structure:
- MiniGrid "carryable" ≈ Crafter "raw_material" (pickup-able)
- MiniGrid "openable" ≈ Crafter "station" (toggleable/usable)
- Abstract: "input + enabler → output"

Cross-domain transfer via SDM: if abstract rule `input + enabler → output`
is learned from MiniGrid, it should apply to Crafter without retraining.

### Crafter QA Battery

- L1: "Can you mine stone?" → yes (with pickaxe)
- L2: "What do you need for iron_pickaxe?" → iron + table
- L3: "Craft wood at table → ?" → wood_pickaxe
- L4: "How to get stone_pickaxe from nothing?" → chop_tree → table → wood_pickaxe → mine_stone → craft_stone_pickaxe

### Crafter Transitions Generation

Use Crafter's `env.get_info()` for symbolic state. Record:
- What the agent had before action
- What action was taken
- What changed (inventory, nearby objects, achievements)

## Files

| File | Change |
|------|--------|
| `src/snks/agent/cls_world_model.py` | v2: remove neocortex substitution, abstract query |
| `src/snks/agent/abstraction_engine.py` | NEW: category discovery + abstract rules |
| `src/snks/agent/crafter_trainer.py` | NEW: Crafter transition generation |
| `src/snks/agent/crafter_encoder.py` | NEW: Crafter situation → VSA encoder |
| `tests/test_stage63_abstraction.py` | abstraction + generalization tests |
| `tests/test_stage63_crafter.py` | Crafter QA battery |
| `experiments/exp119_abstraction.py` | Gate experiment |

## Stage 64 Preview

Curiosity-driven grid agent:
- Affordance queries replace FrontierExplorer
- WM confidence as curiosity signal (low confidence → explore)
- BossLevel ≥50% gate
