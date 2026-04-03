# Stage 60: Causal World Model via Demonstrations

**Date:** 2026-04-03
**Phase:** M4 — Scale (6/8 stages)
**Branch:** `stage60-causal-world-model`
**Depends on:** Stage 59 (VSA bind(X,X)=identity proof)

---

## Goal

Build a causal world model that learns if-then rules from minimal synthetic demonstrations and answers questions about causal relationships. Validation via three levels of QA with increasing complexity.

## Gate Criteria

| Level | Type | Gate | Description |
|-------|------|------|-------------|
| QA-A | True/False facts | >= 90% accuracy on unseen colors | "Does blue key open blue door?" |
| QA-B | Precondition lookup | >= 80% correct answers | "What is needed to open red door?" |
| QA-C | Causal chains | >= 70% correct plans | "How to get behind locked red door?" |

## Approach: VSA Rule Vectors + SDM Lookup

Selected over SDM Transition Graph (reuses Stage 58 which gave NEGATIVE) and Two-Layer approach (overengineering for 5 rules).

### Why This Approach

- Explicit causal rules in VSA — maximum SNKS architectural progress
- Generalization via identity property (proven in Stage 59)
- Direct extension of existing VSACodebook + SDMMemory

## Architecture

### Components

```
CausalWorldModel
+-- codebook: VSACodebook (existing, 512-dim)
+-- rule_store: SDMMemory (existing class, new instance)
+-- learn_from_demo(demo) -> extracts and stores rule
+-- query_true_false(statement) -> bool, confidence
+-- query_precondition(action, effect) -> precond vector
+-- query_chain(goal) -> list[subgoal]
+-- decode(vector) -> str (similarity scan over codebook)

RuleEncoder
+-- encode_rule(precond, action, effect) -> VSA vector
+-- extract_from_demo(demo) -> (precond, action, effect)
+-- encode_with_color_identity(rule) -> VSA vector (hierarchical bind)

Demo = list[DemoStep]
DemoStep = (state: dict, action: str, next_state: dict, reward: float)
```

### Data Flow

```
Synthetic Demonstration (3-5 steps)
    |
RuleEncoder: extracts (precondition, action, effect) triple
    |
VSA: bind(role_precond, precond) + bind(role_action, act) + bind(role_effect, eff)
    |
RuleStore (SDM): writes rule vector
    |
Query: bind(role_X, query) -> SDM lookup -> unbind -> answer
```

### VSA Encoding Scheme

**Roles (fixed in codebook):**
- `role_precond` — what is needed before action
- `role_action` — which action
- `role_effect` — what changes after
- `role_object` — target object
- `role_color` — object color (for same-color via identity)

**Single rule encoding:**
```python
# same_color_unlock rule
precond = bind(role_object, filler("key")) | bind(role_color, filler("red"))
action = filler("open")
effect = bind(role_object, filler("door")) | bind(role_color, filler("red"))

rule = bind(role_precond, precond) | bind(role_action, action) | bind(role_effect, effect)
```

(`|` = bundle/majority vote, `bind` = XOR)

**Note:** `unbind(a, b)` = `bind(a, b)` — XOR is self-inverse. There is no separate unbind function.

### SDM API Pattern

SDMMemory exposes `read_next(state, action)` and `read_reward(state, action)` — both require two arguments. For rule storage, we follow the exp114 pattern: pass rule_address as `state` and `torch.zeros(dim)` as `action`. This avoids modifying the existing SDMMemory class.

```python
# Write rule:
rule_store.write(state=rule_address, action=zeros, next_state=rule_content, reward=reward)
# Read rule:
content, confidence = rule_store.read_next(state=rule_address, action=zeros)
reward = rule_store.read_reward(state=rule_address, action=zeros)
```

### Generalization via Identity

```python
# Training: bind(red, red) = zero_vector -> SDM stores at zero address
# Query: bind(green, green) = zero_vector -> same SDM address -> same rule
# Result: "green key opens green door" learned from red demo only
```

**Hierarchical bind:** color bind happens first (produces identity/non-identity signal), then bundled with other roles. This prevents identity signal from drowning in noise.

## Synthetic Demonstrations

5 demonstrations, each with positive + negative examples:

### Demo 1: same_color_unlock
- (+) agent_has(red_key), facing(red_door, locked) -> open -> red_door unlocked, reward +1
- (-) agent_has(blue_key), facing(red_door, locked) -> open -> FAIL, reward -1

### Demo 2: pickup_requires_adjacent
- (+) adjacent(red_key), carrying(nothing) -> pickup -> agent_has(red_key), reward +1
- (-) not_adjacent(red_key) -> pickup -> FAIL, reward -1

### Demo 3: door_blocks_passage
- (+) facing(red_door, locked) -> forward -> same_position, reward 0
- (+) facing(red_door, unlocked) -> forward -> new_position, reward +1

### Demo 4: carrying_limits
- (-) agent_has(red_key), adjacent(blue_key) -> pickup -> unchanged, reward -1

### Demo 5: open_requires_key
- (-) carrying(nothing), facing(red_door, locked) -> open -> unchanged, reward -1

Objects (key, door, wall, agent) are fixed in VSA codebook — not learned. Only causal rules are learned from demos.

**Negative example encoding:** Negative demos are stored with reward=-1 at the same SDM address pattern. For same_color_unlock: `bind(blue, red) = random_vector` stored with reward=-1, `bind(red, red) = zero_vector` stored with reward=+1. SDM naturally separates these by address.

## Query Engine — Three QA Levels

### QA-A: True/False (facts)

Each rule is stored with reward +1 (positive) or -1 (negative). Query encodes the full statement as an SDM address and reads reward.

```python
# "Does blue key open blue door?" 
address = encode_color_pair(blue, blue)  # bind(blue, blue) = zero_vector
reward = rule_store.read_reward(state=address, action=zeros)
answer = reward > 0  # True
```
Test on unseen colors: train red/blue/yellow, test green/purple/grey.

### QA-B: Precondition lookup

**Strategy: per-rule SDM instances.** Instead of encoding all roles into one vector and unbinding (noisy), each rule type gets its own SDM. Query by rule_type + known parameters, read back the unknown parameter.

```python
# "What is needed to open red door?"
# rule_type = "unlock", known: door_color = red
# SDM for "unlock" rule: address = bind(door_color, key_color), content = key_object
# Query: iterate candidate key_colors, find which gives reward > 0
# bind(red, red) = zero -> reward +1 -> answer: "red key"
```

This avoids deep unbinding from bundled vectors entirely. Each rule type is a separate SDM — 5 rules = 5 small SDMs.

### QA-C: Causal chains (backward chaining)

```python
# "How to get behind locked red door?"
# Scenario 1 (length 4): find_red_key -> pickup -> open_red_door -> pass_through
# Scenario 2 (length 3): pickup_blue_key -> open_blue_door -> pass_through  
# Scenario 3 (length 5): drop_ball -> find_green_key -> pickup -> open_green_door -> pass

# Backward chaining: goal -> which rule has this effect? -> precondition -> repeat
# Rule lookup: iterate rule SDMs, check which has matching effect
# Terminate when precondition is a base state (no further rules needed)
# Max depth: 5
```

## Pipeline Integration

### SNKS Components
- **VSA:** core encoding of rules, objects, colors — active
- **SDM:** rule storage and retrieval — active
- **DAF:** not used (perception not needed for synthetic demos)
- **Planner:** backward chaining over rules — active
- **Language:** not used (Stage 61+)

### Pipeline
```
Synthetic Demo -> RuleEncoder (VSA) -> SDMMemory -> QueryEngine (unbind + backward chain) -> Answer
```

## Phase Position

**M4 progress:** 6/8 stages. This stage advances:
- Learned causal reasoning (not symbolic BFS)
- Demonstration-based learning (architectural pivot from Stage 59)
- Foundation for Stage 61 (demo-guided agent in grid)

## Files

### New
- `src/snks/agent/causal_world_model.py` — CausalWorldModel, RuleEncoder, QueryEngine
- `tests/test_causal_world_model.py` — unit tests for all QA levels
- `src/snks/experiments/exp115_causal_world_model.py` — gate experiments

### Modified
- None (new module, no changes to existing code)

## Experiments

| Exp | What | Gate |
|-----|------|------|
| 115a | QA-A: true/false, 3 train / 3 test colors | >= 90% unseen accuracy |
| 115b | QA-B: precondition queries, all 5 rules | >= 80% correct |
| 115c | QA-C: causal chains, 3 scenarios (length 3-5) | >= 70% correct plans |

## Testing (TDD)

1. `test_rule_encoding_roundtrip` — encode -> decode without loss
2. `test_identity_generalization` — train red, test green
3. `test_qa_true_false_seen` — facts on training colors
4. `test_qa_true_false_unseen` — facts on new colors (>= 90%)
5. `test_qa_precondition` — "what is needed for X?" (>= 80%)
6. `test_qa_chain` — backward chaining plan (>= 70%)
7. `test_negative_examples` — wrong color -> False

## Architecture Decision: Per-Rule SDM vs Single SDM

**Decision:** Per-rule SDM (5 small SDMs, one per rule type).

**Why:** Single SDM with bundled role-filler vectors requires unbinding to extract answers (QA-B). Unbinding from a 3-way bundle at 512-dim produces ~33% noise per component — too noisy for reliable decode. Per-rule SDMs avoid bundling entirely: each SDM stores one rule type, queries are flat bindings (like Stage 59's proven pattern).

**Trade-off:** More SDM instances (5 vs 1), but each is simpler and follows the proven exp114 pattern. Total memory cost is negligible.

## Decode Strategy

`decode(vector) -> str`: compute similarity(vector, filler) for every filler in codebook, return argmax. Threshold: similarity > 0.55 (below = "unknown"). Ties broken by highest similarity.

## Risks

1. **Per-rule SDM scalability** — 5 SDMs is fine, but 50 rules would be unwieldy. Mitigation: Stage 60 scope is 5 rules; unification is a future concern.
2. **Chain depth** — backward chaining iterates over rule SDMs per step, O(rules * depth). Mitigation: max depth=5, 5 rules = 25 lookups max.
3. **Synthetic-to-real gap** — rules learned from synthetic demos may not transfer to MiniGrid. Mitigation: this is Stage 61's problem, Stage 60 validates purely through QA.
4. **Encoding roundtrip fidelity** — hierarchical bind + bundle may lose signal. Mitigation: add test_rule_encoding_roundtrip as first TDD test, increase dim to 1024 if roundtrip fails at 512.
