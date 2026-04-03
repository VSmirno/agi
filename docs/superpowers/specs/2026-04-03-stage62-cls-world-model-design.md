# Stage 62: CLS World Model — Complementary Learning System

## Summary

Replace failed unified SDM with bio-inspired Complementary Learning System.
Two-tier memory: hippocampus (SDM, fast, limited) + neocortex (dict, exact,
unlimited). Consolidation promotes verified rules from SDM to dict.

**Root cause of previous failure:** Binary SDM holds ~200-500 patterns at D=512.
We tried 6000 → 10-30x over capacity. Per-rule SDMs worked because each had
10-50 items.

**Solution:** Neocortex dict stores thousands of verified rules (exact match).
SDM only handles novel/unseen combinations (generalization via VSA).

## Architecture

```
CLSWorldModel
├── hippocampus: SDMMemory(dim=2048, n_locations=5000)
│   └── ~200-500 genuinely novel rules only
├── neocortex: dict[str, Rule]
│   └── Thousands of verified rules, exact lookup
├── consolidator: consolidation loop (SDM → dict)
├── codebook: VSACodebook(dim=2048)
└── encoder: SituationEncoder
```

## Rule

```python
Rule(
    situation_key: str,  # "door_locked_red_key_red_toggle"
    outcome: dict,       # {"result": "door_unlocked", "obj_state": "open"}
    reward: float,       # +1 success, -1 failure
    confidence: int,     # times SDM correctly predicted this
    source: str,         # "synthetic" | "demo" | "consolidated"
)
```

## Training Flow

1. Generate transitions (synthetic + demo)
2. For each transition:
   - Build compound key from situation
   - **Write-on-Surprise:** skip if neocortex already has rule OR SDM predicts correctly
   - Write novel transitions to hippocampus SDM
3. **Consolidation:** replay all training transitions:
   - Query SDM for each
   - If SDM predicts correctly → confidence += 1
   - If confidence ≥ 3 → promote Rule to neocortex dict
4. Result: neocortex has ~4000+ rules, SDM has ~50-200 novel patterns

## Write-on-Surprise

```python
def write(self, transition):
    key = self.make_key(transition.situation, transition.action)
    if key in self.neocortex:
        return  # already consolidated
    predicted = self.hippocampus_predict(transition)
    if predicted == transition.outcome:
        return  # already learned in SDM
    self.hippocampus.write(transition)  # novel!
```

## Query Flow

```
query(situation, action):
  1. Build compound key → neocortex dict lookup
  2. If found → return Rule (100% accurate, zero noise)
  3. If not found → encode VSA → hippocampus SDM read
  4. Decode SDM output → return predicted outcome
```

Neocortex handles known physics. SDM handles unseen combinations
(e.g., purple key + purple door when only red/blue/green were trained).

## QA + Planning

**QA Levels 1-3:** Mostly neocortex (exact match). SDM for unseen colors.
**Level 4 (Planning):** Forward chaining through neocortex dict. Each step
= dict lookup, zero noise, no SDM error compounding. Plans of any length work.

**Planning algorithm:**
```python
for step in range(max_steps):
    for action in [pickup, toggle, drop, forward]:
        key = make_key(state, action)
        rule = neocortex.get(key)
        if rule and rule.reward > 0:
            plan.append(action)
            state = apply(state, rule.outcome)
            break
```

## Generalization via Hippocampus

- Unseen: "purple key + purple door" not in training
- Neocortex miss → SDM read → VSA bind(purple,purple) ≈ bind(red,red) → correct
- After verification → promote to neocortex

## Parameters

| Parameter | Value |
|-----------|-------|
| VSA dim | 2048 |
| SDM locations | 5000 |
| Consolidation threshold | 3 consistent reads |
| Write-on-Surprise | enabled |
| n_amplify (SDM) | 5 |

## Files

| File | Change |
|------|--------|
| `src/snks/agent/cls_world_model.py` | NEW: CLSWorldModel |
| `src/snks/agent/world_model_trainer.py` | unchanged (transitions) |
| `tests/test_stage62_world_model.py` | QA battery + planning tests |
| `experiments/exp118_world_model_qa.py` | Gate experiment |
| `src/snks/agent/unified_world_model.py` | deprecated |

## Gate

QA ≥90% average (4 levels) AND Planning ≥80% (20 scenarios).
