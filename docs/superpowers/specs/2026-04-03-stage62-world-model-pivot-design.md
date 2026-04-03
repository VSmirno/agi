# Stage 62 Pivot: Unified World Model — Domain Understanding First

## Summary

Pivot from BossLevel navigation (44%, stuck) to deep world model quality.
Replace 7 hardcoded per-rule SDMs with a single unified SDM that learns
ALL MiniGrid physics from state transitions. Test via QA battery + 
generative planning. Grid navigation deferred to Stage 63.

**Key hypothesis:** A unified VSA+SDM world model trained on state
transitions can answer arbitrary domain questions and generate correct
plans, without hardcoded rule types.

**Gate:** QA ≥90% average (4 levels) AND Planning ≥80% (20 scenarios)

## Why Pivot

1. BossLevel agent at 44% — 80% of failures from blind exploration
2. NavigationPolicy (learned directions) made it WORSE (16%)
3. Current "world model" = 7 lookup tables, zero understanding
4. Ablation delta=0% — learned model adds nothing over text parsing
5. Real value is in world model, not navigation bug fixes

## Architecture

```
UnifiedWorldModel
├── codebook: VSACodebook(dim=1024)
├── sdm: SDMMemory(n_locations=10000, dim=1024)
├── train_from_transitions(transitions)
├── query(situation, action) → predicted_outcome
├── query_qa(question) → answer
└── query_plan(goal, current_state) → action_sequence
```

Single SDM for all world physics. No per-rule structure.

### Encoding

Everything is facts — role-filler pairs bundled into VSA vectors:

```python
situation = bundle([
    bind(role("agent_carrying"), filler("key_red")),
    bind(role("agent_near"), filler("door_red")),
    bind(role("door_state"), filler("locked")),
    bind(role("action"), filler("toggle")),
])
outcome = bundle([
    bind(role("door_state"), filler("open")),
    bind(role("agent_carrying"), filler("nothing")),
])
```

SDM write: `write(situation, zeros, outcome, reward)`
SDM read: `read_next(situation, zeros) → predicted_outcome`

## Training Pipeline

### Source A: Bot Demo Transitions (~2000-3000)

From 200 Bot episodes, extract state transitions where state CHANGED:
```
before: {agent_pos, carrying, facing_object, facing_state}
action: {forward/toggle/pickup/drop}
after:  {agent_pos, carrying, facing_object, facing_state}
reward: +1 if state changed, 0 if nothing happened
```

Filter: keep transitions where state changed + small sample of
"nothing happened" (negative examples).

### Source B: Synthetic Scenarios (~5000)

Programmatically generate ALL combinations from MiniGrid physics:
```
for obj_type in [key, door, ball, box, wall]:
  for obj_color in colors:
    for obj_state in states:
      for action in [toggle, pickup, drop, forward]:
        for carrying in [nothing, key_red, ball_blue, ...]:
          → spawn in MiniGrid, execute, record result
```

Covers edge cases absent from demos: toggle wall, pickup while
carrying, forward into locked door.

### Total: ~7000-8000 transitions → SDM(10000, dim=1024)

## QA Battery (100 questions, 4 levels)

### Level 1: Object Identity (25 questions) — Gate ≥95%
- "Can you pick up a key?" → yes
- "Can you pick up a wall?" → no
- "Can you pass through an open door?" → yes
- "Can you pass through a locked door?" → no

Encoding: `query(situation=[near_X, action_Y])` → positive/negative outcome.

### Level 2: Preconditions (25 questions) — Gate ≥90%
- "What do you need to open a locked red door?" → red key
- "What do you need to pick up a ball?" → adjacent, not carrying
- "What do you need to drop?" → must be carrying something

Encoding: query with partial situation → decode missing precondition.

### Level 3: Consequences (25 questions) — Gate ≥85%
- "Carrying red key, toggle locked red door. What happens?" → door opens
- "Carrying ball, try pickup key. What happens?" → fail, already carrying
- "Forward into wall. What happens?" → nothing, blocked

Encoding: full situation+action → predict outcome.

### Level 4: Reasoning (25 questions) — Gate ≥80%
- "How to get past a locked blue door?" → find blue key, pickup, go to door, toggle
- "Carrying ball, need key. What to do?" → drop ball, pickup key
- "In room with no visible doors?" → explore / no exit

Encoding: goal + current_state → plan chain via iterative SDM query.

## Generative Planning (20 scenarios) — Gate ≥80%

Each scenario: (initial_state, goal). Model generates plan via forward
chaining: current_state → query best action → predict outcome → repeat.

Scenarios range from simple ("pick up key in same room") to complex
("get ball behind locked door through another room").

Plan scored as correct if: all preconditions met at each step,
final state satisfies goal, no impossible actions.

## What Changes

### Removed
- NavigationPolicy (nav_policy.py) — broken, hurts performance
- 7 per-rule SDMs in CausalWorldModel — replaced by unified SDM
- BossLevel navigation gate — replaced by QA+Planning gate
- FrontierExplorer as primary strategy — deferred to Stage 63

### Kept
- VSACodebook + SDMMemory — core infrastructure
- SpatialMap + GridPathfinder — for Stage 63
- MissionModel — mission parsing works (100% accuracy)
- BossLevelAgent shell — reconnect in Stage 63
- Bot demos + demo generator — source of transitions
- Stage 60-61 tests — regression guard

### New Files
| File | Purpose |
|------|---------|
| `src/snks/agent/unified_world_model.py` | Single-SDM world model |
| `src/snks/agent/world_model_trainer.py` | Transition extraction + synth generation |
| `tests/test_stage62_world_model.py` | QA battery + planning tests |
| `experiments/exp118_world_model_qa.py` | Gate experiment |

## Stage 63 Preview

When world model passes gate:
1. Reconnect to BossLevelAgent
2. Agent queries world model for decisions ("what should I do?")
3. Navigation = pathfinding to where model says to go
4. No separate exploration — model KNOWS what to do
5. Gate: ≥50% BossLevel (reuse exp117)
