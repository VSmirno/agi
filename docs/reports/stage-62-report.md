# Stage 62 Report: BossLevel — M4 Scale Integration Test

**Status:** IN PROGRESS (gate not yet passed)
**Date:** 2026-04-03
**Gate:** ≥50% BabyAI-BossLevel-v0 (50 seeds)
**Current:** 34% (17/50)

## Results

### Exp117a: BossLevel Full (50 seeds)
- **Success rate: 34% (17/50)** — gate ≥50% NOT MET
- Mean steps (success): ~150
- Subgoal completion: 40% (84/209)

### Per-Mission-Type Breakdown
| Type | Rate | Notes |
|------|------|-------|
| Simple (1 action) | ~45% | go_to, pick_up, open |
| Compound (2 actions) | ~30% | X and Y |
| Complex (3+ actions) | ~10% | X then Y after Z |

## Architecture

```
BossLevelAgent
├── MissionModel (NEW)       — VSA+SDM: mission→subgoal sequence
├── CausalWorldModel         — Stage 60 + PUT_NEXT_TO, GO_TO_COMPLETE
├── SpatialMap               — Stage 54 + OBJ_BALL, OBJ_BOX
├── FrontierExplorer         — Stage 55 (unchanged)
├── GridPathfinder           — Stage 47 + solid object avoidance
├── CausalPlanner (generic)  — dynamic subgoal chains from MissionModel
└── SubgoalExecutor          — +DROP, +GO_TO facing, +inventory tracking
```

### Data Flow
1. BabyAI Bot generates 200 demos (100% success rate)
2. MissionModel learns mission→subgoals from demo text (100% extraction accuracy)
3. CausalWorldModel learns 7 rule types (5 existing + 2 new)
4. Runtime: mission → MissionModel → subgoal chain → explore → navigate → interact

## Key Fixes Applied

1. **Pathfinding: solid objects** — keys, balls, boxes now treated as impassable. Was the single biggest bug (15%→36% improvement)
2. **Nearest object selection** — when multiple matching objects exist, picks nearest to agent
3. **GO_TO facing** — agent faces target at adjacency (BabyAI requires this)
4. **Anti-stuck mechanism** — turns after 3 steps at same position
5. **Dynamic key insertion** — when reaching locked door, auto-inserts key-fetch subgoals
6. **Punctuation stripping** — "key," → "key" in mission tokenization

## What Works
- Simple missions near agent (open door in front, go to nearby object)
- Multi-step missions with navigation (pick up key + open door)
- Color matching via CausalWorldModel (VSA identity generalization)
- Exploration of 22x22 grids via FrontierExplorer

## Remaining Issues (for gate ≥50%)

### 1. Exploration Efficiency (biggest gap)
Many failures (seeds 8, 19, 43, 46, 47) show 0 completed subgoals — agent can't find target within step budget. On 22x22 with partial observability, frontier-based exploration can be very slow.

**Potential fixes:**
- Directed exploration toward unexplored rooms
- Priority frontiers (prefer frontiers near doors)
- Larger exploration radius (explore more aggressively early)

### 2. put_next_to Execution
PUT_NEXT_TO missions consistently fail. The agent picks up object but doesn't reliably navigate to target and drop.

### 3. Multi-pickup Sequences
"pick up X and pick up Y" requires drop/pickup cycle. Agent gets stuck when second pickup fails.

### 4. "behind you" / Positional References
Missions like "go to the box behind you" refer to relative positions at episode start. Agent doesn't track initial facing direction.

## Files

| File | Purpose |
|------|---------|
| `src/snks/agent/boss_level_agent.py` | BossLevelAgent |
| `src/snks/agent/mission_model.py` | MissionModel (VSA+SDM) |
| `src/snks/agent/causal_world_model.py` | +2 rule types |
| `src/snks/agent/spatial_map.py` | +OBJ_BALL, +OBJ_BOX |
| `src/snks/agent/pathfinding.py` | +solid objects |
| `scripts/generate_bosslevel_demos.py` | Demo generator |
| `experiments/exp117_bosslevel.py` | Gate experiment |
| `tests/test_stage62_bosslevel.py` | 29 unit tests |
| `_docs/demo_episodes_bosslevel.json` | 200 Bot demos |

## Tests
- 29 unit tests: ALL PASS
- 59 total tests (Stages 60-62): ALL PASS
- Exp117a: 34% (17/50) — FAIL (need ≥50%)
