# Stage 62 Fix: NavigationPolicy — Learned Exploration via VSA+SDM

## Problem

BossLevelAgent at 44% (gate ≥50%). Root cause: blind frontier exploration
wastes 90% of steps. Agent discards 95% of demo information (navigation
trajectories). Ablation delta=0% — current learning adds nothing over
text extraction.

## Solution

Learn a high-level navigation policy from Bot demonstrations. At decision
points, SDM predicts which direction to explore to find the target.
Low-level BFS navigation unchanged.

## Architecture

```
NavigationPolicy (NEW)
├── codebook: VSACodebook(dim=1024)
├── encoder: NavStateEncoder
├── sdm: SDMMemory(n_locations=5000, dim=1024)
└── Methods:
    ├── train_from_demos(demos)     # extract decision points, write to SDM
    ├── predict_direction(state)    # query SDM → direction 0-7
    └── encode_state(features)      # abstract state → VSA vector
```

Replaces FrontierExplorer as primary exploration. FrontierExplorer becomes fallback.

## Decision Point Extraction

From each Bot demo, extract frames where Bot makes strategic decisions:
- **Door toggle:** entering new room
- **Direction change:** Bot turns (not just forward)
- **Object interaction:** pickup, drop

~10-15 decision points per episode × 200 demos = 2000-3000 training examples.

At each decision point, record:
- Abstract state features (see below)
- Ground truth: direction from agent to actual target (0-7, from full grid)

## State Encoding

Abstract features (not raw pixels):

| Feature | Values | Encoding |
|---------|--------|----------|
| agent_quadrant | 0-3 | Which quarter of 22x22 |
| target_type | key/ball/box/door | What we're looking for |
| target_color | 0-5 | Which color |
| explored_ratio | low/mid/high | <25% / 25-60% / >60% |
| n_rooms_visited | few/some/many | 0-2 / 3-5 / 6+ |
| nearest_door_dir | 0-7 or none | Direction to nearest unopened door |

Each feature → `bind(role, filler)`, then bundle all → single 1024-bit VSA vector.

## Direction Encoding

8 directions (N, NE, E, SE, S, SW, W, NW) encoded as VSA fillers.
Direction from agent to target computed from positions in demo.

## SDM Training

```python
for demo in demos:
    for decision_point in extract_decision_points(demo):
        state_vec = encode_state(decision_point.features)
        dir_vec = codebook.filler(f"dir_{direction_to_target}")
        for _ in range(n_amplify):
            sdm.write(state_vec, zeros, dir_vec, 1.0)
```

## Runtime Flow

```
1. Current subgoal target not found on map
2. Encode abstract state → query NavigationPolicy
3. SDM returns direction prediction + confidence
4. If confidence >= threshold:
     Find nearest door/frontier in predicted direction
     BFS navigate there → open door → explore
5. Else (low confidence):
     Fallback to FrontierExplorer (current behavior)
6. Repeat until target found on map
7. Once found → existing BFS navigation + interaction
```

## Scaling

| Parameter | Before | After |
|-----------|--------|-------|
| VSA dim | 512 | 1024 |
| SDM locations | 2000 | 5000 |
| Training examples | ~200 (mission-level) | ~3000 (decision points) |

NavigationPolicy gets its own VSACodebook and SDM — independent from
CausalWorldModel's per-rule SDMs.

## Files

| File | Change |
|------|--------|
| `src/snks/agent/nav_policy.py` | NEW: NavigationPolicy class |
| `src/snks/agent/boss_level_agent.py` | Replace exploration with nav_policy |
| `scripts/generate_bosslevel_demos.py` | Add decision point extraction + direction labels |
| `tests/test_stage62_bosslevel.py` | Add NavigationPolicy tests |

## Gate

≥50% on BabyAI-BossLevel-v0 (50 seeds). Run on minipc.
