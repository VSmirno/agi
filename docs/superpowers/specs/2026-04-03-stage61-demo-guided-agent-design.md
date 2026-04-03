# Stage 61: Demo-Guided Agent

## Summary

Agent that uses CausalWorldModel (Stage 60) for planning in MiniGrid environments
with partial observability. Learns causal rules from demonstrations, explores layout
via FrontierExplorer, builds plans via backward chaining, executes via BFS navigation.

**Key hypothesis:** An agent taught causal rules via demonstrations can plan and
execute in unseen environments without trial-and-error rule discovery.

## Target Environments

- **DoorKey-8x8** — smoke test, gate ≥95% (20 seeds)
- **LockedRoom** — main gate ≥80% (20 seeds), requires color matching
- **Ablation** — with vs without causal model, delta ≥40%

## Architecture

```
DemoGuidedAgent
├── causal_model: CausalWorldModel     # Stage 60, learns from demos on init
├── spatial_map: SpatialMap            # Stage 54, accumulates layout
├── explorer: FrontierExplorer         # Stage 55, explores unknown cells
├── pathfinder: GridPathfinder         # Stage 47, BFS navigation
├── planner: CausalPlanner (NEW)       # backward chaining → executable subgoals
└── executor: SubgoalExecutor (NEW)    # executes subgoals one by one
```

### Episode Lifecycle

1. **Init** — receive demo episodes, call `causal_model.learn_all_rules(colors)`
2. **Explore phase** — FrontierExplorer + SpatialMap until key, door, goal found
3. **Plan phase** — CausalPlanner calls `query_chain("pass_locked_door", color)` → subgoals
4. **Execute phase** — SubgoalExecutor: BFS to target, interaction action
5. **Precondition checks** — `query_can_act()` / `query_precondition()` before each action

## CausalPlanner

Thin wrapper over `CausalWorldModel.query_chain()` + binding subgoals to grid positions.

```python
class CausalPlanner:
    """Converts causal chains into executable subgoals with grid positions."""

    def plan(self, goal: str, color: str, spatial_map: SpatialMap) -> list[ExecutableSubgoal]:
        # 1. query_chain("pass_locked_door", color) → ["find_key", "pickup_key", "open_door", "pass_through"]
        # 2. For each abstract step — bind to position from spatial_map:
        #    - find_key → BFS to key position of correct color
        #    - pickup_key → adjacent cell to key + ACT_PICKUP
        #    - open_door → adjacent cell to door of correct color + ACT_TOGGLE
        #    - pass_through → goal position
        # 3. Color selection: query_color_match() for LockedRoom
```

### ExecutableSubgoal

```python
@dataclass
class ExecutableSubgoal:
    name: str                          # "find_key", "pickup_key", etc.
    target_pos: tuple[int, int] | None # grid position (None if not yet found)
    action_at_target: int | None       # ACT_PICKUP=3, ACT_TOGGLE=5, None
    precondition: str | None           # "adjacent", "has_key", etc.
```

**Key difference from Stage 46 SubgoalExtractor:** Stage 46 extracted subgoals
from observed traces (post-hoc). CausalPlanner generates subgoals *before*
execution from the causal model (pre-hoc). This is real planning.

## SubgoalExecutor

Executes one subgoal at a time. State machine with 3 states:

```
EXPLORE → NAVIGATE → INTERACT
   ↑          │          │
   └──────────┘          │  (subgoal done → next subgoal)
   (target lost)         ↓
```

- **EXPLORE**: target_pos unknown → FrontierExplorer searches for the object
- **NAVIGATE**: target_pos known → BFS to adjacent cell (for key/door) or directly (for goal)
- **INTERACT**: agent adjacent to target → turn to face + execute action (pickup/toggle)

### Precondition checks before INTERACT

- `pickup_key`: `query_can_act("pickup", has_key=False)` — cannot pickup if already carrying
- `open_door`: `query_can_act("open", has_key=True)` — cannot open without key
- Color match in LockedRoom: `query_color_match(key_color, door_color)` — correct key?

## LockedRoom Adaptation

1. Exploration finds door (color visible in obs `[obj_type, color, state]`)
2. CausalPlanner calls `causal_model.query_precondition("open", door_color)` → needed key color
3. Exploration searches for key of correct color via `spatial_map.find_objects_by_color()`
4. If correct key not yet found → continue exploring

### SpatialMap Extension

One new method: `find_objects_by_color(obj_type: int, color: int) -> list[tuple[int, int]]`
Returns positions of objects matching both type and color. Color already stored in channel 1.

## Reused Components (no changes)

- `CausalWorldModel` — Stage 60, used as-is
- `SpatialMap.update()`, `SpatialMap.find_objects()` — Stage 54
- `FrontierExplorer.select_action()` — Stage 55
- `GridPathfinder.find_path()`, `.path_to_actions()` — Stage 47

## Tests

### Unit tests — local `pytest`

1. `test_causal_planner_generates_subgoals` — chain → executable subgoals with positions
2. `test_color_selection_lockedroom` — correct key for door of given color
3. `test_color_selection_unseen_color` — generalization to unseen training colors
4. `test_subgoal_executor_state_machine` — EXPLORE→NAVIGATE→INTERACT transitions
5. `test_precondition_check_blocks_action` — cannot open door without key
6. `test_planner_with_partial_map` — plan updates as exploration progresses

### Gate experiments — ONLY minipc

Deploy: `git push` → `ssh minipc "cd /opt/agi && git pull"` → tmux launch.

- **exp116a**: DoorKey-8x8, 20 seeds, gate ≥95% success rate
- **exp116b**: LockedRoom, 20 seeds, gate ≥80% success rate
- **exp116c**: Ablation — with/without causal model, delta ≥40%

### Metrics

- Success rate (%)
- Mean steps to completion
- Exploration steps vs execution steps (ratio)
- Planning time (ms)

## File Structure

### New files

- `src/snks/agent/demo_guided_agent.py` — DemoGuidedAgent, CausalPlanner, SubgoalExecutor
- `tests/test_stage61_demo_guided_agent.py` — 6 unit tests
- `src/snks/experiments/exp116_demo_guided_agent.py` — gate experiments (minipc)
- `demos/stage-61-demo-guided-agent.html` — web demo

### Changes to existing files

- `src/snks/agent/spatial_map.py` — new method `find_objects_by_color(obj_type, color)`

### No changes

- `causal_world_model.py` — used as-is
- `pathfinding.py` — used as-is
- All other existing files
