# Stage 62: BossLevel — M4 Scale Integration Test

## Summary

Extend DemoGuidedAgent to solve BabyAI BossLevel (22x22, 5 object types, compound
missions) by learning all rules and mission→subgoal mappings from Bot demonstrations.

**Key hypothesis:** A world model trained on Bot demonstrations can decompose arbitrary
BossLevel missions into executable subgoal chains and solve them in unseen layouts.

**Gate:** ≥50% success on BabyAI-BossLevel-v0 (50 seeds)

## Target Environment

- **BabyAI-BossLevel-v0** — 22x22 grid, partial observability (7x7)
- **Object types:** Wall, Door, Key, Ball, Box (5 types)
- **Mission types:** go_to, pick_up, open, put_next_to — plus compound (AND, THEN, AFTER)
- **Max steps:** 1152
- **Examples:**
  - `"pick up a purple ball and open a red door"`
  - `"put the red box next to a grey door and open a purple door after you pick up a key"`
  - `"go to the red key and open the purple door, then pick up the yellow ball"`

## Architecture

```
BossLevelAgent (extends DemoGuidedAgent)
├── causal_model: CausalWorldModel      # Stage 60 + new rule types
│   ├── SAME_COLOR_UNLOCK               # existing
│   ├── PICKUP_ADJACENT                  # existing
│   ├── DOOR_BLOCKS                      # existing
│   ├── CARRYING_LIMITS                  # existing
│   ├── OPEN_REQUIRES_KEY               # existing
│   ├── PUT_NEXT_TO (NEW)               # drop obj adjacent to target
│   └── GO_TO_COMPLETE (NEW)            # adjacent = success
├── mission_model: MissionModel (NEW)    # VSA(mission) → subgoal sequence
│   └── sdm: SDM                        # learns from demo (mission, subgoals) pairs
├── spatial_map: SpatialMap              # Stage 54, scale to 22x22
├── explorer: FrontierExplorer           # Stage 55
├── pathfinder: GridPathfinder           # Stage 47
├── planner: CausalPlanner (EXTENDED)    # generic chains from MissionModel
└── executor: SubgoalExecutor (EXTENDED) # new action types: DROP, GO_TO
```

### Data Flow

1. Bot generates ~200 BossLevel demonstrations
2. From each demo extract: `(mission_text, subgoal_phases, object_interactions, rules_used)`
3. MissionModel learns: `VSA(mission_tokens) → [subgoal_1, subgoal_2, ...]`
4. CausalWorldModel learns new rules from object interactions
5. Runtime: mission → MissionModel → subgoal chain → CausalPlanner binds to map → SubgoalExecutor

### Key Change from Stage 61

CausalPlanner no longer uses hardcoded 4-step chain. It receives arbitrary subgoal
sequences from MissionModel and queries CausalWorldModel for preconditions per subgoal.

## MissionModel

### Mission Encoding

- Tokenize: split by spaces, each token → VSA vector from codebook
- Mission vector = sequential bind with positional encoding: `bind(pos_1, token_1) + bind(pos_2, token_2) + ...`
- Codebook: ~50 words (colors, objects, actions, prepositions, conjunctions)

### Subgoal Vocabulary (learned from demos)

- `GO_TO(type, color)` — navigate to object
- `PICK_UP(type, color)` — pick up object
- `OPEN(type, color)` — open door
- `PUT_NEXT_TO(type1, color1, type2, color2)` — place carried obj next to target
- `DROP()` — free inventory

### Subgoal Sequence Encoding

- Each subgoal → VSA vector: `bind(type_vec, bind(obj_vec, color_vec, pos_in_chain_vec))`
- Sequence = bundle of all subgoals with positional tags
- SDM: address = mission vector, value = subgoal sequence vector

### Retrieval

- New mission → encode → SDM read → decode subgoal sequence
- Decode: unbind positional tags → for each position, similarity search over subgoal vocabulary
- Concrete type/color extracted from mission tokens directly

### Generalization

- "pick up a red ball" and "pick up a blue box" produce similar mission vectors (shared "pick up a")
- SDM returns `[GO_TO(?, ?), PICK_UP(?, ?)]` — specific color/type filled from mission

### Extracting Subgoals from Demo Trajectories

- Track state changes in Bot recording:
  - inventory_changed(+obj) → PICK_UP
  - door_state_changed(locked→open) → OPEN
  - inventory_changed(-obj) + obj_near_target → PUT_NEXT_TO / DROP
  - agent adjacent to mission target → GO_TO complete
- Chronological order of phases = subgoal sequence

### Demo Count

~200 episodes (seeds 0..199), covering major mission type combinations.

## CausalWorldModel Extensions

### New Rule Types

**PUT_NEXT_TO:**
- Precondition: carrying(obj1) = True, adjacent_to(obj2) = True
- Action: DROP
- Effect: obj1 placed next to obj2
- SDM encoding: `bind(CARRYING, obj1_type_color) + bind(ADJACENT, obj2_type_color)` → address

**GO_TO_COMPLETE:**
- Precondition: adjacent_to(target) = True
- Action: none (goal satisfied at adjacency)
- SDM encoding: `bind(ADJACENT, target_type_color)` → reward signal

### Learning from Demos

For each state transition in Bot trajectory:
- `inventory_changed(+obj)` → encode PICKUP rule with context (was adjacent, obj type/color)
- `door_state_changed(locked→open)` → encode OPEN rule (had matching key)
- `inventory_changed(-obj) + obj_near_target` → encode PUT_NEXT_TO rule
- `agent_adjacent_to(mission_target)` → encode GO_TO_COMPLETE

### Extended CausalPlanner

- Receives subgoal sequence from MissionModel
- For each subgoal queries CausalWorldModel:
  - `query_precondition(subgoal_type)` → what's needed
  - `query_can_act(current_state)` → can act now?
- If precondition unmet → inserts sub-subgoals (recursive, max depth=3)
- Binds each subgoal to positions from SpatialMap

**Example 1: "pick up a blue key, then open a green door"**
1. MissionModel → `[PICK_UP(key, blue), OPEN(door, green)]`
2. Planner: PICK_UP(key, blue) → precondition: adjacent → insert GO_TO(key, blue)
3. Planner: OPEN(door, green) → precondition: has_key(green) → need green key
4. Inventory conflict: will be carrying blue key → insert DROP() before PICK_UP(key, green)
5. Final plan: `[GO_TO(key,blue), PICK_UP(key,blue), DROP(), GO_TO(key,green), PICK_UP(key,green), GO_TO(door,green), OPEN(door,green)]`

**Example 2: "put the red box next to a grey door"**
1. MissionModel → `[PICK_UP(box, red), PUT_NEXT_TO(box, red, door, grey)]`
2. Planner: PICK_UP → insert GO_TO(box, red)
3. Planner: PUT_NEXT_TO → precondition: carrying(box,red) + adjacent(door,grey) → insert GO_TO(door, grey)
4. Final plan: `[GO_TO(box,red), PICK_UP(box,red), GO_TO(door,grey), PUT_NEXT_TO(box,red,door,grey)]`

## SubgoalExecutor Extensions

### State Machine

```
EXPLORE → NAVIGATE → INTERACT → DROP
```

### New Action Types

- **DROP:** `env.actions.drop` — for PUT_NEXT_TO: carry obj1, stand adjacent to obj2, drop
- **GO_TO completion:** subgoal complete at adjacency (no interaction needed)

### Inventory Tracking

- Current: boolean `has_key`
- Extended: `carrying: Optional[tuple[str, str]]` — (type, color) of carried object
- For: drop before pick_up of new object, PUT_NEXT_TO validation

### Multi-Object Handling

- Agent carries max 1 object
- If need pick_up(B) but carrying(A): auto drop(A), pick_up(B)
- Planner accounts for this via CARRYING_LIMITS rule from world model

## Demo Generation Pipeline

### Script: `scripts/generate_bosslevel_demos.py`

1. Create BossLevel env with seeds 0..199
2. Run BabyAI Bot on each
3. Record frame-by-frame: agent_pos, agent_dir, action, inventory, grid state
4. Extract subgoal phases from state transitions
5. Save to `_docs/demo_episodes_bosslevel.json`

### Demo Format (extends Stage 57)

```json
{
  "env": "BabyAI-BossLevel-v0",
  "seed": 0,
  "mission": "pick up a blue key, then open a green door",
  "grid_width": 22,
  "grid_height": 22,
  "frames": [
    {
      "step": 0,
      "agent_col": 1, "agent_row": 1, "agent_dir": 0,
      "action": "forward",
      "carrying": "",
      "inventory_type": null, "inventory_color": null
    }
  ],
  "subgoals_extracted": [
    {"type": "GO_TO", "obj": "key", "color": "blue"},
    {"type": "PICK_UP", "obj": "key", "color": "blue"},
    {"type": "GO_TO", "obj": "door", "color": "green"},
    {"type": "OPEN", "obj": "door", "color": "green"}
  ],
  "success": true,
  "total_steps": 183
}
```

## Experiment: Exp117

### 117a: BossLevel Full (MAIN GATE)

- 50 seeds, max_steps=1152
- Gate: ≥50% success rate
- Agent: BossLevelAgent trained on 200 Bot demos

### 117b: Ablation

- Trained MissionModel vs untrained (random subgoals)
- Gate: delta ≥30%

### 117c: Per-Mission-Type Breakdown

- Categories: simple (1 action), compound (2 actions), complex (3+ actions)
- No gate — diagnostic only

## Components Reused from Stage 61

| Component | Source | Changes |
|-----------|--------|---------|
| SpatialMap | Stage 54 | +OBJ_BALL, +OBJ_BOX constants; scale to 22x22 (already dynamic) |
| FrontierExplorer | Stage 55 | No changes |
| GridPathfinder | Stage 47 | No changes |
| CausalWorldModel | Stage 60 | +2 rule types, +learning from state transitions |
| CausalPlanner | Stage 61 | Generic chains from MissionModel |
| SubgoalExecutor | Stage 61 | +DROP action, +GO_TO completion, +inventory tracking |

## Mission Completion Detection

BossLevel has no goal square (unlike DoorKey/LockedRoom). Success is determined by
completing the mission. Detection strategy:

- Each subgoal has a **completion predicate**:
  - GO_TO(obj, color): agent adjacent to obj of that color
  - PICK_UP(obj, color): carrying == (obj, color)
  - OPEN(door, color): door state changed to open
  - PUT_NEXT_TO(o1, c1, o2, c2): o1 adjacent to o2 after drop
  - DROP(): carrying == None
- SubgoalExecutor tracks `completed_subgoals: list[bool]`
- Mission complete when all subgoals in sequence are completed
- Environment also returns `reward > 0` on mission success (ground truth)

## File Structure

### New Files

| File | Purpose |
|------|---------|
| `src/snks/agent/boss_level_agent.py` | BossLevelAgent class |
| `src/snks/agent/mission_model.py` | MissionModel (VSA+SDM for mission→subgoals) |
| `scripts/generate_bosslevel_demos.py` | Bot demo generation pipeline |
| `experiments/exp117_bosslevel.py` | Gate experiment |
| `tests/test_stage62_bosslevel.py` | Unit tests |
| `_docs/demo_episodes_bosslevel.json` | Generated demo data |

### Modified Files

| File | Changes |
|------|---------|
| `src/snks/agent/causal_world_model.py` | +PUT_NEXT_TO, +GO_TO_COMPLETE rule types (7 SDMs total) |
| `src/snks/agent/demo_guided_agent.py` | CausalPlanner: generic chains; SubgoalExecutor: +DROP, +GO_TO, +inventory |
| `src/snks/agent/spatial_map.py` | +OBJ_BALL=6, +OBJ_BOX=7 constants |

## Tests

### Unit Tests (local pytest)

| Test Class | Tests | What |
|------------|-------|------|
| TestMissionEncoding | 4 | Tokenize, encode, decode round-trip; positional encoding |
| TestMissionModel | 4 | Learn from demos, retrieve subgoals, unseen mission generalization |
| TestSubgoalExtraction | 3 | Extract subgoals from Bot trajectory; compound missions; "after" reversal |
| TestNewRuleTypes | 3 | PUT_NEXT_TO preconditions, GO_TO_COMPLETE detection, learning from transitions |
| TestExtendedPlanner | 4 | Generic chains, inventory conflict resolution, recursive precondition insertion, max depth |
| TestExtendedExecutor | 3 | DROP action, GO_TO completion, inventory tracking (type, color) |
| TestBossLevelAgentUnit | 3 | Init with demos, reset, stats |

### Gate Experiments (exp117, run on minipc)

See Experiment section above.

## Metrics

- **Success rate** — primary gate metric (≥50%)
- **Mean steps** — efficiency on successful episodes
- **Exploration ratio** — steps exploring vs executing plan
- **Per-mission-type success** — simple / compound / complex breakdown
- **Subgoal completion rate** — how many subgoals completed even on failed episodes
- **MissionModel decode accuracy** — % of correctly decoded subgoal sequences (on training set)

## Risks

1. **MissionModel generalization** — unseen mission combinations may produce wrong subgoal order. Mitigation: 200 demos cover major patterns, VSA positional encoding preserves structure.
2. **SDM capacity at 22x22** — more objects = more addresses. Mitigation: separate SDM per rule type already scales well.
3. **Compound missions with dependencies** — "X after Y" reverses order. Mitigation: subgoal extraction from demos captures actual execution order, not text order.
4. **Exploration bottleneck on 22x22** — larger grid = longer exploration. Mitigation: FrontierExplorer BFS is O(n), 22x22=484 cells is manageable within 1152 steps.
5. **put_next_to precision** — need to navigate to specific adjacent cell and drop. Mitigation: `_best_adjacent()` from Stage 61 handles neighbor selection.
