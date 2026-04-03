# Stage 62: BossLevel Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-03-stage62-bosslevel-design.md`

## Tasks

### Task 1: Demo Generation Pipeline
**Files:** `scripts/generate_bosslevel_demos.py`, `_docs/demo_episodes_bosslevel.json`

Steps:
1. Create `scripts/generate_bosslevel_demos.py`:
   - Import BabyAI Bot, create BossLevel env seeds 0..199
   - Record frame-by-frame: agent_pos, agent_dir, action, carrying (type+color)
   - Track full grid state per frame for state transition detection
   - Extract subgoals from state transitions (inventory changes, door opens, adjacency)
   - Save to `_docs/demo_episodes_bosslevel.json`
2. Run script, verify output format matches spec
3. Verify: ≥190 successful demos (Bot may fail on some seeds)

**Review checkpoint:** Verify demo format and subgoal extraction quality before proceeding.

### Task 2: SpatialMap Extensions
**Files:** `src/snks/agent/spatial_map.py`

Steps:
1. Add `OBJ_BALL = 6`, `OBJ_BOX = 7` constants
2. Verify existing `find_object()` and `find_objects_by_color()` work with new types
3. Run existing Stage 54 tests to confirm no regressions

### Task 3: MissionModel
**Files:** `src/snks/agent/mission_model.py`

Steps:
1. Create VSA codebook (~50 words: colors, objects, actions, prepositions, conjunctions)
2. Implement `encode_mission(text) -> HRR vector` with positional encoding
3. Implement subgoal vocabulary encoding (GO_TO, PICK_UP, OPEN, PUT_NEXT_TO, DROP)
4. Implement `encode_subgoal_sequence(subgoals) -> HRR vector`
5. Implement `decode_subgoal_sequence(vector) -> list[Subgoal]`
6. SDM integration: `learn(mission_text, subgoals)` and `retrieve(mission_text) -> subgoals`
7. Implement `train_from_demos(demos)` — batch learning from demo episodes

### Task 4: CausalWorldModel Extensions
**Files:** `src/snks/agent/causal_world_model.py`

Steps:
1. Add PUT_NEXT_TO rule type with SDM instance
2. Add GO_TO_COMPLETE rule type with SDM instance (7 SDMs total)
3. Add encoding/query methods for new rules
4. Add `learn_from_transitions(frames)` — learn rules from Bot trajectory state changes
5. Extend `query_precondition()` for new action types
6. Extend `query_chain()` to support dynamic chains (not hardcoded 4-step)

### Task 5: Extended CausalPlanner
**Files:** `src/snks/agent/demo_guided_agent.py` (CausalPlanner class)

Steps:
1. Replace hardcoded `_chain_pass_locked_door()` with generic chain builder
2. Accept subgoal sequence from MissionModel
3. For each subgoal: query CausalWorldModel for preconditions
4. Recursive precondition insertion (max depth=3)
5. Inventory conflict detection: insert DROP() when carrying conflicts
6. Bind subgoals to SpatialMap positions

### Task 6: Extended SubgoalExecutor
**Files:** `src/snks/agent/demo_guided_agent.py` (SubgoalExecutor section)

Steps:
1. Extend inventory tracking: `carrying: Optional[tuple[str, str]]` (type, color)
2. Add DROP action handling in state machine
3. Add GO_TO completion detection (adjacency = done)
4. Add PUT_NEXT_TO execution (navigate adjacent + drop)
5. Mission completion tracking: `completed_subgoals` list

### Task 7: BossLevelAgent
**Files:** `src/snks/agent/boss_level_agent.py`

Steps:
1. Create BossLevelAgent extending DemoGuidedAgent
2. Override init: accept demos, train MissionModel + CausalWorldModel
3. Override `select_action()`: use mission_model for plan, not hardcoded chain
4. Override `reset()`: parse mission from obs, generate plan via MissionModel
5. Handle 22x22 SpatialMap sizing

### Task 8: Unit Tests (TDD)
**Files:** `tests/test_stage62_bosslevel.py`

Steps:
1. TestMissionEncoding: tokenize, encode, decode round-trip; positional encoding (4 tests)
2. TestMissionModel: learn from demos, retrieve subgoals, unseen generalization (4 tests)
3. TestSubgoalExtraction: extract from trajectory; compound; "after" reversal (3 tests)
4. TestNewRuleTypes: PUT_NEXT_TO, GO_TO_COMPLETE, learning from transitions (3 tests)
5. TestExtendedPlanner: generic chains, inventory conflict, recursive preconditions, max depth (4 tests)
6. TestExtendedExecutor: DROP, GO_TO completion, inventory tracking (3 tests)
7. TestBossLevelAgentUnit: init with demos, reset, stats (3 tests)

Run: `.venv/bin/python -m pytest tests/test_stage62_bosslevel.py -v`

### Task 9: Experiment exp117
**Files:** `experiments/exp117_bosslevel.py`

Steps:
1. exp117a: 50 seeds, BossLevelAgent trained on demos, gate ≥50%
2. exp117b: ablation — trained vs untrained MissionModel, gate delta ≥30%
3. exp117c: per-mission-type breakdown (diagnostic)

Run on minipc (read memory for how to deploy).

**Review checkpoint:** Review all gate results before declaring stage complete.

### Task 10: Stage Report
**Files:** `docs/reports/stage-62-report.md`

Steps:
1. Write report with results, architecture diagram, key decisions
2. Update memory with Stage 62 status
3. Commit all files
