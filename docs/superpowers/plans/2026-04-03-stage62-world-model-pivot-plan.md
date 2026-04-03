# Stage 62 Pivot: Unified World Model — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-03-stage62-world-model-pivot-design.md`

## Tasks

### Task 1: Synthetic Transition Generator
**File:** `src/snks/agent/world_model_trainer.py`

Steps:
1. Create `generate_synthetic_transitions()` — spawn MiniGrid scenarios for every (obj_type, color, state, action, carrying) combination
2. Create `extract_demo_transitions(demos)` — from Bot demo frames, extract (before, action, after) where state changed
3. Output: list of `Transition(situation: dict, action: str, outcome: dict, reward: float)`
4. Verify: ≥5000 synthetic + ≥2000 demo transitions

### Task 2: Unified World Model
**File:** `src/snks/agent/unified_world_model.py`

Steps:
1. Create `UnifiedWorldModel(dim=1024, n_locations=10000)`
2. VSA encoding: situation facts as role-filler pairs → bundle → SDM address
3. `train(transitions)` — write all transitions to single SDM
4. `query(situation, action) → outcome` — predict what happens
5. `query_qa(question_type, params) → answer` — structured QA
6. `query_plan(goal, current_state, max_steps=6) → action_list` — forward chaining
7. Verify: imports, basic encode/decode roundtrip

### Task 3: QA Battery Tests
**File:** `tests/test_stage62_world_model.py`

Steps:
1. Level 1: 25 object identity questions (gate ≥95%)
2. Level 2: 25 precondition questions (gate ≥90%)
3. Level 3: 25 consequence questions (gate ≥85%)
4. Level 4: 25 reasoning questions (gate ≥80%)
5. All tests use trained UnifiedWorldModel as fixture
6. Run: `.venv/bin/python -m pytest tests/test_stage62_world_model.py -v`

### Task 4: Generative Planning Tests
**File:** `tests/test_stage62_world_model.py` (additional class)

Steps:
1. 20 planning scenarios (simple → complex)
2. Plan validator: check preconditions, check final state = goal
3. Gate: ≥80% correct plans

### Task 5: Gate Experiment
**File:** `experiments/exp118_world_model_qa.py`

Steps:
1. Train UnifiedWorldModel on synth + demo transitions
2. Run full QA battery (100 questions)
3. Run planning scenarios (20)
4. Report per-level accuracy + overall
5. Run on minipc: `HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python3 experiments/exp118_world_model_qa.py`

### Task 6: Commit and Report
1. Update stage-62-report.md with pivot results
2. Update memory
3. Commit all
