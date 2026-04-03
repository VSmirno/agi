# Implementation Plan: Stage 59 — SDM Learned Color Matching (LockedRoom)

**Spec:** `docs/superpowers/specs/2026-04-03-stage59-sdm-lockedroom-design.md`
**Branch:** `stage59-sdm-lockedroom`

---

## Task 1: Git setup + branch

1. Create branch `stage59-sdm-lockedroom` from main
2. Verify minigrid LockedRoom env works: `python -c "import gymnasium; env = gymnasium.make('MiniGrid-LockedRoom-v0'); env.reset(); print('OK')"`

**Verify:** branch exists, env imports OK

---

## Task 2: LockedRoomEnv wrapper

Create `src/snks/agent/sdm_lockedroom_agent.py` — start with the env wrapper only.

**LockedRoomEnv class:**
- `__init__(self, max_steps=1000)` — wraps `gymnasium.make('MiniGrid-LockedRoom-v0')`
- `reset(seed)` → `(obs_7x7, agent_col, agent_row, agent_dir, carrying_color, mission)`
- `step(action)` → `(obs_7x7, reward, term, trunc, agent_col, agent_row, agent_dir, carrying_color, mission)`
- `_extract(obs)` — reads `env.unwrapped.agent_pos`, `env.unwrapped.carrying`, mission text
- `get_all_doors()` → list of `(color_id, row, col, is_locked, is_open)` from `env.unwrapped.grid` (for oracle/debug)
- `get_all_keys()` → list of `(color_id, row, col)` from `env.unwrapped.grid`
- Grid size: `self.grid_width = 19, self.grid_height = 19` (LockedRoom is always 19x19)

**Key differences from SDMDoorKeyEnv:**
- Returns `carrying_color` (str|None) instead of `has_key` (bool) — need to know KEY COLOR
- Returns `mission` string
- 19x19 grid instead of 5x5

**Verify:** `pytest tests/test_stage59_lockedroom.py::test_env_wrapper` — reset, step, extract carrying color

---

## Task 3: MissionParser

Add to `src/snks/agent/sdm_lockedroom_agent.py`:

**MissionParser class:**
- `parse(mission: str) -> dict` — regex `r"get the (\w+) key from the (\w+) room, unlock the (\w+) door"`
- Returns `{"key_color": str, "room_color": str, "door_color": str}` or `None` if no match
- MiniGrid color names: `red, green, blue, purple, yellow, grey`
- MiniGrid color IDs: `{0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}`
- `color_name_to_id(name: str) -> int` — maps name to MiniGrid color ID
- `color_id_to_name(id: int) -> str` — reverse mapping

**Verify:** `pytest tests/test_stage59_lockedroom.py::test_mission_parser` — test with 5 different seeds

---

## Task 4: ColorStateEncoder

Add to `src/snks/agent/sdm_lockedroom_agent.py`:

**ColorStateEncoder class:**
- `__init__(self, codebook: VSACodebook)` 
- `encode_color_pair(key_color: str, door_color: str) -> torch.Tensor`
  - VSA: `bind(role("key_color"), filler(key_color)) ⊕ bind(role("door_color"), filler(door_color))`
  - This is the core: SDM stores (key_color, door_color) → reward
  - When key_color == door_color, the resulting VSA vector has a consistent pattern across all same-color pairs
- `encode_state(agent_row, agent_col, carrying_color, doors_seen, exploration_pct) -> torch.Tensor`
  - Full state encoding for subgoal selection (like AbstractStateEncoder but with multi-door awareness)

**Why color pair encoding works for generalization:**
- VSA `bind(role("key_color"), filler("red")) ⊕ bind(role("door_color"), filler("red"))` creates a unique vector for red-red
- VSA `bind(role("key_color"), filler("blue")) ⊕ bind(role("door_color"), filler("blue"))` creates a unique vector for blue-blue
- These are NOT similar to each other — SDM cannot generalize from red-red to blue-blue via similarity
- **Fix needed:** encode the RELATIONSHIP, not the colors: `bind(role("match"), filler("same"))` vs `bind(role("match"), filler("different"))`
- OR: encode as `state = VSA(key_color)`, `action = VSA(door_color)`. SDM address = `bind(state, action)`. Same-color pairs activate DIFFERENT locations but all get +reward. SDM reads: for each door_color, `read_reward(VSA(key_color), VSA(door_color))` — the one with same color has accumulated +reward from all previous same-color success episodes across ALL colors.
- **Wait — this actually DOES generalize** because `bind(VSA("red"), VSA("red"))` and `bind(VSA("blue"), VSA("blue"))` share ~50% bits by chance (random vectors). The SDM activation radius catches nearby locations. With 100+ training episodes covering all 6 colors, each same-color pair gets +reward. At read time, ANY same-color pair will have positive reward signal from its own training examples.

**Actual encoding strategy:**
- `state` = `VSA(key_color)` — one VSA vector per color name
- `action` = `VSA(door_color)` — same codebook, one per color name  
- SDM stores reward at `bind(state, action)` address
- At read: query all 6 door colors for held key color → argmax reward
- Each color pair trained independently. No cross-color generalization needed — just need enough training examples (50+ episodes = ~8 per color pair, sufficient)

**Verify:** `pytest tests/test_stage59_lockedroom.py::test_color_encoder` — encode, verify dimensions, verify different pairs give different vectors

---

## Task 5: SDMLockedRoomAgent — exploration + heuristic

Add to `src/snks/agent/sdm_lockedroom_agent.py`:

**SDMLockedRoomAgent class:**
- `__init__(grid_width=19, grid_height=19, dim=512, n_locations=1000, explore_episodes=50, use_mission=True)`
- Uses: SpatialMap(19, 19), FrontierExplorer, GridPathfinder, VSACodebook, SDMMemory, ColorStateEncoder
- Subgoal IDs: SG_EXPLORE=0, SG_GOTO_KEY=1, SG_GOTO_DOOR=2, SG_TOGGLE=3, SG_GOTO_GOAL=4, SG_DROP_KEY=5

**`reset_episode()`** — clear SpatialMap, episode state, but keep SDM

**`select_action(obs_7x7, agent_col, agent_row, agent_dir, carrying_color, mission)`:**
1. `spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)`
2. Check reflexes (same as SDMDoorKeyAgent but with color awareness)
3. Determine target via `_select_subgoal(carrying_color, mission)` 
4. Execute subgoal via BFS navigation (reuse `_execute_subgoal`, `_navigate_to`, `_turn_toward`, `_find_adjacent_walkable` from SDMDoorKeyAgent — COPY, don't import)

**`_select_subgoal(carrying_color, mission)` — EXPLORATION MODE:**
- If `use_mission` and mission parsed → target_key_color known
  - If not carrying → find key of target_key_color via `spatial_map.find_object_by_type_color(OBJ_KEY, color_id)` → GOTO_KEY or EXPLORE
  - If carrying → find locked door via `_find_locked_door()` → GOTO_DOOR or EXPLORE
  - If door open → find goal → GOTO_GOAL or EXPLORE
- If not using mission → heuristic: pick up any key, try locked door, if fail → DROP_KEY, try next key

**`_find_locked_door()` → (row, col, color_id) | None:**
- Iterate `spatial_map.grid`, find door with state==2 (locked)

**`_find_all_doors()` → list of (row, col, color_id, state):**
- Iterate `spatial_map.grid` for OBJ_DOOR cells

**`observe_result(obs_7x7, agent_col, agent_row, agent_dir, carrying_color, reward, success)`:**
- Update spatial_map
- If reward > 0: record (key_color, door_color) → +1.0 in SDM (amplify 10x)
- If toggle failed (agent tried locked door with wrong key): record (key_color, door_color) → -1.0

**Verify:** `pytest tests/test_stage59_lockedroom.py::test_agent_exploration` — agent with mission can solve LockedRoom with exploration heuristic (no SDM needed yet)

---

## Task 6: SDMLockedRoomAgent — SDM planning mode

Extend the agent:

**`_select_subgoal(carrying_color, mission)` — PLANNING MODE (after explore_episodes):**
- If carrying key (color=C):
  - For each known door (color=D, locked): query `sdm.read_reward(VSA(C), VSA(D))`
  - Go to door with highest positive reward (SDM learned which color matches)
  - If no positive signal → fallback: use mission (Phase B) or random (Phase A)
- If not carrying:
  - If use_mission → parse mission → go to correct key
  - If no mission → find any key, pickup (exploration will teach which door)

**`_record_color_transition(key_color, door_color, success)`:**
- `state = codebook.filler(f"key_{key_color}")`
- `action = codebook.filler(f"door_{door_color}")`
- `reward = 1.0 if success else -1.0`
- `sdm.write(state, action, state, reward)` — next_state=state (don't care about next state, only reward)
- Amplify: 10 writes for success, 5 for failure

**`_episode_done(success)`:**
- Increment episode counter
- If done exploring → switch to planning mode

**Verify:** `pytest tests/test_stage59_lockedroom.py::test_sdm_learning` — train 20 episodes, verify SDM read_reward(same_color) > read_reward(different_color) for at least 4/6 colors

---

## Task 7: Unit tests

Create `tests/test_stage59_lockedroom.py`:

1. `test_env_wrapper` — reset, step, get carrying color, get mission
2. `test_mission_parser` — parse 5 seeds, verify color extraction
3. `test_color_encoder` — encode pairs, verify dim=512, verify different pairs differ
4. `test_agent_exploration` — agent with mission solves LockedRoom (heuristic mode), ≥3/5 seeds
5. `test_sdm_learning` — train 20 episodes, verify SDM reward signal distinguishes same vs different color
6. `test_sdm_planning` — train 50 episodes, then eval 10 seeds in planning mode, success > 50%
7. `test_drop_key_recovery` — mock scenario where agent has wrong key, verify DROP_KEY subgoal triggers

**Verify:** `pytest tests/test_stage59_lockedroom.py -v` — all pass

---

## Task 8: Experiment script

Create `src/snks/experiments/exp113_sdm_lockedroom.py`:

**CLI args:**
- `--phase` = `B` (mission) or `A` (no mission)
- `--train-seeds` = range (default 0-49 for B, 0-99 for A)
- `--eval-seeds` = range (default 1000-1199)
- `--max-steps` = 1000
- `--dim` = 512
- `--n-locations` = 1000
- `--output` = JSON file path

**Experiment flow:**
1. Training phase: run train seeds, SDM accumulates
2. Eval phase: run eval seeds, measure success rate
3. Ablation: re-run eval with (a) empty SDM, (b) heuristic-only (random door for no-mission)
4. Report: JSON with success rates, SDM writes count, per-seed results

**Output format:**
```json
{
  "phase": "B",
  "train_seeds": 50,
  "eval_seeds": 200,
  "sdm_writes": 1234,
  "results": {
    "sdm_trained": {"success_rate": 0.85, "mean_steps": 456},
    "heuristic_only": {"success_rate": 0.16, "mean_steps": 890},
    "sdm_untrained": {"success_rate": 0.15, "mean_steps": 900}
  },
  "per_seed": [...]
}
```

**Verify:** Run locally with 3 train + 5 eval seeds, verify JSON output format correct

---

## Task 9: Run experiments on minipc

1. `git push origin stage59-sdm-lockedroom`
2. SSH minipc: `cd /opt/agi && git fetch && git checkout stage59-sdm-lockedroom && git pull`
3. Run exp113a (Phase B, mission): `tmux new -s exp113a "python -m snks.experiments.exp113_sdm_lockedroom --phase B --output /opt/agi/_docs/exp113a_results.json"`
4. Check results, verify gate ≥70%
5. Run exp113b (Phase B ablation)
6. Run exp113c (Phase A, no mission): `--phase A --train-seeds 0-99`
7. Run exp113d (Phase A ablation)
8. Pull results: `git add _docs/exp113*; git commit; git push`

**Verify:** All gates pass

---

## Task 10: Report + merge

1. Write `docs/reports/stage-59-report.md`
2. Update ROADMAP.md — Stage 59 COMPLETE
3. Merge branch to main
4. Update _docs/autonomous-log

**Verify:** ROADMAP updated, report committed

---

## Review Checkpoints

- **After Task 5:** Agent can solve LockedRoom with mission heuristic → validates env wrapper + navigation
- **After Task 7:** All unit tests pass → validates SDM color learning mechanism locally
- **After Task 9:** Experiment results → validates gates on real hardware
