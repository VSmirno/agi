# Crafter Survival Demo — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-07-crafter-survival-demo-design.md`

## Phase 1: Engine Core (no UI)

### Task 1.1: GameSnapshot + DemoEngine skeleton
**Files:** `demos/crafter_demo/engine.py`
**Steps:**
1. Create `demos/crafter_demo/` directory and `__init__.py`
2. Define `GameSnapshot` dataclass (all fields from spec)
3. Define `DemoEngine` class with:
   - `__init__(ckpt_path)` — loads checkpoint (priority chain from spec)
   - `snapshot_lock`, `model_lock` (threading.Lock)
   - `state` enum: playing/paused/stepping
   - `mode`: survival/interactive
   - `target_fps`: int
   - `event_log`: deque(maxlen=200)
4. Implement `DemoEngine.from_checkpoint(path)` — load encoder, detector, ConceptStore with fallback chain
5. Implement `_build_snapshot()` — creates GameSnapshot from current env state
6. Implement minimap extraction: crop 9x9 from `info["semantic"]`, color map, render to base64 PNG
7. Implement confidence flattening: iterate ConceptStore concepts → flat dict

**Verify:** `python -c "from demos.crafter_demo.engine import DemoEngine, GameSnapshot; print('ok')"`

### Task 1.2: Agent loop (tick-based)
**Files:** `demos/crafter_demo/agent_loop.py`
**Steps:**
1. Define `AgentState` dataclass: `plan`, `plan_index`, `nav_phase`, `retry_count`, `auto_reset_at`
2. Implement `tick(engine, agent_state) -> None`:
   - Step 1: observe, detect near, split inventory into survival/items
   - Step 2: reactive check (`rc.check_all()` → dict access)
   - Step 3: plan execution (navigate one step OR execute action)
   - Step 4: auto-plan if no plan (survival mode: lowest need → chain_gen)
   - Step 5: build minimap
   - Step 6: update snapshot under lock
   - Step 7: episode done → log, metrics, schedule auto-reset
3. Implement `navigate_one_step(env, info, target_name) -> bool` — single env.step toward target using semantic map
4. Implement `execute_plan_action(env, step) -> bool` — one env.step for action, check inventory delta
5. Implement `auto_plan(engine, agent_state)` — pick goal based on survival needs or resource progression

**Verify:** Unit test — create env, run 100 ticks, check snapshot updates

### Task 1.3: EnvThread + TrainThread
**Files:** `demos/crafter_demo/engine.py` (extend)
**Steps:**
1. Implement `EnvThread(threading.Thread)`:
   - `run()`: loop with `time.sleep(1/fps)`, calls `tick()`, respects play/pause/step
   - `cmd_queue`: `queue.Queue` for play/pause/step/reset/set_mode/set_goal commands
   - Process commands at start of each tick
2. Implement `TrainThread(threading.Thread)`:
   - `run()`: collect trajectories, train encoder, report progress via callback
   - On completion: acquire `model_lock`, swap encoder+detector, release
   - Reuse training logic from `exp128_text_visual.py` (collect + train phases)
3. Wire into DemoEngine: `start()`, `stop()`, `send_cmd()`

**Verify:** Script that starts engine, sends play, waits 2s, sends pause, prints snapshot count

## Phase 2: FastAPI Server

### Task 2.1: Server skeleton + REST
**Files:** `demos/crafter_demo/server.py`
**Steps:**
1. FastAPI app with `lifespan` handler: create DemoEngine on startup, stop on shutdown
2. Mount static files: `demos/crafter_demo/static/`
3. REST endpoints:
   - `GET /` — serve index.html
   - `GET /api/goals` — return `chain_gen.available_goals()`
   - `GET /api/status` — return current mode, state, episode, step
   - `POST /api/train` — start TrainThread, return 202
4. Add uvicorn runner in `if __name__ == "__main__"`

**Verify:** `curl http://localhost:8421/api/status` returns JSON

### Task 2.2: WebSocket endpoint
**Files:** `demos/crafter_demo/server.py` (extend)
**Steps:**
1. `@app.websocket("/ws")` endpoint
2. Receive loop: parse JSON commands, forward to `engine.send_cmd()`
3. Send loop: read snapshot at target FPS, serialize to JSON, send
4. Handle disconnect gracefully (env thread continues)
5. Handle `train_progress` messages from TrainThread

**Verify:** wscat connect, send `{"cmd":"play"}`, receive frame messages

## Phase 3: Browser UI

### Task 3.1: HTML layout + CSS
**Files:** `demos/crafter_demo/static/index.html`
**Steps:**
1. Three-column top layout: game frame (canvas 512x512), middle panel (survival+inventory+near), right panel (plan+reactive+minimap)
2. Two-column bottom: causal graph (SVG container), metrics (canvas sparklines)
3. Full-width footer: event log (scrollable) + Train button
4. Top bar: title, mode dropdown, play/pause/step buttons, FPS slider
5. CSS Grid layout, dark theme (match stage66 colors), responsive breakpoints

### Task 3.2: WebSocket client + state management
**Files:** `demos/crafter_demo/static/app.js`
**Steps:**
1. WS connect with auto-reconnect (2s delay)
2. Parse incoming messages: `frame` → update all panels, `train_progress` → update train modal
3. Send commands on button clicks
4. State object: last frame data, metrics history (array of last 100 episodes), log buffer
5. Render loop: on each `frame` message, update canvas + all panels

### Task 3.3: Game frame + minimap rendering
**Files:** `demos/crafter_demo/static/app.js` (extend)
**Steps:**
1. Main canvas: decode base64 PNG, draw scaled to 512x512 (nearest-neighbor for pixel art)
2. Minimap canvas: decode base64, draw scaled with color legend
3. Survival bars: colored div bars (green→yellow→red gradient based on value)
4. Inventory: simple key-value list
5. Near detection: badge with confidence value

### Task 3.4: Plan panel + reactive indicator
**Files:** `demos/crafter_demo/static/app.js` (extend)
**Steps:**
1. Plan list: checkmark/arrow/circle icons for done/active/pending steps
2. Reactive overlay: red banner when danger, yellow for survival need
3. Agent action indicator: current action + reason

### Task 3.5: Causal graph (SVG)
**Files:** `demos/crafter_demo/static/causal.js`
**Steps:**
1. Build SVG from confidence dict: nodes = unique concepts, edges = causal links
2. Edge thickness = confidence (0.5 thin → 1.0 thick)
3. Simple force-directed layout (fixed positions for known concepts: tree/stone/coal/iron/table left-to-right)
4. Update on each frame (confidence changes from verification)

### Task 3.6: Metrics sparklines
**Files:** `demos/crafter_demo/static/charts.js`
**Steps:**
1. Canvas-based sparklines for: episode length, resources collected, zombie encounters/survived
2. Rolling window of last 20 episodes
3. Update when episode ends (new data point)

### Task 3.7: Train modal
**Files:** `demos/crafter_demo/static/app.js` (extend)
**Steps:**
1. Modal dialog: epochs input, from-scratch vs finetune toggle
2. Progress bar + loss display during training
3. Auto-close on completion with success toast

## Phase 4: Integration + Polish

### Task 4.1: Copy checkpoints from minipc
**Steps:**
1. Wait for exp128 to complete on minipc
2. `scp minipc:/opt/agi/demos/checkpoints/exp128/ demos/checkpoints/exp128/`
3. Verify loading: start demo, confirm model loads

### Task 4.2: End-to-end test
**Steps:**
1. Start server with checkpoint
2. Open browser, verify all panels render
3. Test survival mode: agent survives, plan executes, reactive triggers
4. Test interactive mode: set goal, observe plan + execution
5. Test train mode: start training, observe progress, verify hot-swap
6. Test step mode: pause, step through, verify frame-by-frame
7. Test reset, mode switching, speed changes

### Task 4.3: Fallback mode (no checkpoint)
**Steps:**
1. Start server without checkpoint
2. Verify banner "No model loaded"
3. Verify random walk + survival bars work
4. Train from UI, verify hot-swap

## Dependencies

```
1.1 ──> 1.2 ──> 1.3 ──> 2.1 ──> 2.2 ──> 3.2
                                          ├──> 3.3
                                          ├──> 3.4
                                          ├──> 3.5
                                          ├──> 3.6
                                          └──> 3.7
        3.1 (parallel with 1.x/2.x)
        4.1 (parallel, when exp128 done)
        4.2 ──> 4.3 (after all above)
```

Tasks 3.3-3.7 can be parallelized after 3.2 is done.
Task 3.1 (HTML/CSS layout) has no code dependencies, can start anytime.
Task 4.1 (checkpoints) is blocked on exp128 completion on minipc.
