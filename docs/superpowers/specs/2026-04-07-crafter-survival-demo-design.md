# Crafter Survival Demo — Design Spec

**Date:** 2026-04-07
**Status:** Draft
**Depends on:** Stage 71 (exp128 checkpoint)

## Goal

Browser-based interactive demo for the Crafter agent pipeline. Default mode: autonomous survival. Shows the full Stage 71 stack — ConceptStore, backward chaining, reactive survival, visual grounding — in real time through a WebSocket-driven UI.

## Modes

1. **Survival** (default) — agent autonomously survives: gathers resources via backward chaining, reacts to danger (flee/attack), manages food/drink/energy needs.
2. **Interactive** — user sets a goal (e.g. "iron_item"), observes the agent plan and execute. Idle if no goal set.
3. **Train** — train encoder from scratch or finetune from checkpoint. Progress shown in UI. Hot-swaps model into running env on completion.

All modes controlled through browser UI buttons — single FastAPI server manages everything.

## Architecture

```
Browser (vanilla JS SPA)
    ↕ WebSocket /ws (JSON, ~15 FPS)
    ↕ REST /api/* (train, goals, checkpoint)
FastAPI server (async, port 8421)
    │
    DemoEngine (shared state)
    ├── EnvThread — game loop (env.step → agent → snapshot)
    ├── TrainThread — encoder training (on demand)
    ├── CrafterPixelEnv
    ├── CNNEncoder + NearDetector
    ├── ConceptStore + ChainGenerator
    └── ReactiveCheck
```

### Threads

| Thread | Role | Lifecycle |
|--------|------|-----------|
| Main | FastAPI async event loop, WS, REST | Always |
| EnvThread | `env.step()` → agent decision → snapshot update | Always (play/pause/step) |
| TrainThread | Collect trajectories → train encoder → hot-swap | On demand |

### Why threads, not processes

Env step is ~5-10ms, detector inference ~2ms — no GIL contention issue for I/O-bound FastAPI. Training runs in burst and swaps model atomically. Multiprocessing would add serialization overhead for shared state (frames, snapshots) with no benefit for a demo.

## WebSocket Protocol

### Server → Client: game frame

Sent every tick in play mode, or once per step command.

```json
{
  "type": "frame",
  "frame": "<base64 PNG 64x64>",
  "step": 1234,
  "episode": 3,
  "mode": "survival",
  "agent": {
    "action": "do",
    "near": "tree",
    "reason": "plan|reactive|explore",
    "plan_step": 2,
    "plan_total": 7
  },
  "survival": {"health": 7, "food": 5, "drink": 8, "energy": 6},
  "inventory": {"wood": 3, "wood_pickaxe": 1},
  "plan": [
    {"label": "collect wood", "status": "done"},
    {"label": "place table", "status": "active"},
    {"label": "make wood pickaxe", "status": "pending"}
  ],
  "reactive": {"action": "flee", "reason": "danger", "target": "zombie"} | null,
  "confidence": {"tree_do_wood": 0.95, "stone_do_stone_item": 0.8},
  "metrics": {
    "episode_length": 234,
    "resources_collected": 12,
    "zombie_encounters": 2,
    "zombie_survived": 2
  },
  "minimap": "<base64 PNG 9x9 semantic>",
  "log": ["[1230] collected wood", "[1232] zombie detected -> flee"]
}
```

### Server → Client: training progress

```json
{
  "type": "train_progress",
  "phase": "JEPA",
  "epoch": 45,
  "total_epochs": 150,
  "loss": 0.023
}
```

### Client → Server: commands

```json
{"cmd": "play"}
{"cmd": "pause"}
{"cmd": "step"}
{"cmd": "set_mode", "mode": "survival"}
{"cmd": "set_goal", "goal": "iron_item"}
{"cmd": "train", "epochs": 150}
{"cmd": "reset"}
{"cmd": "set_speed", "fps": 10}
```

## UI Layout

```
+-------------------------------------------------------------+
|  Crafter Survival Demo          [Survival v] [Play | Pause |Step]
+--------------------+------------------+---------------------+
|                    |  Survival Bars   |  Plan               |
|                    |  HP  [=======..]  |  v collect wood     |
|   Game Frame       |  Food [=====...]  |  > place table      |
|   512 x 512        |  Drink[=======.] |  o make pickaxe     |
|   (canvas)         |  Ener [======..] |  o collect stone    |
|                    |                  |                     |
|                    |  Inventory       |  Reactive           |
|                    |  wood: 3         |  ! zombie -> flee   |
|                    |  pickaxe: 1      |                     |
|                    |                  |  Minimap            |
|                    |  Near Detection  |  [9x9 semantic]     |
|                    |  [tree] 0.94     |                     |
+--------------------+------------------+---------------------+
|  Causal Graph (SVG)              |  Metrics (Sparklines)   |
|  tree--do-->wood                 |  episode length: ~~^~~  |
|  wood--place-->table             |  resources: ~~^~~       |
|  table--make-->pickaxe           |  encounters: ~~         |
|  (line thickness = confidence)   |                         |
+----------------------------------+-------------------------+
|  Event Log                                    [Train btn]  |
|  [1234] collected wood  [1236] flee  [1240] placed table   |
+------------------------------------------------------------+
```

Three columns top, two columns bottom, event log full-width footer. Responsive: collapses to two columns on narrow screens, bottom panels become tabs.

## DemoEngine

```python
@dataclass
class GameSnapshot:
    frame_b64: str          # base64 PNG
    step: int
    episode: int
    mode: str
    agent_action: str
    agent_near: str
    agent_reason: str       # "plan" | "reactive" | "explore"
    plan_step: int
    plan_total: int
    survival: dict          # {health, food, drink, energy}
    inventory: dict
    plan: list[dict]        # [{label, status}]
    reactive: dict | None
    confidence: dict        # {rule_key: float}
    metrics: dict
    minimap_b64: str
    log_lines: list[str]    # last 5 new events
```

### Agent loop (EnvThread)

```
each tick:
  1. near = detector.detect(pixels)
  2. reactive = rc.check_all(near, inventory)
  3. if reactive.action:           # danger or survival need
       execute reactive (flee/attack/seek/sleep)
       log event
  4. elif current_plan:
       execute next plan step (navigate toward target, then act)
       if step succeeds: advance plan_index, verify prediction, log
       if step fails: retry or replan
  5. else (no plan, no reactive):
       survival mode: auto-plan based on lowest need or resource gathering
       interactive mode: idle (wait for set_goal)
  6. update snapshot under lock
  7. if episode done: log cause, update metrics, auto-reset after 2s
```

### Train flow

1. UI sends `{"cmd": "train", "epochs": 150}`
2. FastAPI spawns TrainThread
3. TrainThread: collect trajectories (random walk + controlled) → train encoder → callbacks update shared progress state
4. WS reads progress state → streams `train_progress` messages
5. On completion: atomic swap of encoder + detector under lock
6. EnvThread picks up new model next tick

## Checkpoint Loading

Priority order:
1. `demos/checkpoints/exp128/final/` — full trained model
2. `demos/checkpoints/exp128/phase3/` — trained encoder, no verification
3. `demos/checkpoints/exp128/phase1/` — nav encoder + grounded ConceptStore
4. `demos/checkpoints/stage66.pt` — legacy CNN encoder only
5. No checkpoint — random agent, Train required

Each checkpoint dir contains:
- `encoder.pt` — CNNEncoder state_dict
- `detector.pt` — NearDetector (head + encoder state_dicts)
- `concept_store/concepts.json` + `concept_store/tensors.pt`

Fallback without checkpoint: ConceptStore loaded from `configs/crafter_textbook.yaml` (rules only, no visual grounding). NearDetector absent → agent uses random walk. UI shows "No model — Train to start".

## File Structure

```
demos/crafter_demo/
  server.py          # FastAPI app, WS/REST endpoints, startup, main
  engine.py          # DemoEngine, GameSnapshot, EnvThread, TrainThread
  agent_loop.py      # Survival/interactive agent logic, plan execution
  static/
    index.html       # SPA layout, panels, controls
    app.js           # WS client, state mgmt, render loop
    causal.js        # Causal graph SVG renderer
    charts.js        # Sparkline metrics (vanilla Canvas)
```

## Dependencies

- `fastapi` — async web framework
- `uvicorn` — ASGI server
- `websockets` — transitive via FastAPI

No frontend frameworks. Vanilla JS + Canvas + SVG. No build step.

## Edge Cases

| Case | Behavior |
|------|----------|
| No checkpoint | Banner "No model loaded", random walk, Train button highlighted |
| Episode death | Log cause, metrics update, auto-reset 2s (or click) |
| Train during play | Env continues with current detector, hot-swap on completion |
| WS disconnect | Client auto-reconnect 2s, env thread unaffected |
| Train failure | Error in UI toast, env thread continues |

## Performance

- 64x64 PNG base64 ~ 2-4 KB per frame
- At 15 FPS ~ 60 KB/s bandwidth — negligible
- env.step() ~ 5-10ms, detector ~ 2ms CPU → ceiling ~50 FPS
- 15 FPS target is comfortable

## Non-Goals

- No multi-user support (single env instance)
- No GPU inference in demo (CPU-only, models are small)
- No persistent metrics DB (in-memory per session)
- No mobile-specific UI optimization
