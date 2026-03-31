# Stage 25: Autonomous Goal Composition

**Version:** 1.0
**Date:** 2026-03-31
**Status:** DESIGN APPROVED

---

## Goal

Autonomous multi-step goal decomposition and execution in MiniGrid DoorKey environments. The agent receives a high-level instruction ("use the key to open the door and then get to the goal"), autonomously decomposes it into sub-goals via causal reasoning, executes them, learns from experience, and transfers knowledge across layouts.

### What This Proves

1. **Autonomous decomposition** — agent infers sub-goals without hardcoded task structure
2. **Causal learning** — agent discovers action prerequisites from experience (not seeded)
3. **Incremental learning** — agent improves across episodes within a session
4. **Transfer** — causal knowledge generalizes across grid layouts

### Core Principles

- Concepts first: agent reasons about `key_held`, `door_open` as SKS concepts, not strings
- Learning while doing: no separate training phase, every action updates world model
- Backward chaining: plan from goal to current state, not forward search

---

## Architecture

```
                "use the key to open the door and get to the goal"
                                    │
                                    ▼
                            ┌──────────────┐
                            │  GoalAgent   │
                            │              │
                            │  mode select │
                            └──┬───────┬───┘
                   sequential? │       │ single goal?
                               ▼       ▼
                        Text-driven  Goal-driven
                        (chunker     (backward
                         split)       chaining)
                               │       │
                               ▼       ▼
                        ┌──────────────────┐
                        │  SubGoalExecutor │
                        │                  │
                        │  for each sub:   │
                        │  1. perceive     │
                        │  2. navigate     │
                        │  3. act          │
                        │  4. observe      │
                        │  5. learn        │
                        └──────┬───────────┘
                               │
                      failure? │
                               ▼
                        ┌──────────────┐
                        │  Exploration │
                        │  (try nearby │
                        │   objects,   │
                        │   learn)     │
                        └──────────────┘
```

### Mode Selection

Simple heuristic:
- Instruction contains "then" → **text-driven**: chunker splits at SEQ_BREAK, execute sequentially
- Otherwise → **goal-driven**: backward chaining from final goal

### GoalAgent

Thin orchestrator (~150 lines). Owns:
- `BabyAIExecutor` — single sub-task execution primitive
- `CausalWorldModel` — accumulates causal links across episodes
- `CausalLearner` — before/after observation → causal link updates
- `BlockingAnalyzer` — obstacle detection + resolution via causal query
- `GridPerception` — re-perceive after each action
- `GroundingMap` — persistent between episodes

---

## Backward Chaining

Goal-driven decomposition from final goal to current state:

```
Goal: "get to the goal"
  │
  ├─ perceive → goal at (6,6), agent at (1,5)
  ├─ plan_path(goal) → FAIL (door locked, path blocked)
  │
  ├─ BlockingAnalyzer.find_blocker() → door(locked) at (2,3)
  ├─ BlockingAnalyzer.suggest_resolution() queries CausalWorldModel:
  │     "what action changes door_locked → door_open?"
  │
  ├─ If causal link exists: toggle + key_held → door_open
  │     ├─ Sub-goal: "toggle door" (requires key_held)
  │     ├─ key_held present? → no
  │     ├─ Query: "what produces key_held?" → pickup + key_present
  │     └─ Sub-goal: "pickup key"
  │
  ├─ If NO causal link: (first episode)
  │     ├─ Exploration: try interactions with nearby objects
  │     ├─ Observation: "toggle with key_held → door_open"
  │     └─ CausalLearner records the link
  │
  └─ Final plan (after chaining):
        1. pickup key
        2. go to door
        3. toggle door
        4. go to goal
```

**Recursion depth limit:** 3 (pickup → toggle → goto — maximum for MiniGrid).

### BlockingAnalyzer

Identifies what blocks a path and queries CausalWorldModel for resolution:

```python
@dataclass
class Blocker:
    cell_type: str       # "door"
    cell_color: str      # "yellow"
    pos: tuple[int, int]
    state: str           # "locked"
    sks_id: int          # SKS concept for this object

@dataclass
class SubGoal:
    action: str          # "toggle", "pickup"
    target_word: str     # "door", "key"
    prerequisite: SubGoal | None  # recursive chain

class BlockingAnalyzer:
    def find_blocker(self, grid, agent_pos, target_pos) -> Blocker | None:
        """Scan grid between agent and target for impassable cells."""

    def suggest_resolution(self, blocker, causal_model, current_sks) -> SubGoal | None:
        """Query CausalWorldModel for action that removes blocker."""
```

---

## Causal Learning

Every action is a learning signal. CausalLearner observes before/after state:

```
before_sks = {key_present, door_locked, agent_at_key}
action = pickup (3)
after_sks = {key_held, door_locked}
───────────────────────────────────
delta (added): {key_held}
removed: {key_present, agent_at_key}
───────────────────────────────────
CausalWorldModel.add_link:
  context: {key_present}
  action: pickup
  effect: {key_held}
  strength: 1.0
```

```python
class CausalLearner:
    """Observes action effects and updates CausalWorldModel."""

    def __init__(self, causal_model: CausalWorldModel):
        self._model = causal_model
        self._before: set[int] | None = None

    def before_action(self, current_sks: set[int]) -> None:
        self._before = set(current_sks)

    def after_action(self, action: int, current_sks: set[int]) -> None:
        if self._before is None:
            return
        added = current_sks - self._before
        if added:
            self._model.add_link(
                context_sks=frozenset(self._before),
                action=action,
                effect_sks=frozenset(added),
            )
        self._before = None
```

### Persistence Between Episodes

- `CausalWorldModel` + `GroundingMap` live in GoalAgent, not recreated per episode
- Episode 1: empty model → exploration → learns pickup→key_held, toggle+key→door_open
- Episode 2: model has links → backward chaining succeeds → solves immediately
- Transfer to 6x6: same causal links, different layout → should work

### Key Design Decision: Type-Based SKS

SKS IDs are bound to (object_type, color), NOT position. GridPerception already does this. So causal link "pickup + key_present → key_held" transfers to any key in any position.

---

## Exploration Fallback

When backward chaining cannot find a resolution (empty causal links):

1. Find all interactive objects in the grid (GridPerception.objects)
2. Navigate to each, try pickup/toggle
3. CausalLearner records before/after for each attempt
4. After max_tries=5, retry backward chaining with updated model

Exploration is **targeted** (objects near the obstacle), not random walk.

```python
def explore_near(self, target_pos, max_tries=5) -> bool:
    """Try interactions with objects near target.
    Returns True if any new causal link discovered."""
```

### Typical Learning Trajectory (DoorKey)

| Episode | What happens | CausalWorldModel after |
|---------|-------------|----------------------|
| 1 | goto(goal) → blocked by door → explore → pickup key → learn | pickup+key_present → key_held |
| 1 (cont.) | retry toggle door → success → learn | toggle+key_held+door_locked → door_open |
| 1 (cont.) | goto goal → success | — |
| 2 | goto(goal) → blocked → query model → chain: pickup→toggle→goto → **solve immediately** | — |
| 3 (6x6) | **same links** → plan immediately → solve with different layout | transfer confirmed |

---

## Experiments

### Exp 58: Autonomous Goal Decomposition

**Purpose:** GoalAgent correctly decomposes DoorKey mission into sub-goals.

**Protocol:**
- 20 episodes DoorKey-5x5, varying seeds
- Agent receives original mission text
- Check: correct sub-goals identified (pickup key, toggle door, goto goal)

**Gate:**
```
decomposition_accuracy >= 0.7   # 14/20 correct decompositions
```

### Exp 59: Causal Learning Speed

**Purpose:** How many episodes until agent learns DoorKey rules.

**Protocol:**
- CausalWorldModel starts empty
- Series of 10 episodes DoorKey-5x5
- After each episode: check for links pickup→key_held, toggle+key→door_open

**Gate:**
```
episodes_to_learn <= 5          # all key links learned within 5 episodes
```

### Exp 60: Multi-Trial Success Rate

**Purpose:** Agent improves across episodes (learning curve).

**Protocol:**
- 10 series of 5 episodes each (varying seeds)
- CausalWorldModel preserved within series, reset between series
- Measure success rate at each trial position (1st, 2nd, ..., 5th)

**Gate:**
```
trial_5_success_rate >= 0.6     # by 5th attempt >= 60% of series solved
trial_1_success_rate < trial_5  # learning curve grows (not random)
```

### Exp 61: Transfer Across Layouts

**Purpose:** Knowledge from 5x5 transfers to 6x6.

**Protocol:**
- Phase 1: 5 episodes DoorKey-5x5 (training)
- Phase 2: 10 episodes DoorKey-6x6 with **same** CausalWorldModel
- Phase 3 (control): 10 episodes DoorKey-6x6 with **empty** CausalWorldModel

**Gate:**
```
transfer_success >= 0.3           # 3/10 on 6x6 first-try
transfer_success > control + 0.1  # transfer better than no knowledge
```

---

## Files

| File | Purpose | New/Modified |
|------|---------|-------------|
| `src/snks/language/goal_agent.py` | GoalAgent orchestrator | New |
| `src/snks/language/blocking_analyzer.py` | Obstacle detection + causal resolution | New |
| `src/snks/language/causal_learner.py` | Before/after → causal link updates | New |
| `src/snks/language/babyai_executor.py` | Minor: expose re-perception | Modified |
| `src/snks/agent/causal_model.py` | Add `add_link()` if missing | Modified (if needed) |
| `tests/test_goal_agent.py` | Unit tests with mock env | New |
| `src/snks/experiments/exp58_goal_decomposition.py` | Exp 58 runner | New |
| `src/snks/experiments/exp59_causal_learning.py` | Exp 59 runner | New |
| `src/snks/experiments/exp60_multitrial_success.py` | Exp 60 runner | New |
| `src/snks/experiments/exp61_transfer_layouts.py` | Exp 61 runner | New |

---

## Dependencies

- Stage 24a: RuleBasedChunker (sequential parsing)
- Stage 24b: InstructionPlanner, CausalWorldModel
- Stage 24c: BabyAIExecutor, GridPerception, GridNavigator
- `minigrid>=3.0.0` (DoorKey environments)

---

## Risks

1. **Exploration may not find key** — in large grids, key could be far from door. Mitigation: explore ALL interactive objects, not just nearby ones.
2. **CausalWorldModel context too specific** — link with full before_sks won't match slightly different state. Mitigation: use minimal context (only directly relevant SKS IDs).
3. **Door detection** — GridNavigator returns empty path but doesn't say WHY. BlockingAnalyzer needs to scan grid to find the locked door on the path.
4. **"use the key to open the door"** — original DoorKey mission doesn't parse well with RuleBasedChunker. Goal-driven mode bypasses this by working from final goal, not from text parsing.
