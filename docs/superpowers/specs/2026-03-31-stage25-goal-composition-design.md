# Stage 25: Autonomous Goal Composition

**Version:** 1.2
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
                            │  goal-driven │
                            │  (always)    │
                            └──────┬───────┘
                                   │
                                   ▼
                        ┌────────────────────┐
                        │  Backward Chaining │
                        │                    │
                        │  goal → blocker →  │
                        │  resolve → sub-goal│
                        │  (recursive)       │
                        └──────┬─────────────┘
                               │
                               ▼
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
                        │  (try ALL    │
                        │   objects,   │
                        │   learn)     │
                        └──────────────┘
```

### Mode Selection

Always **goal-driven** (backward chaining). The DoorKey mission text ("use the key to open the door and then get to the goal") contains "then" but the first clause ("use the key to open the door") doesn't parse as a clean imperative — "use" is not in ACTION_PHRASES. Rather than adding fragile text heuristics, GoalAgent always extracts the **final goal** from the instruction (last noun phrase → target object) and works backward from there. This is more robust and aligns with the core principle: reason from goals, not from text.

### GoalAgent

Thin orchestrator (~150 lines). Owns:
- `BabyAIExecutor` — single sub-task execution primitive
- `CausalWorldModel` — accumulates causal links across episodes (requires `CausalAgentConfig`; GoalAgent creates a default config with `causal_min_observations=1` for fast learning)
- `CausalLearner` — before/after observation → causal link updates
- `BlockingAnalyzer` — obstacle detection + resolution via causal query
- `GridPerception` — re-perceive after each action (emits state-dependent SKS, see below)
- `GroundingMap` — persistent between episodes

---

## State-Dependent SKS IDs

GridPerception currently maps `(object_type, color)` → SKS ID. For causal learning, the agent must distinguish **state predicates**: `key_present` (on floor) vs `key_held` (carried), `door_locked` vs `door_open`.

**Extension to GridPerception:** emit state-dependent SKS IDs:

```python
# State-dependent concept IDs (reserved range 50-99).
# These are STABLE IDs — _split_context must treat them as stable
# (not coarsen with % n_bins). See CausalWorldModel section below.
SKS_KEY_PRESENT = 50    # key visible on grid floor
SKS_KEY_HELD = 51       # agent carrying key (env.unwrapped.carrying)
SKS_DOOR_LOCKED = 52    # door exists and is_locked=True
SKS_DOOR_OPEN = 53      # door exists and is_open=True
SKS_GOAL_PRESENT = 54   # goal cell visible

# NOTE: existing SKS_AGENT=1 and SKS_AGENT_CARRYING=2 from grid_perception.py
# are kept but NOT used for causal learning. SKS_KEY_HELD (51) replaces
# SKS_AGENT_CARRYING (2) for causal reasoning — they are semantically
# equivalent but SKS_KEY_HELD is in the stable range.

class GridPerception:
    def perceive(self, grid, agent_pos, agent_dir, carrying=None) -> set[int]:
        """Extended: also emits state predicates.

        - key on floor → SKS_KEY_PRESENT
        - carrying is Key → SKS_KEY_HELD
        - door.is_locked → SKS_DOOR_LOCKED
        - door.is_open → SKS_DOOR_OPEN
        - goal cell → SKS_GOAL_PRESENT

        Args:
            carrying: env.unwrapped.carrying (WorldObject or None).
                      All existing callers (BabyAIExecutor, experiments)
                      must be updated to pass this parameter.
        """
```

This is the minimal extension needed for causal learning to work.

---

## Backward Chaining

Goal-driven decomposition from final goal to current state:

```
Goal: "get to the goal"
  │
  ├─ perceive → goal at (6,6), agent at (1,5)
  ├─ plan_path(goal) → PathResult.BLOCKED (door locked)
  │
  ├─ BlockingAnalyzer.find_blocker() → door(locked) at (2,3)
  ├─ BlockingAnalyzer.suggest_resolution() queries CausalWorldModel:
  │     query_by_effect({SKS_DOOR_OPEN}) → (toggle, {SKS_KEY_HELD, SKS_DOOR_LOCKED})
  │
  ├─ If causal link exists: toggle + key_held → door_open
  │     ├─ Sub-goal: "toggle door" (requires key_held)
  │     ├─ SKS_KEY_HELD in current_sks? → no
  │     ├─ query_by_effect({SKS_KEY_HELD}) → (pickup, {SKS_KEY_PRESENT})
  │     └─ Sub-goal: "pickup key"
  │
  ├─ If NO causal link: (first episode)
  │     ├─ Exploration: try ALL interactive objects in grid
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

### GridNavigator Path Result

Current `plan_path()` returns `[]` for both "already there" and "no path". This is ambiguous. **Fix:** add a return wrapper with proper enum:

```python
import enum

class PathStatus(enum.Enum):
    OK = "ok"
    BLOCKED = "blocked"
    ALREADY_THERE = "already_there"

@dataclass
class PathResult:
    actions: list[int]
    status: PathStatus

class GridNavigator:
    def plan_path_ex(self, grid, agent_pos, agent_dir, target_pos,
                     stop_adjacent=False) -> PathResult:
        if agent_pos == target_pos:
            return PathResult([], PathStatus.ALREADY_THERE)
        path = _bfs(grid, agent_pos, target_pos)
        if path is None:
            return PathResult([], PathStatus.BLOCKED)
        # ... convert to actions ...
        return PathResult(actions, PathStatus.OK)
```

Original `plan_path()` remains for backward compatibility.

### BlockingAnalyzer

Identifies what blocks a path and queries CausalWorldModel for resolution:

```python
@dataclass
class Blocker:
    cell_type: str       # "door"
    cell_color: str      # "yellow"
    pos: tuple[int, int]
    state: str           # "locked"
    sks_id: int          # SKS concept for this object state (e.g. SKS_DOOR_LOCKED)

@dataclass
class SubGoal:
    action: str          # "toggle", "pickup"
    target_word: str     # "door", "key"
    target_sks: int      # SKS ID of target state predicate
    prerequisite: SubGoal | None  # recursive chain

class BlockingAnalyzer:
    def find_blocker(self, grid, agent_pos, target_pos) -> Blocker | None:
        """Scan grid for impassable cells on shortest path line.

        Strategy: BFS from agent to target. If BFS fails, scan all cells
        for locked doors / blocking objects. Return the one closest to
        the agent-target line.
        """

    def suggest_resolution(self, blocker, causal_model, current_sks) -> SubGoal | None:
        """Query CausalWorldModel via query_by_effect() for action
        that produces the desired state change."""
```

---

## CausalWorldModel Extensions

### Issue: No reverse lookup

The existing `CausalWorldModel` supports forward queries: `predict_effect(context, action) → effect`. Backward chaining needs **reverse queries**: `query_by_effect(desired_effect) → (action, required_context)`.

**New method on CausalWorldModel:**

```python
def query_by_effect(
    self, desired_effect_sks: frozenset[int], min_confidence: float = 0.1,
) -> list[tuple[int, frozenset[int], float]]:
    """Reverse lookup: find (action, context) pairs that produce desired effect.

    Scans all stored transitions for ones whose effect_sks overlaps
    with desired_effect_sks. Uses overlap (intersection) matching
    because observe_transition stores symmetric_difference as effect —
    so effect_sks contains both APPEARING and DISAPPEARING SKS IDs.

    Example: toggle door with key produces effect_sks =
    {SKS_DOOR_OPEN, SKS_DOOR_LOCKED, SKS_KEY_HELD} (open appeared,
    locked disappeared, key consumed). Querying for {SKS_DOOR_OPEN}
    matches via overlap.

    Returns:
        List of (action, required_context_sks, confidence) sorted by confidence desc.
    """
```

This is a linear scan over `get_causal_links()` — acceptable for the small number of links in MiniGrid domains.

### Issue: `_split_context` must preserve state-dependent IDs

State-dependent SKS IDs (50-99) must NOT be coarsened by `_split_context`. Fix: extend `_split_context` to treat IDs in range 50-99 as stable (alongside IDs >= `_PERCEPTUAL_HASH_OFFSET`):

```python
_STATE_SKS_RANGE = range(50, 100)

def _split_context(sks: set[int], n_bins: int) -> frozenset[int]:
    stable = frozenset(s for s in sks if s >= _PERCEPTUAL_HASH_OFFSET or s in _STATE_SKS_RANGE)
    unstable = frozenset(s % n_bins for s in sks if s < _PERCEPTUAL_HASH_OFFSET and s not in _STATE_SKS_RANGE)
    return stable | unstable
```

This ensures `SKS_KEY_HELD=51` stays as `51` in context hashes, enabling correct `query_by_effect` matching.

### Issue: CausalLearner API

The existing `observe_transition(pre_sks, action, post_sks)` already computes `symmetric_difference` internally. CausalLearner should pass **full pre/post state** and let the model compute the diff — not pre-compute delta.

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
        # Delegate diff computation to CausalWorldModel.observe_transition()
        self._model.observe_transition(self._before, action, current_sks)
        self._before = None
```

---

## Exploration Fallback

When backward chaining cannot find a resolution (empty causal links):

1. Find **all** interactive objects in the grid (GridPerception.objects)
2. Sort by distance to agent (nearest first, but try all)
3. Navigate to each, try pickup/toggle
4. CausalLearner records before/after for each attempt
5. After max_tries=8 or all objects tried, retry backward chaining with updated model

Exploration tries **all** interactive objects, prioritizing nearby ones first. In DoorKey, the key may be far from the door — limiting to nearby objects could miss it.

```python
def explore(self, max_tries=8) -> bool:
    """Try interactions with all interactive objects in grid.
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
- Warmup phase: 5 episodes DoorKey-5x5 (same GoalAgent, learning enabled) to populate CausalWorldModel
- Test phase: 20 episodes DoorKey-5x5, varying seeds, with the warmed-up model
- Agent receives original mission text
- Check: correct sub-goals identified (pickup key, toggle door, goto goal)
- max_steps_per_episode: 200

**Gate:**
```
decomposition_accuracy >= 0.9   # 18/20 correct (deterministic algorithm with trained model)
```

### Exp 59: Causal Learning Speed

**Purpose:** How many episodes until agent learns DoorKey rules.

**Protocol:**
- CausalWorldModel starts empty
- Series of 10 episodes DoorKey-5x5
- After each episode: check for links that match pickup→key_held, toggle+key→door_open
- max_steps_per_episode: 200

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
- max_steps_per_episode: 200

**Gate:**
```
trial_5_success_rate >= 0.6                    # by 5th attempt >= 60% of series solved
trial_5_success_rate - trial_1_success_rate >= 0.2  # meaningful improvement, not noise
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

**Episode limits:** max_steps_per_episode: 300 (6x6 is larger).

---

## Files

| File | Purpose | New/Modified |
|------|---------|-------------|
| `src/snks/language/goal_agent.py` | GoalAgent orchestrator | New |
| `src/snks/language/blocking_analyzer.py` | Obstacle detection + causal resolution | New |
| `src/snks/language/causal_learner.py` | Before/after → observe_transition() | New |
| `src/snks/language/grid_perception.py` | Add state-dependent SKS IDs (key_held, door_open, etc.) | Modified |
| `src/snks/language/grid_navigator.py` | Add `plan_path_ex()` returning PathResult | Modified |
| `src/snks/language/babyai_executor.py` | Minor: expose re-perception | Modified |
| `src/snks/agent/causal_model.py` | Add `query_by_effect()` reverse lookup | Modified |
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
- CausalAgentConfig from `snks.daf.types` (GoalAgent creates default with `causal_min_observations=1`)

---

## Risks

1. **Exploration may not find key** — in large grids, key could be far from door. Mitigation: explore ALL interactive objects sorted by distance, not just nearby.
2. **CausalWorldModel context too specific** — `_split_context()` coarsens unstable SKS IDs into `n_bins` buckets. State-dependent SKS IDs (50-99 range) are stable → will match across episodes. This is the key reason for introducing them.
3. **Door detection** — GridNavigator returns `PathResult.BLOCKED` but doesn't say where the blocker is. `BlockingAnalyzer.find_blocker()` scans all grid cells for locked doors / impassable objects.
4. **DoorKey mission text** — "use the key to open the door" doesn't parse with RuleBasedChunker. GoalAgent always uses goal-driven mode: extract final goal noun ("goal") and backward chain. No text-driven mode needed.
5. **observe_transition context sensitivity** — `_split_context` coarsens IDs < 10000 with `% n_bins`. State-dependent IDs (50-99) must be treated as stable. Fix specified above: extend `_split_context` with `_STATE_SKS_RANGE`.
6. **symmetric_difference semantics** — `observe_transition` stores symmetric_difference as effect (both appearing AND disappearing IDs). `query_by_effect` uses overlap matching to handle this correctly. Documented in the `query_by_effect` docstring.
7. **Max steps** — all experiments specify `max_steps_per_episode` (200 for 5x5, 300 for 6x6). Episode counts as failure if exceeded.
