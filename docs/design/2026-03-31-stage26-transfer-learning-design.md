# Stage 26: Cross-Environment Causal Transfer

**Version:** 1.1
**Date:** 2026-03-31
**Status:** DESIGN APPROVED

---

## Goal

Prove that causal knowledge learned in one MiniGrid environment transfers to a different environment, accelerating learning. The agent learns door/key mechanics in DoorKey-5x5, then applies this knowledge in MultiRoom — a 3-room environment with multiple doors/keys, plus distractor objects (boxes).

### What This Proves

1. **Causal transfer** — causal links (action + context → effect) generalize across environments
2. **Learning speedup** — transferred knowledge reduces exploration episodes by ≥ 2x
3. **Selective transfer** — agent correctly applies relevant knowledge and ignores irrelevant
4. **Serialization** — causal world model can be saved/loaded (persistent knowledge)

### Core Principles

- Type-based generalization: state predicates (SKS 50-99) are environment-independent
- Transfer via shared causal structure, not surface similarity
- Transfer relies on `query_by_effect()` with state predicates, NOT context hash matching

---

## Architecture

```
    Environment A (DoorKey-5x5)      Environment B (MultiRoom-10x10)
    ┌──────────────────┐            ┌───────────────────────────┐
    │  1 key, 1 door,  │            │  2 keys, 2 doors, goal,  │
    │  1 goal          │            │  2 boxes (distractors)    │
    │  5x5 layout      │            │  3 rooms, 10x10 layout   │
    └────────┬─────────┘            └───────────┬───────────────┘
             │ learn                             │ transfer + learn
             ▼                                   ▼
    ┌──────────────────┐   serialize   ┌───────────────────────┐
    │ CausalWorldModel │ ───────────→ │ CausalWorldModel      │
    │                  │   (JSON)      │ (pre-loaded: key→door │
    │ key_held → door  │               │  + newly learned box) │
    │ door_open → goal │               │                       │
    └──────────────────┘               └───────────────────────┘
```

### Key Design Decision: Transfer Mechanism

Transfer works through **state predicate matching**, not context hashing:

1. GoalAgent encounters a locked door → BlockingAnalyzer detects blocker
2. BlockingAnalyzer calls `causal_model.query_by_effect({SKS_DOOR_OPEN})`
3. Transferred links contain: `action=pickup, context={SKS_KEY_PRESENT} → effect={SKS_KEY_HELD}`
4. These predicates (50-99) are **identical** in DoorKey and MultiRoom
5. Agent reuses the chain without exploration

For novel objects (boxes), no matching links exist → agent explores as usual.

### Action ID Compatibility

GoalAgent uses MiniGrid native action IDs (`ACT_PICKUP=3`, `ACT_TOGGLE=5`).
All experiments use MiniGrid environments (gym `MiniGrid-DoorKey-*`), NOT `CausalGridWorld`.
MultiRoom will also be a standard MiniGrid env. This avoids action space mismatch.

---

## New Components

### 1. CausalModelSerializer (`src/snks/agent/causal_serializer.py`)

Serializes `_TransitionRecord` entries (not just `CausalLink` summaries) to preserve full internal state:

```python
class CausalModelSerializer:
    VERSION = 1

    @staticmethod
    def save(model: CausalWorldModel, path: str, source_env: str = "") -> None:
        """Serialize full internal state to JSON.

        Saves _TransitionRecord entries with original context_sks/effect_sks
        frozensets. Hashes are recomputed on load (Python hash() not stable
        across processes).
        """

    @staticmethod
    def load(path: str, config: CausalAgentConfig | None = None) -> CausalWorldModel:
        """Deserialize causal model from JSON.

        Rebuilds _transitions dict by recomputing hashes from stored
        context_sks/effect_sks. Raises ValueError on version mismatch.
        """
```

Serialized format:
```json
{
  "version": 1,
  "config": {
    "causal_min_observations": 1,
    "causal_confidence_threshold": 0.5,
    "causal_decay": 0.99,
    "causal_context_bins": 64
  },
  "total_observations": 15,
  "baseline_counts": {"<hash_key>": 5},
  "transitions": [
    {
      "action": 3,
      "context_sks": [50, 52],
      "effect_sks": [50, 51],
      "count": 5,
      "total_in_context": 5
    }
  ],
  "source_env": "DoorKey-5x5"
}
```

**Note on baseline_counts:** `_baseline_counts` uses `hash(frozenset)` as keys. These hashes are NOT stable across processes. On load, `_baseline_counts` is reset to empty — this is acceptable because baseline counts only affect decay of *existing* transitions, and a loaded model can recompute baselines from new observations.

### 2. TransferAgent (`src/snks/language/transfer_agent.py`)

Thin wrapper around GoalAgent adding transfer metrics:

```python
@dataclass
class TransferResult:
    success: bool
    steps_taken: int
    explored: bool          # did agent need exploration?
    links_reused: int       # causal links queried and matched
    links_new: int          # new causal links learned this episode

@dataclass
class TransferStats:
    episodes: int
    successes: int
    total_steps: int
    total_links_reused: int
    total_links_new: int
    exploration_episodes: int  # episodes requiring exploration

class TransferAgent:
    def __init__(self, causal_model: CausalWorldModel | None = None):
        """Initialize with optional pre-trained causal model."""

    def run_episode(self, env, instruction: str) -> TransferResult:
        """Run one episode, returns success + transfer metrics."""

    def get_stats(self) -> TransferStats:
        """Aggregate stats across all episodes."""
```

Transfer tracking: instrument GoalAgent's `_explore_all_objects()` and `_execute_subgoals()` to count links_reused vs links_new.

### 3. Enhanced MultiRoom Environment

Current `CausalGridWorld.MultiRoom` is too similar to DoorKey (1 wall, 1 door). Create a proper 3-room MiniGrid env:

```python
class MultiRoomDoorKey(MiniGridEnv):
    """3-room environment: agent must traverse 2 locked doors to reach goal.

    Room 1: agent start + key_1 (yellow)
    Room 2: key_2 (blue) + box distractors
    Room 3: goal

    Walls between rooms 1-2 and 2-3, each with a locked door.
    Tests: agent reuses key→door knowledge, ignores boxes.
    """
```

This lives in `src/snks/env/multi_room.py` (new file, not modifying `causal_grid.py`).

### 4. Extended GridPerception predicates

```python
SKS_BOX_PRESENT = 55     # box visible on grid
```

Box is not interactive in MultiRoom — just an obstacle. Predicate added for completeness.

---

## Experiments

### Exp 62: Cross-Environment Transfer (DoorKey → MultiRoom)

**Protocol:**
1. **Train phase:** GoalAgent runs 5 episodes in `MiniGrid-DoorKey-5x5-v0`, learns causal model
2. **Transfer phase:** TransferAgent with trained causal_model in MultiRoomDoorKey (10 episodes)
3. **Control phase:** TransferAgent with empty causal_model in MultiRoomDoorKey (10 episodes)
4. **Measure:** success rate, mean steps, exploration_episodes

**Gate criteria:**
- Transfer success rate ≥ 70% (10 episodes)
- Learning speedup ≥ 2x (exploration_episodes transfer < control)

### Exp 63: Causal Knowledge Persistence (Save/Load)

**Protocol:**
1. Train GoalAgent in DoorKey-5x5 (5 episodes)
2. Serialize CausalWorldModel to temp JSON
3. Load into new CausalWorldModel instance
4. Run 10 episodes in DoorKey-6x6 with loaded model (transfer group)
5. Run 10 episodes in DoorKey-6x6 with original model (direct group)

**Gate criteria:**
- Success rate difference ≤ 5% between loaded vs direct
- Serialization roundtrip: loaded model `n_links` == original `n_links`

### Exp 64: Selective Transfer — Negative Test (DoorKey → CausalGridWorld PushBox)

**Protocol:**
1. Train GoalAgent in DoorKey-5x5 (5 episodes) — learns key/door mechanics
2. Transfer causal model to PushBox-style environment (no doors/keys, just box + goal)
3. Agent navigates to goal (simple — no blockers in basic PushBox)
4. Track: any toggle/pickup actions attempted on empty cells = "incorrect transfer"

**Gate criteria:**
- No incorrect transfer actions (toggle/pickup on cells without matching objects)
- PushBox steps with transfer model ≤ 1.1x steps without (no degradation)

**Note:** PushBox test uses a simplified GoalAgent that only does navigate-to-goal (no backward chaining needed when path is unblocked).

---

## Implementation Plan

### Phase 1: Serializer
- `CausalModelSerializer` — save/load `_TransitionRecord` with recomputed hashes
- Version check on load
- Unit tests: roundtrip preserves links/counts, version mismatch raises error

### Phase 2: MultiRoom environment
- `MultiRoomDoorKey` — 3-room MiniGrid env with 2 doors, 2 keys, boxes
- Verify GoalAgent works unmodified (same action space, same predicates)

### Phase 3: TransferAgent + metrics
- `TransferAgent` wrapping GoalAgent
- `TransferResult`/`TransferStats` dataclasses
- Track links_reused, links_new, exploration_episodes

### Phase 4: Experiments
- Exp 62: cross-env transfer
- Exp 63: persistence roundtrip
- Exp 64: selective transfer (negative test)

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Python hash() not stable across processes | Serialize context_sks/effect_sks as lists, recompute hashes on load |
| MultiRoom layout too complex for GoalAgent | Start with 3 rooms linearly connected — same key→door pattern twice |
| Box objects confuse exploration fallback | Boxes don't respond to toggle/pickup in MiniGrid — agent learns no-effect quickly |
| Context hash mismatch between envs | Transfer via query_by_effect() on state predicates (50-99), not context matching |

---

## Non-Goals

- Transfer across different action spaces (different games)
- Meta-learning (learning how to transfer) — Stage 32
- Natural language instruction transfer — same instruction format
- Modifying CausalGridWorld action space — experiments use native MiniGrid
