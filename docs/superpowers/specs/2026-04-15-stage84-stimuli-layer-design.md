# Stage 84 — Real Stimuli Infrastructure

**Date:** 2026-04-15
**Status:** Design approved
**Parent:** IDEOLOGY v2 (Category 4 — Stimuli), Stage 83 (VectorWorldModel)
**Approach:** B — vital fix + StimuliLayer extraction

## Motivation

Two concrete problems block correct agent behaviour:

1. **Body vitals always 9.0.** `vector_mpc_agent.py` reads vitals from `info`
   top-level (`info.get(v, 9.0)`), but Crafter places `health/food/drink/energy`
   inside `info["inventory"]`. The default 9.0 fires every step — `VectorState.body`
   is permanently `{health:9.0, food:9.0, ...}`. Sleep never wins on low energy
   because the planner thinks vitals are always full.

2. **`score_trajectory` is a hardcoded mechanism.** The lex-tuple
   `(survived, total_gain, min_vital, -steps)` encodes scoring policy inside the
   mechanism layer (Category 2). Adding curiosity in Stage 85 or aversion in
   Stage 86 requires editing simulation code — a Category 2 violation.

Both stem from the same ideological debt: Category 4 (Stimuli) is unimplemented.

## Root Cause (confirmed by code inspection)

```python
# vector_mpc_agent.py:449-450 — THE BUG
inv = dict(info.get("inventory", {}))           # contains wood, health, food, energy...
body = {v: float(info.get(v, 9.0)) for v in vitals}  # reads from info → 9.0 always
```

Crafter's `info` dict has keys: `achievements`, `discount`, `inventory`,
`player_pos`, `reward`, `semantic`. Vitals are inside `inventory`, not top-level.

## Design

### 1. Fix: separate inv from body

At the top of every step iteration and at episode end:

```python
raw_inv = dict(info.get("inventory", {}))
VITAL_VARS = {"health", "food", "drink", "energy"}
body = {v: float(raw_inv.get(v, 9.0)) for v in vitals}   # from inventory — correct
inv  = {k: v for k, v in raw_inv.items() if k not in VITAL_VARS}  # resources only
```

`HomeostaticTracker.update(prev_inv, inv, ...)` receives inventory without body
variables (correct — tracker observes resource rates, body is tracked separately).
`VectorState(inventory=inv, body=body)` is now semantically honest.

Two locations to update:

**Step start (line ~450):** shown above.

**Episode end (line ~677):** same bug, different default:
```python
# Before (always 0 — default fires because vitals not in top-level info):
body_at_end = {v: float(info.get(v, 0)) for v in vitals}

# After:
raw_inv_end = dict(info.get("inventory", {}))
body_at_end = {v: float(raw_inv_end.get(v, 0.0)) for v in vitals}
```
Without this fix `cause_of_death` is always `"health"` (all vitals appear 0),
corrupting episode metrics.

### 2. StimuliLayer — new file `src/snks/agent/stimuli.py`

```python
@dataclass
class Stimulus:
    """Base class. evaluate(trajectory) returns float score contribution.
    Each stimulus pulls what it needs from trajectory directly
    (final_state, terminated, etc.) — no dead `state` parameter."""
    def evaluate(self, trajectory: VectorTrajectory) -> float:
        raise NotImplementedError

@dataclass
class SurvivalAversion(Stimulus):
    """Large penalty if trajectory terminated (agent died)."""
    weight: float = 1000.0

    def evaluate(self, trajectory) -> float:
        return -self.weight if trajectory.terminated else 0.0

@dataclass
class HomeostasisStimulus(Stimulus):
    """Reward for maintaining vitals above threshold."""
    vital_vars: list[str] = field(default_factory=lambda: ["health","food","drink","energy"])
    weight: float = 1.0

    def evaluate(self, trajectory) -> float:
        final = trajectory.final_state
        if not final:
            return 0.0
        return self.weight * min(final.body.get(v, 0.0) for v in self.vital_vars)

@dataclass
class StimuliLayer:
    stimuli: list[Stimulus] = field(default_factory=list)

    def evaluate(self, trajectory: VectorTrajectory) -> float:
        return sum(s.evaluate(trajectory) for s in self.stimuli)
```

### 3. score_trajectory — accepts StimuliLayer

```python
# vector_sim.py — after
def score_trajectory(
    trajectory: VectorTrajectory,
    stimuli: StimuliLayer | None = None,
) -> tuple:
    """Score trajectory: (stimuli_score, total_gain, -steps).

    stimuli_score subsumes survived + min_vital from old 4-tuple.
    If stimuli=None, falls back to (survived, total_gain, -steps) to
    preserve ordering in existing tests that compare alive > dead.
    """
    total_gain = trajectory.total_inventory_gain()
    steps = len(trajectory.states) - 1
    if stimuli is not None:
        base = stimuli.evaluate(trajectory)
        return (base, total_gain, -steps)
    else:
        survived = 0 if trajectory.terminated else 1
        return (survived, total_gain, -steps)
```

**Note:** the old 4-tuple `(survived, total_gain, min_vital, -steps)` becomes
a 3-tuple. All callers that splice or unpack positionally must be updated (see
File Plan).

Lex ordering preserved: stimuli score dominates, then cumulative gain, then step
economy. Stage 85 adds `CuriosityStimulus` to `StimuliLayer` — zero changes to
mechanism.

### 4. vector_mpc_agent.py — wire StimuliLayer

`run_vector_mpc_episode` receives `stimuli: StimuliLayer | None = None` and
passes it to `score_trajectory`. Default: `StimuliLayer([SurvivalAversion(), HomeostasisStimulus()])`.

The tuple-splice line that builds the final sort key must also be updated:

```python
# Before (line ~584):
sim_score = score_trajectory(traj, vitals)    # 4-tuple
gain = 0 if is_self_action else sim_score[1]
score = (sim_score[0], known, gain) + sim_score[2:]  # (survived, known, gain, min_vital, -steps)

# After:
sim_score = score_trajectory(traj, stimuli=stimuli)  # 3-tuple
gain = 0 if is_self_action else sim_score[1]
score = (sim_score[0], known, gain, sim_score[2])    # (stimuli_score, known, gain, -steps)
```

## File Plan

| File | Change |
|------|--------|
| `src/snks/agent/stimuli.py` | New — `Stimulus`, `SurvivalAversion`, `HomeostasisStimulus`, `StimuliLayer` |
| `src/snks/agent/vector_sim.py` | `score_trajectory`: add `stimuli` param; 4-tuple → 3-tuple |
| `src/snks/agent/vector_mpc_agent.py` | Fix `inv`/`body` split (2 locations); update tuple-splice line; pass `stimuli` |
| `tests/test_stimuli.py` | New — unit tests for all stimulus classes |
| `tests/test_vital_fix.py` | New — integration test: mock info → body != 9.0 |
| `tests/test_vector_sim.py` | Update `score_trajectory` call sites: drop `vital_vars`, add `stimuli=None` |
| `tests/test_vector_mpc.py` | Same — update tuple-unpack assertions |
| `experiments/stage84_eval.py` | New — eval gate on minipc |
| `experiments/diag_stage83.py` | **Archive before merge** — calls `score_trajectory(traj, vital_vars)` positionally; with new signature `vital_vars` would be passed as `stimuli`, causing `AttributeError` at `.evaluate()`. Rename to `_archived_diag_stage83.py` or delete. |
| `experiments/diag_stage83_bugs.py` | **Archive before merge** — same issue. |
| `experiments/stage83_vector_eval.py` | Update — calls `run_vector_mpc_episode`, receives new 3-tuple |

Untouched: `vector_world_model.py`, `vector_bootstrap.py`, `crafter_spatial_map.py`,
`tile_segmenter.py`, `perception.py`.

## Testing

**Unit tests** (`tests/test_stimuli.py`):
- `SurvivalAversion`: `-1000.0` when `terminated=True`, `0.0` when `terminated=False`
- `HomeostasisStimulus`: returns `min_vital` value; `0.0` on empty final state
- `StimuliLayer.evaluate`: sums multiple stimuli correctly
- `score_trajectory(stimuli=None)`: backward compat, returns `(0, gain, -steps)`

**Integration test** (`tests/test_vital_fix.py`):
- Mock `info = {"inventory": {"food": 2.0, "wood": 3}}` → `body["food"] == 2.0`
- `inv` does not contain `"health"`, `"food"`, `"drink"`, `"energy"`

**Eval gate** (`experiments/stage84_eval.py`, minipc):
- `sleep_at_low_energy`: 20 episodes, forced `energy=2` → sleep chosen in >80% of steps
- `no_sleep_at_full_energy`: `energy=9` → sleep chosen in <5% of steps
- `survival_ge_155`: regression — survival mean ≥ 155 (Stage 82 baseline)
- `wood_ge_10pct`: wood ≥3 in ≥10% of episodes

## Connection to IDEOLOGY v2

- **Category 2 (Mechanisms):** `score_trajectory` becomes a pure dispatcher —
  it calls `stimuli.evaluate()` without knowing what stimuli are.
- **Category 4 (Stimuli):** `SurvivalAversion` and `HomeostasisStimulus` are
  the first concrete stimuli objects. They live outside the mechanism.
- **Anti-pattern (hardcoded scoring):** the `min_vital` and `survived` fields in
  the lex-tuple were Category 4 knowledge embedded in Category 2 code. Removed.

## Risks

1. **Lex-tuple ordering change.** Old: `(survived, total_gain, min_vital, -steps)`.
   New: `(stimuli_score, total_gain, -steps)`. Existing eval harness scripts that
   unpack the old 4-tuple will break. Mitigation: update `stage83_vector_eval.py`
   and any other callers before running eval.

2. **`vital_vars` stays in `simulate_forward`.** `simulate_forward` still accepts
   `vital_vars` and passes it to `state.is_dead(vital_vars)`. Do not remove it —
   death detection inside the simulation is a mechanism concern, separate from
   the stimuli scoring layer.

3. **`HomeostaticTracker` one-episode transient.** Previously `inv` included
   health/food/drink/energy, so `observed_rates` accumulated noise from body
   variable deltas. After the fix those keys vanish from `inv`. On the first
   transition episode this creates a one-time spike (`0 - old_value`) in
   `observed_rates` for those keys. Transient — no persistent state corruption.
   Safe to ignore; tracker is not on critical path for Stage 84 gate.
