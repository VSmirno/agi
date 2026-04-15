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

Two locations to update: step start (line ~450) and episode end (line ~677).

### 2. StimuliLayer — new file `src/snks/agent/stimuli.py`

```python
@dataclass
class Stimulus:
    """Base class. evaluate() returns float score contribution."""
    def evaluate(self, state: VectorState, trajectory: VectorTrajectory) -> float:
        raise NotImplementedError

@dataclass
class SurvivalAversion(Stimulus):
    """Large penalty if trajectory terminated (agent died)."""
    weight: float = 1000.0

    def evaluate(self, state, trajectory) -> float:
        return -self.weight if trajectory.terminated else 0.0

@dataclass
class HomeostasisStimulus(Stimulus):
    """Reward for maintaining vitals above threshold."""
    vital_vars: list[str] = field(default_factory=lambda: ["health","food","drink","energy"])
    weight: float = 1.0

    def evaluate(self, state, trajectory) -> float:
        final = trajectory.final_state
        if not final:
            return 0.0
        return self.weight * min(final.body.get(v, 0.0) for v in self.vital_vars)

@dataclass
class StimuliLayer:
    stimuli: list[Stimulus] = field(default_factory=list)

    def evaluate(self, trajectory: VectorTrajectory) -> float:
        return sum(s.evaluate(trajectory.final_state, trajectory) for s in self.stimuli)
```

### 3. score_trajectory — accepts StimuliLayer

```python
# vector_sim.py — after
def score_trajectory(
    trajectory: VectorTrajectory,
    stimuli: StimuliLayer | None = None,
) -> tuple:
    """Score trajectory: (stimuli_score, total_gain, -steps).

    stimuli_score replaces hardcoded (survived, min_vital).
    If stimuli=None, returns (0, total_gain, -steps) — backward compat.
    """
    base = stimuli.evaluate(trajectory) if stimuli else 0.0
    total_gain = trajectory.total_inventory_gain()
    steps = len(trajectory.states) - 1
    return (base, total_gain, -steps)
```

Lex ordering preserved: stimuli score dominates, then cumulative gain, then step
economy. Stage 85 adds `CuriosityStimulus` to `StimuliLayer` — zero changes to
mechanism.

### 4. vector_mpc_agent.py — wire StimuliLayer

`run_vector_mpc_episode` receives `stimuli: StimuliLayer | None = None` and
passes it to `score_trajectory`. Default: `StimuliLayer([SurvivalAversion(), HomeostasisStimulus()])`.

## File Plan

| File | Change |
|------|--------|
| `src/snks/agent/stimuli.py` | New — `Stimulus`, `SurvivalAversion`, `HomeostasisStimulus`, `StimuliLayer` |
| `src/snks/agent/vector_sim.py` | `score_trajectory` signature: add `stimuli` param |
| `src/snks/agent/vector_mpc_agent.py` | Fix `inv`/`body` split (2 locations); pass `stimuli` to `score_trajectory` |
| `tests/test_stimuli.py` | New — unit tests for all stimulus classes |
| `tests/test_vital_fix.py` | New — integration test: mock info → body != 9.0 |
| `experiments/stage84_eval.py` | New — eval gate on minipc |

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

2. **`HomeostaticTracker` receives inv without body vars.** Previously it received
   the full `raw_inv` including health/food. Tracker's `observed_rates` was
   incorrectly tracking body variable deltas as inventory. This is fixed, but
   may change tracker behaviour. Mitigation: tracker is observational only —
   not on critical path for Stage 84 gate.
