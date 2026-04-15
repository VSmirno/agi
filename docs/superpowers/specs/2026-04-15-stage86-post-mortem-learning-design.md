# Stage 86 — Post-Mortem Learning

**Date:** 2026-04-15  
**Status:** Draft  
**Ideological debt:** Principle 6 (System, not agent). Death is currently discarded information. It must become a learning signal.

---

## Context

After each episode the agent dies. Currently `run_vector_mpc_episode` returns `cause="health"` — a single coarse label that discards all causal structure. Stage 86 makes death informative by:

1. Collecting a **damage log** during the episode (per health-decrease event with vitals + nearby entities)
2. **Attributing** damage to sources using temporal-decay weighting (all contributing sources updated proportionally)
3. **Updating** stimulus parameters between episodes so future episodes plan more defensively

The Crafter environment does not expose cause of death. We infer it from `body_deltas` + `entity_tracker` data already available in `run_vector_mpc_episode`.

---

## Crafter Death Mechanics (observed)

- `done=True` when `player.health <= 0`
- Zombie: `-2` hp per contact tick (`dist <= 1`), `-7` if sleeping
- Skeleton: arrow damage at range
- Starvation/dehydration: `food=0` or `drink=0` → `_recover` decrements → `health -= 1` every ~15 ticks
- Lava: `health = 0` instantly

Cause attribution is inferred — not provided by the environment.

---

## Design

### Block 1 — `DamageEvent` + `damage_log`

```python
@dataclass
class DamageEvent:
    step: int
    health_delta: float                           # always < 0
    vitals: dict[str, float]                      # food, drink, energy at event
    nearby_cids: list[tuple[str, int]]            # [(concept_id, dist), ...]
```

Accumulated in `run_vector_mpc_episode` on every step where `health_delta < 0`.

All required data is already in scope:
- `body_deltas.get("health", 0)` — health delta this step
- `vitals` (food, drink, energy values)  
- `entity_tracker.visible_entities()` — entities + positions → compute dist to `player_pos`

Added to the episode return dict as `"damage_log": list[DamageEvent]`.

Also added: `"death_cause": str` — the dominant attributed source ("starvation", "zombie", "skeleton", "dehydration", "unknown"), derived from the attribution dict.

### Block 2 — `PostMortemAnalyzer.attribute()`

```python
def attribute(
    damage_log: list[DamageEvent],
    death_step: int,
    decay: float = 0.02,
) -> dict[str, float]:
```

**Temporal decay weight** for each event: `w = exp(-decay * (death_step - event.step))`.  
Weights are normalised so they sum to 1.0 across all events.

**Source detection per event** (one event may have multiple sources — weight split equally across all detected sources for that event):
- `vitals["food"] < 0.5` → `"starvation"`
- `vitals["drink"] < 0.5` → `"dehydration"`
- any entity with `dist <= 2` → that concept_id (e.g. `"zombie"`, `"skeleton"`)
- no source matched → `"unknown"`

Example: `food=0` and zombie at `dist=1` → both detected → event weight split 50/50 between `"starvation"` and `"zombie"`.

**Return value:** `dict[str, float]` mapping source → fraction of total damage weight. Sum = 1.0.

Example:
```python
{"starvation": 0.28, "zombie": 0.52, "unknown": 0.20}
```

If `damage_log` is empty (episode ended without `health <= 0`, e.g. `done` from step limit), returns `{}`.

### Block 3 — `PostMortemLearner`

```python
@dataclass
class PostMortemLearner:
    food_threshold: float = 3.0
    drink_threshold: float = 3.0
    health_weight: float = 1.0
    lr: float = 0.1

    def update(self, attribution: dict[str, float]) -> None: ...
    def build_stimuli(self, vital_vars: list[str]) -> StimuliLayer: ...
```

**`update(attribution)`** — mutates parameters in-place:
- `"starvation"` share → `food_threshold += lr * share`  
  *(agent becomes more sensitive to food deficit — HomeostasisStimulus fires earlier)*
- `"dehydration"` share → `drink_threshold += lr * share`
- entity share (`"zombie"` + `"skeleton"`) → `health_weight += lr * entity_share`  
  *(agent values health more — HomeostasisStimulus penalises low health harder)*

Parameters are bounded: thresholds clamped to `[1.0, 8.0]`, `health_weight` clamped to `[0.5, 5.0]`.

**`build_stimuli(vital_vars)`** — returns a new `StimuliLayer` instance with current parameters:
```python
StimuliLayer([
    SurvivalAversion(),
    HomeostasisStimulus(
        thresholds={"food": self.food_threshold, "drink": self.drink_threshold},
        weight=self.health_weight,
        vital_vars=vital_vars,
    ),
])
```

### Block 4 — `HomeostasisStimulus` update

Current scoring: `weight * min(body[v] for v in vital_vars)` — rewards absolute vital level.

New scoring with per-vital thresholds:
```python
def evaluate(self, trajectory) -> float:
    final = trajectory.final_state
    if not final:
        return 0.0
    deficit = sum(
        max(0.0, self.thresholds.get(v, 0.0) - final.body.get(v, 0.0))
        for v in self.vital_vars
    )
    return -self.weight * deficit
```

Negative score proportional to how far each vital is below its threshold. Zero when all vitals above thresholds.

`thresholds` defaults to `{}` (no threshold active) — backwards-compatible with existing callers.

### Block 5 — Eval loop

```python
analyzer = PostMortemAnalyzer()
learner = PostMortemLearner()
stimuli = learner.build_stimuli(vital_vars)

for ep in range(n_episodes):
    result = run_vector_mpc_episode(..., stimuli=stimuli)
    if result["damage_log"]:
        attribution = analyzer.attribute(result["damage_log"], result["episode_steps"])
        learner.update(attribution)
    stimuli = learner.build_stimuli(vital_vars)
```

Two eval runs in `stage86_eval.py`:
- `with_pm`: `PostMortemLearner` active (parameters update each episode)
- `without_pm`: fixed stimuli (same as Stage 85 baseline)

---

## Gates

| Gate | Condition | Measurement |
|---|---|---|
| `zombie_deaths_decrease` | `zombie_deaths(ep14-20) < zombie_deaths(ep1-7)` | within `with_pm` run |
| `starvation_decrease` | `starvation_deaths(with_pm) < starvation_deaths(without_pm)` | across two runs |

`zombie_deaths` = count of episodes where `result["death_cause"] == "zombie"`.  
`starvation_deaths` = count of episodes where `result["death_cause"] == "starvation"`.  
`death_cause` = key with highest value in attribution dict; `"alive"` if `damage_log` is empty.
| `survival_holds` | `avg_survival(with_pm) >= 155` | `with_pm` run |

All 3 gates must PASS.

---

## Files Changed

| File | Change |
|---|---|
| `src/snks/agent/post_mortem.py` | NEW — `DamageEvent`, `PostMortemAnalyzer`, `PostMortemLearner` |
| `src/snks/agent/stimuli.py` | `HomeostasisStimulus` — add `thresholds: dict[str, float]` field + new `evaluate()` |
| `src/snks/agent/vector_mpc_agent.py` | Accumulate `damage_log`; add `"damage_log"`, `"death_cause"`, `"episode_steps"` to return dict; rename `"avg_len"` → `"episode_steps"` |
| `experiments/stage86_eval.py` | NEW — 20-ep eval, two runs (with_pm / without_pm), gate checks |
| `tests/test_post_mortem.py` | NEW — unit tests for attribution + learner |

---

## Ideology Check

- **Category 1 (Facts):** unchanged — no new textbook entries
- **Category 2 (Mechanisms):** `vector_mpc_agent` collects `damage_log` — minimal, no new planning logic
- **Category 3 (Experience):** `damage_log` = per-episode observed damage history. `PostMortemLearner.params` = accumulated in-memory adaptation
- **Category 4 (Stimuli):** `HomeostasisStimulus` thresholds + `health_weight` update between episodes — the right place for this to live

`PostMortemLearner` updates only stimuli parameters (Category 4). It does not modify textbook rules (Category 1) or planning mechanics (Category 2). Parameters reset on new run — no cross-run persistence (Stage 86 scope).

---

## Out of Scope (Stage 86)

- Persistence across runs (saves to disk) — Stage 88 (Knowledge Flow)
- `EntityAversionStimulus` (penalises trajectories near enemies in forward sim) — Stage 87
- Early-death analysis (died at step 35 — analyse first 20 steps) — Stage 87
- Per-entity death tracking beyond zombie/skeleton — future
