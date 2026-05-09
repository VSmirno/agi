# Stage91 Local-Affordance Extractor Diff Forensics

## Artifact Paths

- Prior best feasibility baseline:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
- New extractor wiring run:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z`
- Comparison artifacts produced during this investigation:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z/analysis/comparison_summary.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z/analysis/first_divergence_forensics.json`

## Artifact-Proved Facts

### 1. The controller still ranks actions that are missing from the counterfactual set.

Current local source:

- [stage90r_emergency_controller.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_emergency_controller.py:254)

Direct code fact:

- `select_action()` iterates a fixed `allowed_actions` list including `do`.
- If an action is absent from `candidate_outcomes`, it falls back to an empty
  label at lines `273-276`, which defaults to `survived_h=True`, `damage_h=0`,
  and `health_delta_h=0`.

This behavior predates the extractor integration, but it remains active and is
relevant to the regression.

### 2. Seed 17, episode 0 shows an upstream state-path divergence before the first changed rescue choice.

Direct artifact evidence:

- `local_trace_excerpt` step `31` is identical between runs:
  - action `move_left`
  - controller `learner_actor`
  - `player_pos_before=[30,36]`
  - `player_pos_after=[29,36]`
  - `nearest_threat_distances={arrow:null, skeleton:8, zombie:null}`
- First divergent rescue step is `44`.

At rescue step `44`:

| Field | Feasibility baseline | Extractor run |
| --- | --- | --- |
| `primary_regime` | `hostile_contact` | `hostile_near` |
| `nearest_threat_distances.arrow` | `1` | `2` |
| ranked actions | `[move_down, move_right, do, sleep]` | `[move_right, do, sleep, move_left]` |
| chosen action | `move_down` | `move_right` |

Shared-label diffs at that same step:

- `do`: `damage_h 5.0 -> 0.0`, `nearest_hostile_h 0 -> 1`
- `move_right`: `nearest_hostile_h 1 -> 2`, `adjacent_hostile_after_h true -> false`
- `sleep`: `damage_h 5.0 -> 0.0`, `nearest_hostile_h 0 -> 1`

This proves the step-44 change is not only ranking over identical inputs. The
state entering rescue evaluation has already diverged by then.

### 3. Seed 7, episode 3 shows same-state counterfactual-label drift even when the chosen action and actual post-step outcome are unchanged.

Direct artifact evidence at rescue step `9`:

| Field | Feasibility baseline | Extractor run |
| --- | --- | --- |
| `primary_regime` | `local_resource_facing` | `local_resource_facing` |
| `nearest_threat_distances.skeleton` | `5` | `5` |
| ranked actions | `[move_right, do, move_left, move_up]` | `[move_right, do, move_left, move_up]` |
| chosen action | `move_right` | `move_right` |
| actual post outcome | `damage_step=0`, `displacement_step=1`, `nearest_hostile_after=6` | same |

But the predicted labels changed for every shared action:

- `move_right`: `damage_h 5.0 -> 2.0`
- `move_left`: `damage_h 5.0 -> 2.0`
- `move_up`: `damage_h 5.0 -> 2.0`
- `do`: `damage_h 5.0 -> 2.0`

This is the strongest direct evidence that the extractor wiring changed the
counterfactual label path itself, not just logging.

### 4. Several earlier rescue events disappear entirely in the extractor run.

Direct artifact evidence:

- Seed `7`, episode `1`:
  - prior run rescue at step `36`: chosen `move_up`, predicted safe, actual
    `damage_step=0`, `nearest_hostile_after=4`
  - extractor run: no rescue event recorded at step `36`
- Seed `7`, episode `2`:
  - prior run rescue at step `139`: chosen `move_up`, predicted safe, actual
    `damage_step=0`
  - extractor run: no rescue event recorded at step `139`
- Seed `17`, episode `2`:
  - prior run rescue at step `129`: chosen `move_right`, predicted safe, actual
    `damage_step=0`
  - extractor run: no rescue event recorded at step `129`

The stored artifacts do not retain full mid-episode local traces for those
steps in the extractor run, so the exact non-rescue action that caused the
divergence is not directly visible there.

### 5. The new terminal `do` rows are not explained by new controller heuristics or by newly-added local fields being consumed.

Current local source:

- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:944)
- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1026)
- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1732)
- [stage90r_local_affordances.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_local_affordances.py:11)
- [stage90r_emergency_controller.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_emergency_controller.py:280)

Direct code fact:

- The new helper in `stage90r_local_affordances.py` is read-only.
- `vector_mpc_agent.py` still gates `do` target selection on `vf.near_concept`
  at lines `1757-1763`.
- `stage90r_emergency_controller.py` does **not** read any of the new
  `*_local` fields.

Direct artifact evidence at seed `17`, episode `0`, terminal step `144`:

- Both runs choose `do`.
- Both runs assign identical safe `do` ranked components:
  - `damage_h=0.0`
  - `nearest_hostile_h=null`
  - `survived_h=True`
- In both runs, the same-step `local_trace_tail.counterfactual_outcomes`
  omits `do` entirely and only contains:
  - `move_left`
  - `move_right`
  - `move_up`
  - `move_down`
  - `sleep`
- Actual post-step outcome differs sharply:
  - prior run: `damage_step=0.0`, `nearest_hostile_after=5`
  - extractor run: `damage_step=2.0`, `nearest_hostile_after=1`

This proves the terminal seed-17 `do` regression at step `144` is not caused by
the new local fields being consumed by the controller, and not by `do` becoming
newly available in the counterfactual set. `do` is absent from the raw
counterfactual set in both runs and still gets ranked via the controller’s
missing-label fallback.

## First-Divergence Mismatch Table

| Seed / Episode | First divergent rescue step | What changed first | Direct evidence | Readout |
| --- | ---: | --- | --- | --- |
| `17 / 0` | `44` | candidate set, rank order, chosen action, and pre-rescue state | `move_down -> move_right`; `arrow 1 -> 2`; ranked set swaps `move_down` out and `move_left` in | upstream state-path divergence before/at first changed rescue |
| `7 / 3` | `9` | label values only | same state, same ranked actions, same chosen action, same actual post; predicted `damage_h` drops `5 -> 2` for all shared actions | counterfactual-label path changed under identical observed state |
| `7 / 1` | `36` | rescue presence | prior safe `move_up` rescue exists; extractor run has no rescue event at that step | rescue activation/path diverged earlier |
| `7 / 2` | `139` | rescue presence | prior safe `move_up` rescue exists; extractor run has no rescue event at that step | rescue activation/path diverged earlier |
| `17 / 2` | `129` | rescue presence | prior safe `move_right` rescue exists; extractor run has no rescue event at that step | rescue activation/path diverged earlier |

## Answered Questions

### (a) Did the chosen action distribution change because candidate set contents changed, because ranking inputs changed, or because some prior labels/fields shifted?

Artifact-proved answer: all three appear, depending on episode.

- Candidate set contents changed:
  - seed `17`, episode `0`, step `44`
  - old ranked set includes `move_down`; new ranked set does not
- Ranking inputs changed through label shifts:
  - seed `17`, episode `0`, step `44`
  - shared actions get materially different `damage_h` / `nearest_hostile_h`
  - seed `7`, episode `3`, step `9`
  - identical state and identical chosen action, but all shared action labels
    shift from `damage_h=5` to `damage_h=2`
- Rescue-event presence changed:
  - seed `7`, episodes `1` and `2`
  - seed `17`, episode `2`

So the regression is not one single mechanism.

### (b) For the new terminal `do` rows, what did the matching candidate labels look like in the prior best run at the analogous steps?

Strongest exact same-step comparison:

- seed `17`, episode `0`, step `144`
  - old run: chosen action `do`
  - new run: chosen action `do`
  - old and new ranked `do` components are identical and safe-default
  - actual post outcome changes from safe (`damage=0`, `nearest=5`) to unsafe
    (`damage=2`, `nearest=1`)

This means that exact terminal regression is not explained by a changed `do`
label at that step. It is explained by the trajectory/state being different by
the time the same default-safe `do` ranking is applied.

For seed `7`, exact same-step prior rescue rows are not available for the new
terminal `do` steps `174`, `181`, and `208`. The nearest prior terminal rescue
rows for the same episodes remain movement actions instead:

- episode `1`: prior tail ends at step `219` with `move_up`
- episode `2`: prior tail ends at step `185` with `move_down`
- episode `3`: prior tail ends at step `187` with `move_right`

So the new terminal `do` rows on seed `7` emerge only after earlier rescue-path
divergence.

### (c) Is there evidence that extractor wiring changed ordering/presence of counterfactual candidates, mutated state, or altered `do` target availability even though no controller heuristic was intentionally changed?

Artifact-proved facts:

- Yes, ordering/presence changed:
  - seed `17`, episode `0`, step `44`
- Yes, the state entering rescue evaluation differs at the first mismatched
  rescue step:
  - seed `17`, episode `0`, step `44`
- Yes, existing label values shifted under the same state:
  - seed `7`, episode `3`, step `9`
- No direct evidence that extractor wiring changed `do` target availability:
  - `do` remains absent from the raw counterfactual set at seed `17`, episode
    `0`, step `144` in both runs
  - `vector_mpc_agent.py` still gates `do` target selection on `vf.near_concept`
  - controller still ranks missing `do` through fallback defaults

Best-supported inference:

- The regression does **not** look like “the new local fields were consumed by
  the controller and changed scoring.”
- It does look like the extractor integration touched the active
  counterfactual/state path enough to:
  - change observed rescue-entry state on some episodes
  - and change counterfactual labels even where state and executed action match

## Concise Interpretation

The strongest direct root-cause clue is seed `7`, episode `3`, step `9`:
counterfactual labels changed under the same observed state and same actual
executed outcome. That is the narrowest artifact-proved sign that the extractor
integration is not behavior-neutral inside the active counterfactual path.

The strongest direct downstream symptom is seed `17`, episode `0`, step `144`:
`do` is missing from the raw counterfactual set in both runs, yet the controller
still ranks it as safe by default. In the prior run that happened not to be
fatal; in the extractor run the trajectory has already drifted enough that the
same default-safe `do` becomes terminally wrong.

## Smallest Next Testable Hypothesis

Make the extractor diagnostics-only first.

Why:

1. The controller does not consume the new `*_local` fields, so there is no
   intended scoring benefit to justify behavior drift.
2. The helper itself is read-only, but the active counterfactual path changed:
   seed `7`, episode `3`, step `9` proves label drift under identical state.
3. The new run also re-exposes an existing weakness where missing `do`
   candidates still get safe default ranking.

Smallest testable hypothesis:

- Keep only trace-only instrumentation:
  - `local_affordance_scene` in rescue diagnostics
- Remove extractor plumbing from the active counterfactual behavior path first:
  - stop passing `affordance_snapshot` into `_build_local_counterfactual_outcomes`
  - stop attaching new `*_local` label fields to active runtime candidate labels

If that restores behavioral parity with the feasibility baseline, the regression
source is narrowed to the current counterfactual-path integration rather than
the diagnostics themselves.

## Artifact Limits

- Full mid-episode local traces are not persisted; only
  `local_trace_excerpt` and `local_trace_tail` are available.
- That prevents direct proof of the exact non-rescue action between seed `17`
  episode `0` steps `32-43`, even though the first rescue mismatch at step `44`
  clearly shows the state already diverged.
