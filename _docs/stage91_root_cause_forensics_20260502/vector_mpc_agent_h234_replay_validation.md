# Stage91 `vector_mpc_agent.py` H2/H3/H4 Replay Validation

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z`
- Raw eval JSON:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/raw/seed_7_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/raw/seed_17_eval.json`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/analysis/comparison_summary.json`
- Provenance:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/env/import_probe.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/env/py_compile.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/env/runtime_interpreter.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z/env/synced_runtime_sha256.txt`

## Run Shape

The isolated checkout ran under:

`/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python3.11`

Import probe confirms runtime imported from the isolated checkout for:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_mpc_agent.py`

`stage90r_local_affordances.py` was not synced and was not imported. That is
the intended replay surface.

Synced runtime hashes still match prior artifacts for:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/stage90r_emergency_controller.py`

The only `vector_mpc_agent.py` delta versus the feasibility-baseline file is
now H5:

- baseline:
  `near_concept = str(vf.near_concept)`
- replay:
  `action_concept = str(vf.near_concept or "empty")`

So this validation isolates the H2/H3/H4 rollback cleanly.

## Comparison Targets

- Feasibility baseline:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
- Diagnostics-only run:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z`
- Extractor wiring run:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z`

## Before / After Metrics

| Run | Seed | Avg survival | Hostile deaths with terminal rescue | Final hostile rescue nearest-hostile-after | False-safe terminal rows |
| --- | ---: | ---: | ---: | --- | --- |
| feasibility | 7 | `203.5` | `3` | `[3, 2, 1]` | `1 (move_right:1)` |
| diagnostics-only | 7 | `168.0` | `3` | `[4, 2, 1]` | `1 (move_right:1)` |
| extractor wiring | 7 | `196.5` | `3` | `[1, 1, 1]` | `3 (do:3)` |
| H2/H3/H4 replay | 7 | `210.5` | `3` | `[3, 1, 1]` | `1 (move_right:1)` |
| feasibility | 17 | `142.25` | `1` | `[1]` | `1 (do:1)` |
| diagnostics-only | 17 | `134.75` | `2` | `[1, 1]` | `0` |
| extractor wiring | 17 | `128.5` | `3` | `[1, 1, 2]` | `1 (do:1)` |
| H2/H3/H4 replay | 17 | `143.5` | `3` | `[1, 1, 1]` | `1 (do:1)` |

Weak-seed mean `avg_survival`:

- feasibility baseline: `172.875`
- diagnostics-only: `151.375`
- extractor wiring: `162.5`
- H2/H3/H4 replay: `177.0`

## Behavioral Readout

### Seed 7

This replay clearly removed the extractor-family terminal `do` regression.

- `avg_survival` improved to `210.5`, above feasibility `203.5`,
  diagnostics-only `168.0`, and extractor `196.5`.
- The seed-7 terminal false-safe `do` pattern is gone:
  - extractor wiring had `3` false-safe terminal `do` rows
  - H2/H3/H4 replay has `0` terminal `do` rows and `0` false-safe `do` rows
- Terminal outcomes return to the prior movement-heavy shape:
  - episode `0`: `move_up`
  - episode `1`: `move_up`
  - episode `2`: `move_right`
  - episode `3`: `move_right`
- The only visible false-safe terminal row is again the known `move_right`
  pattern at episode `3`, step `187`.

So for seed `7`, reverting H2/H3/H4 materially closed the gap and removed the
bad terminal `do` behavior.

### Seed 17

Seed `17` improved in average survival, but not in terminal-rescue shape.

- `avg_survival` improved to `143.5`, above feasibility `142.25`,
  diagnostics-only `134.75`, and extractor `128.5`.
- But `hostile_deaths_with_terminal_rescue` regressed upward:
  - feasibility: `1`
  - diagnostics-only: `2`
  - H2/H3/H4 replay: `3`
- Terminal rescue actions become all-`do` for the hostile deaths:
  - episode `0`: `do` vs actual `damage_step=2.0`, `nearest_hostile_after=1`
  - episode `1`: `do` vs actual `damage_step=2.0`, `nearest_hostile_after=1`
  - episode `2`: `do` vs actual `damage_step=2.0`, `nearest_hostile_after=1`
- The remaining visible false-safe row is the same pattern class as the
  feasibility baseline:
  - episode `2`, step `173`, `do`
  - predicted `damage_h=0.0`, predicted `nearest_hostile_h=2`
  - actual `damage_step=2.0`, actual `nearest_hostile_after=1`

So for seed `17`, the H2/H3/H4 replay did not solve the terminal `do` failure
class. It restored average survival but not the feasibility-baseline terminal
rescue shape.

## Interpretation

This replay is positive evidence against the claim that H2/H3/H4 were
behavior-neutral.

What this replay proves:

1. Reverting H2/H3/H4 materially changes runtime behavior.
2. The seed-7 terminal `do` regression disappears when H2/H3/H4 are removed.
3. Weak-seed mean survival rises from diagnostics-only `151.375` to `177.0`,
   which is above the prior feasibility mean `172.875`.

What it does **not** prove:

- It does not restore the exact feasibility-baseline behavior surface.
- Seed `17` still ends in the same broad false-safe terminal `do` class, and
  its `hostile_deaths_with_terminal_rescue` stays at `3`.

Best-supported reading:

- H2/H3/H4 were not neutral in practice and were a real contributor to the
  seed-7 regression family.
- The remaining seed-17 terminal `do` pattern is now the clearest unresolved
  behavior delta.
- Since this replay left only H5 as the remaining `vector_mpc_agent.py` delta
  versus the feasibility baseline, H5 is now the narrowest next replay target.

## Smallest Next Testable Hypothesis

Run the second replay from `vector_mpc_agent_delta_narrowing.md`:

- revert H5 only
- keep the H2/H3/H4 rollback in place

What that would prove:

- If seed-17 terminal `do` behavior collapses back toward the feasibility
  baseline, then the remaining mismatch was coming from the `do` target
  normalization hunk.
- If seed-17 still keeps the same terminal `do` failure pattern, then the
  remaining gap is outside H5 and the exact-feasibility-file control replay
  becomes the next clean proof step.
