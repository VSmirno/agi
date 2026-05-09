# Stage91 `vector_mpc_agent.py` H5 Replay Validation

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z`
- Raw eval JSON:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/raw/seed_7_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/raw/seed_17_eval.json`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/analysis/comparison_summary.json`
- Provenance:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/env/import_probe.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/env/py_compile.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/env/runtime_interpreter.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z/env/synced_runtime_sha256.txt`

## Run Shape

This replay used the same six synced runtime files as the prior bounded runs:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/stage90r_emergency_controller.py`

`stage90r_local_affordances.py` was not synced and was not imported.

The replay ran under:

`/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python3.11`

Most important provenance fact:

- current local `vector_mpc_agent.py` is byte-identical to the recovered
  feasibility-baseline file:
  `98c5f83babad7bb97c2cd838605ca950f90d3450b5a08110564db7d7cd3ff368`
- the other five synced runtime file hashes also match the feasibility artifact
  exactly

So this replay is effectively the exact six-file feasibility surface, run again
from a fresh checkout derived from the base verify tree.

## Comparison Targets

- Feasibility baseline:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
- H2/H3/H4 replay:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z`
- Diagnostics-only run:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z`

## Before / After Metrics

| Run | Seed | Avg survival | Hostile deaths with terminal rescue | Final hostile rescue nearest-hostile-after | False-safe terminal rows |
| --- | ---: | ---: | ---: | --- | --- |
| feasibility | 7 | `203.5` | `3` | `[3, 2, 1]` | `1 (move_right:1)` |
| H2/H3/H4 replay | 7 | `210.5` | `3` | `[3, 1, 1]` | `1 (move_right:1)` |
| diagnostics-only | 7 | `168.0` | `3` | `[4, 2, 1]` | `1 (move_right:1)` |
| H5 replay | 7 | `159.0` | `3` | `[1, 2, 1]` | `3 (do:2, move_left:1)` |
| feasibility | 17 | `142.25` | `1` | `[1]` | `1 (do:1)` |
| H2/H3/H4 replay | 17 | `143.5` | `3` | `[1, 1, 1]` | `1 (do:1)` |
| diagnostics-only | 17 | `134.75` | `2` | `[1, 1]` | `0` |
| H5 replay | 17 | `151.5` | `1` | `[2]` | `0` |

Weak-seed mean `avg_survival`:

- feasibility baseline: `172.875`
- H2/H3/H4 replay: `177.0`
- diagnostics-only: `151.375`
- H5 replay: `155.25`

## Behavioral Readout

### Seed 7

H5 replay is clearly negative on seed `7`.

- `avg_survival` drops to `159.0`, below feasibility `203.5`, below H2/H3/H4
  replay `210.5`, and below even the extractor wiring run `196.5`.
- The bad terminal `do` class is reintroduced:
  - episode `1`, step `170`, `do`, predicted safe, actual `damage_step=1.0`,
    `nearest_hostile_after=1`
  - episode `3`, step `208`, `do`, predicted safe, actual `damage_step=1.0`,
    `nearest_hostile_after=1`
- One additional false-safe movement row also appears:
  - episode `2`, step `198`, `move_left`, predicted `damage_h=0.0`,
    actual `damage_step=2.0`, actual `nearest_hostile_after=2`

So H5 reintroduces the exact kind of seed-7 terminal `do` failure that the
H2/H3/H4 replay had removed.

### Seed 17

H5 replay is strongly positive on seed `17`.

- `avg_survival` rises to `151.5`, above feasibility `142.25`, above H2/H3/H4
  replay `143.5`, and above diagnostics-only `134.75`.
- `hostile_deaths_with_terminal_rescue` improves back down to `1`, matching the
  feasibility baseline.
- Visible terminal false-safe rows drop to `0`.
- Final hostile terminal rescue separation improves to `[2]`, better than the
  feasibility baseline’s `[1]`.

The seed-17 replay shape is materially cleaner than either the diagnostics-only
or H2/H3/H4 replay runs.

## Interpretation

This replay is not a net improvement.

What it proves:

1. H5 has real behavioral impact.
2. H5 is positive for seed `17`.
3. H5 is strongly negative for seed `7`, specifically by reintroducing the
   terminal `do` failure family.

The more important root-cause update is this:

- This run used the exact same six synced runtime hashes as the original
  feasibility validation, including a byte-identical `vector_mpc_agent.py`.
- Yet it did **not** reproduce the original feasibility metrics.

That falsifies the earlier assumption that the remaining causal delta lived only
inside these six synced files.

Best-supported conclusion:

- H5 should not be adopted as the active fix. It improves seed `17`, but the
  seed-7 regression cost is too large.
- More importantly, exact-six-file replay not reproducing the original
  feasibility artifact means there is some uncontrolled provenance/runtime
  dependency outside the current six-file sync surface.

## Smallest Next Testable Hypothesis

Do **not** continue tuning `vector_mpc_agent.py` blindly from this point.

Next step should be provenance closure, not another heuristic replay:

1. compare the broader imported runtime closure between
   `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z` and the
   fresh checkout derived from
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
2. identify which unsynced imported modules differ
3. only then rerun the bounded seeds

What that would prove:

- If additional imported runtime files differ, the non-reproduction is explained
  by incomplete sync closure.
- If the broader runtime closure also matches, then the remaining mismatch is in
  environment/interpreter/runtime state rather than repo file content.
