# Stage91 Do-Facing Fix Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated do-facing checkout:
  `/opt/cuda/agi-stage91-do-facing-fix-20260507T213333Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_do_facing_fix_validation_20260507T213333Z`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_do_facing_fix_validation_20260507T213333Z/analysis/comparison_summary.json`
- Raw eval JSON:
  - `raw/seed_7_do_facing_fix_eval.json`
  - `raw/seed_17_do_facing_fix_eval.json`
- Logs:
  - `logs/seed_7_do_facing_fix.log`
  - `logs/seed_17_do_facing_fix.log`
- Provenance:
  - `env/import_probe.json`
  - `env/synced_runtime_diff.patch`
  - `env/synced_runtime_sha256.txt`
  - `env/isolated_checkout_status_after_sync.txt`
  - `env/py_compile.txt`

## Run Shape

The isolated checkout used:

`PYTHONPATH=/opt/cuda/agi-stage91-do-facing-fix-20260507T213333Z/src:/opt/cuda/agi-stage91-do-facing-fix-20260507T213333Z:/opt/cuda/agi-stage91-do-facing-fix-20260507T213333Z/experiments`

Seeds `7` and `17` were run with:

`--mode mixed_control_rescue --n-episodes 4 --max-steps 220 --perception-mode symbolic --local-evaluator /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt --enable-planner-rescue --smoke-lite --record-death-bundle --terminal-trace-steps 32 --max-explanations-per-episode 32`

Only the requested runtime files were synced:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/stage90r_emergency_controller.py`

`stage90r_local_policy.py` was not synced because it is training/dataset label
alignment code and was not required by the runtime eval import path.

One runtime-consistency fix was needed before the remote eval could run:
`vector_mpc_agent.py` had one remaining local-trace call to
`_build_local_counterfactual_outcomes(...)` that still omitted the new required
`facing_concept` keyword. The first remote launch failed with:

`TypeError: _build_local_counterfactual_outcomes() missing 1 required keyword-only argument: 'facing_concept'`

I patched that one omitted call site locally, resynced only
`src/snks/agent/vector_mpc_agent.py`, reran `py_compile`, and then reran the
two HyperPC evals. No controller or simulator heuristics were changed in this
validation task.

## Before / After Metrics

Comparison roots:

- Blocked-move fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_blocked_move_fix_validation_20260507T163659Z`
- Feasibility-label fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
- Do-facing fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_do_facing_fix_validation_20260507T213333Z`

| Run | Seed | Avg survival | Hostile deaths with terminal rescue | Blocked terminal rows | Final rescue nearest-hostile-after | False-safe terminal rows |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| blocked move | 7 | 164.75 | 3 | `2/3` | `[2, 1, 1]` | `0` |
| feasibility labels | 7 | 203.5 | 3 | `2/3` | `[3, 2, 1]` | `1 (move_right:1)` |
| do-facing | 7 | 129.25 | 3 | `3/3` | `[1, 1, 1]` | `3 (do:3)` |
| blocked move | 17 | 151.0 | 3 | `1/3` | `[2, 1, 1]` | `2 (do:2)` |
| feasibility labels | 17 | 142.25 | 1 | `0/1` | `[1]` | `1 (do:1)` |
| do-facing | 17 | 136.0 | 2 | `2/2` | `[1, 1]` | `2 (do:2)` |

Weak-seed mean `avg_survival`:

- blocked move: `157.875`
- feasibility labels: `172.875`
- do-facing: `132.625`

## Terminal Row Readout

Seed 17 did not improve on the target pattern:

- The remaining false-safe terminal `do` row did not disappear. It expanded
  back to `2` false-safe terminal `do` rows, matching the worse blocked-move
  count and regressing from the feasibility-label run's `1`.
- `hostile_deaths_with_terminal_rescue` worsened from `1` to `2`.
- Blocked terminal rows worsened from `0/1` to `2/2`.
- Final rescue separation stayed fully adjacent at `[1, 1]` instead of the
  blocked-move run's `[2, 1, 1]`.
- Current seed-17 terminal rows:
  - episode `0`: `do`, predicted `survived_h=True`, `damage_h=0.0`,
    `nearest_hostile_h=None`, `effective_displacement_h=0.0`,
    `blocked_h=False`, `adjacent_hostile_after_h=False`; actual
    `displacement_step=0`, `nearest_hostile_after=1`, `damage_step=4.0`
  - episode `2`: `do`, predicted `survived_h=True`, `damage_h=0.0`,
    `nearest_hostile_h=None`, `effective_displacement_h=0.0`,
    `blocked_h=False`, `adjacent_hostile_after_h=False`; actual
    `displacement_step=0`, `nearest_hostile_after=1`, `damage_step=1.0`

Seed 7 also regressed sharply:

- `avg_survival` fell from `203.5` after the feasibility-label fix to `129.25`.
- The prior single over-optimistic terminal move row disappeared, but only
  because the terminal pattern collapsed into `3` false-safe terminal `do`
  rows.
- Blocked terminal rows worsened from `2/3` to `3/3`.
- Final rescue separation regressed from `[3, 2, 1]` to `[1, 1, 1]`.
- Current seed-7 terminal rows:
  - episode `1`: `do`, predicted `survived_h=True`, `damage_h=0.0`,
    `nearest_hostile_h=None`, `effective_displacement_h=0.0`,
    `blocked_h=False`, `adjacent_hostile_after_h=False`; actual
    `displacement_step=0`, `nearest_hostile_after=1`, `damage_step=2.0`
  - episode `2`: `do`, predicted `survived_h=True`, `damage_h=0.0`,
    `nearest_hostile_h=None`, `effective_displacement_h=0.0`,
    `blocked_h=False`, `adjacent_hostile_after_h=False`; actual
    `displacement_step=0`, `nearest_hostile_after=1`, `damage_step=2.0`
  - episode `3`: `do`, predicted `survived_h=True`, `damage_h=0.0`,
    `nearest_hostile_h=None`, `effective_displacement_h=0.0`,
    `blocked_h=False`, `adjacent_hostile_after_h=False`; actual
    `displacement_step=0`, `nearest_hostile_after=1`, `damage_step=0.0`

## Interpretation

This validation is negative overall.

The narrow do-facing change does not remove the seed-17 false-safe terminal
`do` pattern. It makes it worse on this bounded weak-seed HyperPC check:
seed `17` rises from `1` false-safe terminal `do` row after the
feasibility-label fix back to `2`, and seed `7` regresses from one residual
false-safe move row to three false-safe terminal `do` rows.

The weak-seed mean `avg_survival` drops from `172.875` after the
feasibility-label fix to `132.625`, below even the blocked-move fix
(`157.875`). The strong seed-7 gains from the feasibility-label fix are not
preserved.

Best readout: switching rescue `do` counterfactuals from `near_concept` to the
facing tile, in the current local runtime stack, does not solve the remaining
root-cause mismatch and instead degrades both weak seeds. The seed-17
false-safe `do` failure remains present, and the counterfactual labels for
terminal `do` rows become less informative, often showing
`nearest_hostile_h=None` and `effective_displacement_h=0.0` while still ranking
`do` as safe and then staying adjacent at `nearest_hostile_after=1`.
