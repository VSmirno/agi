# Stage91 Feasibility-Label Fix Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated feasibility-label checkout:
  `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z/analysis/comparison_summary.json`
- Raw eval JSON:
  - `raw/seed_7_feasibility_label_fix_eval.json`
  - `raw/seed_17_feasibility_label_fix_eval.json`
- Logs:
  - `logs/seed_7_feasibility_label_fix.log`
  - `logs/seed_17_feasibility_label_fix.log`
- Provenance:
  - `env/import_probe.json`
  - `env/synced_runtime_diff.patch`
  - `env/synced_runtime_sha256.txt`
  - `env/isolated_checkout_status_after_sync.txt`
  - `env/py_compile.txt`

## Run Shape

The isolated checkout used:

`PYTHONPATH=/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/src:/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z:/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/experiments`

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

`py_compile` passed for all six synced files. `import_probe.json` confirms the
eval imported `snks`, `vector_*`, `stage90r_emergency_controller`, and
`stage90r_eval_local_policy` from the isolated checkout.

## Before / After Metrics

Comparison roots:

- Movement fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_movement_fix_validation_20260506T221512Z`
- Blocked-move fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_blocked_move_fix_validation_20260507T163659Z`
- Feasibility-label fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`

| Run | Seed | Avg survival | Deaths | Hostile deaths with terminal rescue | Blocked terminal rows | Final rescue nearest-hostile-after | False-safe terminal rows |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| movement fix | 7 | 144.5 | `zombie:4` | 4 | 3/4 | `[1, 1, 1, 1]` | `2 (move_up:2)` |
| blocked move | 7 | 164.75 | `unknown:1, zombie:3` | 3 | 2/3 | `[2, 1, 1]` | `0` |
| feasibility labels | 7 | 203.5 | `dehydration:1, skeleton:1, zombie:2` | 3 | 2/3 | `[3, 2, 1]` | `1 (move_right:1)` |
| movement fix | 17 | 140.0 | `skeleton:1, unknown:1, zombie:2` | 3 | 2/3 | `[1, 1, 1]` | `1 (move_left:1)` |
| blocked move | 17 | 151.0 | `arrow:1, unknown:1, zombie:2` | 3 | 1/3 | `[2, 1, 1]` | `2 (do:2)` |
| feasibility labels | 17 | 142.25 | `dehydration:1, unknown:2, zombie:1` | 1 | 0/1 | `[1]` | `1 (do:1)` |

Weak-seed mean `avg_survival`:

- terminal forensics: `128.375`
- movement fix: `142.25`
- blocked move: `157.875`
- feasibility-label fix: `172.875`

## Terminal Row Readout

Seed 7 improved strongly:

- `avg_survival` rose from `164.75` after the blocked-move fix to `203.5`.
- One episode reached the full `max_steps=220` and died by dehydration rather
  than hostile contact.
- Terminal rescue separation improved from `[2, 1, 1]` to `[3, 2, 1]`.
- `hostile_deaths_with_terminal_rescue` stayed at `3`, so hostile terminal
  failures are not gone, but the final rescue rows are less uniformly adjacent.
- One false-safe row remains: episode `3` chose `move_right`, predicted
  `survived_h=True`, `damage_h=0`, `blocked_h=False`,
  `adjacent_hostile_after_h=False`, and `effective_displacement_h=1`, but
  actually failed to displace, took `2.0` damage, and remained at
  `nearest_hostile_after=1`.

Seed 17 is mixed:

- `avg_survival=142.25`, down from blocked-move `151.0` but still above the
  movement-fix `140.0` and terminal-forensics `108.75` runs.
- `hostile_deaths_with_terminal_rescue` dropped from `3` to `1`.
- Blocked terminal movement rows dropped from `1/3` to `0/1`.
- The remaining hostile terminal row is still a false-safe `do`: episode `2`
  predicted `survived_h=True`, `damage_h=0`, `nearest_hostile_h=2`,
  `adjacent_hostile_after_h=False`, and `effective_displacement_h=1`, but
  actually took `2.0` damage and remained adjacent at
  `nearest_hostile_after=1`.

The seed-17 false-safe `do`/move pattern shrank but did not disappear:

- blocked-move fix: `2` false-safe `do` terminal rows.
- feasibility-label fix: `1` false-safe `do` terminal row.
- false-safe move rows on seed 17 were `0` after this fix.

## Interpretation

The feasibility-label fix is positive on the bounded weak-seed CUDA check. It
raises the weak-seed mean from `157.875` after the blocked-move fix to
`172.875`, with the largest gain on seed `7`.

It also improves the specific target pattern, especially on seed `17`: terminal
hostile deaths with rescue active fall from `3` to `1`, blocked terminal move
rows fall to `0/1`, and the prior two false-safe `do` rows shrink to one.

The fix is still incomplete. Seed `17` survival regresses versus the
blocked-move baseline, and one seed-17 false-safe `do` remains. Seed `7` still
has three hostile deaths with terminal rescue active and one terminal move whose
new feasibility labels were over-optimistic: the label predicted displacement
and non-adjacency, while the actual step stayed blocked/adjacent and took
damage.

Best readout: direct consumption of immediate feasibility labels is a useful
narrow step and improves weak-seed survival overall, but the remaining
false-safe cases are no longer only plain blocked-move/passability misses. They
now point at residual one-step counterfactual mismatch around `do` under contact
and around movement labels that can still predict displacement/separation when
the actual transition does not deliver it.

No additional controller or simulator heuristics were changed in this
validation task.
