# Stage91 Local-Affordance Extractor Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated local-affordance checkout:
  `/opt/cuda/agi-stage91-local-affordance-extractor-20260508T082732Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z/analysis/comparison_summary.json`
- Raw eval JSON:
  - `raw/seed_7_local_affordance_extractor_eval.json`
  - `raw/seed_17_local_affordance_extractor_eval.json`
- Logs:
  - `logs/seed_7_local_affordance_extractor.log`
  - `logs/seed_17_local_affordance_extractor.log`
- Provenance:
  - `env/import_probe.json`
  - `env/synced_runtime_diff.patch`
  - `env/synced_runtime_sha256.txt`
  - `env/isolated_checkout_status_after_sync.txt`
  - `env/py_compile.txt`

## Run Shape

The isolated checkout used:

`PYTHONPATH=/opt/cuda/agi-stage91-local-affordance-extractor-20260508T082732Z/src:/opt/cuda/agi-stage91-local-affordance-extractor-20260508T082732Z:/opt/cuda/agi-stage91-local-affordance-extractor-20260508T082732Z/experiments`

Seeds `7` and `17` were run with:

`--mode mixed_control_rescue --n-episodes 4 --max-steps 220 --perception-mode symbolic --local-evaluator /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt --enable-planner-rescue --smoke-lite --record-death-bundle --terminal-trace-steps 32 --max-explanations-per-episode 32`

Only the requested runtime files were synced:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/stage90r_emergency_controller.py`
- `src/snks/agent/stage90r_local_affordances.py`

`stage90r_local_policy.py` was not synced because it is training/dataset code
and was not required by the runtime eval import path.

`py_compile` passed for all seven synced files. `import_probe.json` confirms the
isolated checkout was the active import root, including
`stage90r_local_affordances.py`.

## Before / After Metrics

Primary comparison root:

- Feasibility-label baseline:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`

Secondary comparison root:

- Blocked-move baseline:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_blocked_move_fix_validation_20260507T163659Z`

| Run | Seed | Avg survival | Deaths | Hostile deaths with terminal rescue | Blocked terminal rows | Final rescue nearest-hostile-after | False-safe terminal rows |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| blocked move | 7 | 164.75 | `unknown:1, zombie:3` | 3 | 2/3 | `[2, 1, 1]` | `0` |
| feasibility labels | 7 | 203.5 | `dehydration:1, skeleton:1, zombie:2` | 3 | 2/3 | `[3, 2, 1]` | `1 (move_right:1)` |
| local affordance extractor | 7 | 196.5 | `dehydration:1, zombie:3` | 3 | 0/3 | `[1, 1, 1]` | `3 (do:3)` |
| blocked move | 17 | 151.0 | `arrow:1, unknown:1, zombie:2` | 3 | 1/3 | `[2, 1, 1]` | `2 (do:2)` |
| feasibility labels | 17 | 142.25 | `dehydration:1, unknown:2, zombie:1` | 1 | 0/1 | `[1]` | `1 (do:1)` |
| local affordance extractor | 17 | 128.5 | `unknown:1, zombie:3` | 3 | 0/3 | `[1, 1, 2]` | `1 (do:1)` |

Weak-seed mean `avg_survival`:

- blocked move: `157.875`
- feasibility labels: `172.875`
- local affordance extractor: `162.5`

## Instrumentation Readout

The new affordance instrumentation is active in runtime artifacts:

- `pre_rescue_state.local_affordance_scene` was present on every recorded rescue
  event:
  - seed `7`: `128/128`
  - seed `17`: `96/96`
- Scene keys seen in rescue traces:
  - `facing_blocked`
  - `facing_concept`
  - `facing_tile`
  - `nearest_hostile_direction`
  - `nearest_hostile_distance`
- Rescue candidate labels carried all six new `*_local` fields on every
  recorded rescue event excerpt:
  - seed `7`: `128/128`
  - seed `17`: `96/96`
- The excerpted/tail local counterfactual traces also carried all six new
  `*_local` fields on every recorded candidate label:
  - seed `7`: `929/929`
  - seed `17`: `713/713`

The generic serialized local trace does **not** preserve the scene snapshot:

- `local_trace_excerpt` + `local_trace_tail` rows with `local_affordance_scene`:
  - seed `7`: `0/256`
  - seed `17`: `0/256`

That matches the current code path: `local_affordance_scene` is attached to the
rescue-oriented trace and step trace, but `build_local_trace_entry()` does not
carry it into the generic local-trace serializer.

## Behavioral Readout

Seed `7` regressed from the prior best bounded baseline, but stayed above the
older blocked-move baseline:

- `avg_survival` fell from feasibility-label `203.5` to `196.5`.
- `hostile_deaths_with_terminal_rescue` stayed at `3`.
- Final rescue separation regressed from `[3, 2, 1]` to `[1, 1, 1]`.
- The prior single false-safe `move_right` row was replaced by `3` false-safe
  terminal `do` rows.

Seed `17` regressed more clearly:

- `avg_survival` fell from feasibility-label `142.25` to `128.5`, below both
  the prior positive baseline and the blocked-move baseline.
- `hostile_deaths_with_terminal_rescue` rose from `1` back to `3`.
- One terminal row reached `nearest_hostile_after=2`, but the other two stayed
  adjacent at `1`.
- False-safe terminal `do` behavior did not disappear; it remained `1`.

## Interpretation

The shared local-affordance extractor validates as instrumentation, not as a
behavior-preserving no-op. The intended local truth fields are present in
emergency traces and candidate labels exactly where they need to be for root
cause inspection.

Behaviorally, this change does **not** preserve the previously best bounded
weak-seed baseline. Weak-seed mean `avg_survival` falls from `172.875` on the
feasibility-label baseline to `162.5`. Seed `7` remains stronger than the
blocked-move baseline, but seed `17` regresses materially and loses the prior
drop in `hostile_deaths_with_terminal_rescue`.

The clearest regression signature is that terminal false-safe behavior swings
back toward `do`:

- seed `7`: `1` false-safe terminal row on the feasibility-label baseline
  becomes `3`, all `do`.
- seed `17`: the remaining false-safe terminal row stays `do`.

Best readout: the extractor itself is wired correctly, but the current runtime
integration is not yet behavior-neutral relative to the prior positive
feasibility-label baseline. The new local fields are usable for forensics, but
this revision should not replace the prior best bounded baseline as the active
Stage91 validation winner.

No additional controller or simulator heuristics were changed in this task.
