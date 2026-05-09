# Stage91 Blocked-Move Fix Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated blocked-move checkout:
  `/opt/cuda/agi-stage91-blocked-move-fix-20260507T163659Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_blocked_move_fix_validation_20260507T163659Z`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_blocked_move_fix_validation_20260507T163659Z/analysis/comparison_summary.json`
- Raw eval JSON:
  - `raw/seed_7_blocked_move_fix_eval.json`
  - `raw/seed_17_blocked_move_fix_eval.json`
- Provenance:
  - `env/import_probe.json`
  - `env/synced_runtime_diff.patch`
  - `env/synced_runtime_sha256.txt`
  - `env/isolated_checkout_status_after_sync.txt`

## Run Shape

The isolated checkout used:

`PYTHONPATH=/opt/cuda/agi-stage91-blocked-move-fix-20260507T163659Z/src:/opt/cuda/agi-stage91-blocked-move-fix-20260507T163659Z:/opt/cuda/agi-stage91-blocked-move-fix-20260507T163659Z/experiments`

Seeds `7` and `17` were run with `CUDA_VISIBLE_DEVICES=0/1` and:

`--mode mixed_control_rescue --n-episodes 4 --max-steps 220 --perception-mode symbolic --local-evaluator /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt --enable-planner-rescue --smoke-lite --record-death-bundle --terminal-trace-steps 32 --max-explanations-per-episode 32`

Only the requested runtime files were synced into the isolated checkout:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/stage90r_emergency_controller.py`

`py_compile` passed for all five synced files on HyperPC. `import_probe.json`
confirms imports resolved from the isolated checkout.

## Before / After Metrics

Comparison roots:

- Terminal forensics:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_terminal_forensics_20260506T171559Z`
- Movement fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_movement_fix_validation_20260506T221512Z`
- Threat-ranking non-solution:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_threat_ranking_fix_validation_20260507T104724Z`
- Blocked-move fix:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_blocked_move_fix_validation_20260507T163659Z`

| Run | Seed | Avg survival | Deaths | Hostile deaths with terminal rescue | Blocked terminal rows | Final rescue nearest-hostile-after |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| terminal forensics | 7 | 148.0 | `zombie:4` | 4 | 2/4 | `[1, 1, 2, 1]` |
| movement fix | 7 | 144.5 | `zombie:4` | 4 | 3/4 | `[1, 1, 1, 1]` |
| threat ranking | 7 | 116.25 | `unknown:1, arrow:1, zombie:1, skeleton:1` | 3 | 3/3 | `[1, 1, 1]` |
| blocked move | 7 | 164.75 | `unknown:1, zombie:3` | 3 | 2/3 | `[2, 1, 1]` |
| terminal forensics | 17 | 108.75 | `skeleton:2, zombie:1, unknown:1` | 3 | 3/3 | `[1, 1, 1]` |
| movement fix | 17 | 140.0 | `skeleton:1, zombie:2, unknown:1` | 3 | 2/3 | `[1, 1, 1]` |
| threat ranking | 17 | 119.75 | `skeleton:1, arrow:1, zombie:1, unknown:1` | 3 | 2/3 | `[5, None, 1]` |
| blocked move | 17 | 151.0 | `arrow:1, zombie:2, unknown:1` | 3 | 1/3 | `[2, 1, 1]` |

Weak-seed mean `avg_survival`:

- terminal forensics: `128.375`
- movement fix: `142.25`
- threat ranking: `118.0`
- blocked move: `157.875`

## Terminal Row Readout

Seed 7 after the blocked-move fix:

- Terminal hostile rows fell from `4` in the movement-fix run to `3`.
- Blocked terminal rows fell from `3/4` to `2/3`.
- One terminal-attributed row reached `nearest_hostile_after=2`; the other two
  remained adjacent at `1`.
- The previous false zero-damage safe movement labels disappeared on this seed:
  `false_zero_damage_safe_rows=0` versus `2` in movement fix.

Seed 17 after the blocked-move fix:

- Terminal hostile rows stayed at `3`.
- Blocked terminal rows fell from `2/3` to `1/3`.
- One row reached `nearest_hostile_after=2`; two remained adjacent at `1`.
- False-safe terminal labels still remain, but now on non-move choices:
  two final rescue rows chose `do`, predicted `survived_h=True` and
  `damage_h=0`, then took `5.0` and `2.0` actual damage.

## Interpretation

The blocked-move/passability fix is positive on this bounded weak-seed check:
both weak seeds improved versus the movement-fix baseline, and the combined
weak-seed mean improved from `142.25` to `157.875`.

It also directly reduces the target mismatch pattern. Blocked final rescue rows
drop on both seeds, and each seed now has one terminal rescue step where actual
nearest-hostile separation reaches `2` instead of remaining uniformly adjacent.

It is not a complete Stage91 fix. `hostile_deaths_with_terminal_rescue` remains
`3` on both weak seeds, terminal actual separation still stays at `1` for most
rows, and seed 17 still shows false-safe one-step labels through `do` choices
rather than blocked movement. The best readout is that adjacent passability was
a real local-rollout fidelity gap and the narrow fix helps, but remaining
terminal failures still involve local counterfactual reliability/action ranking
under immediate danger.

No further controller or simulator heuristics were changed in this validation
task.
