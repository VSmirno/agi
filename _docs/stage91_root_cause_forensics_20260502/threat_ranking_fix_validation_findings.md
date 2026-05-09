# Stage91 Threat-Ranking Fix Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated threat-ranking checkout:
  `/opt/cuda/agi-stage91-threat-ranking-fix-20260507T104724Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_threat_ranking_fix_validation_20260507T104724Z`
- Comparison artifacts:
  - `analysis/comparison_summary.json`
  - `commands/seed_7_threat_ranking_fix.cmd`
  - `commands/seed_17_threat_ranking_fix.cmd`
  - `logs/seed_7_threat_ranking_fix.log`
  - `logs/seed_17_threat_ranking_fix.log`
  - `raw/seed_7_threat_ranking_fix_eval.json`
  - `raw/seed_17_threat_ranking_fix_eval.json`
  - `env/import_probe.json`
  - `env/synced_runtime_diff.patch`
  - `env/synced_runtime_sha256.txt`

## Run Shape

Both runs used the isolated checkout as the import root:

`PYTHONPATH=/opt/cuda/agi-stage91-threat-ranking-fix-20260507T104724Z/src:/opt/cuda/agi-stage91-threat-ranking-fix-20260507T104724Z:/opt/cuda/agi-stage91-threat-ranking-fix-20260507T104724Z/experiments`

with `CUDA_VISIBLE_DEVICES=0` and:

`--mode mixed_control_rescue --n-episodes 4 --max-steps 220 --perception-mode symbolic --local-evaluator /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt --enable-planner-rescue --smoke-lite --record-death-bundle --terminal-trace-steps 32 --max-explanations-per-episode 32`

Only the requested runtime files were synced into the isolated checkout:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/stage90r_emergency_controller.py`

## Before / After Metrics

Prior comparison roots:

- Terminal forensics:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_terminal_forensics_20260506T171559Z`
- Movement-fix validation:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_movement_fix_validation_20260506T221512Z`

### Seed 7

| Run | Avg survival | Deaths | Rescue rate | Emergency safety | Override mix | Hostile deaths with terminal rescue | Terminal hostile actions |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| terminal forensics | 148.0 | `zombie:4` | 0.475 | 281 | `independent:179, planner:77, learner:20, advisory:5` | 4 | `move_down:2, move_up:2` |
| movement fix | 144.5 | `zombie:4` | 0.424 | 245 | `independent:168, planner:57, learner:15, advisory:5` | 4 | `move_down:2, move_up:2` |
| threat ranking fix | 116.25 | `unknown:1, arrow:1, zombie:1, skeleton:1` | 0.310 | 144 | `independent:100, planner:26, learner:17, advisory:1` | 3 | `move_left:3` |

Seed 7 got materially worse versus both priors:

- `-28.25` avg survival versus the movement-fix run.
- `-31.75` avg survival versus terminal forensics.
- Mean rescue use dropped sharply, and one episode died as `unknown` with no rescues.

The terminal hostile actions did become cleaner in the narrow final-step sense: the hostile/arrow terminal final actions were all `move_left`, with no final `sleep` or `do`. That did not translate into survival.

### Seed 17

| Run | Avg survival | Deaths | Rescue rate | Emergency safety | Override mix | Hostile deaths with terminal rescue | Terminal hostile actions |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| terminal forensics | 108.75 | `skeleton:2, zombie:1, unknown:1` | 0.359 | 156 | `independent:78, advisory:41, planner:26, learner:11` | 3 | `move_down:2, move_left:1` |
| movement fix | 140.0 | `skeleton:1, zombie:2, unknown:1` | 0.457 | 256 | `independent:146, advisory:62, planner:42, learner:6` | 3 | `sleep:1, move_left:2` |
| threat ranking fix | 119.75 | `skeleton:1, arrow:1, zombie:1, unknown:1` | 0.461 | 221 | `independent:85, advisory:66, planner:48, learner:22` | 3 | `move_left:2, move_down:1` |

Seed 17 stayed above terminal forensics but lost most of the movement-fix gain:

- `-20.25` avg survival versus the movement-fix run.
- `+11.0` avg survival versus terminal forensics.
- The movement-fix terminal `sleep` death disappeared, and all hostile/arrow final terminal actions were movement actions.

## Combined Readout

Mean weak-seed avg survival:

- Terminal forensics: `128.375`
- Movement fix: `142.25`
- Threat ranking fix: `118.0`

The threat-ranking fix reduced some bad terminal-choice signatures:

- Final terminal `sleep`/`do` hostile actions dropped from `1` after the movement fix to `0`.
- Independent emergency choices dropped on both weak seeds:
  - seed 7: `168 -> 100`
  - seed 17: `146 -> 85`
- Hostile deaths still had terminal rescue active; `hostile_deaths_without_terminal_rescue=0` for both new runs.

But the survival outcome is negative:

- Seed 7 regressed badly and introduced an unrecovered `unknown` death plus an `arrow` death.
- Seed 17 regressed versus the movement-fix run despite removing the terminal `sleep` final action.
- Hostile/arrow terminal failures still occur with rescue active, just with less obviously bad final actions.

## Interpretation

The fix validates part of the hypothesis: the emergency ranking can be steered away from final `sleep`/`do` choices under immediate threat, and the override-source mix becomes less dominated by `independent_emergency_choice`.

It does **not** validate as a Stage91 survival fix. The weak-seed mean fell below both prior comparisons, and terminal rescue outcomes remain poor. The best current readout is that terminal action ranking was one real failure surface, but making the final action look more movement-like is insufficient and may perturb trajectories into other failure modes.

No additional controller or simulator heuristics were changed in this validation task.
