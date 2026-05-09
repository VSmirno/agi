# Seed 7 Terminal Rescue Counterfactual Mismatches

Scope: weak seed `7` only, existing HyperPC artifacts only. No new evals were run.

Artifact roots inspected:

- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_terminal_forensics_20260506T171559Z`
- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_movement_fix_validation_20260506T221512Z`
- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_threat_ranking_fix_validation_20260507T104724Z`

Included rows are only seed-7 terminal `zombie` / `skeleton` / `arrow` deaths where rescue was active at the terminal local step or within the last two rescue steps. The threat-ranking run episode 0 is excluded because it died `unknown` without terminal hostile/arrow rescue evidence.

## Evidence Table

Columns: `pred_*` are from the chosen action in `ranked_emergency_actions.components` at the last rescue step. `actual_dmg` and `actual_near` are from that rescue row's `post_rescue_outcome.damage_step` and `post_rescue_outcome.nearest_hostile_after`.

| Run | Ep | Len | Death | Step | Trigger | Choice | Planner | Override | pred_surv | pred_dmg | pred_escape | pred_near | actual_dmg | actual_near | Blocked | Pos before -> after | Prev rescue |
| --- | --: | --: | --- | --: | --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --- | --- | --- |
| terminal | 0 | 168 | zombie | 167 | low_vitals | move_down | move_right | advisory_aligned_safety | F | 2 | 1 | 2 | 2 | 1 | T | [29,21] -> [29,21] | move_down; pred 2/2 |
| terminal | 1 | 55 | zombie | 54 | low_vitals | move_down | move_up | advisory_aligned_safety | F | 2 | 1 | 2 | 2 | 1 | F | [23,29] -> [23,30] | move_up; pred 0/2 |
| terminal | 2 | 171 | zombie | 170 | low_vitals | move_up | move_up | planner_aligned_safety | T | 0 | 1 | 2 | 2 | 1 | F | [25,23] -> [25,22] | move_left; pred 2/1 |
| terminal | 3 | 198 | zombie | 197 | low_vitals | move_up | move_up | planner_aligned_safety | T | 0 | 1 | 2 | 1 | 1 | T | [39,12] -> [39,12] | move_right; pred 0/2 |
| movement | 0 | 168 | zombie | 167 | low_vitals | move_down | move_right | advisory_aligned_safety | F | 2 | 1 | 2 | 2 | 1 | T | [29,21] -> [29,21] | move_down; pred 2/2 |
| movement | 1 | 55 | zombie | 54 | low_vitals | move_down | move_up | advisory_aligned_safety | F | 2 | 1 | 2 | 2 | 1 | F | [23,29] -> [23,30] | move_up; pred 0/2 |
| movement | 2 | 146 | zombie | 145 | low_vitals | move_up | move_left | independent_emergency_choice | T | 0 | 1 | 2 | 2 | 1 | T | [7,30] -> [7,30] | move_up; pred 0/2 |
| movement | 3 | 209 | zombie | 208 | low_vitals | move_up | move_down | independent_emergency_choice | T | 0 | 1 | 2 | 1 | 1 | T | [38,17] -> [38,17] | move_up; pred 0/2 |
| threat | 1 | 56 | arrow | 55 | low_vitals | move_left | move_up | independent_emergency_choice | F | 2 | 0 | 1 | 2 | 1 | T | [21,31] -> [21,31] | do; pred 0/ |
| threat | 2 | 189 | zombie | 188 | low_vitals | move_left | move_right | learner_aligned_safety | F | 0 | 1 | 2 | 1 | 1 | T | [0,32] -> [0,32] | move_left; pred 0/2 |
| threat | 3 | 163 | skeleton | 162 | low_vitals | move_left | move_left | planner_aligned_safety | F | 2 | 0 | 1 | 2 | 1 | T | [34,32] -> [34,32] | move_down; pred 2/1 |

## Direct-Evidence Patterns

1. Rescue activation is not the missing piece for these terminal hostile/arrow failures.

All 11 included terminal hostile/arrow deaths have rescue active at the last rescue step; every last trigger is `low_vitals`. The actual terminal damage is positive in every row, and `actual_near=1` in every row.

2. The dominant concrete mismatch is blocked or non-escaping movement under terminal threat.

Eight of 11 rows have `blocked=T`, meaning the chosen emergency movement left the player in the same tile. Those eight rows still take `actual_dmg` of `1` or `2` and remain adjacent to a hostile/projectile. In six of those eight blocked rows, the chosen action predicted `pred_escape=1` and/or `pred_near=2`, but actual separation stayed `1`.

3. The pre-threat-ranking and movement-fix runs contain false-safe labels.

Rows `terminal ep2`, `terminal ep3`, `movement ep2`, and `movement ep3` predict `pred_surv=T`, `pred_dmg=0`, `pred_escape=1`, `pred_near=2`; actual execution takes `1-2` damage and remains at distance `1`. The movement fix changed trajectories but did not remove the terminal mismatch on seed 7.

4. The threat-ranking fix made final actions look cleaner but did not make them executable.

Threat-ranking rows choose only `move_left`, avoiding final `sleep`/`do`, but all three included hostile/arrow terminal deaths are blocked and remain at `actual_near=1`. This is direct evidence that movement-like final action choice is insufficient without passability/execution fidelity.

## Likely Local Code Surfaces

Proved from artifacts:

- The last rescue step's selected movement often does not execute in the environment (`blocked=T`, unchanged position), while the rescue ranking still assigns favorable or tolerable local labels.
- The selected emergency action is produced by the rescue ranking path logged as `ranked_emergency_actions`, `rescue_action`, `planner_action`, and `override_source`.
- The actual terminal damage/blocked status is recorded after environment execution in the death bundle.

Strong inference from local code:

- `src/snks/agent/vector_sim.py:76-97` and `src/snks/agent/vector_sim.py:425-486` are the strongest exact surface. `VectorState.move_player()` unconditionally changes simulated position for `move_*`, and `_advance_dynamic_entities()` applies that simulated movement before hostile/projectile damage. There is no visible passability/collision check analogous to the environment's blocked move outcome.
- `src/snks/agent/vector_mpc_agent.py:1018-1027` builds the local counterfactual outcomes used by rescue. `src/snks/agent/vector_mpc_agent.py:1040-1083` applies and records the emergency selection. `src/snks/agent/vector_mpc_agent.py:1764-1813` generates the `damage_h`, `survived_h`, `escape_delta_h`, and `nearest_hostile_h` labels from simulated final state.
- `src/snks/agent/stage90r_emergency_controller.py:271-315` ranks actions from those labels. It penalizes damage and rewards survival/escape, but it has no direct terminal passability/blocked-action feature unless that is already reflected in the candidate labels.
- `src/snks/agent/vector_mpc_agent.py:1448-1475` records the rescue row's actual post-step `damage_step` and `nearest_hostile_after` used by this table. `src/snks/agent/vector_mpc_agent.py:1254-1279` records the death-bundle `blocked_move`, `actual_damage`, and nearest hostile diagnostics. These are diagnostic evidence, not the decision source.
- Nearby dataset label generation has the same kind of label surface in `src/snks/agent/stage90r_local_policy.py:500-561`: `damage_h`, `survived_h`, `escape_delta_h`, and `nearest_hostile_h` are derived from observed horizon endpoints. If trained/advisory labels are used to steer rescue, this is a secondary surface to audit for blocked-move representation.

## Root-Cause Readout

The strongest supported root-cause hypothesis is a local counterfactual execution-fidelity gap: terminal emergency rollouts can evaluate a movement as increasing separation or avoiding damage when the real environment blocks that movement or still leaves the player adjacent to a hostile/projectile. The direct artifact signal is the repeated combination of favorable/tolerable `ranked_emergency_actions` labels, `blocked=T`, unchanged position, `actual_dmg>0`, and `actual_near=1`.

This does not prove a single-line bug in `vector_sim.py`, because the artifacts do not include a per-tile simulator passability trace. It does prove that the rescue decision consumes labels that are not reliable at the terminal action boundary for seed 7, and the local code surfaces above are the narrowest places where those labels are generated, ranked, and compared to actual execution.
