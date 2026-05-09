# Seed 17 Terminal Rescue Counterfactual Mismatch

Scope: existing HyperPC artifacts only, seed `17` only. No new evals were run.

Artifact roots:

- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_terminal_forensics_20260506T171559Z`
- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_movement_fix_validation_20260506T221512Z`
- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_threat_ranking_fix_validation_20260507T104724Z`

Rows below include only hostile/arrow-labeled seed-17 episodes whose terminal tail had an active final rescue event. The prediction columns are for the chosen final rescue action from `candidate_outcome_excerpt`/`ranked_emergency_actions`; actual columns are from the same rescue event's `post_rescue_outcome`.

## Evidence Table

| Run | Ep | Len | Death | Last rescue step | Trigger | Chosen | Planner | Override | Pred survived_h | Pred damage_h | Pred escape_delta_h | Pred nearest_hostile_h | Actual damage_step | Actual nearest_hostile_after | Prev rescue |
| --- | --: | --: | --- | --: | --- | --- | --- | --- | --- | --: | ---: | --- | --: | --- | --- |
| terminal forensics | 0 | 59 | skeleton | 58 | hostile_contact | move_down | move_up | advisory_aligned_safety | True | 0.0 | 1 | 2 | 4.0 | 1 | step 57 move_down, pred survived=False damage=4.0 |
| terminal forensics | 1 | 143 | skeleton | 142 | low_vitals | move_down | move_up | advisory_aligned_safety | False | 2.0 | 0 | 1 | 2.0 | 1 | step 141 move_down, pred survived=False damage=2.0 |
| terminal forensics | 2 | 194 | zombie | 193 | low_vitals | move_left | move_left | planner_aligned_safety | True | 0.0 | 0 | 1 | 2.0 | 1 | step 192 move_left, pred survived=True damage=0.0 |
| movement fix | 0 | 198 | skeleton | 197 | low_vitals | sleep | move_left | independent_emergency_choice | False | 0.0 | 1.0 | 2 | 5.0 | 1 | step 196 move_down, pred survived=False damage=2.0 |
| movement fix | 1 | 150 | zombie | 149 | low_vitals | move_left | move_down | advisory_aligned_safety | False | 2.0 | 0 | 1 | 2.0 | 1 | step 148 move_down, pred survived=False damage=2.0 |
| movement fix | 2 | 173 | zombie | 172 | low_vitals | move_left | move_up | advisory_aligned_safety | True | 0.0 | 1 | 2 | 2.0 | 1 | step 171 move_left, pred survived=True damage=0.0 |
| threat-ranking fix | 0 | 53 | skeleton | 52 | hostile_contact | move_left | move_up | learner_aligned_safety | True | 5.0 | 0 | 4 | 6.0 | 5 | step 51 move_left, pred survived=True damage=0.0 |
| threat-ranking fix | 1 | 220 | arrow | 219 | low_vitals | move_down | move_down | planner_aligned_safety | False | 0.0 | None | None | 0.0 | None | step 218 move_left, pred survived=False damage=0.0 |
| threat-ranking fix | 2 | 167 | zombie | 166 | low_vitals | move_left | move_left | planner_aligned_safety | True | 0.0 | 1 | 2 | 2.0 | 1 | step 165 move_left, pred survived=True damage=0.0 |

Note on threat-ranking episode 1: the episode-level `death_cause` is `arrow`, but the death bundle reports `env_cause=alive`, final body `health=2.0`, and the last five captured steps have `actual_damage=0.0` with no nearest hostile distances. Treat this row as an attribution/accounting inconsistency, not as direct evidence of terminal projectile damage.

## Direct-Evidence Patterns

### 1. False-safe one-step rescue labels

Proved by rows where the chosen final rescue action predicted no damage and survival, but the same step took hostile damage:

- terminal ep0: `move_down` predicted `survived_h=True`, `damage_h=0.0`, `nearest_hostile_h=2`; actual `damage_step=4.0`, `nearest_hostile_after=1`, death `skeleton`.
- terminal ep2: `move_left` predicted `survived_h=True`, `damage_h=0.0`; actual `damage_step=2.0`, `nearest_hostile_after=1`, death `zombie`.
- movement ep2: `move_left` predicted `survived_h=True`, `damage_h=0.0`, `nearest_hostile_h=2`; actual `damage_step=2.0`, `nearest_hostile_after=1`, death `zombie`.
- threat ep2: `move_left` predicted `survived_h=True`, `damage_h=0.0`, `nearest_hostile_h=2`; actual `damage_step=2.0`, `nearest_hostile_after=1`, death `zombie`.

This is direct evidence that the local one-step counterfactual can be wrong at terminal hostile contact. It is not merely a long-horizon strategy failure.

### 2. Correctly unsafe labels still permit terminal rescue failure

Proved by rows where the final rescue label already predicted damage/death, and actual execution also took damage:

- terminal ep1: predicted `survived_h=False`, `damage_h=2.0`, nearest hostile stays `1`; actual `damage_step=2.0`, death `skeleton`.
- movement ep1: same predicted/actual `damage=2.0`, death `zombie`.

This proves another narrower point: rescue remains active but may have no good one-step option or may select the least bad option too late. It does not by itself prove the counterfactual is wrong.

### 3. Sleep/non-move can still win under terminal pressure

Proved by movement ep0: final rescue selected `sleep` via `independent_emergency_choice` while planner wanted `move_left`; the prediction said `survived_h=False` but `damage_h=0.0`, `escape_delta_h=1.0`, `nearest_hostile_h=2`; actual damage was `5.0` and nearest hostile after was `1`.

The movement fix did not eliminate the ranking path that can choose a non-movement action at terminal hostile pressure.

### 4. Hostile death accounting can overstate terminal hostile evidence

Proved for threat-ranking ep1: episode-level `death_cause=arrow`, but `death_trace_bundle.env_cause=alive` at step `220`, with final `health=2.0` and no hostile/damage signal in the final captured steps. This row should not be used as direct terminal arrow-damage evidence.

## Likely Code Surfaces

### Proved surfaces

- `src/snks/agent/vector_mpc_agent.py:1721-1816` builds the exact one-step local counterfactual labels used in the table: `survived_h`, `damage_h`, `escape_delta_h`, and `nearest_hostile_h`.
- `src/snks/agent/vector_mpc_agent.py:1028-1083` feeds those labels into rescue activation/selection and stores `ranked_emergency_actions` plus `candidate_outcome_excerpt`.
- `src/snks/agent/vector_mpc_agent.py:1448-1487` records the actual same-step rescue outcome: `damage_step` and `nearest_hostile_after`.
- `src/snks/agent/stage90r_emergency_controller.py:271-331` ranks emergency actions from those labels. This is the direct action-selection surface for false-safe or low-quality labels.
- `experiments/stage90r_eval_local_policy.py:704-727` classifies hostile terminal-rescue counters from episode-level `death_cause` alone. This is directly implicated by the threat ep1 `death_cause=arrow` / `env_cause=alive` inconsistency.

### Strongest inference

The strongest remaining root-cause surface is the local counterfactual rollout fidelity feeding emergency ranking:

- `src/snks/agent/vector_sim.py:425-486` advances dynamic entities and applies proximity/projectile damage in simulated rollouts.
- `src/snks/agent/vector_sim.py:489-513` chooses textbook movement behavior when tracked hostile velocity is absent.
- `src/snks/agent/vector_mpc_agent.py:1721-1816` converts that rollout into the rescue labels.

Reason: four final-rescue rows across all three artifact roots predicted a chosen movement action as one-step safe, while the environment immediately applied hostile damage and left the nearest hostile at `1`. That pattern can only come from a mismatch before or inside label generation, or from stale/incomplete dynamic entity state supplied to it. The movement fix improving seed 17 but leaving false-safe rows confirms missing hostile movement was one real source, but not the only remaining one.

### Secondary inference

Emergency ranking is a consumer and amplifier, not the primary proven source, for false-safe rows. `stage90r_emergency_controller.py:271-331` reasonably gives large positive weight to `survived_h=True`, low `damage_h`, and positive `escape_delta_h`; if those labels are wrong, the selected action will be wrong. The `sleep` row shows ranking also needs scrutiny under terminal pressure, but the broader repeated mismatch points first to label/rollout fidelity.

### Label/accounting inference

For hostile summary metrics and future tables, use `death_trace_bundle.env_cause`, final body, and terminal actual damage as guards around episode-level `death_cause`. The `death_cause=arrow` / `env_cause=alive` row maps to:

- `src/snks/agent/vector_mpc_agent.py:1571-1585`, where `death_cause` is derived from post-mortem attribution and bundled with `env_cause`.
- `experiments/stage90r_eval_local_policy.py:704-727`, where hostile counters are derived from `death_cause` without checking whether the terminal env cause was actually death.

## Bottom Line

Proved: seed 17 hostile terminal failures are not missing-rescue failures. Rescue is active at the final step, but the selected rescue action often fails to prevent immediate damage. In multiple rows, the one-step prediction says the chosen action is safe when actual execution immediately takes hostile damage.

Inference: the highest-value local surface is still counterfactual rollout/label fidelity in `vector_sim.py` plus `_build_local_counterfactual_outcomes` in `vector_mpc_agent.py`. Emergency ranking should be hardened after the labels are trustworthy, and terminal hostile summary metrics should guard against attribution-only `death_cause` rows whose death bundle says the environment was still alive.
