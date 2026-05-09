# Stage91 Terminal Forensics Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout: `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated instrumented checkout: `/opt/cuda/agi-stage91-terminal-forensics-20260506T171559Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_terminal_forensics_20260506T171559Z`
- Synced instrumentation evidence:
  - `env/instrumented_checkout.txt`
  - `env/base_commit.txt`
  - `env/instrumented_eval_sha256.txt`
  - `env/instrumented_eval_diff.patch`
  - `env/instrumented_checkout_status.txt`
- Commands:
  - `commands/seed_7_terminal_forensics.cmd`
  - `commands/seed_17_terminal_forensics.cmd`
- Logs:
  - `logs/seed_7_terminal_forensics.log`
  - `logs/seed_17_terminal_forensics.log`
- Raw eval JSON:
  - `raw/seed_7_terminal_forensics_eval.json`
  - `raw/seed_17_terminal_forensics_eval.json`

## Run Shape

Both runs used the isolated checkout as the import root:

`PYTHONPATH=/opt/cuda/agi-stage91-terminal-forensics-20260506T171559Z/src:/opt/cuda/agi-stage91-terminal-forensics-20260506T171559Z:/opt/cuda/agi-stage91-terminal-forensics-20260506T171559Z/experiments`

and `CUDA_VISIBLE_DEVICES=0` with:

`--mode mixed_control_rescue --n-episodes 4 --max-steps 220 --perception-mode symbolic --local-evaluator /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt --enable-planner-rescue --smoke-lite --record-death-bundle --terminal-trace-steps 32 --max-explanations-per-episode 32`

## Summary Results

- Seed 7:
  - `avg_survival=148.0`
  - deaths: `zombie=4`
  - episode lengths: `168, 55, 171, 198`
  - rescue counts: `84, 22, 46, 129`
  - `rescue_rate=0.475`
  - controller distribution: `emergency_safety=281, learner_actor=118, planner_bootstrap=193`
  - override sources: `independent_emergency_choice=179, planner_aligned_safety=77, learner_aligned_safety=20, advisory_aligned_safety=5`
- Seed 17:
  - `avg_survival=108.75`
  - deaths: `skeleton=2, zombie=1, unknown=1`
  - episode lengths: `59, 143, 194, 39`
  - rescue counts: `35, 94, 27, 0`
  - `rescue_rate=0.359`
  - controller distribution: `emergency_safety=156, learner_actor=114, planner_bootstrap=165`
  - override sources: `independent_emergency_choice=78, advisory_aligned_safety=41, planner_aligned_safety=26, learner_aligned_safety=11`

## Findings

1. Hostile weak-seed deaths are not primarily caused by rescue stopping near terminal danger.

   Seven of eight episodes died to hostile causes. All seven hostile deaths had rescue events, and their terminal local trace ended with `rescue_applied=True` on the final step. The final-step rescue triggers were `low_vitals` or `hostile_contact`, with nearest hostile distance usually `1`.

   The single no-rescue episode was seed 17 episode 3: `death_cause=unknown`, `health=0`, no hostile distances in the final local trace, and no rescue events. That is a separate non-hostile/unknown failure mode, not evidence that hostile rescue activation stopped.

2. Terminal emergency action selection often chooses actions that do not escape actual danger.

   Final-step examples:

   - seed 7 episode 0: picked `move_down`, actual `damage_step=2.0`, `nearest_hostile_after=1`, died to zombie.
   - seed 7 episode 1: picked `move_down`, actual `damage_step=2.0`, `nearest_hostile_after=1`, died to zombie.
   - seed 7 episode 3: picked `move_up`, actual `damage_step=1.0`, `nearest_hostile_after=1`, died to zombie.
   - seed 17 episode 0: picked `move_down`, actual `damage_step=4.0`, `nearest_hostile_after=1`, died to skeleton.
   - seed 17 episode 1: picked `move_down`, actual `damage_step=2.0`, `nearest_hostile_after=1`, died to skeleton.
   - seed 17 episode 2: picked `move_left`, actual `damage_step=2.0`, `nearest_hostile_after=1`, died to zombie.

   The final 8 rescue actions repeatedly oscillate among local moves and sometimes include `do`/`sleep`; they do not form a stable escape trajectory.

3. The one-step counterfactual trace is not reliably locally correct at terminal danger.

   In several final decisions, the ranked/candidate trace predicted the chosen action as one-step safe, but actual execution still took terminal damage:

   - seed 7 episode 2: candidate `move_up` predicted `damage_h=0`, `survived_h=True`, `nearest_hostile_h=2`; actual selected `move_up` took `damage_step=2.0`.
   - seed 7 episode 3: candidate `move_up` predicted `damage_h=0`, `survived_h=True`, `nearest_hostile_h=2`; actual selected `move_up` took `damage_step=1.0`.
   - seed 17 episode 0: candidate `move_down` predicted `damage_h=0`, `survived_h=True`, `nearest_hostile_h=2`; actual selected `move_down` took `damage_step=4.0`.
   - seed 17 episode 2: candidate `move_left` predicted `damage_h=0`, `survived_h=True`; actual selected `move_left` took `damage_step=2.0`.

   This is stronger than "locally right but strategically wrong" for those terminal steps: the local one-step candidate outcome itself disagrees with observed execution.

## Current Root-Cause Readout

The best-supported failure mode is **terminal emergency action-selection/counterfactual reliability**, not missing rescue activation. Rescue keeps activating in hostile terminal danger, but the selected emergency action often fails to increase separation or prevent immediate damage; in multiple cases the one-step counterfactual ranking predicts safety that the actual environment transition does not deliver.

No controller fix was attempted in this pass.
