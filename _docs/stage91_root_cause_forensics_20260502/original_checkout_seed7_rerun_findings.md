# Original Checkout Seed 7 Rerun Findings

## Artifact

- Rerun artifact root:  
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_original_checkout_seed7_rerun_20260509T075059Z`
- Original feasibility checkout used directly:  
  `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`

Comparison targets:

1. Original feasibility artifact  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
2. Seed-7 launch-envelope replay  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_seed7_launch_envelope_replay_validation_20260509T074218Z`
3. H2/H3/H4 replay  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z`

## What Was Rerun

This task reran seed `7` directly from the original feasibility checkout itself, not from a reconstructed copy.

Provenance files:

- checkout path:  
  `.../env/original_checkout_path.txt`
- base commit:  
  `.../env/base_commit.txt`
- checkout `HEAD`:  
  `.../env/original_checkout_head.txt`
- checkout git status:  
  `.../env/original_checkout_git_status.txt`
- runtime file hashes from that checkout:  
  `.../env/original_checkout_runtime_sha256.txt`
- import probe:  
  `.../env/import_probe.json`
- interpreter path:  
  `.../env/runtime_interpreter.txt`
- python version:  
  `.../env/python_version.txt`
- launch envelope summary:  
  `.../env/launch_envelope.txt`
- exact rerun command:  
  `.../commands/seed_7_rerun.cmd`
- original archived command reference:  
  `.../commands/original_seed7_command_reference.cmd`

Matched envelope:

- `CUDA_VISIBLE_DEVICES=0`
- interpreter `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`
- `PYTHONPATH=/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/src:/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z:/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/experiments`
- working directory `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`
- same six runtime hashes as the archived feasibility artifact

The original checkout is still on base commit `71d1e298b9d4bd050c7aa9772d99638b59b9b347` with the same six modified runtime files:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/stage90r_emergency_controller.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/vector_world_model.py`

## Result

Log:

- `.../logs/seed_7.log`

Episodes:

- `ep 0`: `57`, `unknown`
- `ep 1`: `220`, `dehydration`
- `ep 2`: `170`, `zombie`
- `ep 3`: `188`, `zombie`

Seed-7 summary comparison:

| Run | avg_survival | hostile_deaths_with_terminal_rescue |
|---|---:|---:|
| Archived feasibility | 203.5 | 3 |
| Launch-envelope replay | 153.75 | 3 |
| Original-checkout rerun | 158.75 | 2 |
| H2/H3/H4 | 210.5 | 3 |

## Terminal Action / Death Shape

Comparison summary:

- `.../analysis/comparison_summary.json`

### Archived feasibility

- ep `1`: `skeleton`, step `220`, rescue `move_up`, `damage_step=0.0`, `displacement_step=0`, `nearest_hostile_after=3`
- ep `2`: `zombie`, step `186`, rescue `move_down`, `damage_step=2.0`, `displacement_step=1`, `nearest_hostile_after=2`
- ep `3`: `zombie`, step `188`, rescue `move_right`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`

### Original-checkout rerun

- ep `2`: `zombie`, step `170`, rescue `do`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`
- ep `3`: `zombie`, step `188`, rescue `move_right`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`

Key differences from archived feasibility:

- ep `1` changed from hostile terminal `skeleton` to non-hostile `dehydration`
- ep `2` regressed from `move_down` at `186` with `nearest_hostile_after=2` to terminal `do` at `170` with `nearest_hostile_after=1`
- ep `3` stayed on the same `move_right` terminal shape as feasibility

Relative to the launch-envelope replay:

- original-checkout rerun improved `avg_survival` slightly (`158.75` vs `153.75`)
- both still retain the bad seed-7 terminal `do` family
- original-checkout rerun is still far from the archived feasibility result

## Interpretation

This is the strongest closure result so far.

Running directly from the original feasibility checkout itself did **not** reproduce the archived feasibility seed-7 behavior.

That means:

1. the gap is not caused only by using a reconstructed checkout
2. the gap is not caused only by the six-file sync surface
3. the gap is not caused only by the previously suspected launch-envelope drift

The original checkout is therefore **not replay-stable now**, even under the archived command shape.

## What This Proves

Proved:

- the original checkout still contains the same six modified runtime files and hashes
- the rerun used the original checkout path directly
- the rerun used the archived seed-7 command shape with fresh env capture
- the output still diverged sharply from the archived feasibility seed-7 result

Inference:

- the remaining cause is outside the tracked repo-local runtime files and outside simple checkout reconstruction
- the most plausible remaining buckets are:
  - external nondeterminism in the runtime stack
  - mutable non-source state outside the six-file surface and outside the checked repo-local Python closure
  - ambient environment or driver/library state that was not preserved in the original artifact

## Concise Conclusion

The archived feasibility seed-7 result is **not reproducible now even from its own original checkout**.

That moves the root-cause search out of repo-local code replay and into external state / nondeterminism territory.

## Smallest Next Closure Step

The next clean discriminator is not another code replay.

It is an immediate second rerun from the same original checkout and same command, with the same provenance capture, to test whether the original checkout is internally run-to-run stable right now.

If that second rerun drifts again, the remaining problem is operational nondeterminism or hidden mutable external state, not a missed file.
