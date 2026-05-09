# Seed 7 Launch-Envelope Replay Findings

## Artifact

- Replay artifact root:  
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_seed7_launch_envelope_replay_validation_20260509T074218Z`
- Isolated checkout used for replay:  
  `/opt/cuda/agi-stage91-seed7-launch-envelope-replay-20260509T074218Z`

Comparison targets:

1. Feasibility baseline  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
2. H5 replay  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z`
3. H2/H3/H4 replay  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h234_replay_validation_20260509T070248Z`

## Reconstructed Launch Envelope

Recorded provenance:

- command:  
  `.../commands/seed_7_launch_envelope_replay.cmd`
- envelope summary:  
  `.../env/launch_envelope.txt`
- interpreter:  
  `.../env/runtime_interpreter.txt`
- python version:  
  `.../env/python_version.txt`
- checkout path:  
  `.../env/isolated_checkout.txt`
- synced runtime hashes:  
  `.../env/synced_runtime_sha256.txt`
- runtime diff vs base commit:  
  `.../env/runtime_status_vs_base.txt`
- import probe:  
  `.../env/import_probe.json`
- reconstruction notes / residual uncertainty:  
  `.../env/reconstruction_notes.txt`

Matched exactly or near-exactly:

- `CUDA_VISIBLE_DEVICES=0`
- interpreter path `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`
- `PYTHONPATH=<checkout>/src:<checkout>:<checkout>/experiments`
- standalone `/opt/cuda/...` checkout working directory shape
- seed `7`, symbolic perception, smoke-lite, planner rescue, local-evaluator path, episode/step limits
- six synced runtime hashes exactly match the feasibility artifact, including  
  `vector_mpc_agent.py = 98c5f83babad7bb97c2cd838605ca950f90d3450b5a08110564db7d7cd3ff368`

Remaining uncertainty:

- this is a new checkout path, not the original feasibility checkout path
- the original feasibility artifact did not preserve a full shell/env dump
- the original feasibility validation ran seed `7` and seed `17` concurrently on separate pinned GPUs; this replay ran seed `7` only, as requested
- the checkout was reconstructed from the verify tree plus six-file sync, not copied byte-for-byte from the original feasibility checkout directory

## Seed 7 Result

Replay log:

- `.../logs/seed_7.log`

Episodes:

- `ep 0`: `57`, `unknown`
- `ep 1`: `220`, `skeleton`
- `ep 2`: `150`, `zombie`
- `ep 3`: `188`, `zombie`

Key metric comparison:

| Run | avg_survival | hostile_deaths_with_terminal_rescue |
|---|---:|---:|
| Feasibility | 203.5 | 3 |
| H5 | 159.0 | 3 |
| H2/H3/H4 | 210.5 | 3 |
| Launch-envelope replay | 153.75 | 3 |

So the exact-envelope replay did **not** recover the feasibility seed-7 behavior. It also did not close the gap versus H5; it regressed further from `159.0` to `153.75`.

## Terminal Rescue Shape

Comparison summary artifact:

- `.../analysis/comparison_summary.json`

### Feasibility

- ep `1`: `move_up`, `skeleton`, step `220`, `damage_step=0.0`, `displacement_step=0`, `nearest_hostile_after=3`
- ep `2`: `move_down`, `zombie`, step `186`, `damage_step=2.0`, `displacement_step=1`, `nearest_hostile_after=2`
- ep `3`: `move_right`, `zombie`, step `188`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`

### H5

- ep `1`: `do`, `zombie`, step `171`, `damage_step=1.0`, `displacement_step=0`, `nearest_hostile_after=1`
- ep `2`: `move_left`, `zombie`, step `199`, `damage_step=2.0`, `displacement_step=1`, `nearest_hostile_after=2`
- ep `3`: `do`, `zombie`, step `209`, `damage_step=1.0`, `displacement_step=0`, `nearest_hostile_after=1`

### H2/H3/H4

- ep `1`: `move_up`, `skeleton`, step `220`, `damage_step=0.0`, `displacement_step=0`, `nearest_hostile_after=3`
- ep `2`: `move_right`, `zombie`, step `214`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`
- ep `3`: `move_right`, `zombie`, step `188`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`

### Launch-envelope replay

- ep `1`: `move_up`, `skeleton`, step `220`, `damage_step=0.0`, `displacement_step=1`, `nearest_hostile_after=3`
- ep `2`: `do`, `zombie`, step `150`, `damage_step=1.0`, `displacement_step=0`, `nearest_hostile_after=1`
- ep `3`: `move_right`, `zombie`, step `188`, `damage_step=2.0`, `displacement_step=0`, `nearest_hostile_after=1`

## False-Safe Terminal Rows

The current death bundles in these artifacts do not serialize `terminal_rows`, so the old row-wise false-safe table cannot be reproduced directly from `death_trace_bundle`.

What is visible from terminal rescue traces:

- the replay removed one of the H5 terminal `do` failures, but not the class:
  - H5 had terminal `do` on ep `1` and ep `3`
  - launch-envelope replay still has terminal `do` on ep `2`
- feasibility and H2/H3/H4 do not show a terminal `do` row in seed `7`

So the bad seed-7 terminal `do` family shrank relative to H5, but it was not eliminated.

## Interpretation

This is a useful negative result.

What the replay **did** show:

- changing the launch envelope changes behavior materially
- ep `1` snapped back toward the feasibility family:
  - same `skeleton` terminal episode
  - same `move_up`
  - same final `nearest_hostile_after=3`
- ep `3` also snapped back to the feasibility/H2/H3/H4 family:
  - `move_right`
  - step `188`
  - `nearest_hostile_after=1`

What it **did not** show:

- launch-envelope matching alone did not recover feasibility seed `7`
- overall survival got worse than H5 (`153.75` vs `159.0`)
- ep `2` became a much earlier terminal `do` failure (`150`) instead of the feasibility `move_down` at `186`

So launch-envelope drift was **not sufficient** to explain the original non-reproduction gap.

## Best-Supported Next Step

The remaining uncertainty is now narrower:

1. the exact original feasibility checkout directory may still matter in ways not captured by the six-file sync plus source-tree closure check
2. ambient process / ROCm / shell state not preserved in the feasibility artifact may still matter

The smallest next closure experiment is:

- rerun **seed 7 only** directly from the original feasibility checkout  
  `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`
- use the original recorded command shape with fresh env capture added
- compare that direct rerun against this replay

That would separate:

- reconstructed-envelope drift from
- true external nondeterminism or hidden non-source state tied to the original checkout/process environment
