# Seed 7 Repeatability Probe Findings

## Artifact

- Probe artifact root:  
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_seed7_repeatability_probe_20260509T140437Z`
- Original checkout used for every trial:  
  `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`

Reference comparisons:

1. Archived feasibility seed 7  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
2. Prior original-checkout rerun  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_original_checkout_seed7_rerun_20260509T075059Z`

## Preflight Runtime State

Preflight file:

- `.../env/preflight_runtime_flags.json`

Visible runtime/backend state:

- `python`: `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`
- `python_version`: `3.11.15`
- `torch_version`: `2.5.1+cu121`
- `torch_version_cuda`: `12.1`
- visible GPU under `CUDA_VISIBLE_DEVICES=0`: `NVIDIA GeForce RTX 3090`
- `cuda_device_count`: `1`
- `cudnn_enabled`: `true`
- `cudnn_benchmark`: `false`
- `cudnn_deterministic`: `false`
- `deterministic_algorithms_enabled`: `false`
- `deterministic_algorithms_warn_only`: `false`
- `CUBLAS_WORKSPACE_CONFIG`: unset
- `PYTHONHASHSEED`: unset

Interpretation:

- there is no evidence of explicit determinism enforcement
- the runtime is allowed to use nondeterministic algorithms

## Launch Envelope

Envelope file:

- `.../env/launch_envelope.txt`

Shared launch shape for all trials:

- `CUDA_VISIBLE_DEVICES=0`
- `PYTHONPATH=/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/src:/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z:/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/experiments`
- `PWD=/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`

Each trial recorded:

- command under `trial_N/commands/seed_7_trial_N.cmd`
- shell env under `trial_N/env/remote_shell_env.txt`
- full raw eval under `trial_N/raw/seed_7_eval.json`
- log under `trial_N/logs/seed_7.log`

## Trial-by-Trial Results

Comparison summary:

- `.../analysis/comparison_summary.json`

### Archived feasibility baseline

- `avg_survival=203.5`
- ep `2`: `zombie`, step `186`, terminal rescue `move_down`, `nearest_hostile_after=2`

### Prior original-checkout rerun

- `avg_survival=158.75`
- ep `2`: `zombie`, step `170`, terminal rescue `do`, `nearest_hostile_after=1`

### Trial 1

- `avg_survival=164.5`
- deaths: `unknown`, `dehydration`, `zombie`, `zombie`
- ep `2`: `zombie`, step `193`, terminal rescue `do`, `damage_step=1.0`, `nearest_hostile_after=2`
- ep `3`: `zombie`, step `188`, terminal rescue `move_right`

### Trial 2

- `avg_survival=189.5`
- deaths: `dehydration`, `skeleton`, `zombie`, `zombie`
- ep `1`: early `skeleton`, step `137`, terminal rescue `do`
- ep `2`: `zombie`, step `192`, terminal rescue `do`, `damage_step=1.0`, `nearest_hostile_after=1`
- ep `3`: `zombie`, step `209`, terminal rescue `do`

### Trial 3

- `avg_survival=164.25`
- deaths: `unknown`, `skeleton`, `zombie`, `zombie`
- ep `1`: `skeleton`, step `220`, terminal rescue `move_up`, `nearest_hostile_after=3`
- ep `2`: `zombie`, step `171`, terminal rescue `do`, `damage_step=2.0`, `nearest_hostile_after=1`
- ep `3`: `zombie`, step `209`, terminal rescue `do`

## Spread

Across the 3 serial trials:

- `avg_survival` values: `164.5`, `189.5`, `164.25`
- mean: `172.75`
- min: `164.25`
- max: `189.5`
- spread: `25.25`

This is material run-to-run variation from the same checkout and same launch envelope.

## Most Important Pattern

The reruns are **not** stable around one exact non-203.5 result.

They vary materially, but they also show one consistent failure mode:

- episode `2` stayed in the bad terminal `do` family in **all 3 trials**
- it never returned to the archived feasibility `move_down` terminal shape

So:

1. there is real nondeterministic spread in overall seed-7 survival
2. the archived feasibility ep-2 `move_down` ending still looks like a different regime, not just a common sample from the current rerun distribution

## Concise Interpretation

Two conclusions are now supported at the same time:

1. **The original checkout is nondeterministic under replay.**  
   Same checkout + same command + same device pinning produced a `25.25`-point spread in `avg_survival`.

2. **The archived `203.5` seed-7 result still looks special.**  
   None of the 3 trials reproduced it, and ep `2` never returned to the archived `move_down` terminal behavior.

The archived result therefore does not look like a simple stable center of the current replay distribution.

## Bottom Line

- repeated runs are **not stable**
- they vary materially
- but they vary around a degraded regime, not around the archived `203.5`
- the ep-2 terminal action is especially telling: current reruns consistently end in `do`-family failure, not the archived `move_down` family

## Smallest Next Step

The next useful closure step is operational, not another code replay:

- capture and compare lower-level runtime state that can drift between runs, such as:
  - driver / CUDA runtime details
  - library load order / linked shared objects
  - process-level RNG state sources if exposed
  - any mutable cache or model-loading side effects outside the repo tree

If a single next probe is needed, the highest-value one is a paired rerun with extra runtime-state capture before and after each trial, while keeping the code and command fixed.
