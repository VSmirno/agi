# Seed 7 Deterministic Backend Probe Findings

## Artifact

- Probe artifact root:  
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_seed7_deterministic_backend_probe_20260509T141402Z`
- Original checkout used:  
  `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`

Comparison anchors:

1. Archived feasibility seed 7 (`avg_survival=203.5`)  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
2. Prior repeatability probe  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_seed7_repeatability_probe_20260509T140437Z`

## What Was Attempted

Wrapper launch path:

- `.../commands/deterministic_wrapper.py`

Strict target settings:

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.set_float32_matmul_precision("highest")`
- env `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- env `PYTHONHASHSEED=0`
- same original checkout, `CUDA_VISIBLE_DEVICES=0`, interpreter path, `PYTHONPATH`, and cwd

Preflight strict flags:

- `.../env/preflight_strict_runtime_flags.json`

## Strict Determinism Result

Strict mode is **not viable** for this evaluation stack on the current runtime.

Dedicated strict failure capture:

- exit code: `.../strict_failure_capture/env/exit_code.txt`
- active strict flags: `.../strict_failure_capture/env/active_runtime_flags.json`
- exact failure log: `.../strict_failure_capture/logs/strict_trial.log`

Blocking error:

```text
RuntimeError: kthvalue CUDA does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True)'.
```

The failure occurs during `VectorWorldModel` initialization:

- `src/snks/agent/vector_world_model.py`
- `CausalSDM._calibrate_radius()`
- `dists_flat.kthvalue(...)`

So strict deterministic algorithms cannot be used without changing code or changing the device/implementation path.

## Strongest Workable Deterministic-Controlled Subset

Because strict mode failed, the probe fell back to a relaxed subset:

- `cudnn.deterministic = True`
- `cudnn.benchmark = False`
- `allow_tf32 = False` for cudnn and matmul
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `PYTHONHASHSEED=0`
- `torch.use_deterministic_algorithms(False)`

Active relaxed flags for the two completed trials:

- `.../trial_1/env/active_runtime_flags.json`
- `.../trial_2/env/active_runtime_flags.json`

Those files confirm:

- `cudnn_deterministic=true`
- `cudnn_benchmark=false`
- `deterministic_algorithms_enabled=false`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `PYTHONHASHSEED=0`

## Trial Results

Comparison summary:

- `.../analysis/comparison_summary.json`

### Trial 1

- `avg_survival=204.25`
- deaths: `zombie`, `dehydration`, `zombie`, `zombie`
- ep `0`: terminal `do`, step `220`
- ep `2`: terminal `sleep`, step `168`, `nearest_hostile_after=1`
- ep `3`: terminal `do`, step `209`

### Trial 2

- `avg_survival=157.0`
- deaths: `unknown`, `zombie`, `skeleton`, `zombie`
- ep `1`: terminal `do`, step `175`
- ep `2`: terminal `do`, step `187`, `nearest_hostile_after=1`
- ep `3`: terminal `do`, step `209`

## Spread

Relaxed deterministic-controlled trials:

- values: `204.25`, `157.0`
- mean: `180.625`
- spread: `47.25`

Reference spread from the prior nondeterminism probe:

- values: `164.5`, `189.5`, `164.25`
- spread: `25.25`

So this backend-controlled subset did **not** collapse the spread. In this small sample, the spread is actually larger.

## Episode-2 Terminal Shape

Archived feasibility ep `2`:

- `zombie`
- step `186`
- terminal rescue `move_down`
- `nearest_hostile_after=2`

Relaxed deterministic-controlled ep `2`:

- trial `1`: `sleep`, step `168`, `nearest_hostile_after=1`
- trial `2`: `do`, step `187`, `nearest_hostile_after=1`

So the deterministic-controlled subset did **not** restore the archived `move_down` family. It stayed in bad terminal behavior, just with a new `sleep` variant in one trial.

## Interpretation

There are two clear outcomes from this probe:

1. **Strict deterministic algorithms are blocked by an unsupported CUDA op.**  
   This is proved by the captured `kthvalue` failure.

2. **The strongest practical relaxed subset does not stabilize the run.**  
   Trial-to-trial spread remains large, and the episode-2 terminal shape still does not return to the archived feasibility family.

One trial (`204.25`) came numerically close to the archived `203.5`, but it is not the same regime:

- archived run had ep `2` terminal `move_down`
- relaxed deterministic trial had ep `2` terminal `sleep`
- archived run did not have ep `0` terminal `zombie` at `220`

So a near-match in `avg_survival` alone is not enough to claim reproduction.

## Bottom Line

- strict deterministic torch/CUDA controls are **not possible** here without code changes, because `kthvalue` on CUDA is nondeterministic for this PyTorch build
- the strongest workable deterministic backend subset did **not** collapse the replay spread
- it also did **not** recover the archived episode-2 `move_down` family

## Smallest Next Step

The remaining work should stay outside repo-local behavior changes and focus on lower-level runtime state:

- capture library / driver / shared-object provenance
- compare allocator / cache / process state between runs
- consider a CPU-only or alternate-device control path for a small smoke probe, if practical, to separate CUDA nondeterminism from higher-level stochasticity
