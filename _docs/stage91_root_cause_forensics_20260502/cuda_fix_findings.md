# HyperPC CUDA Path Fix Findings

## Artifact Paths

- Remote CUDA-fix artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/cuda_fix_venv_20260502`
- Import probes:
  - `env/bare_python_import_probe.json`
  - `env/repo_pythonpath_python_import_probe.json`
- Repro commands:
  - `commands/gpu_min_repro.cmd`
  - `commands/gpu_min_repro_fixed.cmd`
  - `commands/gpu_seed7_full_fixed.cmd`
- Logs:
  - `logs/gpu_min_repro.log`
  - `logs/gpu_min_repro_fixed.log`
  - `logs/gpu_seed7_full_fixed.log`
- Outputs:
  - `diff/gpu_min_repro_fixed.json`
  - `diff/gpu_seed7_full_fixed.json`

## Environment / Import Provenance

- SSH environment had no active `CONDA_PREFIX`, `VIRTUAL_ENV`, or `PYTHONPATH`.
- The interpreter in use was still the conda env executable:
  `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`
- Bare import provenance resolved `snks` from host source:
  - `snks -> /opt/cuda/agi/src/snks/__init__.py`
  - `snks.agent.stage90r_local_model -> /opt/cuda/agi/src/snks/agent/stage90r_local_model.py`
  - `snks.agent.vector_mpc_agent -> /opt/cuda/agi/src/snks/agent/vector_mpc_agent.py`
- Self-contained provenance with `PYTHONPATH=/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/src`
  resolved those same modules from the verify checkout instead.
- Root drift source:
  `/opt/cuda/miniforge3/envs/agi-stage90r-py311/lib/python3.11/site-packages/__editable__.snks-0.1.0.pth`
  contains:
  `/opt/cuda/agi/src`

Conclusion: the main drift source is the editable-install `.pth` in the conda env, not shell activation leakage.

## Exact GPU Failure

Canonical repro command before the fix:

```bash
PYTHONPATH=/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/src \
/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python \
experiments/stage90r_eval_local_policy.py \
  --mode mixed_control_rescue \
  --n-episodes 1 \
  --max-steps 20 \
  --seed 7 \
  --perception-mode symbolic \
  --local-evaluator /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt \
  --enable-planner-rescue \
  --smoke-lite \
  --out .../diff/gpu_min_repro.json
```

Stack trace:

- `experiments/stage90r_eval_local_policy.py:628`
- `src/snks/agent/vector_mpc_agent.py:996`
- `src/snks/agent/stage90r_local_model.py:1311`
- `src/snks/agent/stage90r_local_model.py:971`
- `src/snks/agent/stage90r_local_model.py:948`
- `torch.nn.modules.sparse.py:190`
- `torch.nn.functional.py:2551`

Error:

`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

## Root Cause

- `run_vector_mpc_episode()` defaults `local_advisory_device='cpu'`
- `rank_local_actor_candidates()` creates `class_ids`, `confidences`, `body_vec`,
  `inv_vec`, and action indices on that `device`
- the evaluator artifact was loaded onto CUDA
- therefore CPU tensors were fed into a CUDA-resident embedding layer at
  `LocalActionEvaluator.encode_state_context()` / `tile_embedding(class_ids)`

The first concrete failing boundary is:

- tensors created on CPU in
  `src/snks/agent/stage90r_local_model.py:1245-1253, 1273, 1309`
- consumed by CUDA weights in
  `src/snks/agent/stage90r_local_model.py:948`

## Scoped Fix

Only the Stage90R mixed-control eval call site was changed:

- `experiments/stage90r_eval_local_policy.py`
  - pass `local_advisory_device=device` into `run_vector_mpc_episode()`

This keeps runtime behavior scoped to the eval path and aligns the advisory tensor
device with the evaluator device already selected by the eval script.

## Validation

Focused local tests:

- `./.venv/bin/python -m pytest tests/test_stage90r_eval_local_policy.py tests/test_vector_mpc_agent.py`
- result: `6 passed`

GPU repro after the fix:

- `logs/gpu_min_repro_fixed.log`
- result:
  - `ep  0: len=  20 death=alive rescues=0`
  - exit code `0`

Full seed-7 GPU compare after the fix:

- `logs/gpu_seed7_full_fixed.log`
- `diff/gpu_seed7_full_fixed.json`
- summary:
  - `avg_survival=157.75`
  - `rescue_rate=0.437`
  - `learner_control_fraction=0.222`
  - `planner_dependence=0.452`
  - `controller_distribution={emergency_safety: 276, learner_actor: 140, planner_bootstrap: 215}`
  - `episode_lengths=[168, 55, 210, 198]`

## Remaining Caveat

The bare recorded command still imports drifted host code unless the verify
checkout `src` is explicitly prepended. The code fix resolves the CUDA/CPU device
bug, but a truly self-contained canonical invocation still needs one of:

- `PYTHONPATH=/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/src ...`
- or reinstalling the conda env’s editable `snks` package against the verify checkout.
