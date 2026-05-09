# Runtime Provenance Closure Findings

## Scope

Compared:

1. Feasibility artifact root  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`
2. H5 replay artifact root  
   `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_vector_mpc_h5_replay_validation_20260509T071426Z`

Goal: explain why the H5 replay did not reproduce the original feasibility metrics even though the six explicitly synced runtime files matched exactly.

## Proved Facts

### 1) The broader source tree does **not** differ between the two isolated checkouts

Feasibility checkout path from provenance:

- `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`  
  from `.../hyper_stage91_feasibility_label_fix_validation_20260507T181856Z/env/isolated_checkout.txt`

Direct tree diff excluding only generated noise showed no source/config delta:

- only extra entry in feasibility tree: `_docs`
- only extra entry in H5 replay checkout: `run_stage91.sh`

There was no additional Python or config source drift under `src/`, `experiments/`, or `configs/`.

### 2) The feasibility checkout itself records only six tracked runtime edits

From:

- `.../hyper_stage91_feasibility_label_fix_validation_20260507T181856Z/env/isolated_checkout_status_after_sync.txt`

The dirty tracked files are exactly:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/stage90r_emergency_controller.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/vector_world_model.py`

That rules out a hidden seventh tracked runtime patch inside the feasibility checkout.

### 3) The imported `_build_runtime()` closure is identical between the two checkouts

Using the same interpreter and `PYTHONPATH`, I imported `stage90_quick_slice._build_runtime(seed=7, checkpoint=None, crop_world=False)` in both trees and compared every loaded repo-local module by module name and sha256.

Result:

- module set equal: `True`
- module count: `70` vs `70`
- sha mismatches: `0`

This proves the two checkouts load the same repo-local Python closure for runtime construction.

Key matching imported files/hashes:

- `experiments/stage89_eval.py` `55a1aa50e0bba794b2fa3694f881fa2816a40e3139896fe9c909bf3cdae8888d`
- `experiments/stage90_quick_slice.py` `c04244083d35a0988614df3d8fb94058a0d79201200e02907ddd27b39f1d0d2b`
- `src/snks/agent/stage90r_local_model.py` `defe0cf7beee63c09f6d9f41184ed5fa2cf44646d4423bc8ffa0520fa5800b9d`
- `src/snks/agent/vector_bootstrap.py` `5817936171cf60227a4f3112388aa8c1eb0fb3e81507d75ae5e2d8f635a13717`
- `src/snks/agent/vector_mpc_agent.py` `98c5f83babad7bb97c2cd838605ca950f90d3450b5a08110564db7d7cd3ff368`
- `src/snks/agent/vector_world_model.py` `6ccff24a5c2464972abb8c5245b38110f5e5fe7504dd125027ee44277ed9f1ef`

### 4) Current local unsynced imported runtime files are not the gap

Current local hashes match the feasibility checkout for the non-synced imported files I checked:

- `experiments/stage89_eval.py`
- `experiments/stage90_quick_slice.py`
- `src/snks/agent/stage90r_local_model.py`
- `src/snks/encoder/tile_segmenter.py`
- `src/snks/agent/perception.py`
- `src/snks/agent/post_mortem.py`

The current local branch does have a modified `src/snks/agent/stage90r_local_policy.py`, but that file is not on the mixed-control rescue path:

- mixed-control rescue imports `stage90r_local_model`, not `stage90r_local_policy`: [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:610)
- the explicit `stage90r_local_policy` import is on the local-only canary path: [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:261)
- `stage90r_local_model.py` references `stage90r_local_policy` only inside dataset split support code, not evaluator loading: [stage90r_local_model.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_local_model.py:399)

So there is no evidence that a currently unsynced-but-imported local Python module explains the replay gap.

### 5) Checkpoint choice and high-level eval config match

Both raw eval JSONs record the same:

- `perception_mode: symbolic`
- same `local_evaluator` checkpoint path  
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt`
- `smoke_lite: true`
- same episode/step counts

The only visible metadata difference in the JSON header is the `baseline_reference.comparison_path`, and `baseline_reference.available` is `false` in both runs, so that path is not behaviorally active.

### 6) Interpreter identity is the same

Feasibility command uses:

- `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`

H5 replay provenance records:

- `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python3.11`

These are the same interpreter:

- `python -> python3.11`
- both report `Python 3.11.15`

## Execution-Context Differences

### 7) Feasibility run recorded explicit per-seed GPU pinning

From:

- `.../hyper_stage91_feasibility_label_fix_validation_20260507T181856Z/commands/seed_7_feasibility_label_fix.cmd`
- `.../hyper_stage91_feasibility_label_fix_validation_20260507T181856Z/commands/seed_17_feasibility_label_fix.cmd`

Feasibility used:

- seed 7: `CUDA_VISIBLE_DEVICES=0`
- seed 17: `CUDA_VISIBLE_DEVICES=1`
- `PYTHONPATH=<checkout>/src:<checkout>:<checkout>/experiments`

### 8) The H5 replay artifact did not preserve equivalent command/env provenance

H5 replay root contains no `commands/` folder and no recorded `CUDA_VISIBLE_DEVICES` state. Its env provenance only records:

- `env/runtime_interpreter.txt`
- `env/synced_runtime_sha256.txt`
- `env/import_probe.txt`
- `env/py_compile.txt`

So the H5 replay did **not** preserve the same launch envelope audit trail.

### 9) The observed timing pattern is consistent with a changed launch envelope

Feasibility logs:

- seed 7 finished in `114s`
- seed 17 finished in `84s`

H5 replay logs:

- both seeds ran out to about `323s`

That is a large envelope change. It does not prove causality by itself, but it is consistent with a different GPU visibility/concurrency setup.

## Relevant Source Points

- `_device()` chooses generic `"cuda"` when any GPU is visible: [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:46)
- mixed-control rescue uses that device for the local evaluator and advisory path: [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:610)
- `_build_runtime()` uses `pick_device()` for the world-model/runtime stack: [stage90_quick_slice.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90_quick_slice.py:67)
- `pick_device()` also selects generic `"cuda"` when any GPU is visible: [tile_segmenter.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/encoder/tile_segmenter.py:111)

These source points mean `CUDA_VISIBLE_DEVICES` is a real behavior-shaping input, not just a performance tweak.

## Answers To The Requested Questions

### (1) What imported files differ between the two isolated checkouts or runtime imports?

**Proved:** none in the repo-local runtime closure I could reproduce.

- The full source tree diff found no source/config delta.
- The `_build_runtime()` imported module set matched exactly (`70` modules, `0` hash mismatches).

### (2) Did any unsynced but imported module change locally between the original feasibility run and now?

**Proved for the checked mixed-control rescue path:** no.

The non-synced imported files checked locally match the feasibility checkout hashes. The one notable current local extra change, `stage90r_local_policy.py`, is not on the mixed-control rescue execution path.

### (3) Does execution context differ?

**Proved:** yes.

- Feasibility recorded explicit `CUDA_VISIBLE_DEVICES` pinning per seed.
- H5 replay artifact did not preserve equivalent launch provenance.
- Interpreter binary target is effectively the same.
- High-level eval config and checkpoint path match.

Most plausible material context delta: GPU visibility / device selection / concurrency envelope.

### (4) Did the original feasibility run likely depend on a broader dirty worktree than the six explicitly synced files?

**Proved:** not in the tracked repo-local source/config tree.

The feasibility checkout’s tracked dirty set is exactly the six synced files, and its broader source tree matches the later H5 replay checkout.

## Interpretation

The earlier assumption was too narrow in one specific way: the gap is not in `vector_mpc_agent.py`, and not in another missed repo-local Python module edit either.

The remaining evidence points to **runtime envelope drift**, with the highest-value candidate being device visibility and concurrency:

1. source closure matches
2. interpreter matches
3. checkpoint path matches
4. feasibility explicitly pinned GPUs per seed
5. H5 replay did not preserve equivalent command/env provenance
6. H5 replay timing changed by roughly 3x

That is enough to stop blind source replays.

## Smallest Next Closure Experiment

Do **not** expand the Python sync surface first. That is now low value.

Instead run a single-seed envelope replay that closes only the remaining high-power context variable:

1. Start from a fresh checkout whose full source/config tree matches the feasibility checkout.
2. Keep the already matched six runtime file hashes exactly as in the feasibility artifact.
3. Re-run **seed 7 only** first.
4. Use the exact feasibility launch shape:
   - `CUDA_VISIBLE_DEVICES=0`
   - `PYTHONPATH=<checkout>/src:<checkout>:<checkout>/experiments`
   - `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`
   - no paired concurrent seed on the same envelope
5. Record `commands/*.cmd`, env snapshot, and import probe in the artifact root.

What that would prove:

- If seed 7 snaps back toward the feasibility trace shape, the gap was launch-envelope drift.
- If it still matches the bad H5 replay shape, then the next closure target is non-source external state outside the repo tree, not another Python module.
