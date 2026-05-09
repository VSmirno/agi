# Stage 91 CUDA Canonical Runbook

**Date:** 2026-05-02  
**Scope:** HyperPC eval path for Stage90R/Stage91 mixed-control rescue runs

## Goal

Зафиксировать корректный self-contained способ запускать Stage90R/Stage91 eval
на CUDA на `HYPERPC`, не полагаясь на drifted import path из host env.

## What Was Wrong

Две независимые проблемы были смешаны в один симптом:

1. **Import drift**
   - interpreter env: `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`
   - editable install file:
     `/opt/cuda/miniforge3/envs/agi-stage90r-py311/lib/python3.11/site-packages/__editable__.snks-0.1.0.pth`
   - этот `.pth` указывал на `/opt/cuda/agi/src`
   - значит bare command импортировал `snks` из host repo, а не из verify checkout

2. **GPU device mismatch**
   - evaluator загружался на CUDA
   - mixed-control advisory tensors строились на CPU
   - падение:
     `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

## Code Fix

В eval path `mixed_control_rescue` теперь явно прокидывается:

- `local_advisory_device=device`
- deterministic eval RNG через `_eval_episode_rng(...)`

Код:
- [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:54)
- [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:647)
- [test_stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/tests/test_stage90r_eval_local_policy.py:1)

## Canonical CUDA Invocation

Use the verify checkout as the import root. Do not rely on the env's editable
`snks` install to pick the right repo.

Example:

```bash
VERIFY=/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z
PY=/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python

cd "$VERIFY"
PYTHONPATH="$VERIFY/src:$VERIFY:$VERIFY/experiments" \
CUDA_VISIBLE_DEVICES=0 \
"$PY" experiments/stage90r_eval_local_policy.py \
  --mode mixed_control_rescue \
  --n-episodes 4 \
  --max-steps 220 \
  --seed 7 \
  --perception-mode symbolic \
  --local-evaluator "$VERIFY/_docs/stage90r_seed7_actor_selection_probe3.pt" \
  --enable-planner-rescue \
  --smoke-lite \
  --terminal-trace-steps 32 \
  --record-death-bundle \
  --out "$VERIFY/_docs/stage91_gpu_eval_seed7.json"
```

For root-cause reruns, keep `--terminal-trace-steps` and
`--record-death-bundle` enabled. The default head excerpts are useful for
early-pattern inspection, but the regression now needs terminal-step evidence
from weak episodes.

## Verification Evidence

Remote artifact root:

- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/cuda_fix_venv_20260502`

Key files:

- minimal repro command:
  `commands/gpu_min_repro.cmd`
- fixed minimal repro:
  `commands/gpu_min_repro_fixed.cmd`
- fixed minimal repro log:
  `logs/gpu_min_repro_fixed.log`
- fixed minimal repro output:
  `diff/gpu_min_repro_fixed.json`
- full seed-7 GPU compare command:
  `commands/gpu_seed7_full_fixed.cmd`
- full seed-7 GPU compare log:
  `logs/gpu_seed7_full_fixed.log`
- full seed-7 GPU compare output:
  `diff/gpu_seed7_full_fixed.json`

Observed fixed GPU result:

- `avg_survival = 157.75`
- `rescue_rate = 0.437`
- `planner_dependence = 0.452`
- `learner_control_fraction = 0.222`

## Remaining Caveat

Это runbook для **correct CUDA invocation**, а не доказательство Stage 91
robustness. Multi-seed regression и residual nondeterminism остаются отдельной
проблемой и должны разбираться уже поверх этого canonical GPU path.
