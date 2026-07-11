# Stage9X-B3 Implementation Plan — Generalized Option-Failure Retrieval

Date: 2026-07-11  
Spec: `docs/superpowers/specs/2026-07-11-stage9x-generalized-option-failure-retrieval-design.md`

## Objective

Turn exact option-failure memory into trace-visible generalized failure retrieval, while staying inside the same `VectorWorldModel.memory` SDM and avoiding hand-written crisis policy.

## Phase 0 — Baseline freeze

- Record current branch/head.
- Keep `4436bcc` artifacts as negative baseline:
  - gen1 dies at 159;
  - gen2 dies at 103;
  - no negative scoring in gen2 before divergence.

No code changes in this phase.

## Phase 1 — Abstraction helpers

Files:

- `src/snks/agent/vector_world_model.py`
- tests in `tests/agent/test_world_model_outcome_role.py`

Work:

- add context abstraction helper;
- add option abstraction helper;
- keep deterministic ordering;
- return named abstraction levels.

Tests:

- abstraction removes only intended fields;
- equivalent contexts produce identical abstraction dicts;
- exact level remains unchanged.

## Phase 2 — Multi-level failure write/read

Files:

- `src/snks/agent/vector_world_model.py`
- `tests/agent/test_world_model_outcome_role.py`

Work:

- on negative `learn_option_outcome`, write failure to exact and abstract levels;
- on `predict_option_outcome`, read exact failure first, then abstract failures, then aggregate exact;
- include retrieval metadata.

Tests:

- exact failure still works;
- abstract failure is retrieved when exact misses;
- aggregate survived does not mask sparse failure;
- physics/primitive outcome roles remain isolated.

## Phase 3 — Stimulus trace and scaling

Files:

- `src/snks/agent/stimuli.py`
- `src/snks/agent/vector_mpc_agent.py`
- `tests/agent/test_outcome_stimulus.py`

Work:

- scale negative score by retrieval specificity;
- expose retrieval level in selected-row recall and candidate debug;
- keep positive recall at zero contribution.

Tests:

- abstract failure contributes weaker negative score than exact failure;
- positive abstract recall contributes zero;
- trace includes retrieval level and final contribution.

## Phase 4 — Precursor credit

Files:

- `src/snks/agent/vector_mpc_agent.py`
- `tests/agent/test_outcome_recorder.py`

Work:

- extend `_OptionOutcomeRecorder` to retain recent selected option snapshots;
- on real terminal failure or recorder-critical vital failure, write negative outcomes for recent precursors;
- include distance-to-failure metadata if feasible.

Tests:

- terminal failure writes multiple precursor failures;
- farthest precursor is weaker or trace-marked;
- no terminal/critical failure does not write negative precursor credit.

## Phase 5 — Local smoke

Run:

- focused unit tests;
- in-process smoke with neighboring contexts;
- short paired probe with `--smoke-lite` only for import/plumbing, not evidence.

Exit:

- no import/runtime errors;
- negative retrieval works in a neighboring context.

## Phase 6 — CUDA paired validation

Use git-flow only:

- commit branch;
- push;
- isolated CUDA checkout;
- no source scp;
- direct interpreter with `PYTHONPATH=$PWD/src`.

Run:

```bash
PYTHONPATH=$PWD/src \
CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONHASHSEED=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python \
experiments/probe_stage9x_option_arbitration.py \
  --seed 17 \
  --gen1-max-steps 350 \
  --gen2-max-steps 200 \
  --out output_to_user/stage9x_option_arbitration_probe/paired_seed17_b3_cuda_350_200.json
```

Required report:

- exact commit;
- CUDA device;
- gen1/gen2 terminal status;
- gen2 negative scoring count;
- first retrieval level used;
- first candidate-ranking divergence;
- ablation comparison if available.

## Stop conditions

Stop and reframe if:

- generalized retrieval produces negative scoring but no candidate ranking change;
- candidate ranking changes but via unrelated baseline randomness;
- broad abstract failures suppress all exploration;
- two behavioral attempts fail after unit/smoke gates pass.

Do not continue by tuning weights without a trace-level causal explanation.

