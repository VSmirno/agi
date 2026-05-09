# Stage91 Local-Affordance Diagnostics-Only Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- New remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z`
- Raw eval JSON:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/raw/seed_7_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/raw/seed_17_eval.json`
- Comparison artifact:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/analysis/comparison_summary.json`
- Provenance:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/env/import_probe.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/env/py_compile.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/env/runtime_interpreter.txt`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_diagnostics_only_validation_20260508T122803Z/env/synced_runtime_sha256.txt`

## Runtime Provenance

The eval ran from the isolated checkout under:

`/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python3.11`

Import probe confirms the active imports came from the isolated checkout for:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/stage90r_local_affordances.py`

Synced runtime hashes match the prior feasibility baseline for:

- `experiments/stage90r_eval_local_policy.py`
- `src/snks/agent/vector_world_model.py`
- `src/snks/agent/vector_bootstrap.py`
- `src/snks/agent/vector_sim.py`
- `src/snks/agent/stage90r_emergency_controller.py`

Direct hash comparison shows the only active runtime delta versus the prior
feasibility baseline was `src/snks/agent/vector_mpc_agent.py`, plus the added
diagnostics module `src/snks/agent/stage90r_local_affordances.py`.

## Source Mapping

Current local source matches the intended diagnostics-only rollback:

- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:944)
  builds `local_affordance_snapshot`.
- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1080)
  keeps `pre_rescue_state.local_affordance_scene`.
- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1730)
  no longer accepts or uses `affordance_snapshot` in
  `_build_local_counterfactual_outcomes()`.
- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1811)
  candidate labels contain only the feasibility-label fields
  (`effective_displacement_h`, `blocked_h`, `adjacent_hostile_after_h`) and no
  `*_local` additions.
- [stage90r_local_affordances.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_local_affordances.py:11)
  remains a read-only scene extractor.
- [stage90r_emergency_controller.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_emergency_controller.py:254)
  still consumes only the feasibility-label fields, not any `*_local` fields.

## Before / After Metrics

Primary comparison root:

- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`

Secondary comparison root:

- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_local_affordance_extractor_validation_20260508T082732Z`

| Run | Seed | Avg survival | Hostile deaths with terminal rescue | Final rescue nearest-hostile-after | False-safe terminal rows |
| --- | ---: | ---: | ---: | --- | --- |
| feasibility baseline | 7 | `203.5` | `3` | `[3, 2, 1]` | `1 (move_right:1)` |
| extractor wiring | 7 | `196.5` | `3` | `[1, 1, 1]` | `3 (do:3)` |
| diagnostics-only | 7 | `168.0` | `3` | `[4, 2, 1]` | `1 (move_right:1)` |
| feasibility baseline | 17 | `142.25` | `1` | `[1]` | `1 (do:1)` |
| extractor wiring | 17 | `128.5` | `3` | `[1, 1, 2]` | `1 (do:1)` |
| diagnostics-only | 17 | `134.75` | `2` | `[1, 1]` | `0` |

Weak-seed mean `avg_survival`:

- feasibility baseline: `172.875`
- extractor wiring: `162.5`
- diagnostics-only: `151.375`

## Diagnostics Presence

Artifact-proved:

- `pre_rescue_state.local_affordance_scene` is still present on every recorded
  rescue event kept in `rescue_trace` plus `rescue_trace_tail`:
  - seed `7`: `161/161`
  - seed `17`: `160/160`
- Generic `local_trace_excerpt` and `local_trace_tail` rows still do not carry
  `local_affordance_scene`:
  - seed `7`: `0/256`
  - seed `17`: `0/256`

So the diagnostics contract was preserved where this validation actually
depends on it: in `pre_rescue_state`.

## Behavioral Readout

### Seed 7

- The diagnostics-only rollback removed the extractor-run false-safe terminal
  `do` pattern.
- Terminal hostile separation improved from extractor `[1, 1, 1]` to
  diagnostics-only `[4, 2, 1]`.
- The visible false-safe pattern returned to the prior feasibility-baseline
  shape: one terminal `move_right` row at episode `3`, step `187`.
- But overall survival did **not** recover. `avg_survival` fell to `168.0`,
  which is below both feasibility `203.5` and extractor `196.5`.

### Seed 17

- Diagnostics-only improved on the regressed extractor run:
  - `avg_survival`: `128.5 -> 134.75`
  - `hostile_deaths_with_terminal_rescue`: `3 -> 2`
  - false-safe terminal rows: `1 -> 0`
- But it still did not recover the prior feasibility baseline:
  - `avg_survival`: `134.75` versus `142.25`
  - `hostile_deaths_with_terminal_rescue`: `2` versus `1`
- One baseline-aligned row did come back exactly: episode `0` ends with the
  same terminal rescue action as feasibility baseline (`move_left` at step
  `146`, `nearest_hostile_after=6`, `damage_step=0.0`).

## Interpretation

Making the shared local-affordance extractor diagnostics-only did preserve the
intended diagnostics surface, and it did remove the worst extractor-specific
terminal `do` regression pattern.

But it did **not** restore the prior best bounded weak-seed baseline.

The evidence is narrower than “the extractor still affects behavior”:

1. The active runtime surface versus the feasibility baseline differs only in
   `vector_mpc_agent.py` plus the added diagnostics module.
2. The bad extractor-run terminal `do` pattern on seed `7` disappeared after
   the diagnostics-only rollback, which is consistent with removing the
   extractor from active counterfactual labels.
3. Even after that rollback, weak-seed mean survival regressed further to
   `151.375`, so the current `vector_mpc_agent.py` behavior still does not
   match the prior feasibility baseline.

Best-supported conclusion: the diagnostics-only rollback fixed the explicit
extractor-label contamination, but the current `vector_mpc_agent.py` state still
contains some additional behavioral delta versus the exact feasibility-baseline
version.

## Smallest Next Testable Hypothesis

Diff the current `src/snks/agent/vector_mpc_agent.py` against the exact
feasibility-baseline version and replay only that file change set.

Reason:

- Provenance rules out the other synced runtime files.
- The terminal pattern improvement says the diagnostics-only rollback mattered.
- The remaining survival gap says the current `vector_mpc_agent.py` still is
  not behavior-equivalent to the feasibility-baseline `vector_mpc_agent.py`.

That is the smallest credible next step before changing controller or simulator
heuristics again.
