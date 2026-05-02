# Stage 90R Report — Emergency Safety Controller

**Spec:** `docs/superpowers/specs/2026-05-02-stage90r-emergency-safety-controller-design.md`  
**Plan:** `docs/superpowers/plans/2026-05-02-stage90r-emergency-safety-controller-plan.md`  
**Implementation commit:** `71d1e29` (`feat(stage90r): add emergency safety controller`)  
**Date:** 2026-05-02

## Goal

Recover online `mixed_control_rescue` robustness by making emergency behavior a
first-class control layer instead of a patch on top of actor/planner
disagreement, while migrating only the emergency-relevant Crafter facts needed
by that path into textbook/config.

## What Changed

Implementation changes:

- added `src/snks/agent/stage90r_emergency_controller.py`
- wired first-class emergency arbitration into
  `src/snks/agent/vector_mpc_agent.py`
- added narrow `emergency_control` facts to `configs/crafter_textbook.yaml`
- exposed textbook-backed emergency config access in
  `src/snks/agent/crafter_textbook.py`
- extended rescue telemetry in `src/snks/agent/stage90r_local_policy.py`
- extended rescue eval summary in `experiments/stage90r_eval_local_policy.py`
- added focused tests in `tests/test_stage90r_emergency_controller.py`

Behavioral contract after this stage:

- emergency activation no longer depends mainly on `actor != planner`
- emergency selection runs under a safety-first local ranking
- normal path remains isolated when emergency mode is inactive

## Validation

Focused local validation:

- `uv run --extra dev pytest -q tests/test_stage90r_emergency_controller.py tests/test_vector_mpc_agent.py tests/test_stage90r_local_policy.py`
- result: `23 passed`

Bounded online compare:

- host: `cuda@192.168.98.56` (`HYPERPC`)
- clean checkout: `/opt/cuda/agi-stage90r-verify-71d1e29`
- compare artifact:
  `/opt/cuda/agi-stage90r-verify-71d1e29/_docs/hyper_stage90r_emergency_controller_compare/71d1e29_seed7_smokelite_symbolic_cpu4_eval.json`
- checkpoint reused:
  `_docs/local_stage90r_actor_selection_probe3/seed7_epochs3.pt`
- run shape:
  - `mode=mixed_control_rescue`
  - `perception_mode=symbolic`
  - `smoke_lite=true`
  - `seed=7`
  - `n_episodes=4`
  - `max_steps=220`
  - `enable_planner_rescue=true`
  - CPU-only via `CUDA_VISIBLE_DEVICES=""`

Frozen rescue baseline:

- source:
  `_docs/hyper_stage90r_mixed_control_rescue_compare/comparison_summary.json`
- `9083357 avg_survival = 179.25`

Candidate result (`71d1e29`):

- `avg_survival = 190.0`
- `death_cause_breakdown = {"zombie": 2, "skeleton": 1, "arrow": 1}`
- `controller_distribution = {"emergency_safety": 335, "learner_actor": 145, "planner_bootstrap": 280}`
- `planner_dependence = 0.453`
- `learner_control_fraction = 0.191`
- `rescue_rate = 0.441`
- `early_hostile_deaths_without_rescue = 0`
- `hostile_deaths_without_rescue = 0`
- `rescue_override_source_distribution = {"advisory_aligned_safety": 7, "independent_emergency_choice": 251, "learner_aligned_safety": 13, "planner_aligned_safety": 64}`

## Interpretation

The primary acceptance gate is satisfied:

- aggregate bounded `mixed_control_rescue` improved above the frozen `9083357`
  baseline (`190.0 > 179.25`)
- the early hostile deaths that previously bypassed rescue dropped to `0`

The JSON's embedded `fallback_criterion.status` remained `UNKNOWN`, but that is
an artifact-format issue rather than a behavioral failure:

- the clean hyper checkout did not carry the local baseline summary into the
  payload
- the frozen baseline value remains available from the existing repository
  compare artifact
- manual comparison against that frozen value is sufficient to judge the stage

## Stage Review

**Ideological debt addressed:** rescue no longer depends primarily on incidental
actor/planner disagreement and no longer lives only as a late patch on the
normal control path

**Layer changed:** `mechanisms` with a bounded `facts` migration for
emergency-relevant Crafter knowledge

**What changed:** introduced a first-class emergency safety controller, moved
its required world facts into textbook/config, and added rescue-side telemetry
rich enough to explain activation and override behavior

**Evidence of improvement:** focused tests passed; bounded HyperPC compare
improved aggregate survival from `179.25` to `190.0`; hostile deaths without
any rescue intervention fell to `0`

**Why this is architectural, not tactical:** the gain comes from a new control
contract and explicit fact/mechanism separation for the emergency path, not from
adding another disagreement-trigger tweak or a new Crafter-specific branch in
generic planning code

**Knowledge flow outcome:** emergency-relevant hostile/resource facts now have a
textbook/config home for this path; rescue telemetry now records activation and
selection explanations for future analysis

**Remaining assumptions / walls:** compare proof is CPU-only; the separate
online GPU-path hang remains unresolved; textbook migration is intentionally
bounded and not repo-wide; the broader stimuli-layer question remains open

**Decision:** `PASS`
