# Stage91 Next Findings

## Direct Remote Evidence

- Canonical artifact root inspected directly on HyperPC:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/canonical_gpu_multiseed_8ae6e57_20260502`
- Canonical import probe confirms self-contained imports from the verify checkout, not `/opt/cuda/agi/src`:
  - `snks -> /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/src/snks/__init__.py`
  - `stage90r_local_model -> /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/src/snks/agent/stage90r_local_model.py`
  - `vector_mpc_agent -> /opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/src/snks/agent/vector_mpc_agent.py`
- The verify checkout itself is still at git `HEAD=71d1e298b9d4bd050c7aa9772d99638b59b9b347`, with `experiments/stage90r_eval_local_policy.py` modified in-place.
- Remote diff shows exactly the two known eval-path edits layered on top of that checkout:
  - add deterministic `_eval_episode_rng(...)`
  - pass `local_advisory_device=device`
- Canonical multiseed summary on HyperPC confirms the regression:
  - `candidate_mean_avg_survival=172.95`
  - `baseline_avg_survival=179.25`
  - `status=PARTIAL`

## What Is Now Proven

### 1. The remaining regression is not explained by missing rescue activation.

Direct canonical weak-seed summaries contradict that theory:

- Seed 7:
  - `avg_survival=152.25`
  - `rescue_rate=0.476`
  - `learner_control_fraction=0.199`
  - `planner_dependence=0.456`
  - `controller_distribution={emergency_safety:290, learner_actor:121, planner_bootstrap:198}`
  - `death_cause_breakdown={zombie:4}`
  - `rescue_override_source_distribution={advisory_aligned_safety:8, independent_emergency_choice:181, learner_aligned_safety:21, planner_aligned_safety:80}`
- Seed 17:
  - `avg_survival=158.25`
  - `rescue_rate=0.434`
  - `learner_control_fraction=0.237`
  - `planner_dependence=0.409`
  - `controller_distribution={emergency_safety:275, learner_actor:150, planner_bootstrap:208}`
  - `death_cause_breakdown={arrow:1, dehydration:1, unknown:1, zombie:1}`
  - `rescue_override_source_distribution={advisory_aligned_safety:43, independent_emergency_choice:151, learner_aligned_safety:30, planner_aligned_safety:51}`

These are weak runs with heavy rescue activity, not weak runs where rescue rarely fires.

### 2. The weak seeds correlate with more emergency takeover and less planner dependence than the stronger seeds.

Canonical cross-seed comparison:

- Seed 7: `avg_survival=152.25`, `rescue_rate=0.476`, `planner_dependence=0.456`
- Seed 17: `avg_survival=158.25`, `rescue_rate=0.434`, `planner_dependence=0.409`
- Seed 27: `avg_survival=183.5`, `rescue_rate=0.319`, `planner_dependence=0.512`
- Seed 37: `avg_survival=194.5`, `rescue_rate=0.194`, `planner_dependence=0.537`
- Seed 47: `avg_survival=176.25`, `rescue_rate=0.397`, `planner_dependence=0.448`

The weakest seeds are not the least rescued. They are among the most rescued, and they also spend less time in planner-dependent control.

### 3. Seed-level episode data shows rescue often fires long before death and still does not prevent failure.

Direct canonical episode facts:

- Seed 7 episodes:
  - `(episode 0) steps=168 death=zombie rescues=84 first_rescue_step=33`
  - `(episode 1) steps=55 death=zombie rescues=22 first_rescue_step=33`
  - `(episode 2) steps=188 death=zombie rescues=55 first_rescue_step=132`
  - `(episode 3) steps=198 death=zombie rescues=129 first_rescue_step=9`
- Seed 17 episodes:
  - `(episode 0) steps=180 death=arrow rescues=110 first_rescue_step=5`
  - `(episode 1) steps=220 death=dehydration rescues=128 first_rescue_step=11`
  - `(episode 2) steps=194 death=zombie rescues=37 first_rescue_step=28`
  - `(episode 3) steps=39 death=unknown rescues=0`

This is strong evidence that the regression includes runs where rescue activates early and often yet still fails to preserve survival.

## What Is Still Inference, Not Proof

### 1. “Terminal rescue ineffectiveness” remains plausible but is not proven from the canonical multiseed payload itself.

Reason: the evaluated code shape did **not** save terminal tails or death bundles.

- The evaluated code path only stored:
  - `rescue_trace[: args.max_explanations_per_episode]`
  - `local_trace_excerpt[: args.max_explanations_per_episode]`
- Direct canonical raw payload confirms that episodes contain only:
  - `rescue_trace`
  - `local_trace_excerpt`
  - no `rescue_trace_tail`
  - no `local_trace_tail`
  - no `death_trace_bundle`

The first canonical excerpts are capped at 8 entries, so they show early rescue behavior only. They do not show the final rescue decisions before death.

### 2. The current local working tree has richer trace capture, but that was not part of the canonical HyperPC run.

- Current local file [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:700) now writes:
  - `rescue_trace_tail`
  - `local_trace_tail`
  - `death_trace_bundle`
- But the evaluated code shape in the canonical run did not include those fields. The remote checkout was `71d1e29` plus the 7-line deterministic-RNG / `local_advisory_device` patch only.

So the separate terminal-forensics task is the right next evidence source. The canonical multiseed artifacts alone cannot prove whether late rescue stopped firing, chose the wrong action, or optimized only one-step safety near terminal failure.

## Revised Best-Supported Hypothesis

The prior note was too loose. The stronger hypothesis is now:

The remaining Stage91 regression is associated with **overactive emergency-safety takeover dominated by independent emergency choices**, which correlates with reduced planner-dependent control and weak survival on seeds 7 and 17.

This is directly supported by:

- high `emergency_safety` counts in the weak seeds
- high `rescue_rate` in the weak seeds
- low `planner_dependence` in the weak seeds
- dominance of `independent_emergency_choice` in weak-seed override-source distributions
- repeated hostile deaths despite early rescue onset

What is **not** yet proven is whether the underlying defect is:

- rescue activation threshold too eager
- rescue action ranking choosing locally safe but strategically bad actions
- emergency takeover starving planner recovery paths
- or a separate terminal failure mode that only terminal traces will expose

## Exact Code Surfaces Now Most Implicated

### Behavior surfaces

- [stage90r_emergency_controller.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_emergency_controller.py:191)
  - weighted emergency activation score and threshold crossing
  - this is where high hostile/low-vital pressure can drive frequent takeover
- [stage90r_emergency_controller.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_emergency_controller.py:271)
  - emergency action utility ranking
  - especially important because weak seeds are dominated by `independent_emergency_choice`, not planner-aligned safety
- [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1018)
  - emergency activation is fed by one-step local counterfactual outcomes
- [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:661)
  - `local_counterfactual_horizon=1`
  - this keeps emergency evidence explicitly one-step local during eval

### Measurement / interpretation surfaces

- evaluated code shape at `8ae6e57`:
  - `experiments/stage90r_eval_local_policy.py:689-691` only stored first-8 `rescue_trace` and `local_trace_excerpt`
  - that omission is why canonical multiseed cannot directly answer the terminal-step question
- [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:742)
  - `planner_dependence` counts planner bootstrap, planner rescue, and planner-aligned emergency only
  - independent emergency choices reduce this metric by construction, so it is both a real behavioral signal and an interpretation surface
- [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:686)
  - `hostile_death_without_rescue = death_cause in {"zombie", "skeleton", "arrow"} and not rescue_trace`
  - still too weak to detect “rescued many times, still died to hostile threat”

## Sharper Next Step

Do not rerun multiseed here.

Use the already-running terminal-forensics task to answer exactly one question:

For weak seeds 7 and 17, in the final 16-32 steps before death, did emergency safety:

- stop activating,
- keep activating but choose non-escaping actions,
- or keep choosing one-step-safe actions that worsen longer-horizon survival?

That is the smallest next test that can distinguish activation failure from action-selection failure.
