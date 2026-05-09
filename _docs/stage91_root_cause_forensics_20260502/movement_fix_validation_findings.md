# Stage91 Movement-Fix Validation Findings

## Artifact Paths

- HyperPC host: `cuda@192.168.98.56`
- Base verify checkout: `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z`
- Isolated movement-fix checkout:
  `/opt/cuda/agi-stage91-movement-fix-20260506T221512Z`
- Remote artifact root:
  `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_movement_fix_validation_20260506T221512Z`
- Commands:
  - `commands/seed_7_movement_fix.cmd`
  - `commands/seed_17_movement_fix.cmd`
- Raw eval JSON:
  - `raw/seed_7_movement_fix_eval.json`
  - `raw/seed_17_movement_fix_eval.json`
- Environment / provenance:
  - `env/base_commit.txt`
  - `env/instrumented_checkout.txt`
  - `env/instrumented_checkout_status.txt`
  - `env/instrumented_checkout_diff.patch`
  - `env/instrumented_sha256.txt`

## Change Under Test

Validated a narrow local-sim fix only:

- `vector_bootstrap.py`: load textbook passive `movement` rules into the model
- `vector_world_model.py`: persist `movement_behaviors`
- `vector_sim.py`: when a dynamic hostile has no inferred `velocity`, advance it
  by textbook `movement_behaviors` such as `chase_player`

This was motivated by terminal forensics showing false-safe one-step rescue
labels. Before the fix, textbook-seeded `zombie`/`skeleton` behavior could be
missing from the local counterfactual rollout whenever the tracker did not
provide a velocity.

## Result Summary

Compared against the prior terminal-forensics run under:

- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_terminal_forensics_20260506T171559Z`

### Seed 7

- Before:
  - `avg_survival=148.0`
  - `rescue_rate=0.475`
  - `planner_dependence` not the main signal; override distribution was heavily
    dominated by `independent_emergency_choice=179`
- After movement fix:
  - `avg_survival=144.5`
  - `rescue_rate=0.424`
  - `planner_dependence=0.465`
  - `controller_distribution={emergency_safety:245, learner_actor:121, planner_bootstrap:212}`
  - `rescue_override_source_distribution={advisory_aligned_safety:5, independent_emergency_choice:168, learner_aligned_safety:15, planner_aligned_safety:57}`
  - `death_cause_breakdown={zombie:4}`
  - `hostile_deaths_after_prior_rescue=4`
  - `hostile_deaths_with_terminal_rescue=4`
  - `hostile_deaths_without_terminal_rescue=0`

Readout: the fix slightly reduced rescue rate and independent emergency choices,
but did **not** improve survival on this seed.

### Seed 17

- Before:
  - `avg_survival=108.75`
  - `rescue_rate=0.359`
  - `controller_distribution={emergency_safety:156, learner_actor:114, planner_bootstrap:165}`
  - `rescue_override_source_distribution={independent_emergency_choice:78, advisory_aligned_safety:41, planner_aligned_safety:26, learner_aligned_safety:11}`
- After movement fix:
  - `avg_survival=140.0`
  - `rescue_rate=0.457`
  - `planner_dependence=0.391`
  - `controller_distribution={emergency_safety:256, learner_actor:127, planner_bootstrap:177}`
  - `rescue_override_source_distribution={advisory_aligned_safety:62, independent_emergency_choice:146, learner_aligned_safety:6, planner_aligned_safety:42}`
  - `death_cause_breakdown={skeleton:1, unknown:1, zombie:2}`
  - `hostile_deaths_after_prior_rescue=3`
  - `hostile_deaths_with_terminal_rescue=3`
  - `hostile_deaths_without_terminal_rescue=0`

Readout: this seed improved materially in survival, but still fails through
hostile terminal deaths despite rescue remaining active.

## Combined Readout

- The movement-behavior fix is **not a no-op**. It changes weak-seed behavior
  materially.
- It does **not** cleanly solve Stage91:
  - one weak seed got worse (`7`)
  - one weak seed improved substantially (`17`)
- Across the two validated weak seeds, mean `avg_survival` improved from
  `128.375` to `142.25`, but this improvement is uneven and still well below
  the strong-seed range.
- In all hostile failures after the fix, the new summary fields still show:
  - `hostile_deaths_after_prior_rescue > 0`
  - `hostile_deaths_with_terminal_rescue > 0`
  - `hostile_deaths_without_terminal_rescue = 0`

So the prior Stage91 conclusion still holds: terminal hostile failures are not
mainly caused by rescue failing to activate.

## Interpretation

Best-supported update:

1. Missing hostile movement in local one-step rollout was a real fidelity bug.
2. Fixing that bug alone does not make the emergency path robust.
3. The remaining dominant surface is still emergency action selection /
   counterfactual reliability under corrected hostile movement, not rescue
   activation presence.

The fix likely removed one source of false-safe labels, but weak-seed behavior
still depends heavily on how emergency ranking consumes those labels and on what
other state-fidelity gaps remain in the local rollout.

## Next Step

Do not add more simulator heuristics blindly.

The next narrow test should compare terminal ranked rescue actions before/after
this movement fix on the improved seed (`17`) and the regressed seed (`7`) to
identify whether:

- the corrected rollout changed the top-ranked action set in the right
  direction but the utility ranking is still wrong, or
- one-step labels remain locally inaccurate even after hostile movement is
  restored.
