# Stage91 Current Conclusions

Date: 2026-05-08

This note records the current accepted/rejected conclusions from the Stage91
weak-seed rescue forensics and bounded HyperPC validations.

## Current Best Direction

The strongest current baseline is:

- movement-behavior fix in `vector_sim`
- blocked-move / occupied-tile fix in `vector_sim`
- feasibility-label additions plus direct feasibility-aware rescue scoring

Best bounded weak-seed result so far:

- weak-seed mean `avg_survival=172.875`
- source run:
  `_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`

This is the current best validated direction of travel.

## What Is No Longer the Main Problem

- Missing rescue activation is not the main root cause.
- Rescue often remains active at terminal hostile deaths.
- Pure controller-ranking tweaks without better local counterfactual semantics
  are not the right layer to optimize first.

## Accepted Conclusions

### 1. Missing hostile movement in local rollout was a real bug

Validated positive but insufficient.

Effect:
- changed weak-seed behavior
- did not close Stage91 on its own

### 2. Blocked / occupied movement was a real terminal mismatch

Validated positive.

Effect:
- reduced blocked terminal rescue rows
- improved weak-seed survival versus the movement-only fix

### 3. First-class local feasibility labels are useful

Validated positive overall.

Effect:
- improved weak-seed mean from `157.875` to `172.875`
- strongly improved `seed 7`
- reduced the remaining false-safe pattern on `seed 17`, but did not remove it

## Rejected Directions

### 1. Threat-ranking tweak

Rejected.

Result:
- improved one narrow terminal symptom
- regressed weak-seed survival materially

Conclusion:
- not a viable active baseline

### 2. `do`-facing fix in rescue counterfactuals

Rejected on bounded weak-seed validation.

Result:
- weak-seed mean regressed from `172.875` to `132.625`
- residual false-safe terminal `do` rows got worse, not better

Conclusion:
- do not carry this as the active candidate direction

## Current Readout

The project is now past "find any promising direction" and into "narrow the
remaining false-safe terminal action mismatch."

The best current interpretation is:

- local rescue needs truthful immediate execution/feasibility facts
- the chooser should stay learnable
- remaining failures are now a smaller tail of false-safe one-step rescue
  judgments, not the broad earlier failure mode

## Next Step Constraint

Do not move into:

- environment tuning
- broad controller heuristics
- large rescue-policy rewrites

Stay inside the current paradigm:

- explicit local feasibility/perception is allowed
- direction choice remains learned
- each next fix should be narrow and validated first on weak seeds `7` and `17`
