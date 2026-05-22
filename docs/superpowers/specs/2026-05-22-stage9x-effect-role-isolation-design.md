# Stage9X Effect-Role Isolation Repair

## Evidence

Stage9X interaction completion now reaches the first tree and gathers wood, but
the seed17 validation trace gets stuck after the goal switches to
`craft_wood_sword`. By step 60 the inventory has enough wood, yet the selected
plan remains baseline/self motion and never emits `place_table` or
`make_wood_sword`.

The root-cause probe separated planner logic from world-model recall:

- A fresh textbook-seeded model produces a valid
  `chain:place_table+make_wood_sword` candidate for `Goal("craft_wood_sword")`,
  and that chain scores above baseline.
- The canonical HyperPC world model at
  `/opt/cuda/agi-entity-promotion-20260513T101914Z/_docs/hyper_entity_promotion_20260513T101914Z/wm/seed17.pt`
  decodes `predict("wood_sword", "make")` as `{"damage_h": 2}` instead of the
  textbook crafting effect.
- The same contaminated decode appears after recording outcomes, while a fresh
  full-profile textbook model decodes the crafting effect as inventory gain.

The failure is therefore not a missing candidate generator. The chain exists,
but forward simulation reads an outcome-like vector from the action-effect
address, sees no `wood_sword` progress, and baseline wins scoring.

## Invariant

The `VectorWorldModel` keeps one `CausalSDM`, but every semantic payload must
have an explicit role in the address:

```text
bind(bind(concept_vec, action_vec), role__EFFECT__)          -> physics effect
bind(bind(concept_vec, action_vec), role__OUTCOME_H__)       -> primitive outcome
bind(bind(option_context_vec, option_vec), role__OPTION...)  -> option outcome
```

No physics-effect read or write may use the legacy unrole address
`bind(concept_vec, action_vec)`.

## Implementation Plan

1. Add a stable effect address role, `__EFFECT__`, and a version marker,
   `effect_address_version = 1`, to saved world-model payloads.
2. Route `learn`, `predict`, and `batch_predict` through the effect-role address.
3. Leave primitive outcome and option outcome reads/writes unchanged except for
   their continued separation from the new effect role.
4. On load, treat missing `effect_address_version` as a legacy model. Preserve
   loaded vectors, memory content, requirements, and metadata, then seed textbook
   action/passive effect facts into the new effect-role channel. This repairs
   canonical models without deleting their old raw-address or outcome memories.

## Validation

Local validation covers:

- outcome writes do not alter `predict("wood_sword", "make")`;
- legacy/polluted snapshots load with `wood_sword/make` repaired to a crafting
  effect rather than `damage_h`;
- with wood available and no adjacent table, the repaired model gives
  `chain:place_table+make_wood_sword` positive craft progress and the chain
  scores above baseline;
- existing vector world-model, outcome stimulus/recorder, and vector MPC tests
  remain green.

The next required gate is a HyperPC seed17 episode-0 video validation. That is
outside this local implementation task.
