# Stage9X Combat Affordance Closure

**Date:** 2026-05-22
**Status:** Implemented locally, pending HyperPC video gate
**Base commit:** `7cd8e596 fix(stage9x): verify effect-role repair before trusting snapshots`

## Forensic Evidence

HyperPC validation for `7cd8e59` showed real progress: repair smoke passed,
the table+sword chain executed, a sword was crafted at step `86`, and
`armed_melee=True` remained true through the terminal window.

The post-sword forensic task `017fca49` showed the remaining death is not
capability loss:

- steps `87-178` all retain `armed_melee=True`;
- the sword remains in inventory;
- terminal rows repeatedly select armed hostile `do` while the hostile is
  adjacent but not on the facing tile;
- examples include steps `160/162/166/168/170/174/176/178` with
  `is_adjacent=True` and `is_facing_target=False`;
- step `178` ends with `HP 1 -> 0`.

Primary failure: the combat interaction violates action geometry. `do` only
acts on the facing tile, so adjacent-but-not-facing hostile attacks are blind
no-ops. Emergency safety can amplify the bug by replacing planner alignment
with an unqualified `do`.

Secondary failure: target arbitration can still prefer an adjacent zombie while
longer-range skeleton/arrow pressure is lethal. That is intentionally not
solved here.

## Scope

This fix is a combat-affordance closure, not a score tweak.

It extends the existing interaction-completion mechanism so declared hostile
`do` rules with `effect.remove_entity` use the same approach/align/act contract
as resource interactions:

```text
target not adjacent -> navigate/approach
target adjacent but not facing -> alignment move
target adjacent and facing -> do
```

The mechanism is general over action geometry and declared effects. Combat is
identified from textbook facts (`do <target>` with `effect.remove_entity` and
requirements such as `wood_sword`), not by adding a zombie-specific policy.

## Behavioral Contract

When armed and a hostile `do` target is adjacent:

- if the target is not on the facing tile, selected primitive must be an
  alignment move, not `do`;
- if the target is on the facing tile, `do` is allowed;
- emergency safety must preserve that contract when it selects or keeps `do`;
- trace rows should expose `interaction_completion.reason` as an alignment or
  emergency-alignment reason.

When the hostile is not adjacent, existing approach/navigation behavior may
continue.

## Emergency Handling

Emergency safety still ranks primitive actions, but an emergency-selected `do`
is passed through a hostile-alignment guard. If the selected `do` corresponds
to a feasible adjacent remove-entity target that is not faced, the executed
primitive becomes the alignment move. The rescue trace records
`combat_alignment_override` with the original action, executed action, target,
and facing facts.

Counterfactual construction now keeps a `do` row when a feasible adjacent
tracked/spatial target exists even if `near_concept` is empty. This gives the
emergency comparison visibility into attack as an option without expanding into
a full combat planner.

## Non-Goals

- No broad target arbitration rewrite.
- No skeleton-vs-zombie combat planner.
- No survival score retuning.
- No HyperPC video validation in this local task.

## Validation

Focused tests cover:

- armed adjacent zombie not facing emits alignment, not `do`;
- armed adjacent zombie facing emits `do`;
- emergency-selected `do` preserves required hostile alignment;
- counterfactuals include adjacent hostile `do` when `near_concept` is empty;
- existing tree interaction completion behavior remains covered by the prior
  interaction-completion tests.

Known follow-up: push the branch and run the HyperPC seed17 ep0 video gate.
