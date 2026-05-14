# Stage9X Capability and Parent-Goal Reactivation Plan

## Goal

Close the AGI-level gap exposed by the seed17 gen2 trace: the agent can create a tool (`wood_sword`) but does not reliably convert the acquired capability into situated action, goal reactivation, or survival arbitration.

This is not a Crafter tuning pass. Crafter is the testbed. The target mechanism is general:

```text
blocked parent goal -> instrumental subgoal -> capability acquired -> parent goal resumes under new affordances
```

## Success Criteria

For the validated seed17 ep0 line:

- After successful `make_wood_sword`, the trace records an explicit capability state such as `armed_melee=true`.
- The goal layer can explain why `craft_wood_sword` was active: which parent goal or threat context requested it.
- After `wood_sword` is acquired, the parent goal is reactivated or explicitly abandoned with a reason.
- Near-hostile behavior changes conditionally:
  - unarmed + hostile close: avoid or craft weapon;
  - armed + hostile close + viable health: engage or hold tactical position;
  - armed + low health: avoid;
  - critical drink/food: survival goal can override fight.
- Validation is based on trace evidence and seed17 ep0 video, not aggregate survival alone.

## Non-Goals

- Do not hardcode `if wood_sword and zombie then do`.
- Do not add Crafter-only combat policy as the primary mechanism.
- Do not optimize multiseed score before the seed17 causal trace is understood.
- Do not introduce a second parallel value-function substrate.

## Phase 1: Trace Audit, Step 20 to Death

| Task | Effort | Done Criteria |
| --- | ---: | --- |
| Extract post-sword timeline from `20260514_seed17_ep0_promoted_station_fix.json` | 1-2h | Table from step 20 to death with inventory, body, active goal, plan origin, primitive, controller, rescue trigger, nearest hostiles, near concept |
| Identify goal handoff after sword craft | 1-2h | Clear answer: does `craft_wood_sword` terminate, and what goal replaces it? |
| Audit hostile encounters while armed | 1-2h | For every step with `wood_sword>0` and hostile within range 1-3: planner origin, selected primitive, rescue override, outcome |
| Audit hydration arbitration | 1-2h | First step where `drink<3`, goal at that step, candidate plans, why water acquisition did or did not take control |
| Audit outcome writes for armed combat | 1-2h | Evidence whether `(zombie, do)` / `(skeleton, do)` outcomes are written or recalled differently with `wood_sword>0` |

Gate: do not implement Capability Stimulus until the trace says which boundary failed: goal selector, planner candidates, rescue override, outcome substrate, or hydration arbitration.

## Phase 2: Minimal Design

| Task | Effort | Done Criteria |
| --- | ---: | --- |
| Define `CapabilityState` | 2-4h | Small structured state derived from inventory/body/world facts, e.g. `armed_melee`, `has_pickaxe`, `can_place_station` |
| Define parent-goal relation | 2-4h | Goal records optional `parent_goal`, `blocked_by`, `requested_capability`, and completion condition |
| Define Capability Stimulus | 2-4h | Scoring signal consumes capability + local context without naming Crafter-only rules in the scoring core |
| Define trace schema | 1-2h | `local_trace` shows capability state, parent goal, handoff reason, and arbitration reason |

Design rule: capability is a general affordance layer. Crafter mappings live in textbook/config facts, not in planner scoring code where avoidable.

## Phase 3: Minimal Implementation

| Task | Effort | Done Criteria |
| --- | ---: | --- |
| Add capability extraction | 2-4h | Unit tests: `wood_sword>0 -> armed_melee`; no weapon -> false |
| Add goal provenance/handoff fields | 4-8h | Craft subgoal can point back to threat parent; trace shows the relation |
| Add parent reactivation on subgoal success | 4-8h | After `wood_sword` gain, blocked combat/survival goal is reconsidered in the next decision cycle |
| Add Capability Stimulus | 4-8h | Armed-vs-unarmed threat scoring changes in isolated unit tests |
| Add hydration override guard | 2-4h | Critical vitals can override combat even when armed |

Keep each patch narrow. If two iterations fail to change seed17 post-sword behavior, stop and reframe the architecture.

## Phase 4: Validation

| Validation | Done Criteria |
| --- | --- |
| Focused tests | Capability extraction, parent handoff, armed hostile scoring, hydration override pass |
| Seed17 ep0 full-profile video | Produced with perception overlay and trace |
| Trace audit after implementation | Shows sword craft, capability acquisition, parent reactivation, and explicit decision around zombie/hydration |
| Video self-inspection | Frames around first sword, first armed hostile encounter, first low-drink event, death/survival |

Primary pass condition is not "kills zombie". It is:

```text
The trace explains why the agent used, deferred, or ignored the acquired capability.
```

Killing a zombie is a useful behavioral confirmation, but the architectural success condition is coherent capability-conditioned control.

## Risks

| Risk | Mitigation |
| --- | --- |
| Capability layer becomes Crafter-specific | Keep mappings in textbook/config; core uses capability predicates |
| Parent-goal machinery becomes a planner rewrite | Start with one-level parent pointer and trace-visible handoff only |
| Emergency rescue masks combat decisions | Trace controller/rescue on every armed hostile encounter before changing scoring |
| Hydration and combat fight each other | Add explicit arbitration reason to trace before changing weights |
| Success measured only by survival | Require video and step trace around capability use |

## Immediate Next Step

Run Phase 1 trace audit on:

```text
output_to_user/gen2_trace_audit/20260514_seed17_ep0_promoted_station_fix.json
```

Produce a concise report:

- first sword acquisition;
- goal and controller timeline after sword;
- armed hostile encounters;
- hydration goal activation/failure;
- exact failed boundary classification.
