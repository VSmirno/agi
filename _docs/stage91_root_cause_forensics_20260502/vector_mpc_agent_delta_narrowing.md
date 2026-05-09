# Stage91 `vector_mpc_agent.py` Delta Narrowing

## Scope

This document narrows the remaining Stage91 behavior delta to exact hunks in:

- [src/snks/agent/vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py)

No repo code was edited in this task. No new broad evals were run.

## Baseline Recovery

Recovered exact feasibility-baseline file from HyperPC:

- baseline source:
  `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/src/snks/agent/vector_mpc_agent.py`
- baseline hash:
  `98c5f83babad7bb97c2cd838605ca950f90d3450b5a08110564db7d7cd3ff368`

Current local file:

- local source:
  `/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py`
- local hash:
  `399384d66bb0d2beaaa040e59a257eebb585c14635160fb0a6c2707236ec8320`

This matches prior artifact provenance:

- feasibility artifact synced hash:
  `98c5f83babad7bb97c2cd838605ca950f90d3450b5a08110564db7d7cd3ff368`
- diagnostics-only artifact synced hash:
  `399384d66bb0d2beaaa040e59a257eebb585c14635160fb0a6c2707236ec8320`

## Compact Semantic Diff

The diff against the feasibility-baseline file is much smaller than expected.
There are only five semantic hunks:

| Hunk | Current location | Baseline location | Change | Initial classification |
| --- | --- | --- | --- | --- |
| `H1` | [vector_mpc_agent.py:66](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:66) | baseline import block | add `build_local_affordance_snapshot` import | likely behavior-neutral diagnostics |
| `H2` | [vector_mpc_agent.py:944](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:944) | baseline had no snapshot call here | build `local_affordance_snapshot(...)` each step before rescue evaluation | likely rescue-path-affecting despite intended diagnostics-only use |
| `H3` | [vector_mpc_agent.py:1088](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1088) | baseline `pre_rescue_state` had no scene field | add `pre_rescue_state.local_affordance_scene` | likely behavior-neutral diagnostics |
| `H4` | [vector_mpc_agent.py:1311](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1311) | baseline step trace had no scene field | add `step_trace.local_affordance_scene` | likely behavior-neutral diagnostics |
| `H5` | [vector_mpc_agent.py:1755](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1755) | baseline `near_concept = str(vf.near_concept)` | change `do` target normalization to `str(vf.near_concept or "empty")` | likely counterfactual-label-affecting, but weak evidence so far |

## Exact Hunk Readout

### H1: Import only

Baseline did not import the affordance helper. Current file adds:

```python
from snks.agent.stage90r_local_affordances import build_local_affordance_snapshot
```

This is just plumbing for later hunks and has no standalone behavioral meaning.

### H2: Per-step snapshot construction

Current file adds this immediately after `env_facing_before` and before rescue
evaluation:

```python
local_affordance_snapshot = build_local_affordance_snapshot(
    player_pos=player_pos,
    spatial_map=spatial_map,
    dynamic_entities=observed_dynamic_entities,
    last_move=prev_move,
)
```

Source facts:

- It runs every step.
- It runs before actor ranking, emergency feature evaluation, and emergency
  action selection.
- The helper implementation in
  [stage90r_local_affordances.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/stage90r_local_affordances.py:11)
  is read-only by inspection:
  - reads `spatial_map._map`
  - reads `spatial_map.is_blocked()`
  - iterates `dynamic_entities`
- `CrafterSpatialMap.is_blocked()` is itself read-only:
  [crafter_spatial_map.py:46](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/crafter_spatial_map.py:46)

This hunk is therefore *intended* to be diagnostics-only, but it is the only
remaining per-step pre-decision delta in the file.

### H3: Rescue diagnostics field

Current file adds:

```python
"local_affordance_scene": dict(local_affordance_snapshot.get("scene", {})),
```

inside `rescue_pending["pre_rescue_state"]`, after rescue activation has already
been determined and after `emergency_selection` has already been made.

This is post-selection bookkeeping.

### H4: Step-trace diagnostics field

Current file adds:

```python
"local_affordance_scene": dict(local_affordance_snapshot.get("scene", {})),
```

inside `step_trace.append(...)`.

This executes after `env.step(primitive)` and after the post-step state has
already been observed. It is post-action telemetry only.

### H5: `do` target normalization

Baseline:

```python
near_concept = str(vf.near_concept)
if near_concept in {"None", "empty", "unknown"}:
    continue
target = near_concept
```

Current:

```python
action_concept = str(vf.near_concept or "empty")
if action_concept in {"None", "empty", "unknown"}:
    continue
target = action_concept
```

This is the only remaining hunk that directly changes the inputs to
`_build_local_counterfactual_outcomes()` and therefore can directly alter raw
candidate-set contents or `do` target selection.

## Evidence-Based Classification

### Likely behavior-neutral diagnostics

#### `H1`, `H3`, `H4`

Reasoning:

- `H1` is import-only.
- `H3` happens after emergency activation and action selection.
- `H4` happens after `env.step(...)`.
- Existing artifact evidence already proved that the emergency controller does
  not consume any new local-affordance fields directly, and diagnostics-only
  validation confirmed the scene field survives only in the rescue-oriented
  diagnostics surface.

These hunks are low explanatory power for either:

- same-state counterfactual-label drift, or
- first divergent rescue-step choice.

### Likely counterfactual-label-affecting

#### `H5`

Reasoning:

- It lives inside `_build_local_counterfactual_outcomes()`.
- It is the only remaining hunk that can directly change whether a `do`
  counterfactual candidate exists and what `target` it uses.
- If it fires on a step where `vf.near_concept` is falsey but not already one
  of the sentinel strings, then candidate presence and downstream emergency
  ranking can diverge immediately.

But current evidence against high impact:

- Retained `local_trace_excerpt` and `local_trace_tail` rows from feasibility,
  extractor, and diagnostics-only artifacts show no blank `near_concept`
  values.
- All retained values are already string labels such as `'empty'`, `'stone'`,
  `'tree'`, `'zombie'`, and `'water'`.
- In the retained artifact surface, this makes baseline and current behavior
  equivalent for the common cases we can actually see.

So `H5` is a real behavior hunk, but currently low explanatory power.

### Likely rescue-path-affecting

#### `H2`

Reasoning:

- It is the only delta that runs before rescue scoring and selection on every
  step.
- Diagnostics-only validation still failed to restore the feasibility baseline
  even though all `*_local` label wiring had already been removed.
- After diagnostics-only rollback, the only file-level pre-decision delta left
  in `vector_mpc_agent.py` is the snapshot build itself.

This creates a tension:

- Source inspection says `H2` should be behavior-neutral.
- File-level provenance says some remaining behavior delta still exists in
  `vector_mpc_agent.py`.

Because `H2` is the only pre-decision hunk left with any meaningful reach, it
has the highest explanatory power even though the source looks pure.

### Unclear

There is no additional unclear hunk beyond `H2` and `H5`. The file diff is
that small.

## Mapping Hunks to Existing Artifact Evidence

### Same-state label drift from `local_affordance_extractor_diff_forensics`

Artifact fact:

- seed `7`, episode `3`, rescue step `9` kept the same pre-state, same ranked
  actions, same chosen action, and same actual outcome, but predicted
  `damage_h` shifted for all shared actions.

Implication for current narrowing:

- That earlier same-state label drift was explained by active extractor wiring
  into counterfactual labels.
- Diagnostics-only rollback removed that active wiring.
- Therefore the only remaining direct counterfactual-construction delta is `H5`.

This is why `H5` remains worth one isolated replay even though retained traces
do not show blank `near_concept` values.

### Upstream trajectory divergence from diagnostics-only validation

Artifact fact:

- Diagnostics-only validation removed the extractor-run terminal `do` pattern,
  but weak-seed mean still did not recover.
- Provenance showed all synced runtime files still matched feasibility baseline
  except `vector_mpc_agent.py` plus the added diagnostics module.

Implication for current narrowing:

- `H3` and `H4` are post-selection/post-step telemetry only.
- `H5` looks weak on retained trace values.
- `H2` is the only hunk left that touches the live step loop before emergency
  evaluation.

That makes `H2` the highest-value replay target even though the helper itself
appears read-only.

## Recommended Minimal Replay Order

### Replay A: revert `H2` + `H3` + `H4` together, keep `H5`

What to change:

- Remove `build_local_affordance_snapshot(...)`
- Remove both `local_affordance_scene` telemetry fields
- Keep the `do` normalization hunk unchanged

Why this is first:

- `H2` cannot be removed alone without breaking `H3` and `H4`.
- This replay removes the entire diagnostics-only affordance snapshot path from
  `vector_mpc_agent.py` while preserving the tiny `do` normalization hunk.
- It is the highest explanatory-power probe for the remaining mismatch because
  it removes the only pre-decision per-step delta cluster.

What it would prove:

- If a bounded recheck snaps back toward feasibility-baseline behavior, then
  the snapshot path itself was not behavior-neutral in practice, despite
  looking pure by source inspection.
- If behavior stays mismatched, the remaining live delta is probably `H5` or a
  non-file provenance issue.

### Replay B: keep `H1`-`H4` as-is, revert only `H5`

What to change:

- Restore baseline `do` gating:
  `near_concept = str(vf.near_concept)`

Why this is second:

- `H5` is the only direct counterfactual candidate-construction hunk left.
- It is small and isolated.
- Existing retained traces make it low probability, but it is still the only
  explicit behavior hunk inside `_build_local_counterfactual_outcomes()`.

What it would prove:

- If a bounded recheck changes candidate presence or rescue ranking, then some
  unretained steps do hit falsey non-sentinel `vf.near_concept` values.
- If nothing changes, `H5` can be deprioritized hard.

### Replay C: revert `H2`-`H5` together and recover the exact feasibility-baseline file

What to change:

- Restore `vector_mpc_agent.py` byte-for-byte to the recovered feasibility
  baseline.

Why this is third:

- This is the clean control after the two smaller probes.
- It tells us whether the remaining mismatch really belongs to this file at all.

What it would prove:

- If exact baseline file restores the prior bounded behavior, the culprit is
  somewhere inside `H2`-`H5`.
- If exact baseline file still does not restore behavior under the current
  validation harness, then the earlier “remaining delta is only
  `vector_mpc_agent.py`” assumption is incomplete and the provenance problem is
  outside this file.

## Concrete Narrowing Conclusion

The exact file diff is not broad. It narrows to one meaningful tension:

- `H5` is the only explicit behavior hunk, but retained artifacts do not show
  the input pattern that would make it matter often.
- `H2` is intended to be diagnostics-only, but it is the only remaining
  pre-decision step-loop delta and therefore the strongest explanation for the
  residual mismatch if the file itself is still causal.

So the smallest credible narrowing sequence is:

1. replay `H2`+`H3`+`H4` out together
2. if needed, replay `H5` separately
3. if still ambiguous, restore the exact baseline file as the control

That sequence is ordered by explanatory power, not by superficial code size.
