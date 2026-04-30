# Stage90R Current Stage Checkpoint

## What Is Now Proven

### 1. The original deterministic mixed-control collapse was real and understood

The old `move_up x6` failure on tiny `mixed_control` was not random noise.
It came from:

- pathological tiny split behavior
- missing threat/resource support in train
- contradictory planner supervision on repeated state signatures

That narrow bug class is now closed.

### 2. The guidance/agreement path improved the narrow offline baseline

`9083357` established the first working narrow baseline:

- deterministic `seed 7` path promotes a checkpoint
- hyper multiseed `mixed_control` is promotion-robust `5/5`
- old terminal actor `move_up 4/4` collapse no longer survives to the end

But `9083357` remained behavior-fragile:

- intermediate actor reflexes still varied by seed
- transient `move_up`, `move_right`, and similar detours still appeared

### 3. The actor-selection-contract redesign improved the offline behavior further

`ca12f40` made the actor contract more state- and candidate-aware:

- transient `move_right` reflexes disappeared
- transient `move_left` reflexes disappeared
- transient `move_up` reflexes shrank to seed `31` only
- promotion robustness remained `5/5`

So `ca12f40` is a stronger narrow offline `mixed_control` baseline than
`9083357`.

## What Failed

### 4. Better offline mixed-control did not automatically mean better rescue behavior

Bounded online `mixed_control_rescue` on hyper showed a regression:

- `9083357`: `avg_survival=179.25`, `rescue_rate=0.132`
- `ca12f40`: `avg_survival=116.0`, `rescue_rate=0.121`

Root cause is not mainly “rescue fires and chooses badly.”
The better explanation is:

- some `ca12f40` runs die before rescue ever becomes eligible
- the tighter actor-selection contract reduces early actor/planner disagreement
  opportunities in hostile states
- rescue therefore fails earlier at the eligibility layer

### 5. The first rescue-contract fix was only a partial repair

`7544240` added a structural consensus-danger rescue override.

It did fix the narrow hole it targeted:

- `ca12f40` episode 0: `27` steps, `0` rescues
- `7544240` episode 0: `58` steps, `3` rescues

But it did not recover aggregate rescue-side behavior:

- `7544240`: `avg_survival=110.75`, `rescue_rate=0.124`

So eligibility repair alone is not enough.

## Separate Adjacent Risk

### 6. Hyper GPU online rescue path is still a shared runtime problem

For both `9083357` and `ca12f40`:

- standalone evaluator CUDA probes succeed
- live symbolic `run_vector_mpc_episode` on hyper GPU hangs before episode 0
- the stall sits inside the online actor-ranking evaluator path

This is a real neighboring issue, but it is not specific to the latest branch.

## Baseline Decision

At the end of this stage:

- `9083357` remains the wider proven baseline for online/rescue-side safety
- `ca12f40` is the stronger narrow offline `mixed_control` baseline
- `7544240` is an informative partial rescue fix, not a new global baseline

## Next Structural Debt

The next problem is no longer:

- split policy
- soft teacher aggregation
- world-prior plumbing
- actor agreement scheduling
- one more rescue-trigger condition tweak

The remaining structural debt is:

- rescue needs its own emergency selector / safety-first controller
- it should not depend primarily on incidental `actor vs planner` disagreement
- it should choose emergency actions from an explicit safety objective

That is the correct starting point for the next phase.
