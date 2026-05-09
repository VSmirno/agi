# Null-target `do` fix — validation findings

**Date:** 2026-05-10
**Local branch:** stage90r-world-model-guardrails
**Remote artifact root:** `/opt/cuda/agi-stage91-null-target-do-fix-20260509T202758Z/`
**Eval invocation:** strict-determinism wrapper, CPU-only (`CUDA_VISIBLE_DEVICES=""`), `PYTHONHASHSEED=0`,
torch 2.5.1+cu121, `cudnn_deterministic=true`, `float32_matmul_precision=highest`.
1 trial × 4 episodes per seed (spread is 0 by construction).

## Patch summary

### 1. `src/snks/agent/vector_mpc_agent.py:1746–1752`
Drop the silent `continue` for `do` when `near_concept ∈ {"None", "empty",
"unknown"}`. Target now stays `"self"`, so `simulate_forward` runs a real
no-op `do` rollout and produces a real label.

```diff
-            if near_concept in {"None", "empty", "unknown"}:
-                continue
-            target = near_concept
+            if near_concept not in {"None", "empty", "unknown"}:
+                target = near_concept
+            # else: target stays "self"; simulate_forward handles a
+            # null-target `do` as a no-op for the player while still
+            # advancing dynamic entities.
```

### 2. `src/snks/agent/stage90r_emergency_controller.py:272–283`
Backstop in `select_action`: skip any action whose outcome is missing or
has no `label` field instead of falling through to default best-case
fields (which awarded a phantom +6.0 utility).

```diff
-            label = dict(outcome_by_action.get(action, {}).get("label", {}))
+            outcome = outcome_by_action.get(action)
+            if outcome is None or "label" not in outcome:
+                continue
+            label = dict(outcome.get("label", {}))
```

## Active runtime flags

`mode=strict`, `cudnn_deterministic=true`, `cuda_is_available=false`,
`PYTHONHASHSEED=0`, identical across all five seeds. Per-seed snapshots in
`trial/flags/seed_<S>_settings.json`. Eval flags: `--mode mixed_control_rescue
--n-episodes 4 --max-steps 220 --perception-mode symbolic --enable-planner-rescue
--smoke-lite --terminal-trace-steps 32 --record-death-bundle`. Evaluator
checkpoint: `agi-stage91-determ-rebase-C-do_facing-…/_docs/stage90r_seed7_actor_selection_probe3.pt`.

## Per-seed table

| seed | this run | Phase A | Δ Phase A | Phase B | M2 (s17) |
|------|---------:|--------:|----------:|--------:|---------:|
| 7    | **139.00** | 155.50 | **−16.50** | — | — |
| 17   | **150.25** | 141.50 | **+8.75**  | 150.50 | 174.75 |
| 27   | **198.25** | 208.25 | **−10.00** | — | — |
| 37   | **201.50** | 210.75 | **−9.25**  | — | — |
| 47   | **158.75** | 203.00 | **−44.25** | — | — |
| weak (7,17)         | 144.625 | 148.50 | −3.88 | — | — |
| strong (27,37,47)   | 186.17  | 207.33 | **−21.17** | — | — |
| overall             | 169.55  | 183.70 | **−14.15** | — | — |

## Seed 17 ep0 terminal trace — freeze pattern is **not** gone

`episode_steps=184`, `death_cause=skeleton`, `controller_distribution =
{emergency_safety: 121, learner_actor: 24, planner_bootstrap: 39}`,
`n_rescue_events=121`.

Steps 168–183 (16 consecutive ticks before death), agent at `[30, 54]`,
zombie at distance 1, action `do` with `target=None`, no movement, periodic
2.0 dmg from skeleton arrows:

```
step=168 pos [30,54]→[30,54] prim=do target=None blocked=False z_dist=1 dmg=0.0
…
step=171 pos [30,54]→[30,54] prim=do target=None blocked=False z_dist=1 dmg=2.0
…
step=183 pos [30,54]→[30,54] prim=do target=None blocked=False z_dist=1 dmg=2.0
```

Comparison with the f6ad2b2 reference (commit log): freeze trap at `[34,39]`,
6 `do` ticks, target=None, identical structure. **The trap relocated from
`[34,39]` to `[30,54]` and lengthened from 6 to 16+ ticks.** Structural
failure mode is unchanged.

### Per-action utility decomposition at the freeze trap (rescue_trace_tail)

```
step=178 trig=low_vitals  rescue=do  policy=independent_emergency_choice
    do          util = -23.00  target=None
    sleep       util = -24.10
    move_down   util = -25.20
    move_left   util = -25.45
…steps 179–183: identical ranking, do wins by ~2 utility every step.
```

**Backstop is taking effect.** `do/None` is no longer awarded the phantom
+6.0; its rollout-derived utility is −23.00. **But it still wins**, because
movement candidates score worse (≈ −25.2 to −25.45) — they include penalty
contributions for adjacent threats / inadmissible/blocked moves that the
no-op rollout avoids. Ranking gap: ~2.0–2.5 in favour of `do/None` every
step.

## Verdict

| Outcome band | This run |
|---|---|
| seed 17 ≥ 170 (clear win) | no |
| 160 ≤ seed 17 < 170 (substantial closure) | no |
| **150 ≤ seed 17 < 160 (partial; check terminal trace)** | **yes — and trace shows freeze still appears** |
| seed 17 < 150 (didn't work) | no |
| **Any other seed regresses by >5** | **yes — seeds 7, 27, 37, 47 all regress (−16.5, −10, −9.25, −44.25)** |

**Bottom line: freeze trap is NOT closed.** The original phantom +6.0
default-best-case path is genuinely fixed (verified: `do/None` now scores
−23 from a real rollout, not +6 from defaults). But removing the phantom
just exposed a deeper modelling bug: with a real rollout label, `do/None`
*still* wins emergency rerank at distance-1 hostile contact because the
no-op rollout looks artificially favourable next to movement rollouts that
include exposure/inadmissibility penalties. The freeze relocated and got
worse (16+ ticks instead of 6) — and three strong seeds plus seed 7
regressed by 9–44 points each, dropping the overall mean by 14.15.

## Recommended next probe (no patch yet)

1. **Capture a per-action label dump** at one freeze step (e.g. step 178 of
   seed-17 ep0): full label dict for `do/None`, `move_down`, `move_left`,
   `move_right`, `move_up`, `sleep`. Specifically confirm `survived_h`,
   `damage_h`, `health_delta_h`, `inadmissible_h`, `blocked_h`, and any
   penalty contribution that puts movement at −25 vs `do/None` at −23.
2. **Check whether movement rollouts carry an inadmissible/blocked penalty
   that no-op `do` is structurally exempt from.** If yes, the comparison is
   apples-to-oranges: the fix should bias the *utility function*, not just
   the rollout invocation. Candidates: penalise `do` when `target=None &&
   nearest_hostile_dist ≤ 1` (it explicitly buys you nothing); raise the
   movement-toward-escape weight when current cell is adjacent to hostile.
3. **Before any patch**, replay the seed-17 freeze step with the planner's
   move suggestion forced: does taking `move_right` / `move_up` at step 168
   actually escape? If yes, the rollout label is wrong (movement should
   *survive better*, not score 2.5 worse). That points to the world model
   simulator under-weighting "leave the hostile's reach" or over-weighting
   transient inadmissibility.
4. **Strong-seed regression** is independent evidence the fix has changed
   ranking globally (not just at the freeze trap). Worth eyeballing one
   strong-seed episode where the controller used to pick movement and now
   picks `do/None` to see if the same utility-gap inversion is at work.

Per task constraints, no further patches applied this round.
