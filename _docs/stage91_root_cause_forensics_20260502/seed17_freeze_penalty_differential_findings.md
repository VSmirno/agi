# Seed-17 Freeze Penalty Differential — Forensic Findings

Inputs: `null_target_fix_seed_{17,47}_eval.json`, `phaseA_seed_17_eval.json`,
`stage90r_emergency_controller.py:270–325`, `vector_mpc_agent.py:1730–1827`.

## TL;DR

The primary component differential that lets `do/None` win at the seed-17
post-fix freeze is **`blocked_penalty` (−3.0)**, applied to movement candidates
only. `do` (and `sleep`) get `blocked_h=False` by construction even when they
produce zero displacement and zero resource gain, while a stationary attempted
move gets `blocked_h=True`. With moves and `do` predicting identical outcomes
otherwise (same damage, same `adjacent_hostile_after_h`, same final hostile
distance), this −3.0 wedge alone is enough to pin the controller into an
absorbing freeze on a stone wall.

## Step 165 — null-target-fix seed 17 episode 0 (full decomposition)

Position: `[30, 54]`, near_concept = **stone**, body 7 HP / food 4 / drink 2,
zombie d=1, skeleton d=3, arrow d=3.

| action      | survived | dmg | h_delta | esc | nrst_h | displ | blocked | adj_after | resrc | adv_rk | plan | learn | utility |
|-------------|---------:|----:|--------:|----:|-------:|------:|--------:|----------:|------:|-------:|-----:|------:|--------:|
| do          | F        | 7.0 | −7.0    | 0.0 | 1      | 0     | **F**   | T         | 0     | None   | F    | F     | **−45.50** |
| sleep       | F        | 7.0 | −7.0    | 0.0 | 1      | 0     | F       | T         | 1     | 2      | F    | F     | −46.60 |
| move_down   | F        | 7.0 | −7.0    | 0.0 | 1      | 0     | **T**   | T         | 2     | 0      | T    | F     | −47.50 |
| move_left   | F        | 7.0 | −7.0    | 0.0 | 1      | 0     | **T**   | T         | 1     | 1      | F    | T     | −48.05 |

(top‑4 ranked actions; full 6 not in JSON, but earlier non‑damage freeze steps
152–155 show the same shape with `do=−52.5` vs `move_down=−54.7`.)

### Component delta: do − move_down = +2.00

| component         | do    | move_down | Δ (do better by) |
|-------------------|------:|----------:|-----------------:|
| survived bonus    | −12   | −12       | 0                |
| −4·damage         | −28   | −28       | 0                |
| 1.75·escape_delta | 0     | 0         | 0                |
| 0.5·health_delta  | −3.5  | −3.5      | 0                |
| min(0.5,0.1·rg)   | 0     | 0.2       | −0.2             |
| advisory_bonus    | 0     | 0.6       | −0.6             |
| planner_bonus     | 0     | 0.2       | −0.2             |
| learner_penalty   | 0     | 0         | 0                |
| sleep_threat      | 0     | 0         | 0                |
| **blocked_penalty** | **0** | **−3.0** | **+3.0** ← dominant |
| adjacent_penalty  | −2    | −2        | 0                |
| displacement_bonus| 0     | 0         | 0                |
| **total**         | **−45.5** | **−47.5** | **+2.0** |

Only one component gives `do` an advantage: the −3.0 blocked penalty that
movement candidates pay because they're hemmed in by stone. The advisory /
planner / resource bonuses on `move_down` claw back 1.0 of that, leaving net
+2.0 for `do`. Same wedge persists at steps 152–183, with the magnitude varying
only by which advisory/planner/learner alignments happen to land on which move.

## Why `do` escapes blocked_h — the root code

`vector_mpc_agent.py:1818–1820`:

```python
"blocked_h": bool(
    primitive.startswith("move_") and displacement_h == 0
),
```

`displacement_h` is 0 for `do`, `sleep`, **and** failed moves. The label-builder
only flags moves. The raw counterfactual at step 165 confirms this — see the
`do`/`stone` rollout label dumped from `local_trace_tail`:

```
do  -> displacement_h=0, resource_gain_h=0, inventory_delta={}, survived_h=False,
       damage_h=7, adjacent_hostile_after_h=True, blocked_h=False
move_down -> identical except blocked_h=True, resource_gain_h=2
```

`do` produced **zero displacement, zero resource gain, zero inventory change** —
functionally indistinguishable from a blocked move — yet inherits the "not a
move" exemption from blocked. The controller scores it as the safest option.

## Cross-check — null-target-fix seed 47 episode 0

The seed-47 freeze regime is dominated by **sleep**, not `do` (e.g. step 76
selects sleep, util=6.3). The mechanism is the same family:

| action     | survived | dmg | blocked | adj_after | resrc | adv_rk | utility |
|------------|---------:|----:|--------:|----------:|------:|-------:|--------:|
| sleep      | T        | 0.0 | F       | F         | 1     | 2      | **+6.30** |
| do         | T        | 0.0 | F       | F         | 0     | None   | +6.00 |
| move_left  | T        | 0.0 | **T**   | F         | 0     | 1      | +3.35 |
| move_down  | T        | 5.0 | F       | F         | 1     | 0      | −15.40 |

Differential sleep − move_left = +2.95 ≈ −3.0 blocked_penalty + 0.45 advisory −
0.1 sleep_threat. The blocked penalty is again the dominant wedge against the
only non-self-harm move available. Once the controller picks sleep/do, the
freeze becomes self-reinforcing because subsequent steps continue to evaluate
moves as "blocked" against the same wall.

First do-streak step on seed 47 (step 179): do=+6.0 vs move_left=+3.7 — same
shape; +2.3 wedge from blocked_penalty − advisory bonus.

## Cross-check — Phase A seed 17 step 139 (sanity)

Pre-fix do "phantom" verified:

| action    | survived | dmg | blocked | adj_after | utility |
|-----------|---------:|----:|--------:|----------:|--------:|
| do        | **T**    | 0.0 | F       | F         | **+6.00** (all label defaults) |
| sleep     | F        | 2.0 | F       | T         | −25.85 |
| move_down | F        | 2.0 | T       | T         | −26.95 |
| move_left | F        | 2.0 | T       | T         | −27.20 |

`do` on Phase A step 139 has *every* label at its default (`survived_h=True`,
`damage_h=0`, `escape_delta_h=None`, `nearest_hostile_h=None`) — the signature
of the pre-fix `continue` that skipped `do` in `vector_mpc_agent.py:1746–1749`
when `near_concept ∈ {None, empty, unknown}`. Matches the prior diagnostic
(commit 39f960b9) exactly. This run is *not* the post-fix wedge — it's the
phantom default. Two different bugs, same victim action.

Post-fix, Phase A's phantom `+6.0` is gone, but the **blocked_penalty wedge**
described above remains and reproduces the freeze under different conditions.

## Code lines responsible

1. `vector_mpc_agent.py:1818–1820` — `blocked_h` is asymmetric: only
   movement primitives can ever be flagged blocked; `do`/`sleep` always get
   `blocked_h=False` regardless of whether the action was effectively a no-op.
2. `stage90r_emergency_controller.py:289` — `blocked_penalty = -3.0`
   unconditionally; combined with (1), this is a 3.0-utility bonus to any
   stationary self-action vs any failed movement, whenever the rollout
   predicts equal damage outcomes.
3. (Pre-fix only, related but distinct) `vector_mpc_agent.py:1746–1749` — the
   `continue` that skipped `do` rollouts and produced default-label phantom
   utility. Already addressed by the null-target patch but appears to have
   been reverted again per the task description.

## Recommended fix surface (proposal — not applied)

Three viable candidates; (A) is the smallest local fix and addresses the
asymmetry directly.

**(A) Promote `blocked_h` to an "ineffective action" flag at the label site.**
In `vector_mpc_agent.py:1818–1820`:

```python
"blocked_h": bool(
    displacement_h == 0
    and resource_gain == 0
    and not inventory_delta
    and primitive != "sleep"   # sleep's "no-op" is intentional rest
),
```

Cost: trivial; benefit: removes the −3.0 wedge that lets a no-op `do` against a
wall beat any blocked move with identical predicted damage. Risk: a productive
`do` against a tree/cow that yields resource_gain or inventory delta is
correctly *not* flagged. Sleep stays exempt because it's dispatched through a
separate `sleep_threat_penalty` already and its no-op character is by design.

**(B) Re-tune `blocked_penalty` in the controller (e.g. −0.5 instead of −3.0).**
Cost: cheaper to implement; benefit: weakens the wedge but keeps the same
asymmetry, so `do/None` still systematically wins ties. Worse than (A).

**(C) Make blocked relative — only penalize a blocked move when at least one
unblocked move exists.** Cost: extra controller logic; benefit: principled —
"prefer non-blocked moves where possible, otherwise treat blocked moves as
zero". Effectively equivalent to (A) for the freeze case but doesn't fix the
underlying labeller asymmetry, so other call sites that read the label still
see the same bias.

Recommendation: **(A)**. It fixes the bug at its source — the label is wrong
about what a "blocked" outcome means — and every downstream consumer becomes
correct automatically.

## What's missing from the JSONs (none, but flag for future)

Per-action labels at every emergency step are present (`ranked_emergency_actions`
holds top-4; `local_trace_*.counterfactual_outcomes` holds all 5–6). No extra
log lines needed for this analysis. If `ranked_emergency_actions` were ever
truncated below all six allowed actions, add a `full_ranked_emergency_actions`
field to keep the bottom of the ranking auditable.
