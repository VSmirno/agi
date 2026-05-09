# Broadened blocked_h Fix Validation — Findings

Run: `agi-stage91-blocked-h-broaden-20260509T211630Z`
Artifact root (HyperPC): `/opt/cuda/agi-stage91-blocked-h-broaden-20260509T211630Z/trial/`
Date: 2026-05-09 (eval) / 2026-05-10 (write-up)

The validation subagent timed out at 3600s while writing markdown; eval runs
themselves completed for all 5 seeds. Analysis below was done locally on the
pulled JSONs.

## Patches under test (all reverted after validation)

1. `src/snks/agent/vector_mpc_agent.py:1746–1752` — null-target `do`
   no longer skipped; falls back to `target="self"`. From task 7c0ae280.
2. `src/snks/agent/vector_mpc_agent.py:1818–1822` — broadened blocked_h:
   ```python
   "blocked_h": bool(
       displacement_h == 0
       and resource_gain == 0
       and not inventory_delta
       and primitive != "sleep"
   ),
   ```
   PRIMARY FIX under test.
3. `src/snks/agent/stage90r_emergency_controller.py:271–281` — backstop:
   actions with missing outcome are skipped. From task 7c0ae280.

37/37 local tests passed pre-validation.

## Per-seed results (avg_survival, 4 episodes × 220 max-steps)

| seed | this run | Phase A | Δ |
|------|---------:|--------:|----:|
| 7    | 126.00   | 155.50  | **−29.50** |
| 17   | **167.75** | 141.50 | **+26.25** |
| 27   | 192.00   | 208.25  | −16.25 |
| 37   | 208.25   | 210.75  | −2.50 |
| 47   | 197.50   | 203.00  | −5.50 |

Aggregates:
- weak (7+17): 146.88 vs 148.50 (Δ −1.62)
- strong (27+37+47): 199.25 vs 207.33 (Δ −8.08)
- overall: 178.30 vs 183.80 (Δ −5.50)

## Verdict: partial signal, net-negative

Mixed outcome. The broadened rule did close the seed-17 freeze-trap as
intended (+26.25, into the «substantial closure» band), confirming the
penalty-differential diagnosis from b39fb937. But it caused a structural
regression on seed 7 (−29.50) and a noticeable hit on seed 27 (−16.25).
Net overall −5.50; not a fix to ship.

## Why seed 7 regressed (the smoking gun)

Per-episode for seed 7:

| ep | Phase A length | Phase A death | post len | post death | Δ |
|----|--------------:|---------------|---------:|-----------|----:|
| 0  | 57            | unknown       | 57       | unknown   | 0   |
| 1  | 175           | zombie        | 209      | zombie    | +34 |
| 2  | 181           | skeleton      | 137      | zombie    | −44 |
| 3  | 209           | zombie        | 101      | **unknown** | **−108** |

ep 3 is the dominant loss. The post-fix run dies of `unknown` cause (i.e.
food/water depletion, not hostile death) at step 101. Last 20 steps before
death are pure movement primitives — no `do`, no `sleep`. The agent wanders,
never eats, never drinks, dies of starvation/dehydration. 34 rescue events
in 101 steps (one third of all steps), so emergency_safety is hyperactive.

Phase A on the same episode survives 209 steps and dies to a zombie — i.e.
the agent actually *acts* (mines, eats, drinks via `do`) under the original
rules. The broadened blocked_h killed productive `do` along with
ineffective `do`.

## Mechanism

The new rule sets `blocked_h=True` whenever `displacement_h==0 AND
resource_gain==0 AND not inventory_delta` regardless of action primitive.
This was supposed to flag only no-op `do` against empty tiles, but it also
flags `do` candidates whose **rollout** under-predicts resource_gain or
inventory_delta. The world model is not always confident enough to commit
to a non-zero resource gain in the one-step horizon, especially for water
drinking, sleep recovery, or partial mining. So `do` candidates that would
be productive in the real env get flagged as ineffective in the rollout
and earn the −3.0 blocked_penalty. Result: agent avoids `do` even when it
would feed/water/mine, and starves.

The penalty is also applied globally, not only under emergency context, so
even outside hostile-pressure situations the agent now systematically
under-prefers `do`. That's a structural bias on the entire trajectory, not
just freeze-trap steps.

## Recommendation (NOT applied yet)

Move the broadened-blocked_h logic from the **label producer** in
`vector_mpc_agent.py:1818–1822` into the **scoring step** in
`stage90r_emergency_controller.py:280–305`, and gate it on actual emergency
context:

```python
# inside select_action, alongside existing blocked/adjacent penalties
ineffective_no_op = (
    effective_displacement == 0.0
    and resource_gain == 0.0
    and not label.get("inventory_delta_h")
    and action != "sleep"
)
in_emergency = bool(adjacent_after or (nearest_h is not None and nearest_h <= 1))
emergency_no_op_penalty = -3.0 if ineffective_no_op and in_emergency else 0.0
```

This keeps the label semantically pure (`blocked_h` = "you tried to move
and could not"), restricts the freeze-prevention to actual hostile-contact
cases, and leaves productive idle `do` (eating, drinking, mining) free of
penalty when there's no threat.

The risk this still leaves open: under emergency with `do` against e.g. a
food source (where eating is genuinely the right move), the rule would
penalise it. Mitigation: also exempt cases where `health_delta_h > 0`
(i.e. the rollout predicts the action restores vitals).

## Next steps (proposed)

1. Apply the gated emergency-only no-op penalty above plus the
   `health_delta_h > 0` exemption.
2. Re-validate canonical multiseed.
3. Specifically check:
   - seed 17 holds at ≥165 (freeze-trap remains closed)
   - seed 7 ep 3 returns to `do`-using productive behavior
   - no new strong-seed regressions

## Artifacts (locally pulled from HyperPC)

- `_docs/stage91_root_cause_forensics_20260502/transfer/blocked_h_broaden_seed_{7,17,27,37,47}_eval.json`
- `_docs/stage91_root_cause_forensics_20260502/transfer/phaseA_seed_7_eval.json`
