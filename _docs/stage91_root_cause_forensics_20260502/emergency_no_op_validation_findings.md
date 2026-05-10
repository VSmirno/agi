# Emergency-Gated No-Op Penalty Validation — Findings

Run: `agi-stage91-emergency-no-op-20260510T014154Z`
Artifact root (HyperPC): `/opt/cuda/agi-stage91-emergency-no-op-20260510T014154Z/_docs/emergency_no_op/`
Date: 2026-05-10 (eval; subagent task 3e30276c failed at 80s on API quota, so eval was driven directly via ssh)

## Patches under test (all reverted after validation)

1. `vector_mpc_agent.py:1746–1752` — null-target `do` no longer skipped.
2. `stage90r_emergency_controller.py:271–281` — backstop: missing-outcome
   actions excluded.
3. `stage90r_emergency_controller.py:289–308` — emergency-gated
   ineffective-action penalty −3.0, applied only when
   `nearest_hostile_now ≤ 2` AND action is genuinely no-op
   (no displacement, no resource gain, no inventory delta, no positive
   health delta, not already-blocked, not sleep).

11/11 controller + vector_mpc tests passed pre-validation.

## Per-seed results

| seed | this | Phase A | Δ_A | broaden (af5d579c) | Δ_broaden |
|------|-----:|--------:|----:|-------------------:|----------:|
| 7    | 140.50 | 155.50 | −15.00 | 126.00 | +14.50 |
| 17   | 141.75 | 141.50 | +0.25 | 167.75 | −26.00 |
| 27   | 181.25 | 208.25 | **−27.00** | 192.00 | −10.75 |
| 37   | 196.75 | 210.75 | −14.00 | 208.25 | −11.50 |
| 47   | 159.25 | 203.00 | **−43.75** | 197.50 | −38.25 |

Aggregates:
- weak (7+17): 141.12 vs Phase A 148.50 (Δ −7.38)
- strong (27+37+47): 179.08 vs Phase A 207.33 (Δ **−28.25**)
- overall: 163.90 vs Phase A 183.80 (Δ **−19.90**)

## Verdict: NET-NEGATIVE — worse than even the broaden attempt

Outcome bands triggered:
- Seed 17 < 160 (freeze trap NOT closed; +0.25 vs Phase A is noise-band).
- Three strong seeds regressed by >5 (27 by 27, 37 by 14, 47 by 44).

The gating on hostile contact (≤2 tiles) was supposed to make the rule
more conservative than the broaden attempt. Instead the result was worse
on strong seeds and didn't close seed 17. Two structural issues:

1. **Seed 17 ep 0 changed shape but didn't extend.** It's now 184 steps,
   skeleton death (vs 57 steps `do` freeze in Phase A) — meaning the
   freeze trap at ep 0 itself broke. But the run lost on episodes 1–3,
   netting +0.25.

2. **Strong seeds destabilised.** Seed 47 is the worst at −43.75. This
   isn't a freeze trap — strong seeds don't have the do-spam pattern.
   The penalty is firing in non-emergency-trap contexts (chase
   encounters where `do` is genuinely a no-op for one tick, e.g.
   when player is mid-flee and happens to face nothing).

## Iteration history on this surface

| attempt | mechanism | seed 17 | overall vs A |
|---------|-----------|--------:|-------------:|
| Phase A baseline | (no fix) | 141.50 | 0 |
| 1. gate adjacent_penalty on displacement | local rule, wrong-signed | 141.50 | −3.15 |
| 2. drop continue + backstop | label producer | 150.25 | −14.15 |
| 3. broaden blocked_h | label producer, always | 167.75 | −5.50 |
| 4. emergency-gated no-op (this) | controller scoring, gated | 141.75 | **−19.90** |

Three iterations have all been net-negative or worse. Each fix correctly
addressed the immediately-observed bug, but the next round of validation
exposed a different failure mode. The freeze trap is not a single point
bug — it's an emergent equilibrium of multiple interacting components
(world-model rollout fidelity, label semantics, emergency controller
ranking, planner/learner mix), and surgical local fixes keep shifting
the equilibrium in unpredictable ways.

## Stop and step back

Recommendation: stop iterating on label semantics / scoring tweaks for
seed 17. The seed-17 regression is reproducible, but every local fix
creates regressions elsewhere. Three options for the user:

A. **Accept the current Phase A baseline** as the validated stage-91
   plateau (overall=183.80, weak=148.50, strong=207.33) and treat the
   −9 cost on seed 17 as a known limitation. Move on.

B. **Reconsider whether the EmergencySafetyController should activate
   at all** in the freeze-trap pattern. The trap occurs because emergency
   takes over and ranks `do` highest; if emergency activation is too
   eager when the agent has no good options anyway, maybe the right
   move is to NOT take over. This is a strategic question, not a
   scoring tweak.

C. **Improve the world-model rollout fidelity** so that `do/None`
   honestly predicts adjacent_after=True under chase, with damage. Then
   ranking would naturally prefer moves. But this requires deeper
   changes to vector_sim's hostile-action interaction.

## Artifacts

- Locally pulled JSONs:
  `_docs/stage91_root_cause_forensics_20260502/transfer/emergency_no_op_seed_{7,17,27,37,47}_eval.json`
- Remote: `/opt/cuda/agi-stage91-emergency-no-op-20260510T014154Z/`
