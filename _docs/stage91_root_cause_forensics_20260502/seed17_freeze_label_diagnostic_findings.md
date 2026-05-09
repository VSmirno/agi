# Seed-17 Freeze-Trap: Per-Action Utility Decomposition

**Inputs read** (all local copies on `/home/yorick/.../transfer/`):
- `phaseA_seed_17_eval.json` (HEAD with full fix stack, no gate-fix)
- `M3_seed_17_eval.json` (sanity, byte-identical to Phase A on this episode)
- `M2b_seed_17_eval.json` (chase_player ON, no feasibility scoring)
- `gatefix_trial1_seed_17_eval.json` (cross-check)

**Source code referenced (read-only)**:
- `src/snks/agent/vector_mpc_agent.py` — `_build_local_counterfactual_outcomes`, lines 1721–1827
- `src/snks/agent/stage90r_emergency_controller.py` — `EmergencySafetyController.select_action`, lines 244–343

Episode 0, seed 17, terminates step 145 (`death_cause=zombie`). Player camps at `[34, 39]` with health=2 from step ~138 onward; zombie at distance 1 (and arrow at distance 1 from step 141 on). Six consecutive emergency-overridden `do` actions, then death.

## Step 139 — utility decomposition

`local_trace_tail[step=139]`: action=`do`, pos_before=pos_after=`[34, 39]`, body={health:2, food:4, drink:3, energy:7}, `near_concept='empty'`, `nearest_threat_distances={zombie:1, skeleton:4, arrow:1}`.

`counterfactual_outcomes` actions present at step 139: **`[move_left, move_right, move_up, move_down, sleep]` — `do` is ABSENT**.

`ranked_emergency_actions` (top 4 of 6 after sort, from `rescue_trace_tail`):

| action     | survived | damage | health_Δ | escape_Δ | nearest_h | displ | blocked | adj_after | resource | utility | dominant component |
|------------|----------|--------|----------|----------|-----------|-------|---------|-----------|----------|---------|--------------------|
| do         | True     | 0.0    | 0.0      | None     | None      | 0.0   | False   | False     | 0.0      | **+6.00** | DEFAULT label (no rollout produced) — base survival bonus alone |
| sleep      | False    | 2.0    | -2.0     | -1.0     | 0         | 0.0   | False   | True      | 1.0      | -25.85  | -12 (died) -8 (damage) -1.5 (sleep_threat) -2 (adj) -1.75 (escape) |
| move_down  | False    | 2.0    | -2.0     | -1.0     | 0         | 0.0   | True    | True      | 2.0      | -26.95  | -12 -8 -1.75 -2 -3 (blocked) |
| move_left  | False    | 2.0    | -2.0     | -1.0     | 0         | 0.0   | True    | True      | 1.0      | -27.20  | -12 -8 -1.75 -2 -3 (blocked) |

(`move_right`, `move_up` excluded by sort — both worse than these top-4.)

`do` wins by **+33.05** over the runner-up.

## Step 142 — utility decomposition

`local_trace_tail[step=142]`: action=`do`, pos `[34,39]`→`[34,39]`, body unchanged, `near_concept='arrow'`, threats `{zombie:1, skeleton:4, arrow:1}`.

`counterfactual_outcomes` at step 142: **all 6 actions present, including `do`** (`do` label: survived=False, damage=2, blocked=False, adj_after=True).

`ranked_emergency_actions`:

| action     | survived | damage | escape_Δ | nearest_h | displ | blocked | adj_after | resource | utility | dominant component |
|------------|----------|--------|----------|-----------|-------|---------|-----------|----------|---------|--------------------|
| do         | False    | 2.0    | 0.0      | 1         | 0.0   | False   | True      | 0.0      | **-23.00** | base -12, dmg -8, adj -2, no blocked penalty |
| sleep      | False    | 2.0    | 0.0      | 1         | 0.0   | False   | True      | 1.0      | -24.10  | same as do but +sleep_threat penalty -1.5; resource +0.1 |
| move_down  | False    | 2.0    | 0.0      | 1         | 0.0   | True    | True      | 2.0      | -25.20  | adds -3 blocked penalty |
| move_left  | False    | 2.0    | 0.0      | 1         | 0.0   | True    | True      | 1.0      | -25.45  | -3 blocked, -0.25 less resource |

At step 142 `do` still wins, but only by **+1.10** over sleep, **+2.20** over moves — and now via the legitimate channel: `do` is the only non-blocked, non-sleep candidate, so it avoids both `blocked_penalty=-3` (moves) and `sleep_threat_penalty=-1.5` (sleep).

## Per-action label inspection across the freeze window

| step | near_concept | `do` in counterfactual_outcomes? | `do` utility | `do` survived_h | `do` adj_after_h |
|------|--------------|----------------------------------|--------------|-----------------|------------------|
| 139  | empty        | **NO**                            | +6.00 (default) | True (default) | False (default) |
| 140  | empty        | **NO**                            | +6.00 (default) | True (default) | False (default) |
| 141  | arrow        | yes                              | -23.00       | False           | True             |
| 142  | arrow        | yes                              | -23.00       | False           | True             |
| 143  | empty        | **NO**                            | +6.00 (default) | True (default) | False (default) |
| 144  | empty        | **NO**                            | +6.00 (default) | True (default) | False (default) |

For moves at steps 139/140/143/144: `blocked_h=True` and `adj_after_h=True` are real (rollout did run). For `sleep` at all six steps: rollout ran, produced `survived_h=False, damage_h=2, adj_after_h=True`. The asymmetry is **not** in the world model's hostile dynamics — it is that `do` is selectively excluded from rollout when `near_concept ∈ {None, empty, unknown}`.

This confirms the prior attribution doc's claim was **mechanically wrong**: `adjacent_hostile_after_h` is False for `do` not because the rollout produced a different end_hostile, but because **no rollout was performed and the field defaulted to False**.

## Cross-check against M2b (no feasibility scoring, no blocked/adj penalties)

M2b component schema lacks `blocked_h` and `adjacent_hostile_after_h`, but the same defaulting pattern is visible. M2b episode 0 step 174 (terminal):

```
do          util=+6.00  survived=True  damage=0  escape_Δ=None  nearest_h=None  (DEFAULT)
move_right  util=+6.00  survived=True  damage=0  escape_Δ=0     nearest_h=2     (rollout ran)
move_left   util=-24.95 survived=False damage=3  ...
```

The `do` row has the exact `escape_delta=None / nearest_h=None` signature of a never-run rollout. **The defect is upstream of feasibility scoring and predates the adjacent-penalty addition.** It exists wherever `EmergencySafetyController.select_action` consumes counterfactual outcomes that omit `do`.

## Root cause: which mechanism makes `do` win

When the player faces an empty / unknown / out-of-map tile, the world-model rollout for the `do` primitive is **skipped entirely** (no entry produced in `counterfactual_outcomes`). The `EmergencySafetyController` then scores `do` from a missing label, where every safety field defaults to its "best case":

```
survived_h=True   damage_h=0.0   blocked_h=False
adjacent_hostile_after_h=False   escape_delta_h=None (treated as 0.0)
```

Plugged into the utility formula, this gives `(+6.0) − 0 + 0 + … = +6.00` exactly — a phantom safe action that sits ~30 points above any rollout result that includes a one-hit-kill from the adjacent zombie.

The agent then commits to `do`, the env executes a no-op `do` against an empty tile, the zombie advances and hits, and the next planning tick repeats — a 6-step camp ending in death.

## Specific code lines implicated

1. **`src/snks/agent/vector_mpc_agent.py:1746–1749`** — silent skip of the `do` candidate:
   ```python
   if primitive == "do":
       near_concept = str(vf.near_concept)
       if near_concept in {"None", "empty", "unknown"}:
           continue   # ← outcome never appended
       target = near_concept
   ```
   Skipping is the upstream defect. A skipped candidate is **not** equivalent to "do is unsafe"; downstream code interprets absence as best-case.

2. **`src/snks/agent/stage90r_emergency_controller.py:272–282`** — defaults treat a missing label as a survival-grade outcome:
   ```python
   label = dict(outcome_by_action.get(action, {}).get("label", {}))
   survived = bool(label.get("survived_h", True))   # ← default True
   damage = float(label.get("damage_h", 0.0))
   blocked = bool(label.get("blocked_h", False))
   adjacent_after = bool(label.get("adjacent_hostile_after_h", False))
   ```
   These two defects compose: the rollout is silently skipped, and the scorer then treats "no data" as "best-case survival".

3. (Tangentially) `vector_mpc_agent.py:1818–1823` — `blocked_h` and `adjacent_hostile_after_h` are only computed inside this builder; nothing downstream re-derives them, so any candidate without an outcome bypasses both penalties unconditionally. The earlier gate-fix attempt (`f6ad2b2`) tried to enforce adjacent_penalty more strongly but, because `do` carries `adj_after=False` from the default branch (never from a rollout), the gate's wrong sign reflected the wrong mental model — even with a correct sign, gating on `adj_after` cannot reach `do` while the rollout itself is missing.

## Recommended fix surface (NOT applied)

Pick the minimal one. Order of preference:

1. **Always run a `do` rollout, even on empty tiles.** Drop the `continue` at 1746–1749. Use `target="self"` (or a sentinel) when `near_concept` is empty/None/unknown; let `simulate_forward` execute a real `do` no-op, advance hostile entities, and produce a label with `damage_h≥0`, `adjacent_hostile_after_h` reflecting true zombie chase, `survived_h` reflecting actual one-hit-kill outcomes. This is a one-line semantic change and matches what `sleep` already does. It removes the phantom-safe artifact at its source.

2. **Treat missing candidates as inadmissible, not best-case.** In `select_action` (272–282), if `outcome_by_action.get(action) is None`, either skip the action from `ranked` entirely or assign a "no_rollout_penalty" floor (e.g., utility = -∞ / large negative). This is defensive and useful regardless of (1), but on its own it would make `do` flatly unavailable on empty-facing turns — losing the legitimate "stand and absorb a hit" move at step 142 — so prefer (1) as the primary fix.

3. **Both (1) and (2)**, with (2) acting as a backstop for any future candidate that may legitimately have to skip its rollout.

Do not touch the `adjacent_penalty` gate sign; it is downstream of the actual asymmetry and tuning it cannot fix the missing-rollout root cause.

## Limitations

None for the diagnostic itself — Phase A artifact carries `ranked_emergency_actions` with full `components`, `local_trace_tail` carries `counterfactual_outcomes` with per-action labels, and the source code matches. M2b's component schema is shallower (no `blocked_h` / `adjacent_hostile_after_h`) but still preserves the missing-rollout fingerprint via `escape_delta_h=None / nearest_hostile_h=None`. No additional eval dump is required to settle the question.
