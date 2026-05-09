# Adjacent-Penalty Gate-Fix Validation — Findings

Run: `agi-stage91-adjacent-penalty-gate-20260509T195137Z`
Artifact root: `/opt/cuda/agi-stage91-adjacent-penalty-gate-20260509T195137Z/_docs/hyper_adjacent_penalty_gate_validation_20260509T195137Z/trial_1/`
Date: 2026-05-09 (run) / 2026-05-10 (write-up)

## Patch summary

`src/snks/agent/stage90r_emergency_controller.py:290`

Before:
```python
adjacent_penalty = -2.0 if adjacent_after else 0.0
```
After:
```python
adjacent_penalty = -2.0 if adjacent_after and effective_displacement > 0.0 else 0.0
```

Three required patches verified present in the run dir:
- `dists_flat.cpu().kthvalue` in `vector_world_model.py:143`
- `_install_crafter_determinism_patch` in `crafter_pixel_env.py:46,65`
- `adjacent_after and effective_displacement > 0.0` in `stage90r_emergency_controller.py:290`

## Active runtime flags (per seed, identical)

- mode: `strict`
- python: `3.11.15`, torch `2.5.1+cu121`, CUDA `12.1`, cuDNN `90100`
- device: RTX 3090, `CUDA_VISIBLE_DEVICES=0`
- `cudnn.deterministic=True`, `cudnn.benchmark=False`, TF32 disabled both paths
- `torch.use_deterministic_algorithms(True)`, `float32_matmul_precision='highest'`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`

Captured at `trial_1/env/active_runtime_flags_seed_<S>.json`.

## Per-seed table (avg_survival, 4 episodes × 220 max-steps)

| seed | This run (gate-fix) | Phase A | Phase B | M2 (seed 17 only) | Δ vs Phase A |
|------|---------------------|---------|---------|-------------------|--------------|
| 7    | 155.50              | 155.50  | —       | —                 | 0.00         |
| 17   | **141.50**          | 141.50  | 150.50  | 174.75            | **0.00**     |
| 27   | 209.75              | 208.25  | —       | —                 | +1.50        |
| 37   | 201.25              | 210.75  | —       | —                 | **−9.50**    |
| 47   | 194.75              | 203.00  | —       | —                 | **−8.25**    |

Aggregates:
- weak (7+17) = **148.50** (Phase A 148.50, Δ 0.00)
- strong (27+37+47) = **201.92** (Phase A 207.33, Δ −5.42)
- overall = **180.55** (Phase A 183.70, Δ −3.15)

## Seed-17 ep0 terminal trace — before vs after

Phase A (per attribution task 22384772): agent at health=2, zombie 1 tile away,
6 consecutive `do` selections; freeze trap.

This run, ep0 (length 145, death = zombie). From `death_trace_bundle.recent_steps`:

```
step prim  blk    pos          h_before h_after  zd
130  do    False  [34,39]      9.0      9.0      –
131  do    False  [34,39]      9.0      9.0      –
132  do    False  [34,39]      9.0      9.0      –
133  do    False  [34,39]      9.0      9.0      –
134  do    False  [34,39]      9.0      9.0      –
135  do    False  [34,39]      9.0      9.0      –
136  do    False  [34,39]      9.0      9.0      –
137  do    False  [34,39]      9.0      9.0      3      ← zombie enters detection radius
138  do    False  [34,39]      9.0      2.0      2      ← first hit (−7 hp)
139  do    False  [34,39]      2.0      2.0      1
140  do    False  [34,39]      2.0      2.0      1
141  do    False  [34,39]      2.0      2.0      1
142  do    False  [34,39]      2.0      2.0      1
143  do    False  [34,39]      2.0      2.0      1
144  do    False  [34,39]      2.0      0.0      1      ← death
```

The seed-17 ep0 trajectory is **byte-identical** to Phase A's frozen-`do` trap:
agent never leaves [34,39], picks 15 consecutive `do` from step 130 onward, takes
all damage standing still, dies at step 144. The gate fix did not perturb the
controller's choice in this episode.

The same freeze pattern now also appears in seed 37 ep0 (12+ consecutive `do` at
[27,35] with zombie at distance 1, terminal frames identical structurally) — a
trap that was previously absent from seed 37 under Phase A.

## Verdict

**Outcome band: gate fix did NOT help (seed 17 < 150) AND it over-relaxes the
safety preference (seed 37: −9.50, seed 47: −8.25; both regress by > 5).**

Two failure conditions from the brief triggered simultaneously:

1. Seed 17 sits at exactly 141.50 — bit-identical to Phase A. The `do`-spam
   freeze trap is unchanged.
2. Seeds 37 and 47 lost 9.50 and 8.25 respectively. Strong-seed mean dropped
   from 207.33 → 201.92.

Why the fix failed to dislodge seed 17: the original hypothesis was that moves
were unfairly penalised against `do`. After the gate, *moves that succeed*
still earn −2.0 (they have `effective_displacement > 0` and remain adjacent
under chase), while `do` (zero displacement) earns 0. This actually *widens*
the gap in favour of staying put — the opposite of what was intended. The
seed-17 trace confirms: zero perturbation. The strong-seed regressions are
consistent: episodes that previously moved out cleanly are now nudged toward
the same `do`-on-the-spot pattern (visible in seed 37 ep0).

## Bottom line + recommended next step

Bottom line: **the gate fix is wrong-signed for the freeze trap.** It removes
penalty from the *winning* (frozen) action and keeps it on the *losing*
(escape) action, so it can only entrench staying. Seed 17 measured zero change;
seeds 37/47 measured net negative. Do not ship.

Recommended next probe (diagnosis only, do not patch yet):

- Re-examine the seed-17 attribution log: confirm the actual per-action utility
  decomposition pre-fix. If `do` truly earned 0 pre-fix while moves earned −2,
  the issue is that `adjacent_hostile_after_h` is computed differently for
  `do` than for moves (maybe via different world-model rollouts). The right
  fix is then to align the label semantics, not to gate the penalty on
  displacement.
- Alternative direction: penalise *staying adjacent for ≥N consecutive frames*
  (a frozen-do detector) and/or add a positive bonus for the action that
  reduces `nearest_hostile_dist` more than `do`. Both move the gradient
  toward escape rather than removing it from stays.
- Specifically diagnose seed 37 ep0 first-divergent step against Phase A's
  seed 37 ep0 trace to confirm the regression entry-point. Phase A's seed 37
  trace is in the verify-71d1e29 / Phase A artifact set; under the gate fix
  the divergence should appear at the first step where pre-fix preferred a
  move (penalty −2 in both branches → tie broken by displacement_bonus) and
  post-fix prefers `do` (move now uniquely penalised).

## Artifacts (remote)

- raw evals: `…/trial_1/raw/seed_{7,17,27,37,47}_eval.json`
- runtime flags: `…/trial_1/env/active_runtime_flags_seed_*.json`
- logs: `…/trial_1/logs/seed_*.log`
- commands: `…/trial_1/commands/seed_*_trial_1.cmd`
- wrapper: `…/commands/deterministic_wrapper.py`
