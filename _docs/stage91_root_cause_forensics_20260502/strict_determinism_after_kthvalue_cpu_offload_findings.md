# Strict Determinism After kthvalue CPU Offload — Findings

**Probe timestamp:** 2026-05-09T14:57:26Z
**Host:** HyperPC (cuda@192.168.98.56), RTX 3090
**Interpreter:** `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python` (CPython 3.11.15, torch 2.5.1+cu121, cuDNN 9.1.0, CUDA 12.1)
**Patch under test:** `src/snks/agent/vector_world_model.py:143` —
`dists_flat.cpu().kthvalue(target_pct_idx).values.item()` (one-shot CPU offload in
`CausalSDM._calibrate_radius`).

## Artifact paths

- **Remote checkout (isolated, hardlink-cloned from feasibility checkout, with patched file overlaid):**
  `/opt/cuda/agi-stage91-strict-determinism-kthvalue-cpu-20260509T145726Z`
- **Remote artifact root:**
  `/opt/cuda/agi-stage91-strict-determinism-kthvalue-cpu-20260509T145726Z/_docs/hyper_stage91_strict_determinism_kthvalue_cpu_20260509T145726Z`
- **Wrapper:** `…/commands/deterministic_wrapper.py` (verbatim copy of the prior probe wrapper
  at `…hyper_stage91_seed7_deterministic_backend_probe_20260509T141402Z/commands/deterministic_wrapper.py`)
- **Per-trial:** `…/trial_{1,2,3}/raw/seed_7_eval.json`, `…/trial_{N}/logs/seed_7.log`,
  `…/trial_{N}/commands/seed_7_trial_{N}.cmd`, `…/trial_{N}/env/active_runtime_flags.json`,
  `…/trial_{N}/env/exit_code.txt`
- **Preflight:** `…/env/preflight_strict_runtime_flags.json`,
  `…/env/preflight_strict_stdout.txt`, `…/env/preflight_strict_stderr.txt`

Patch verification:
```
$ grep -n 'dists_flat.cpu().kthvalue' .../src/snks/agent/vector_world_model.py
143:        radius = int(dists_flat.cpu().kthvalue(target_pct_idx).values.item())
```

## Active strict runtime flags (preflight + each trial)

| Flag | Value |
|---|---|
| `mode` | strict |
| `deterministic_algorithms_enabled` | **true** |
| `deterministic_algorithms_warn_only` | false |
| `cudnn_enabled` | true |
| `cudnn_deterministic` | true |
| `cudnn_benchmark` | false |
| `allow_tf32_matmul` | false |
| `allow_tf32_cudnn` | false |
| `float32_matmul_precision` | highest |
| `cuda_device_name` | NVIDIA GeForce RTX 3090 |
| `cudnn_version` | 90100 |
| `torch_version` | 2.5.1+cu121 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| `PYTHONHASHSEED` | `0` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `PYTHONPATH` | `…/src:…:…/experiments` |

Identical snapshot in `preflight_strict_runtime_flags.json` and in every
`trial_{N}/env/active_runtime_flags.json`.

## Per-trial results (seed 7, n_episodes=4, max_steps=220, smoke-lite, mixed_control_rescue, planner-rescue ON)

All three trials exited with code **0** under strict
`torch.use_deterministic_algorithms(True)`. **No new blocker error surfaced.**

### Aggregate

| Trial | avg_survival | rescue_rate | planner_dependence | learner_control_fraction |
|---|---|---|---|---|
| 1 | **157.00** | 0.495 | 0.417 | 0.185 |
| 2 | **212.00** | 0.447 | 0.454 | 0.206 |
| 3 | **171.25** | 0.463 | 0.426 | 0.206 |

### Per-episode (steps / death_cause / first_rescue / last_rescue / n_rescue)

| ep | Trial 1 | Trial 2 | Trial 3 |
|---|---|---|---|
| 0 | 57 / unknown / —/—/0 | 220 / dehydration / 33/219/51 | 57 / unknown / —/—/0 |
| 1 | 220 / dehydration / 33/219/165 | 220 / skeleton / 33/219/171 | 220 / dehydration / 33/219/165 |
| 2 | 163 / zombie / 122/162/41 | 220 / zombie / 149/219/52 | 220 / zombie / 149/219/47 |
| 3 | 188 / zombie / 9/187/105 | 188 / zombie / 9/187/105 | 188 / zombie / 9/187/105 |

### Death-cause breakdown

| Trial | unknown | dehydration | skeleton | zombie |
|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 2 |
| 2 | 0 | 1 | 1 | 2 |
| 3 | 1 | 1 | 0 | 2 |

## Spread

- **Strict-determinism (this probe):** min=157.00, max=212.00, **spread = 55.00**, mean=180.08
- Repeatability (no determinism flags): 164.5 / 189.5 / 164.25 → spread **25.25**
- Relaxed-deterministic-backend (cudnn det + no TF32, `use_deterministic_algorithms(False)`):
  204.25 / 157.0 → spread **47.25**

**Strict mode did NOT collapse the spread; if anything it widened it (55.00 ≥ 47.25 ≥ 25.25).**
The two-sample comparison is small but the conclusion is qualitative: enabling strict CUDA
determinism does not stabilise seed-7 avg_survival.

## Determinism is partial, not absent

Episode 3 (`episode_id=3`) is **byte-identical across all three strict trials**:
length 188, zombie, first_rescue=9, last_rescue=187, n_rescue=105. Trial-1 and Trial-3
also agree on episodes 0 and 1 byte-for-byte. So the seeded RNGs *do* produce
reproducible behaviour for some rollouts — it is not raw GPU jitter on every step.

The divergence is concentrated in **episodes 0 and 2**:

- **ep0** alternates between a 57-step early death (trials 1, 3) and a full 220-step run
  with 51 planner rescues (trial 2) — a binary outcome flip, not a numerical drift.
- **ep2** ranges from 163 to 220 steps with different rescue trigger counts.

This is the signature of a **non-seeded stochastic source that mutates its state
across episode resets but is occasionally identical between runs** (e.g. process
start-time entropy, an unseeded `random` / `numpy` /env-side RNG, dataloader-worker
hashing, async timing, or a per-call CUDA workspace allocator effect that is *not*
covered by `torch.use_deterministic_algorithms(True)` — e.g. a deterministic op that
is still order-sensitive across kernel launches under different concurrent state).

A binary "early death vs. full episode" flip on ep0 between otherwise-deterministic
runs strongly suggests **a few critical action draws diverge near a decision
boundary**, not pervasive numerical noise.

## Next-blocker

**None at the kthvalue level.** The CPU-offload patch fully unblocks
`torch.use_deterministic_algorithms(True)` for the Stage91 mixed_control_rescue eval
stack on this checkout — strict trials run end-to-end with no
`RuntimeError: ... does not have a deterministic implementation`.

`strict_failure_capture/` is empty. Logs per trial are 5 lines each (per-episode
summary only) — no warnings, no errors.

## Bottom line

1. **Fix verified.** `dists_flat.cpu().kthvalue(...)` makes Stage91 eval compatible
   with `torch.use_deterministic_algorithms(True)` on torch 2.5.1+cu121 / RTX 3090.
   No other deterministic-algorithm assertions fired in 3 strict trials × 4 episodes
   × 220 steps.
2. **Spread did not collapse** — it actually grew to 55.0 (vs 25.25 raw / 47.25
   relaxed-deterministic). CUDA backend nondeterminism is **not** the source of the
   seed-7 avg_survival instability.
3. **Partial determinism observed** — episode 3 is bit-identical across all three
   strict trials; divergence is localised to episodes 0 and 2 and behaves like a
   small-number-of-decisions flip near a boundary, not pervasive jitter.
4. **Next probe direction:** the remaining stochastic source is upstream of the
   CUDA kernels. Candidates to chase next, in order of likelihood:
   - Python `random` / `numpy` global RNG not seeded (or reseeded with non-fixed entropy
     per episode/reset) inside the env or planner-rescue path.
   - Dataloader / worker hashing not pinned (PyTorch worker `generator=` and
     `worker_init_fn` for any `DataLoader` used in eval).
   - Time-/PID-based seeds inside emergency_safety or planner_bootstrap fallbacks.
   - Hash-randomised set/dict iteration affecting tie-break ordering when scoring
     candidate actions (PYTHONHASHSEED=0 should cover this — verify it propagates to
     all subprocesses; we saw it set in the wrapper's parent only).

   Concrete next step: bisect the divergence by adding deterministic checkpoints
   (hash of observation, hash of policy logits, hash of chosen action) at
   step granularity for ep0 between trial-1 and trial-2; the first divergent step
   identifies the offending RNG site.
