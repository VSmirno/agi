# Stage91 Determinism Root Cause and Fix — Findings

**Probe timestamp:** 2026-05-09T16:03:13Z
**Host:** HyperPC (cuda@192.168.98.56), RTX 3090
**Interpreter:** `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python` (CPython 3.11.15, torch 2.5.1+cu121, cuDNN 9.1.0, CUDA 12.1)
**Local checkout:** `/home/yorick/agi-stage90r-world-model-guardrails` (branch `stage90r-world-model-guardrails`)

## TL;DR (Bottom line)

Stage91 mixed_control_rescue eval was nondeterministic across runs *not*
because of a CUDA backend leak (already closed by the kthvalue CPU offload)
but because the upstream **`crafter` environment iterates a `set` of object
instances inside `Env._balance_chunk` / `Env._balance_object`**. Set
iteration order over Python objects depends on `id(obj)` (memory address),
which `PYTHONHASHSEED` does **not** cover and which differs across
processes — so the same env seed could despawn a different zombie/skeleton/
cow on the first balance tick of a process, propagating into a completely
different observation stream a few steps later.

**Fix:** monkey-patch `crafter.env.Env._balance_chunk` from inside our
`CrafterPixelEnv` wrapper to receive `objs` sorted by `(pos.x, pos.y,
type_name)`. One-shot, idempotent, applied at module import.

**Validation:** under strict `torch.use_deterministic_algorithms(True)`,
seed-7 mixed_control_rescue eval (4 episodes × 220 steps) is now
**byte-identical across 3 consecutive trials** (avg_survival = 155.5
exactly, per-episode lengths/death_causes/rescue counts match
character-for-character). Seeds 17 and 27 each reproduce identically across
2 trials. 68/68 unit tests pass on HyperPC under the patched checkout.

## Static audit results

`grep` across `src/snks/agent/`, `experiments/stage90*`, dependencies of
the `mixed_control_rescue` path for: stdlib `random`, `numpy.random`,
`torch.rand*`/`torch.Generator`, `time.time`/`monotonic`/`getpid`,
`os.urandom`, untracked `set`/`dict` iteration in tie-break code, and
unseeded `DataLoader`s.

Findings:

- All `np.random.RandomState` / `torch.Generator` instances on the eval
  path are seeded explicitly:
  - `experiments/stage90r_eval_local_policy.py:54` `_eval_episode_rng(base_seed, episode_index)` → `np.random.RandomState(seed+ep)`
  - `src/snks/agent/vector_world_model.py:104,280` torch.Generator with `manual_seed(seed)`
  - `src/snks/agent/crafter_pixel_env.py:75` passes `seed=seed` into `crafter.Env`
- No `time.time()` / `os.getpid()` / `os.urandom()` seeding inside the
  episode loop or any controller (`vector_mpc_agent`, `vector_sim`,
  `stage90r_emergency_controller`, `vector_bootstrap`).
- `EmergencySafetyController.select_action` ranks candidates by stable
  `list.sort(key=...)` over a fixed-order `allowed` list — deterministic
  given inputs.
- The unseeded `np.random.RandomState()` fallback at
  `vector_mpc_agent.py:635` is dead in this path because the eval always
  passes `rng=_eval_episode_rng(...)`.
- No `DataLoader` is constructed during eval.

This left the env-side as the only plausible suspect; static audit alone
did not isolate it because the `set` is inside the third-party `crafter`
package (`/opt/cuda/miniforge3/envs/agi-stage90r-py311/lib/python3.11/
site-packages/crafter/`), not our repo.

## Dynamic hash-bisect

Isolated bisect checkout:
`/opt/cuda/agi-stage91-determinism-bisect-20260509T155656Z` (hardlink
clone of feasibility checkout, with the kthvalue CPU-offload patch
overlaid).

Patched `src/snks/agent/vector_mpc_agent.py` to emit a per-step JSONL
record (env-gated by `STAGE91_HASH_LOG`) containing `step`, `primitive`,
`control_origin`, `planner_primitive`, `learner_action`, `player_pos`,
SHA1 digests of `body`, `inv`, sorted `vf.visible_concepts()`, plus
`pred_best_loss`, `pred_baseline_loss`, `nearest_threats`,
`actor_non_progress_streak`, `rescue_trigger`. Insertion site: just
after `action_counts[primitive] += 1` (fully formed step state, before
`env.step`).

Two strict-determinism trials of seed 7 ep0 (n_episodes=1):

```
trial 1: 220 steps, dehydration, 51 rescues
trial 2:  57 steps, unknown,      0 rescues
```

First divergence: **step index 12** — same `player_pos=[27,35]`, same
`body`, same `inv`, identical primitives through step 11, but:

| field | trial 1 | trial 2 |
|---|---|---|
| `vis_h` | `03c4c38c023c` | `27bb630bfe63` |
| `nearest_threats[zombie]` | `3` | `None` |

i.e. by the time the env returned the obs after step 11, the `semantic`
field already differed — a zombie was present near the player in trial 1
and absent in trial 2. Player state, inventory, and action history were
identical → divergence is environment-internal.

## Identified source

`crafter/engine.py:36` declares `self._chunks = collections.defaultdict(set)`
(per-chunk object containers). `crafter/env.py:88-95` calls

```python
if self._step % 10 == 0:
    for chunk, objs in self._world.chunks.items():
        self._balance_chunk(chunk, objs)
```

and `crafter/env.py:158-178` `_balance_object` does

```python
random = self._world.random
creatures = [obj for obj in objs if isinstance(obj, cls)]
...
elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
    obj = creatures[random.randint(0, len(creatures))]
    ...
    self._world.remove(obj)
```

`creatures` order is the iteration order of the chunk's `set` of object
instances. Python uses `id(obj) // 16` for the hash of an instance with
the default `__hash__`; `PYTHONHASHSEED` only fixes hashing of `str`,
`bytes`, and a few other built-ins — it does **not** affect identity
hashing. Memory addresses come from CPython's malloc+free-list reuse
patterns, which differ across processes regardless of seeds. So the
*same* `random.randint` index picks a *different* zombie/skeleton/cow on
the first balance tick, producing a divergent world state from step ~10
onward — exactly matching the observed divergence at step 12 (the first
balance tick is step 10; the resulting world state surfaces in the next
observation).

This is consistent with the original probe's observation that ep3 was
byte-identical across trials (no balance tick happened to alter creature
counts in a way that changed eventual outcome) while ep0/ep2 flipped
(one despawn picked differently was sufficient to swap a kill outcome
with a survive outcome).

## Patch description

**File:** `src/snks/agent/crafter_pixel_env.py`
(local: `/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/crafter_pixel_env.py`)

Added a module-level idempotent monkey-patch right after the `crafter`
import block:

```python
_CRAFTER_DETERMINISM_PATCHED = False

def _install_crafter_determinism_patch() -> None:
    global _CRAFTER_DETERMINISM_PATCHED
    if _CRAFTER_DETERMINISM_PATCHED or not HAS_CRAFTER:
        return
    import crafter.env as _crafter_env
    _orig_balance_chunk = _crafter_env.Env._balance_chunk

    def _deterministic_balance_chunk(self, chunk, objs):
        sorted_objs = sorted(
            objs,
            key=lambda o: (int(o.pos[0]), int(o.pos[1]), type(o).__name__),
        )
        return _orig_balance_chunk(self, chunk, sorted_objs)

    _crafter_env.Env._balance_chunk = _deterministic_balance_chunk
    _CRAFTER_DETERMINISM_PATCHED = True

_install_crafter_determinism_patch()
```

**Properties:**
- Smallest blast radius: changes only the iteration order of `objs`
  inside the despawn pick (and the spawn target-count check, which is
  `len(creatures)` — order-invariant). The RNG draws themselves are
  untouched.
- Single seed yields identical world dynamics as before *for any one
  process* — `sorted(...)` is a stable, total order keyed on world
  geometry. So the patch preserves single-seeded semantics to the same
  extent that any one prior process did, while collapsing cross-process
  variance.
- Idempotent (`_CRAFTER_DETERMINISM_PATCHED` flag) and safe in absence
  of `crafter` (`HAS_CRAFTER` guard).

## Validation results

Validation checkout:
`/opt/cuda/agi-stage91-determinism-validation-20260509T160313Z`
Artifact root:
`/opt/cuda/agi-stage91-determinism-validation-20260509T160313Z/_docs/hyper_stage91_determinism_validation_20260509T160313Z`

Strict-determinism wrapper (verbatim from prior probe; sets
`use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, no TF32, `float32_matmul_precision='highest'`,
env `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`).

Canonical command per trial: `experiments/stage90r_eval_local_policy.py
--mode mixed_control_rescue --n-episodes 4 --max-steps 220 --seed S
--perception-mode symbolic --enable-planner-rescue --smoke-lite
--record-death-bundle --terminal-trace-steps 32
--max-explanations-per-episode 32 --local-evaluator
/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt`.

### Seed 7 (3 trials, target seed)

| Trial | avg_survival | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|---|
| 1 | **155.50** | 57 / unknown / 0 | 175 / zombie / 119 | 181 / skeleton / 71 | 209 / zombie / 118 |
| 2 | **155.50** | 57 / unknown / 0 | 175 / zombie / 119 | 181 / skeleton / 71 | 209 / zombie / 118 |
| 3 | **155.50** | 57 / unknown / 0 | 175 / zombie / 119 | 181 / skeleton / 71 | 209 / zombie / 118 |

**Spread = 0.0** (was 55.0 before patch). Every per-episode field
matches across trials.

### Bisect verification (n_episodes=1, seed=7, with hash logging)

After patch, three consecutive bisect runs produced byte-identical
JSONL hash logs:

```
fc12be85aa35d8c4794c250e6c9ef1c1  trial_1.jsonl
fc12be85aa35d8c4794c250e6c9ef1c1  trial_2.jsonl
fc12be85aa35d8c4794c250e6c9ef1c1  trial_3.jsonl
```

(Per-step `vis_h`/`nearest_threats` now stable; before patch, divergent
at step idx 12.)

### Generalisation: seed 17 (2 trials)

| Trial | avg_survival | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|---|
| 1 | **141.50** | 145 / zombie / 86 | 220 / dehydration / 104 | 162 / zombie / 33 | 39 / unknown / 0 |
| 2 | **141.50** | 145 / zombie / 86 | 220 / dehydration / 104 | 162 / zombie / 33 | 39 / unknown / 0 |

### Generalisation: seed 27 (2 trials)

| Trial | avg_survival | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|---|
| 1 | **208.25** | 173 / zombie / 31 | 220 / arrow / 148 | 220 / zombie / 25 | 220 / alive / 20 |
| 2 | **208.25** | 173 / zombie / 31 | 220 / arrow / 148 | 220 / zombie / 25 | 220 / alive / 20 |

### Unit tests

On HyperPC, with explicit `PYTHONPATH` to the validation checkout:

```
tests/test_crafter_pixel_env_67.py            16 passed
tests/test_vector_sim.py                      23 passed
tests/test_stage90r_emergency_controller.py    3 passed
tests/test_stage63_crafter.py                 21 passed
tests/test_vector_mpc_agent.py                 5 passed
============================== 68 passed in 76.59s
```

## Artifact paths

- **Local patched file:**
  `src/snks/agent/crafter_pixel_env.py` (lines 21-65 added)
- **Bisect checkout (with hash-log instrumentation in vector_mpc_agent.py):**
  `/opt/cuda/agi-stage91-determinism-bisect-20260509T155656Z`
  - Bisect logs and one-episode evals: `_docs/bisect/trial_{1,2,3}.jsonl`,
    `_docs/bisect/trial_{1,2,3}_eval.json`,
    `_docs/bisect/deterministic_wrapper.py`
- **Validation checkout (clean: only kthvalue + crafter-determinism patches):**
  `/opt/cuda/agi-stage91-determinism-validation-20260509T160313Z`
  - Artifacts: `_docs/hyper_stage91_determinism_validation_20260509T160313Z/`
    - `seed_{7,17,27}/trial_{N}/raw/seed_{S}_eval.json`
    - `seed_{7,17,27}/trial_{N}/logs/seed_{S}.log`
    - `seed_{7,17,27}/trial_{N}/commands/seed_{S}_trial_{N}.cmd`
    - `seed_{7,17,27}/trial_{N}/env/active_runtime_flags.json`
    - `seed_{7,17,27}/trial_{N}/env/exit_code.txt`
    - `commands/deterministic_wrapper.py`

Patch verification on the validation checkout:

```
$ grep -n '_install_crafter_determinism_patch\|_deterministic_balance_chunk' \
    /opt/cuda/agi-stage91-determinism-validation-20260509T160313Z/src/snks/agent/crafter_pixel_env.py
46:def _install_crafter_determinism_patch() -> None:
54:    def _deterministic_balance_chunk(self, chunk, objs):
61:    _crafter_env.Env._balance_chunk = _deterministic_balance_chunk
65:_install_crafter_determinism_patch()
```

## Bottom line

Strict-determinism for Stage91 mixed_control_rescue eval is now closed
end-to-end. The two-layer fix is:

1. `vector_world_model.py:143` — `dists_flat.cpu().kthvalue(...)` (prior
   patch; unblocks `torch.use_deterministic_algorithms(True)`).
2. `crafter_pixel_env.py:21-65` — `_install_crafter_determinism_patch()`
   (this patch; eliminates env-side set-iteration nondeterminism).

With both applied, seed 7 produces avg_survival=155.50 across 3
consecutive strict trials (per-episode byte-identical), and the same
behaviour generalises across seeds 17 and 27.
