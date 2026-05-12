# Outcome-Role Extension to VectorWorldModel — PCCS Step 1

**Date:** 2026-05-12
**Status:** Design (pre-implementation, supersedes the reverted episodic-substrate design)
**Companion docs:** [`docs/IDEOLOGY.md`](../../IDEOLOGY.md), [`docs/architecture-report-2026-05-11.md`](../../architecture-report-2026-05-11.md)

---

## 1. Why this replaces the episodic-substrate approach

The first cross-episode learning attempt built a *separate* `CausalSDM`
instance keyed by an 11-role bundled `decision_vec`. Phase-1 systematic-
debugging probe (substrate loaded after one full episode of writes, queried
at typical step-0 contexts) confirmed: every query returned conf=1.0 with
**identical** decoded outcome (`survived=True damage=3 died_to=None`) across
ten different `plan_origin`s and five different `active_goal`s. Reason: the
12-role bundle leaves ~91% similarity between any two candidates that differ
only in `plan_origin`; SDM activates the same locations, returns the same
content. Adding a constant to every candidate's `base` score is a no-op on
lexicographic ranking, so `gen2` reverted byte-identically to the
no-episodic baseline.

The deeper problem behind the symptom: a second `CausalSDM` keyed by
`(context+plan_origin) → outcome` is *value-function approximation via kNN*
duplicated next to the world model. It pulls against the project's PCCS
principle — *one HDC substrate, multiple roles*. The right move is to give
the existing world-model SDM a new role rather than build a parallel one.

## 2. Mechanism

Today `VectorWorldModel.predict(concept, action) → (effect_vec, conf)` reads
the address `bind(c, a)` from the SDM. We add a parallel role-binding for
the same `(concept, action)` pair that stores the *realised trajectory
outcome* over a short horizon `H`:

```
bind(concept_vec, action_vec)                       → effect_vec   (existing)
bind(bind(concept_vec, action_vec), role_outcome_h) → outcome_vec  (new)
```

`outcome_vec` is a small HDC bundle:

| Role | Encoding | Why |
|---|---|---|
| `role_survived_h` | binary `bind(role, concept("alive"|"dead"))` | Headline survival signal |
| `role_damage_h` | thermometer scalar (clipped 0..10) via `encode_scalar` | Soft cost |
| `role_died_to` | concept vector of death cause, else `none` | Postmortem-grade negative signal |

Writes happen H=5 env steps after each decision (same pending-ring pattern
as before, but writing into the existing `model.memory` rather than into a
separate substrate object). On death, all pending in-flight snapshots flush
with `died_to=cause_of_death`. Saves piggyback on the existing
`VectorWorldModel.save()` — one `.pt` per seed contains *everything*
(concepts, actions, roles including `outcome_h`, addresses, content,
near_requirements, action_requirements). Loading the next episode restores
all of it.

## 3. Per-candidate-plan retrieval at scoring time

For each candidate plan, the planner computes a `(concept, action)` pair:

- **`do` plans** (`single:tree:do`, `single:water:do`, etc.): `concept = plan.steps[0].target`, `action = "do"`.
- **`make` / `place` plans**: `concept = plan.steps[0].target`, `action = plan.steps[0].action`. (E.g. `make_wood_pickaxe`: concept=`wood_pickaxe`, action=`make`.)
- **`sleep`** plan: `concept = "self"`, `action = "sleep"`.
- **Motion plans** (`self:move_left`, etc., including `motion_chain:` variants): `concept = vf.near_concept` (facing tile), `action = plan.steps[0].action`. This is the agreed answer to Q2 — use facing context as the concept for motion so substrate distinguishes "walk left next to tree" from "walk left next to lava".
- **`baseline`** plan (empty): `concept = vf.near_concept`, `action = "noop"`.

The new `OutcomeStimulus.evaluate(trajectory)` resolves the pair from
`trajectory.plan` plus a per-step context-holder (set by
`run_vector_mpc_episode` each step with `near_concept`), calls
`model.predict_outcome(concept, action) → (score, confidence)`, decodes the
outcome, and returns the composite signal:

```
weight * confidence * (
    survived_bonus if survived else -died_penalty
    - damage_unit * damage_h
    - death_cause_penalty if died_to is not None
)
```

Because XOR-binding `bind(c, a, role_outcome)` produces *orthogonal* SDM
addresses for different `(c, a)` pairs (no bundle dilution), per-candidate
queries hit different memory regions and return genuinely different
outcomes when the agent has had different experiences with each pair. This
is exactly the property the previous design lacked.

## 4. Per-seed save/load lifecycle

Q1 was: persist everything in one file. Agreed.

`run_vector_mpc_episode` gets a new optional kwarg
`world_model_path: Path | None = None`. At episode start, if the path
exists, `model.load(world_model_path)` runs after `load_from_textbook`. The
existing `load_state_dict` on the inner `CausalSDM` is additive — concept
and role vectors get replaced, but content counters accumulate. That is the
desired semantics: textbook bootstrap remains in place, online learning
from past episodes is layered on top, and the current episode's writes
further accumulate. At episode end, `model.save(world_model_path)` writes
the merged state. One file per seed
(`_docs/world_model_seed_{seed}.pt`), gitignored.

Storage cost: existing `model.save` already serialises `concepts`,
`actions`, `roles`, `memory.state_dict()` and the requirement dicts.
Adding outcome-role writes only grows the SDM content counters
(`memory.content` is `(n_locations, dim) = (50000, 16384) → 3.2 GB`). The
file is large but no larger than today's runtime memory. Acceptable for
research workflow; compression / quantisation is a separate later concern.

## 5. What this design deliberately does NOT do

- **No second SDM.** That mistake is the reason this spec exists. One
  `VectorWorldModel`, one `memory: CausalSDM`, multiple roles inside it.
- **No bundled context vector.** Each `(concept, action)` pair gets its
  own SDM address. No "facing + visible + inv + body + goal + plan_origin"
  super-bundle. Bundle dilution killed the previous design.
- **No marginal/normalised stimulus** (mean-subtract, differential query).
  Once retrieval differs per candidate naturally, those workarounds are
  unnecessary.
- **No textbook promotion** (that's a separate roadmap item).
- **No Kuramoto binding yet** (replace XOR binding with phase coupling is
  PCCS-step-3, this is PCCS-step-1).

## 6. Risks and checks

| Risk | Detection | Mitigation |
|---|---|---|
| Outcome-role writes pollute physics-role reads | `predict(concept, action)` confidence stays in normal range across an episode (compare with/without outcome writes on a probe trace) | Roles are XOR-bound; address `bind(c, a, role_outcome)` is orthogonal to `bind(c, a)`. Test asserts this. |
| Per-seed file unbounded growth | `os.path.getsize` over multiple generations | SDM `n_locations` is fixed; address-collision saturates rather than grows file. |
| Determinism break | Two-run byte-compare under strict eval | Outcome writes happen via the same `model.learn` machinery already covered by Stage 91 determinism guarantees. No new RNG. |
| Save/load drops requirement dicts | Existing save/load roundtrip tests | `model.save` already includes `action_requirements`, `near_requirements`, `proximity_ranges`. Test extends to include outcome-role roundtrip. |
| Concept "self" for sleep and "noop" for baseline collide with other meanings | grep concept vocabulary | `self` is already used by `self_actions` in plan generation; will not collide with any tile concept since tiles use semantic names. |

## 7. Files touched

New:
- `tests/agent/test_world_model_outcome_role.py` (~120 LOC)

Modified:
- `src/snks/agent/vector_world_model.py` — `learn_outcome`, `predict_outcome`, `_decode_outcome` helpers; `_ensure_role("outcome_h")` plus value roles; save/load already covers.
- `src/snks/agent/stimuli.py` — replace `EpisodicMemoryStimulus` with `OutcomeStimulus` (similar interface, queries model directly instead of substrate).
- `src/snks/agent/vector_mpc_agent.py` — replace episodic lifecycle with outcome lifecycle (load/learn/save via `model`, not via separate substrate object). Removes the old `EpisodicSubstrate` integration entirely.
- `experiments/record_stage91_seed_video.py` — replace `--enable-episodic` and `--episodic-substrate-path` with `--enable-outcome-learning` and `--world-model-path`.

Removed:
- `src/snks/agent/episodic_substrate.py` (already deleted in revert commit upstream; new design replaces it).
- `tests/agent/test_episodic_substrate.py`, `tests/agent/test_episodic_stimulus.py` (will be re-created against the new mechanism).

## 8. Implementation plan

1. **VectorWorldModel outcome-role extension + tests.** Add
   `learn_outcome`, `predict_outcome`, outcome-vec encoder/decoder. Tests
   cover roundtrip, orthogonality vs physics predict, save/load
   roundtrip. ~1 hr.
2. **OutcomeStimulus.** Replace previous stimulus. Per-candidate resolves
   `(concept, action)` via the rules in §3. Tests cover positive recall,
   death recall, per-candidate differentiation. ~30 min.
3. **Lifecycle hooks.** Add `world_model_path` kwarg, load/save calls,
   pending-ring buffer for outcome writes. Reuse the existing recorder
   pattern; rewrite to operate on `model.learn_outcome` rather than
   `substrate.write`. ~1 hr.
4. **Recorder CLI.** Update `record_stage91_seed_video.py` flags. ~15 min.
5. **Local smoke test.** Run with a fake env on seed 17 ep 0 for 10
   steps to validate import paths and per-step calls before HyperPC.
   Per the workflow rule learned last session. ~5 min.
6. **HyperPC gen1+gen2 validation.** Two episodes seed 17 ep 0
   full-profile. Compare trajectories. Expected: gen2 trajectory ≠ baseline
   (the previous failure mode) AND gen2 ≠ gen1 (substrate is contributing).
   ~10 min wall time.
7. **5-seed multiseed.** Aggregate gen1 vs gen2 metrics across seeds
   7,17,27,37,47. Compare survival, productive_do, crafting events.
   ~30 min wall.
8. **Determinism check.** Two runs same seed both with substrate loaded
   → byte-identical traces. ~10 min.
9. **Docs.** Append section to architecture report; tick the roadmap item
   in README. ~15 min.

Total target ~3-4 hours of focused work plus ~1 hour HyperPC wall time.

## 9. What "approved" looks like

Maintainer agrees that:
- One substrate with a new role is the right encoding (not a parallel SDM).
- The `(concept, action)` resolution rules in §3 cover all candidate-plan
  shapes the planner emits today.
- Per-seed save/load via existing `VectorWorldModel.save()` is acceptable
  (the ~3 GB file size is acknowledged).
- The deliberately-excluded items (marginal queries, textbook promotion,
  Kuramoto binding) are correctly out of scope.

Once approved: implement step 1 with tests, present, then proceed through
the rest of the plan.
