# Episodic Substrate — Cross-Episode Learning via HDC-Encoded Decision Snapshots

**Date:** 2026-05-12
**Status:** Design (pre-implementation)
**Author:** SNKS architecture line
**Companion docs:** [`docs/IDEOLOGY.md`](../../IDEOLOGY.md), [`docs/architecture-report-2026-05-11.md`](../../architecture-report-2026-05-11.md)

---

## 1. What problem this solves

Today the agent forgets everything between episodes. The textbook-bootstrapped
`VectorWorldModel` (SDM) holds env-physics transition facts that came from the
textbook anyway; online `model.learn(...)` writes during an episode are lost
at the next `reset()`.

The user-visible failure mode: agent dies in a recurring situation (no sword,
zombie close, table near, wood ≥ 1) by choosing to flee instead of craft, and
**learns nothing from it** — next episode in the same situation it makes the
same choice. The "cross-episode SDM persistence" idea was the wrong frame for
this gap, because SDM stores `(concept, action) → effect_vec` — env physics,
not *decision quality in context*.

The right gap is **Experience layer** (category 3 of the ideology) — there is
no live persistent representation of "I have been in this kind of situation
before, this is what happened". This spec fills that gap.

## 2. The mechanism in one paragraph

At every decision point the agent bundles the current contextual state plus
its chosen plan into a single binary HDC vector — `decision_vec`. After a
short horizon `H` (or at episode-end, whichever comes first) it bundles the
observed outcome — damage taken, items gained, whether the agent survived the
horizon, near-concept transitions, death cause if applicable — into a single
binary HDC vector `outcome_vec`. A persistent **EpisodicSDM** writes the
association `decision_vec → outcome_vec`. At the next planning step the agent
builds a candidate `decision_vec` for *each candidate plan*, queries
EpisodicSDM, and gets back a confidence-weighted predicted outcome. That
prediction feeds a new `EpisodicMemoryStimulus` which contributes to
`score_trajectory.base`, biasing plan selection toward choices whose past
analogues survived and away from analogues that died. EpisodicSDM is saved
to disk per-seed at episode end and loaded at the next episode of that seed.

Everything reuses existing infrastructure (`CausalSDM`, `bind`, `bundle`,
`StimuliLayer`). No new symbolic rules. No textbook promotion. Knowledge
that accumulates is *contextual outcome statistics in HDC space*.

## 3. Architecture

### 3.1 New module — `src/snks/agent/episodic_substrate.py`

```python
class EpisodicSubstrate:
    """Per-seed HDC episodic memory of decision contexts and their outcomes."""

    memory: CausalSDM                    # second SDM instance, separate from world model
    roles: dict[str, torch.Tensor]       # one HDC role vector per context feature
    concept_vecs: dict[str, torch.Tensor]  # shared with VectorWorldModel.concepts where overlap

    def encode_decision(state, plan_origin) -> torch.Tensor: ...
    def encode_outcome(rollout_states, rollout_events) -> torch.Tensor: ...
    def write(decision_vec, outcome_vec, surprise: float) -> None: ...
    def query(decision_vec) -> tuple[torch.Tensor | None, float]: ...  # (outcome_vec, confidence)
    def save(path: Path) -> None: ...
    def load(path: Path) -> bool: ...
```

### 3.2 `decision_vec` composition (the "B / standard" set from Q1)

A single binary HDC vector, dim equal to world-model SDM (16384), built as
`bundle([bind(role_X, value_X) for each X])` with these features:

| Role | Value encoding | Why included |
|---|---|---|
| `role_facing_tile` | concept vector of vf.near_concept | What the player is about to interact with |
| `role_visible_set` | bundle of visible-concept vectors | Surrounding scene gist |
| `role_inv_wood` | thermometer-encoded wood count (clipped 0..10) | Crafting feasibility signal |
| `role_inv_has_weapon` | binary 0/1 HDC (wood_sword OR wood_pickaxe present) | Fight-vs-flee key bit |
| `role_inv_has_table` | binary 0/1 (table in adj or in nearby spatial_map) | Craft-readiness bit |
| `role_body_health` | bucketed (low / mid / high) | Risk tolerance |
| `role_body_food` | bucketed | Forces "eat first" awareness |
| `role_body_drink` | bucketed | Same for drink |
| `role_body_energy` | bucketed | Same for sleep |
| `role_active_goal` | concept vector of goal.id | What we currently want |
| `role_plan_origin` | concept-encoded plan-origin tag (`single:tree:do`, `chain:place_table+make_wood_sword`, `motion_chain:...`, `baseline`) | The choice we are making |
| `role_nearest_threat_dist` | bucketed Manhattan distance (0 / 1-2 / 3-5 / 6+ / None) | Tactical pressure |

Total: 12 roles. With dim=16384 the standard SDM crosstalk floor (~0.11
under full profile) stays below the 0.2 confidence gate, so unwritten queries
return `(None, 0.0)` instead of phantom matches.

Mutable state (player_pos, step_index) is intentionally **excluded** —
generalising across world positions is exactly what we want.

### 3.3 `outcome_vec` composition

| Role | Value encoding | Why |
|---|---|---|
| `role_survived_h` | binary (alive at step `t+H`) | Headline signal |
| `role_damage_h` | thermometer (clipped 0..10) | Soft cost |
| `role_items_gained` | bundle of `bind(item_concept, +1)` for each new inventory item over H | What this plan-class produces |
| `role_near_changed` | concept vector of `near_concept` at `t+H` if different, else `same` | Did the decision move us productively? |
| `role_died_to` | concept vector of death-cause if died inside horizon, else `none` | Postmortem-grade signal |

Horizon `H = 5` steps. Short enough to keep credit assignment tight; long
enough that "craft now → fight next" gets one full payoff cycle.

### 3.4 Write rule

Every decision point produces a snapshot. After `H` real env steps (or
episode end) the agent bundles the realised outcome and writes:

```
episodic.memory.write(decision_vec, outcome_vec)
```

A pending-snapshots ring buffer with capacity `H` holds in-flight decision
vectors awaiting their outcome. When death occurs, all pending snapshots are
written with their partial outcomes and the death-cause role populated.

Surprise-weighted reinforcement: a snapshot whose `outcome_vec` had high
prediction surprise (low confidence in the EpisodicSDM query at write time)
is written twice — same content, two SDM addresses — so high-information
events get a stronger trace. Routine matches with confidence ≥ 0.6 get one
write. This is the substrate's analogue of "salient memories consolidate".

### 3.5 Read rule — `EpisodicMemoryStimulus`

New stimulus in `src/snks/agent/stimuli.py`:

```python
@dataclass
class EpisodicMemoryStimulus(Stimulus):
    substrate: EpisodicSubstrate
    weight: float = 1.0
    confidence_threshold: float = 0.3

    def evaluate(self, trajectory: VectorTrajectory) -> float:
        # Build decision_vec for trajectory.plan in trajectory.states[0]
        decision_vec = self.substrate.encode_decision_from_traj(trajectory)
        outcome_vec, confidence = self.substrate.query(decision_vec)
        if outcome_vec is None or confidence < self.confidence_threshold:
            return 0.0
        # Decode survived_h, damage_h, items_gained from outcome_vec
        decoded = self.substrate.decode_outcome(outcome_vec)
        # Composite signal: survived bonus minus damage penalty minus death-cause cost
        signal = (
            (1.0 if decoded["survived_h"] else -3.0)
            - 0.25 * decoded.get("damage_h", 0)
            - (5.0 if decoded.get("died_to") not in (None, "none") else 0.0)
        )
        return self.weight * confidence * signal
```

Added to the default StimuliLayer alongside SurvivalAversion, Homeostasis,
Curiosity. Weight starts at `1.0` (small relative to SurvivalAversion=1000)
so episodic memory biases choices without dominating safety.

### 3.6 Lifecycle hooks (`vector_mpc_agent.run_vector_mpc_episode`)

| Hook point | Action |
|---|---|
| Start of episode | `substrate.load(path=f"_docs/episodic_seed_{seed}.pt")`. If missing, start fresh — no fallback to textbook (textbook seeds world-model SDM, not episodic). |
| After scoring, before exec | For the chosen plan, build `decision_vec`, push onto pending-ring with a `write_due_at = current_step + H` marker. |
| After every env.step | Pop any pending snapshot whose `write_due_at == current_step` and write its outcome to substrate. |
| On death | Flush all pending snapshots; populate `died_to` role. |
| End of episode | `substrate.save(path=...)`. |

### 3.7 Where this fits in the existing pipeline

```
env.step → info["semantic"]
  ↓
PERCEPTION → spatial_map.update
  ↓
WORLD MODEL  (predict, simulate_forward)
  ↓
GOAL SELECTOR
  ↓
PLAN GENERATION
  ↓
SIMULATE + SCORE  ← EpisodicMemoryStimulus reads here  (NEW)
  ↓
RANK + RESCUE
  ↓
ACT
  ↓
LEARN  (world-model SDM update)         (existing)
  ↓
EPISODIC WRITE  (pending-ring flush)     (NEW)
```

Reads happen during score_trajectory (one query per candidate plan per step).
Writes happen at fixed lag H after the decision was made. No new threads, no
parallelism — same per-step loop.

## 4. What this design deliberately does NOT do

- **No promotion to textbook** — facts in textbook stay categories 1 facts.
  Episodic memory stays category 3 (Experience). The two layers communicate
  only through scoring, not through `promoted_hypotheses.yaml`.
- **No phase-coupling yet** — PCCS with Kuramoto binding is the long-term
  target (see roadmap item 6 in the architecture report). This spec uses
  vanilla XOR-bind / majority-bundle from existing `CausalSDM`. Phase coupling
  arrives in a later spec; the episodic substrate is built such that its
  binding mechanism can be swapped without changing the read/write interface.
- **No counterfactual replay** — the agent does not generate "what if I had
  chosen X instead" trajectories at write time. It writes only the path it
  actually took. Counterfactual analysis can be layered on as a separate
  feature later.
- **No goal-progress tweak** — `Goal.progress` and `score_trajectory` stay as
  they are. Episodic memory enters scoring through `StimuliLayer` only.
- **No emergency-controller override** — the rescue layer keeps its current
  authority. Episodic stimulus is one signal among many in the planner's
  ranking; it does not bypass `EmergencySafetyController`.

## 5. Risks and how they get checked

| Risk | Symptom in trace | Mitigation |
|---|---|---|
| SDM crosstalk at 12 roles | Unwritten queries return conf > 0.2 with garbage outcomes | Full-profile only (dim=16384, n_locations=50000). Verified at spec stage by probing `episodic.query(arbitrary_unwritten_vec)`. If conf > 0.2, drop roles to the minimum set (A from Q1). |
| Stimulus weight dominates safety | Agent ignores hostiles to recreate a past-successful craft | Start weight=1.0; SurvivalAversion stays at 1000. Death-cause role in decoded outcome already penalises -5 per query. |
| Per-seed file growth unbounded | Disk usage grows linearly across hundreds of episodes | Cap SDM at `n_locations=50000`; SDM's address-collision mechanism naturally consolidates. Snapshot file size is fixed by `n_locations × dim`. |
| Bad-credit assignment | "Successful craft" got credit because zombie wandered off, not because crafting was right | Accept noise at H=5; longer-horizon attribution is a follow-up (CuriosityStimulus + DeathHypothesis already track death-context). |
| Test-determinism breakage | Strict-determinism eval no longer byte-identical because episodic substrate adds RNG draws | Substrate writes are deterministic given fixed `learn()` sequence; no new RNG. Verified by running same seed twice and asserting outcome traces match. |

## 6. Files touched

New:
- `src/snks/agent/episodic_substrate.py` (~250 LOC)
- `tests/agent/test_episodic_substrate.py` (~150 LOC)

Modified:
- `src/snks/agent/stimuli.py` — add `EpisodicMemoryStimulus`
- `src/snks/agent/vector_mpc_agent.py` — load/save hooks, pending-ring flush, EpisodicStimulus wired into default `StimuliLayer`
- `experiments/record_stage91_seed_video.py` — optional `--episodic-substrate-path` flag (default uses `_docs/episodic_seed_{seed}.pt`)

No other files touched. The four-category boundary is preserved.

## 7. Implementation plan

Steps in order; each step ends with a passing test gate. Implementation skill
(`writing-plans`) will turn this into a detailed plan document; the headings
below are the milestone backbone.

**Step 1 — `EpisodicSubstrate` skeleton + unit tests**
- Class with `encode_decision`, `encode_outcome`, `write`, `query`, `save`,
  `load`. Backed by a fresh `CausalSDM` instance.
- Roles defined as a frozen dict; HDC role vectors generated deterministically
  from role-name hashes (RNG seeded by `dim` + role name).
- Unit tests:
  - Encode-decode roundtrip with single role.
  - Write + query returns the written outcome at confidence > 0.9.
  - Unwritten query returns confidence < 0.2 (crosstalk floor under full
    profile).
  - Save → load roundtrip preserves substrate state.

**Step 2 — `EpisodicMemoryStimulus`**
- Subclass `Stimulus`; uses substrate.query on trajectory's first-state
  decision_vec.
- Decodes outcome_vec into named fields via existing
  `VectorWorldModel.decode_effect`-style helper (factored into substrate).
- Unit tests:
  - Untrained substrate → stimulus returns 0.0.
  - Substrate trained with `survived=True, damage=0` for given decision →
    stimulus returns +1.0.
  - Substrate trained with `died_to=zombie` → stimulus returns ≤ -5.0.

**Step 3 — Wire into `run_vector_mpc_episode`**
- Load at episode start (`_docs/episodic_seed_{seed}.pt` or fresh).
- Push decision snapshot to pending ring after action selection.
- Flush pending snapshots whose `write_due_at == current_step` after env.step.
- Death path: flush remaining with partial outcomes.
- Save at episode end.

**Step 4 — Single-seed video validation**
- Run seed 17 ep 0 full-profile twice in succession.
- First run: substrate file absent → fresh write.
- Second run: substrate loaded → expected behaviour change at any decision
  point that recurs (e.g. the make_wood_sword step at ~48 should now have
  positive episodic signal).
- Deliver mp4 + JSON; per the existing workflow rule, watch for changed
  trajectory shape, not just survival numbers.

**Step 5 — 5-seed multiseed**
- 5 seeds × 2 generations (gen 1 fresh, gen 2 with persisted substrate).
- Compare aggregate metrics: avg_survival, productive_do, crafting events,
  death-cause distribution.
- Expectation: gen 2 ≥ gen 1 on productive_do and crafting events; survival
  may be similar or slightly higher.

**Step 6 — Determinism check under strict eval**
- Same seed, two runs, byte-compare traces. Substrate must not break the
  Stage 91 determinism guarantee.

**Step 7 — Document in architecture report**
- Append section to `docs/architecture-report-2026-05-11.md` describing
  the now-live Experience-layer mechanism.
- Update README roadmap item 1 status: in progress → done.

## 8. What "approved" looks like

Maintainer reviews this doc, agrees that:
- The mechanism (decision_vec → outcome_vec association in a separate SDM,
  read via stimulus, written at horizon H) is sound.
- The role set is the right starting point (B-tier, 12 roles).
- The lifecycle integration matches the existing pipeline.
- The deliberately-excluded items (textbook promotion, phase coupling,
  counterfactual replay) are correctly out of scope.

Once approved: hand the plan to `writing-plans` skill to flesh out each
implementation step into actionable tasks; then code.
