# Stage 87 — Curiosity About Death

**Date:** 2026-04-15  
**Status:** Draft  
**Ideological debt:** Principle 6 (System, not agent). Curiosity currently means "explore the new". Stage 87 redefines it: "reduce uncertainty about why I die".

---

## Context

After Stage 86, the agent has PostMortemLearner adapting stimuli thresholds based on death attribution. Stage 87 adds a complementary signal: **death-relevant curiosity**.

Currently curiosity = expected prediction surprise (`avg_surprise`). This is exploration for its own sake — the agent is rewarded for visiting unfamiliar states, regardless of whether those states are relevant to understanding death.

Stage 87 reformulation:
```
U_curiosity(s) = weight * avg_surprise(s) * death_relevance(s)
```

Where `death_relevance(s)` = how much does trajectory `s` expose conditions correlated with past deaths?

The mechanism: the system forms **DeathHypotheses** between episodes by correlating death causes with vital states at death time. The active hypothesis defines `death_relevance`.

---

## Observed Pattern (from Stage 86 data)

Stage 86 attribution shows zombie is dominant death cause. Combined with DamageEvent.vitals, we can check: when agent dies from zombie, what was food level? Hypothesis:
```
"I die from zombie more often when food < 3 than when food > 6."
Reason: low food → agent deprioritises defensive goals → zombie exposure.
Next episode: explore states with food ≈ 3 to test this.
```

---

## Design

### Block 1 — `DeathHypothesis`

```python
@dataclass
class DeathHypothesis:
    cause: str              # "zombie", "starvation", "skeleton", etc.
    vital: str              # "food", "health", "drink", "energy"
    threshold: float        # vital below this correlates with this cause
    n_supporting: int       # episodes: dominant_cause==cause AND vital<threshold at death
    n_observed: int         # total episodes where dominant_cause==cause
    
    @property
    def is_verifiable(self) -> bool:
        return self.n_observed >= 3 and self.n_supporting >= 2
    
    @property
    def support_rate(self) -> float: ...
    
    def death_relevance(self, trajectory: VectorTrajectory) -> float:
        """High relevance if trajectory visits states near threshold.
        Returns [1.0, 2.0]: 1.0 = neutral, 2.0 = high relevance."""
        min_vital = min(s.body.get(self.vital, 9.0) for s in trajectory.states)
        proximity = max(0.0, 1.0 - abs(min_vital - self.threshold) / 3.0)
        return 1.0 + proximity
```

### Block 2 — `HypothesisTracker`

Records `(dominant_cause, vitals_at_death)` per episode. After each record, rebuilds hypotheses by correlating cause with whether each vital was below a fixed threshold.

Tested thresholds: `{food: 3.0, drink: 3.0, health: 4.0, energy: 2.0}`.

```python
class HypothesisTracker:
    def record(self, attribution: dict[str, float], vitals_at_death: dict[str, float]) -> None: ...
    def active_hypothesis(self) -> DeathHypothesis | None:
        """Return highest-support verifiable hypothesis."""
    def n_verifiable(self) -> int:
        """How many verifiable hypotheses have been formed."""
```

A hypothesis is **verifiable** when `n_observed >= 3 and n_supporting >= 2` (observed cause ≥3 times, ≥2 of those with the correlated vital condition).

### Block 3 — `CuriosityStimulus` update

`stimuli.py` — extend the existing stub:

```python
@dataclass
class CuriosityStimulus(Stimulus):
    weight: float = 0.1
    hypothesis: "DeathHypothesis | None" = None

    def evaluate(self, trajectory: VectorTrajectory) -> float:
        surprise = trajectory.avg_surprise()
        relevance = self.hypothesis.death_relevance(trajectory) if self.hypothesis else 1.0
        return self.weight * surprise * relevance
```

### Block 4 — `PostMortemLearner.build_stimuli()` update

```python
def build_stimuli(
    self,
    vital_vars: list[str],
    hypothesis: "DeathHypothesis | None" = None,
) -> StimuliLayer:
    stimuli = [SurvivalAversion(), HomeostasisStimulus(...)]
    if hypothesis is not None:
        stimuli.append(CuriosityStimulus(hypothesis=hypothesis))
    return StimuliLayer(stimuli)
```

### Block 5 — Eval loop (`stage87_eval.py`)

```python
learner = PostMortemLearner()
tracker = HypothesisTracker()
analyzer = PostMortemAnalyzer()
stimuli = learner.build_stimuli(vitals)

for ep in range(n_episodes):
    result = run_vector_mpc_episode(..., stimuli=stimuli)
    damage_log = result.get("damage_log", [])
    attribution = analyzer.attribute(damage_log, result["episode_steps"])
    vitals_at_death = damage_log[-1].vitals if damage_log else {}
    
    learner.update(attribution)
    tracker.record(attribution, vitals_at_death)
    
    active = tracker.active_hypothesis()
    stimuli = learner.build_stimuli(vitals, hypothesis=active)
```

---

## Gates

| Gate | Condition | Measurement |
|---|---|---|
| `hypothesis_formed` | `tracker.n_verifiable() >= 1` after 20 episodes | HypothesisTracker state |
| `curiosity_active_episodes` | ≥5 episodes where active_hypothesis != None | count per run |
| `survival_holds` | avg_survival(with_hypothesis) >= 155 | episode_steps mean |

All 3 must PASS.

---

## Files Changed

| File | Change |
|---|---|
| `src/snks/agent/death_hypothesis.py` | NEW — `DeathHypothesis`, `HypothesisTracker` |
| `src/snks/agent/stimuli.py` | `CuriosityStimulus` — add `hypothesis` field + death_relevance weighting |
| `src/snks/agent/post_mortem.py` | `PostMortemLearner.build_stimuli()` — add `hypothesis` param + `CuriosityStimulus` |
| `experiments/stage87_eval.py` | NEW — 20-ep eval with HypothesisTracker + 3 gates |
| `tests/test_stage87.py` | NEW — unit tests for DeathHypothesis + HypothesisTracker |

---

## Ideology Check

- **Category 3 (Experience):** `HypothesisTracker._records` = per-episode death observations. `DeathHypothesis` = derived correlation. Runtime-only, no persistence.
- **Category 4 (Stimuli):** `CuriosityStimulus.death_relevance` — a stimulus weight, not a hardcoded reflex. Changes what the MPC scores, not how it plans.
- Does NOT modify textbook rules (Category 1) or planning mechanics (Category 2).

The hypothesis is **observational, not causal** — it correlates vital_level with death_cause, does not claim mechanism. The agent explores near the threshold, learns from the resulting experience.

---

## Out of Scope (Stage 87)

- Causal graph hypothesis (A → B → death) — Stage 88+ research
- Persistence of hypotheses across runs — Stage 88 (Knowledge Flow)
- Hypothesis rejection / updating on disconfirming evidence — Stage 88+
- EntityAversionStimulus — deferred to Stage 88 (post-mortem avoidance)
