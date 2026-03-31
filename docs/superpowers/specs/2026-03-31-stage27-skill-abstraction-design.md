# Stage 27: Skill Abstraction — Hierarchical Macro-Actions

**Version:** 1.1
**Date:** 2026-03-31
**Status:** DESIGN APPROVED

---

## Goal

Extract reusable **skills** (named macro-actions) from learned causal knowledge. Instead of re-discovering sub-goal chains each episode via backward chaining, the agent recognizes recurring patterns and packages them as named, composable skills with verified preconditions and effects.

### What This Proves

1. **Skill extraction** — agent autonomously identifies reusable action patterns from causal links
2. **Skill reuse** — using cached skills reduces primitive action count by ≥ 1.5x
3. **Skill composition** — composite skills chain sub-skills (e.g., `solve_doorkey` = `pickup_key` + `open_door` + `navigate_to_goal`)
4. **Skill transfer** — extracted skills work in new environments (MultiRoom)

### Core Principles

- Skills are **named generalizations** over SubGoal chains, not hardcoded
- Extraction is **automatic** from CausalWorldModel after threshold observations
- Skills are verified by **state predicates** (pre/post conditions)
- Composition via **precondition/effect matching**, not manual sequencing

---

## Architecture

```
                  CausalWorldModel
                  (learned links)
                        │
                        │ extract (after N observations)
                        ▼
                  ┌─────────────┐
                  │ SkillLibrary│
                  │             │
                  │ pickup_key  │
                  │ open_door   │
                  │ solve_dk    │◄── compose (chain matching)
                  └──────┬──────┘
                         │
                         │ match preconditions
                         ▼
                  ┌──────────────┐
                  │  SkillAgent  │
                  │              │
                  │  1. match    │
                  │  2. execute  │
                  │  3. verify   │
                  │  4. fallback │
                  └──────────────┘
```

---

## New Components

### 1. Skill (`src/snks/language/skill.py`)

```python
@dataclass
class Skill:
    name: str                          # "pickup_key", "open_door"
    preconditions: frozenset[int]      # required SKS before execution
    effects: frozenset[int]            # expected SKS changes after
    terminal_action: int | None        # primitive action (3=pickup, 5=toggle) or None for composite
    target_word: str                   # "key", "door", "goal"
    sub_skills: list[str] | None       # composite: ordered list of skill names
    success_count: int = 0
    attempt_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.attempt_count, 1)

    @property
    def is_composite(self) -> bool:
        return self.sub_skills is not None and len(self.sub_skills) > 0
```

### 2. SkillLibrary (`src/snks/language/skill_library.py`)

```python
class SkillLibrary:
    def __init__(self): ...

    def extract_from_causal_model(self, model: CausalWorldModel, min_confidence: float = 0.7) -> int:
        """Auto-extract skills from high-confidence causal links.
        Returns number of new skills extracted."""

    def compose_skills(self) -> int:
        """Find skill chains where skill_A.effects ⊇ skill_B.preconditions.
        Create composite skills. Returns count of new composites."""

    def find_applicable(self, current_sks: set[int], goal_sks: frozenset[int]) -> list[Skill]:
        """Find skills whose preconditions are met and effects include goal."""

    def get(self, name: str) -> Skill | None: ...
    def register(self, skill: Skill) -> None: ...

    @property
    def skills(self) -> list[Skill]: ...
```

**Extraction logic:**
- For each CausalLink with confidence ≥ threshold:
  - `context_sks` ∩ state_range(50-99) → preconditions
  - `effect_sks` ∩ state_range(50-99) → **raw_effects** (symmetric difference: includes both added and removed SKS)
  - Filter effects: only keep SKS IDs that represent the DESIRED state change (e.g., SKS_KEY_HELD=51 from pickup, SKS_DOOR_OPEN=53 from toggle). This is done by intersecting with known "positive predicates" {51, 53, 54} rather than "pre-state predicates" {50, 52}.
  - Map action → target_word (pickup→"key", toggle→"door")
  - Name = f"{action_name}_{target_word}"
- Dedup: if skill with same (preconditions, effects, action) exists, increment count

**Composition logic (max_depth=3):**
- For each pair (skill_A, skill_B) where skill_A.effects ∩ skill_B.preconditions ≠ ∅:
  - Create composite: sub_skills=[A.name, B.name], preconditions=A.preconditions, effects=B.effects
- Chain extension (depth ≤ 3): if composite AB exists and AB.effects ∩ C.preconditions ≠ ∅, create ABC
- No further chaining beyond depth 3 (matches MAX_CHAIN_DEPTH in blocking_analyzer.py)

### 3. SkillAgent (`src/snks/language/skill_agent.py`)

Extends GoalAgent with skill-first execution:

```python
@dataclass
class SkillEpisodeResult(EpisodeResult):
    """Extends EpisodeResult with skill tracking."""
    skills_used: list[str] = field(default_factory=list)
    skills_total: int = 0

class SkillAgent(GoalAgent):
    def __init__(self, env, skill_library: SkillLibrary | None = None, **kwargs):
        super().__init__(env, **kwargs)
        self._library = skill_library or SkillLibrary()

    def run_episode(self, instruction: str, max_steps: int = 300) -> SkillEpisodeResult:
        """Skill-first: try matching skills before backward chaining.

        Overrides GoalAgent.run_episode entirely (no hook needed — clean override).
        After episode completes, calls _after_episode() for skill extraction.
        """

    def _try_skill(self, skill: Skill, max_steps: int) -> tuple[bool, int, float]:
        """Execute a skill (primitive or composite). Returns (success, steps, reward)."""

    def _after_episode(self):
        """Post-episode: extract new skills from updated causal model."""
```

**Execution priority:**
1. Check SkillLibrary for applicable composite skill → execute
2. Check SkillLibrary for applicable primitive skill → execute
3. Fallback to GoalAgent backward chaining + exploration
4. After episode: extract new skills from causal model

---

## Experiments

### Exp 65: Skill Extraction from DoorKey

**Protocol:**
1. Run SkillAgent for 10 episodes in DoorKey-5x5
2. After each episode, call `extract_from_causal_model()` + `compose_skills()`
3. Measure: number of skills extracted, composite skills formed

**Gate criteria:**
- ≥ 2 primitive skills extracted (pickup_key, open_door)
- ≥ 1 composite skill formed (solve_doorkey chain)
- Skills have success_rate ≥ 0.8

### Exp 66: Skill Reuse Speedup

**Protocol:**
1. **Phase A:** SkillAgent runs 5 warmup episodes in DoorKey-5x5 (learns + extracts skills)
2. **Phase B:** SkillAgent runs 10 test episodes in DoorKey-5x5 with extracted skills
3. **Control:** GoalAgent (no skills) runs 10 episodes in DoorKey-5x5
4. Measure: mean steps, exploration_episodes

**Gate criteria:**
- Skill agent mean_steps ≤ 0.67 × control mean_steps (≥ 1.5x speedup)
- Skill agent exploration_episodes ≤ 1 in test phase (10 episodes)

### Exp 67: Skill Transfer to MultiRoom

**Protocol:**
1. SkillAgent learns + extracts skills in DoorKey-5x5 (5 episodes)
2. Transfer: SkillLibrary + CausalWorldModel to MultiRoomDoorKey (custom env from Stage 26, `src/snks/env/multi_room.py`)
3. Run 10 episodes, measure skill reuse and success

**Gate criteria:**
- Success rate ≥ 0.9 (skills from DoorKey apply in MultiRoom)
- ≥ 1 skill reused per episode (not just causal transfer)

---

## Implementation Plan

### Phase 1: Skill + SkillLibrary
- Skill dataclass
- SkillLibrary: register, get, find_applicable
- extract_from_causal_model
- compose_skills
- Unit tests

### Phase 2: SkillAgent
- Extends GoalAgent
- Skill-first execution in run_episode
- _try_skill for primitive and composite skills
- _after_episode extraction
- Unit tests

### Phase 3: Experiments
- Exp 65, 66, 67
- Report + ROADMAP update

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Skill extraction too aggressive (noise) | min_confidence=0.7, min_count=3 |
| Composite skill ordering wrong | Verify precondition/effect chain before creating |
| Skills don't transfer (different object positions) | Skills use predicates not positions |
| Overhead of skill matching > savings | find_applicable is O(n_skills), trivial for <20 skills |

---

## Non-Goals

- Learning skills from demonstration (that's Stage 30: Few-Shot Learning)
- Automatic skill naming from natural language (manual mapping sufficient)
- Skill discovery in continuous action spaces (grid-world only)
