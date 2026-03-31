# Stage 30: Few-Shot Learning — Design Specification

**Date:** 2026-03-31
**Status:** Draft v1.0
**Branch:** `stage30-few-shot-learning`
**Experiments:** exp74, exp75, exp76

---

## 1. Motivation

Stages 25–29 taught the agent to learn from its own experience: backward chaining, causal learning, skill extraction, analogical reasoning, curiosity. But a truly capable agent should also learn by **observing others**. Few-shot learning from demonstrations is a core human cognitive ability: a child watches an adult open a door once and can replicate the action.

**What this stage proves:** СНКС agent can learn to solve a task from 1–3 demonstrations, without any prior experience, by extracting causal knowledge and skills from observed trajectories.

---

## 2. Architecture

### 2.1 Demonstration Recording

A `Demonstration` is a recorded trajectory: a sequence of `(sks_before, action, sks_after)` transitions observed from a demonstrator agent.

```python
@dataclass
class DemoStep:
    sks_before: set[int]   # perceived state before action
    action: int             # action taken
    sks_after: set[int]     # perceived state after action

@dataclass
class Demonstration:
    steps: list[DemoStep]
    goal_instruction: str   # what the demo was trying to achieve
    success: bool           # whether the demo achieved the goal
```

### 2.2 DemonstrationRecorder

Records a trajectory by wrapping an existing agent's episode:

```
Agent.run_episode(instruction) → (result, demonstration)
```

Implementation: monkey-patch `env.step()` to capture sks transitions via GridPerception, same as CausalLearner but passive (observing, not doing).

### 2.3 FewShotLearner

Core module. Takes 1–N demonstrations and produces:
1. **CausalWorldModel** — causal links extracted from observed transitions
2. **SkillLibrary** — skills extracted from the causal model
3. Composed skills via `SkillLibrary.compose_skills()`

Key insight: `CausalWorldModel.observe_transition()` already works with any (pre, action, post) triple — it doesn't matter whether the agent performed the action or observed it. So we feed demonstration steps directly into the causal model.

```python
class FewShotLearner:
    def learn_from_demonstrations(self, demos: list[Demonstration]) -> tuple[CausalWorldModel, SkillLibrary]:
        model = CausalWorldModel(config)
        for demo in demos:
            for step in demo.steps:
                model.observe_transition(step.sks_before, step.action, step.sks_after)
        library = SkillLibrary()
        library.extract_from_causal_model(model)
        library.compose_skills()
        return model, library
```

### 2.4 FewShotAgent

Extends CuriosityAgent. Before acting, receives demonstrations and learns from them. Then uses the standard skill → analogy → backward chaining → explore pipeline.

```python
class FewShotAgent(CuriosityAgent):
    def learn_from_demos(self, demos: list[Demonstration]) -> None:
        learner = FewShotLearner()
        model, library = learner.learn_from_demonstrations(demos)
        # Merge into agent's existing knowledge
        self._causal_model = model
        self._library = library
```

---

## 3. Experiments

### Exp 74: One-Shot Skill Extraction
- **Setup:** Expert agent (pre-trained CuriosityAgent) solves DoorKey 1 time. Record demo.
- **Test:** FewShotLearner extracts skills from single demo. Verify:
  - `pickup_key` skill extracted with correct preconditions/effects
  - `toggle_door` skill extracted
  - Composite `pickup_key+toggle_door` composed
- **Gate:** skill_extraction_accuracy >= 0.9 (fraction of expected skills found)

### Exp 75: Few-Shot Goal Completion
- **Setup:** Record 1, 2, 3 demos from expert on DoorKey-5x5.
- **Test:** FewShotAgent learns from demos, then solves 10 unseen DoorKey layouts.
- **Gate:**
  - 1 demo: success_rate >= 0.5
  - 3 demos: success_rate >= 0.8

### Exp 76: Few-Shot Cross-Environment Transfer
- **Setup:** Record 3 demos from expert on DoorKey (key/door).
- **Test:** FewShotAgent + AnalogicalReasoner solves CardGate (card/gate) without any CardGate demos.
- **Gate:** cross_env_success >= 0.7

---

## 4. Files

| File | Description |
|------|-------------|
| `src/snks/language/demonstration.py` | DemoStep, Demonstration, DemonstrationRecorder |
| `src/snks/language/few_shot_learner.py` | FewShotLearner: demos → causal model + skills |
| `src/snks/language/few_shot_agent.py` | FewShotAgent extends CuriosityAgent |
| `tests/test_few_shot.py` | Unit tests |
| `src/snks/experiments/exp74_one_shot_skill.py` | Exp 74 |
| `src/snks/experiments/exp75_few_shot_goal.py` | Exp 75 |
| `src/snks/experiments/exp76_few_shot_transfer.py` | Exp 76 |
| `demos/stage-30-few-shot.html` | Web demo |

---

## 5. Design Decisions

1. **Reuse CausalWorldModel** — no new learning mechanism needed. The existing `observe_transition()` works for observed actions, not just self-performed ones. This is philosophically correct: observing someone else's action provides the same causal evidence.

2. **DemonstrationRecorder wraps env.step** — lightweight, doesn't require modifying any agent code. Records the same data CausalLearner would see.

3. **FewShotAgent inherits CuriosityAgent** — gets the full stack: skills, analogies, curiosity, backward chaining. Few-shot knowledge bootstraps the causal model, curiosity fills in gaps.

4. **No gradient-based meta-learning** — stays true to СНКС philosophy: knowledge = causal links in world model, learning = observation + local rules.
