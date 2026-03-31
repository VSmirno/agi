# Stage 24b: Instruction Planning + Real QA Backends — Design Doc

**Дата:** 2026-03-31
**Статус:** Pending implementation
**Зависимости:** Stage 24a (InstructionParser), Stage 22 (GroundedQA), Stage 11 (StochasticSimulator), Stage 6 (CausalWorldModel)

---

## Цель

1. InstructionPlanner: parsed instruction → action plan через CausalWorldModel
2. Реальные QA backends: адаптеры CausalWorldModel/StochasticSimulator для GroundedQA
3. Проверка каузальных цепочек ("open door" → нужен ключ → pickup first)

## Scope

Полуинтегрированный. CausalWorldModel заполняется через observe_transition() вручную (синтетические переходы). Нет реального DAF/MiniGrid.

---

## Компоненты

### InstructionPlanner

```python
class InstructionPlanner:
    """Converts parsed instruction chunks into action plan."""

    def __init__(
        self,
        grounding_map: GroundingMap,
        causal_model: CausalWorldModel,
        simulator: StochasticSimulator,
    ) -> None: ...

    def plan(
        self,
        chunks: list[Chunk],
        current_sks: set[int],
    ) -> list[int]:
        """Convert chunks to action sequence.

        For sequential instructions (SEQ_BREAK), plans each sub-instruction.
        For single instructions, uses simulator.find_plan_stochastic() if
        goal SKS is known, otherwise returns single action.
        """
```

Pipeline:
1. Split chunks by SEQ_BREAK
2. For each sub-instruction: resolve OBJECT/ATTR → goal SKS via GroundingMap
3. Resolve ACTION → action_id via action name mapping
4. If goal requires prerequisite (causal chain), plan prerequisite first
5. Return flat action sequence

### CausalQABackend

```python
class CausalQABackend:
    """Factual QA via CausalWorldModel.get_causal_links()."""

    def __init__(self, causal_model: CausalWorldModel, grounding_map: GroundingMap) -> None: ...

    def query(self, roles: dict[str, int]) -> QAResult | None:
        """Find causal links involving the queried ACTION+OBJECT."""
```

Scans causal links for matching action+context. Returns answer SKS from effect.

### SimulationQABackend

```python
class SimulationQABackend:
    """Simulation QA via StochasticSimulator.sample_effect()."""

    def __init__(self, simulator: StochasticSimulator, current_sks: set[int]) -> None: ...

    def query(self, roles: dict[str, int]) -> QAResult | None:
        """Simulate action effect from current state."""
```

### Action Name Mapping

Bidirectional mapping action_name ↔ action_id. Already exists as dict in experiments (MINIGRID_ACTIONS). InstructionPlanner receives it as constructor param.

---

## Файлы

```
src/snks/language/
├── planner.py               🆕  InstructionPlanner
└── qa_backends.py           🆕  CausalQABackend, SimulationQABackend

tests/
└── test_instruction_planner.py  🆕  ~12 unit-тестов

src/snks/experiments/
├── exp54b_plan_correctness.py   🆕
└── exp55_causal_chains.py       🆕
```

---

## Эксперименты

### Exp 54b: Plan Correctness

Synthetic CausalWorldModel with known transitions:
- pickup(key) in room_with_key → key_held
- open(door) with key_held → door_open
- goto(ball) → near_ball

Instructions → expected action sequences. Gate: accuracy > 0.7

### Exp 55: Causal Chains

"open the door" when key not held → planner should produce [pickup_key, open_door].
Tests that InstructionPlanner detects prerequisites via CausalWorldModel.

Gate: chain_accuracy > 0.5

---

## Не входит в scope

- MiniGrid execution (Stage 24c)
- Real DAF perception
- MetacogMonitor-based reflective backend (deferred — not enough value for synthetic test)
