# Stage 28: Analogical Reasoning

**Date:** 2026-03-31
**Status:** v1.0
**Автор:** autonomous-dev pipeline

---

## Цель

Агент должен решать структурно новые задачи по аналогии с известными навыками.
Конкретно: обученный на `key→door` агент переносит паттерн на `card→gate` без переобучения.

---

## Архитектурный подход: Predicative Analogy + Static Role Registry

### Ключевая идея

Аналогия = изоморфизм между предикатными паттернами двух Skill-ов в SKS-пространстве.
Skill `pickup_key` имеет структуру `{SKS_KEY_PRESENT} → {SKS_KEY_HELD}`.
Skill для card имеет структуру `{SKS_CARD_PRESENT} → {SKS_CARD_HELD}`.
Сходство определяется через статически зарегистрированный `ROLE_REGISTRY`.

### Новые SKS предикаты (55-58)

```python
SKS_CARD_PRESENT = 55   # purple key на полу
SKS_CARD_HELD    = 56   # агент несёт purple key
SKS_GATE_LOCKED  = 57   # purple door заблокирован
SKS_GATE_OPEN    = 58   # purple door открыт
```

Диапазон 50-99 — `_STATE_SKS_RANGE` в CausalWorldModel, попадают в `stable` контекст.

### ROLE_REGISTRY

```python
ROLE_REGISTRY: dict[str, tuple[int, int]] = {
    "instrument_present": (SKS_KEY_PRESENT, SKS_CARD_PRESENT),
    "instrument_held":    (SKS_KEY_HELD, SKS_CARD_HELD),
    "blocker_locked":     (SKS_DOOR_LOCKED, SKS_GATE_LOCKED),
    "blocker_open":       (SKS_DOOR_OPEN, SKS_GATE_OPEN),
}
```

Similarity: `matched_roles / total_predicates` где matched_roles = кол-во предикатов
из source_skill, имеющих аналог в ROLE_REGISTRY.

### AnalogyMap и AnalogicalReasoner

```python
@dataclass
class AnalogyMap:
    source_skill_name: str
    adapted_skill: Skill          # готовый skill с заменёнными предикатами
    sks_mapping: dict[int, int]   # {source_sks: target_sks}
    role_mapping: dict[str, str]  # {"key": "card", "door": "gate"}
    similarity: float

class AnalogicalReasoner:
    def find_analogy(library, target_sks, threshold=0.7) -> list[AnalogyMap]
    def adapt_skill(skill, role_mapping) -> Skill
```

### Тестовая среда CardGateWorld

`CardGateWorld` использует `Key("purple")` и `Door("purple", is_locked=True)`.
В `GridPerception.perceive()` ветка по `(cell.type, cell.color)`:
- `("key", "purple")` → `SKS_CARD_PRESENT`
- `("door", "purple", locked)` → `SKS_GATE_LOCKED`
- `("door", "purple", open)` → `SKS_GATE_OPEN`

Несущий агент: если `carrying.type == "key" and carrying.color == "purple"` → `SKS_CARD_HELD`.

В `GridPerception.find_object("card")` ищет `Key("purple")`.

### Интеграция в SkillAgent

Новый шаг в `run_episode()` между `find_applicable` и backward chaining:

```
1. Прямой путь к goal — продолжить
2. find_applicable(current_sks, goal_sks) — применить skill (attempt==0)
3. [NEW] AnalogicalReasoner.find_analogy(...) — применить adapted skill (attempt==0)
4. Backward chaining (GoalAgent)
5. Explore
```

---

## Файлы

### Новые
- `src/snks/language/role_registry.py` — ROLE_REGISTRY константы
- `src/snks/language/analogical_reasoner.py` — AnalogyMap, AnalogicalReasoner
- `src/snks/env/card_gate_world.py` — CardGateWorld(MiniGridEnv)
- `src/snks/experiments/exp68_analogy_found.py`
- `src/snks/experiments/exp69_analogy_solve.py`
- `src/snks/experiments/exp70_regression.py`
- `tests/test_analogical_reasoner.py`

### Изменения
- `src/snks/language/grid_perception.py` — SKS 55-58, цветовая ветка, find_object("card"/"gate")
- `src/snks/language/blocking_analyzer.py` — gate как blocker
- `src/snks/language/skill_agent.py` — шаг аналогии

---

## Gate-критерии экспериментов

| Exp | Gate | Threshold |
|-----|------|-----------|
| 68  | analogy similarity | >= 0.7 |
| 68  | analogy found | True |
| 69  | success rate on CardGateWorld | >= 0.8 |
| 69  | avg analogies used per episode | >= 1 |
| 70  | regression: DoorKey success rate | >= 0.9 |
| 70  | regression: MultiRoom success rate | >= 0.8 |
