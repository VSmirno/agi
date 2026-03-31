# Stage 29: Curiosity-Driven Exploration

**Date:** 2026-03-31
**Status:** v1.0

---

## Цель

Агент должен автономно исследовать среду без внешней награды, мотивируясь новизной состояний.

---

## Архитектура

### CuriosityModule

Хранит счётчик посещений `state_key → count`.
Внутренняя награда: `r_int = 1.0 / (1 + count)`.

State key = `frozenset(sks) | {agent_pos_hash}` — включает позицию агента чтобы различать клетки с одинаковыми SKS предикатами (пустая комната).

```python
class CuriosityModule:
    def observe(state_key: frozenset) -> float   # returns r_int, increments count
    def intrinsic_reward(state_key: frozenset) -> float  # peek without updating
    def n_distinct() -> int                       # unique states seen
    def most_novel_direction(grid, agent_pos, agent_dir) -> int  # best next action
```

### CuriosityAgent

Extends SkillAgent. Переопределяет `_explore()`:
- Вместо random: для каждого направления вычислить `intrinsic_reward(next_state)`
- Выбрать действие с максимальной ожидаемой новизной
- Используется только когда все остальные пути провалились

### ZeroRewardWrapper

`gymnasium.Wrapper` — обнуляет `reward` кроме терминального успеха.
Для Exp 73.

---

## Gate-критерии

| Exp | Gate | Threshold |
|-----|------|-----------|
| 71 | n_distinct states ≥ N | ≥ 10 (Empty-5x5, 200 steps) |
| 71 | r_int(new) formula | == 1.0 |
| 71 | r_int(repeated) formula | == 0.5 |
| 72 | curious/random distinct ratio | ≥ 1.3 |
| 72 | curious coverage | ≥ 0.25 (25% клеток) |
| 73 | success rate (0 external reward) | ≥ 0.7 |

---

## Файлы

### Новые
- `src/snks/language/curiosity_module.py`
- `src/snks/language/curiosity_agent.py`
- `src/snks/experiments/exp71_curiosity_unit.py`
- `src/snks/experiments/exp72_curiosity_vs_random.py`
- `src/snks/experiments/exp73_curiosity_goal.py`
- `tests/test_curiosity_module.py`
