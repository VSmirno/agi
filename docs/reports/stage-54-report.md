# Stage 54: Partial Observability

## Результат: PASS

**Ветка:** `stage54-partial-observability`
**Milestone:** M4 — Масштаб (первый этап)

## Что доказано
- Агент решает DoorKey-5x5 с 7x7 partial observation (100% на 200 random layouts)
- SpatialMap корректно накапливает частичные наблюдения во всех 4 направлениях
- FrontierExplorer эффективно исследует неизвестные области
- Переключение explore→plan работает: агент исследует до обнаружения объектов, затем BFS к цели
- FullyObsWrapper shortcut устранён (TD-006 partially addressed)

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 108a | SpatialMap coverage | 25.0/25 (100%) | — | INFO |
| 108b | DoorKey-5x5 partial obs (200 seeds) | 200/200 = 100% | ≥80% | PASS |
| 108c | Ablation: full vs random | 100% vs 0% | — | INFO |

Mean steps: 23.5 (vs 16 steps с full obs в Stage 47)

## Ключевые решения

1. **SpatialMap + FrontierExplorer** — аккумуляция 7x7 partial views в абсолютную карту. Optimistic planning: неизвестные клетки = пустые.
2. **Координатная трансформация** — MiniGrid obs encoding `img[i,j] = view(col=i, row=j)`, agent at `view(3,6)`. Верифицировано для всех 4 направлений.
3. **Adjacent-cell navigation** — для key/door BFS ведёт к смежной клетке (нельзя встать на key в MiniGrid), затем turn-to-face + pickup/toggle.
4. **No SDM integration** — чисто символический подход. SDM integration отложена до Stage 58.
5. **5x5 grid + 7x7 view = high coverage** — на 5x5 grid 7x7 view покрывает большую часть карты за 1-2 шага. Настоящий тест partial obs будет в Stage 55 (MultiRoom с partial obs).

## Ограничения

- **5x5 too small** — 7x7 view на 5x5 grid означает высокое покрытие. Stage 55 (MultiRoom-N3, ~25x25 grid) будет настоящим вызовом.
- **TD-006 partially closed** — DoorKey с partial obs работает. MultiRoom с partial obs (gate ≥60%) остаётся в TD-006 до Stage 55.
- **Deterministic BFS** — без learning, без generalization. Чисто алгоритмический planning.

## Веб-демо
- `demos/stage-54-partial-obs.html` — Canvas с двумя видами: реальная карта + карта в памяти агента. Replay 3 эпизодов.

## Файлы изменены
- `src/snks/agent/spatial_map.py` — NEW: SpatialMap, FrontierExplorer, view_to_world
- `src/snks/agent/partial_obs_agent.py` — NEW: PartialObsAgent, PartialObsDoorKeyEnv
- `tests/test_stage54_partial_obs.py` — NEW: 34 теста
- `src/snks/experiments/exp108_partial_obs.py` — NEW: gate experiments
- `demos/stage-54-partial-obs.html` — NEW: web demo
- `demos/index.html` — UPDATED: добавлена карточка Stage 54
- `docs/superpowers/specs/2026-04-02-stage54-partial-observability-design.md` — NEW: spec

## Следующий этап
- **Stage 55: Exploration Strategy** — MultiRoom-N3 с partial obs. Gate: ≥60% success. Это настоящий тест partial obs — grid ~25x25, агент видит только 7x7.
