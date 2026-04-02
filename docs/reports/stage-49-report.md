# Stage 49: Multi-Room навигация

## Результат: PASS

**Ветка:** `stage49-multi-room`
**Merge commit:** pending

## Что доказано

- MultiRoomNavigator решает 100% из 200 случайных MultiRoom-N3 раскладок (gate ≥60%)
- BFS pathfinding масштабируется на 25x25 grid без модификаций
- Реактивный подход (BFS + toggle при подходе к двери) проще и надёжнее, чем SubgoalPlanning для задач без ключей
- Среднее 16.3 шагов, максимум 25, p95=21 — агент действует оптимально
- **M1 (Генерализация) COMPLETE** — все gate-критерии выполнены

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 108a | BFS pathfinding (50 layouts) | 100% | 100% | PASS |
| 108b | Navigation (200 layouts) | 100% (200/200), mean 16.3 steps | ≥60% | PASS |
| 108c | Steps analysis | mean=16.3, min=10, max=25, p95=21 | ≤150 | PASS |

## Ключевые решения

1. **Reactive BFS вместо SubgoalPlanning** — MultiRoom не требует ключей, только toggle дверей. BFS с `allow_door=True` находит путь через закрытые двери, агент открывает их реактивно при подходе. Минимальный код, максимальная robustness.
2. **FullyObsWrapper с транспонированием** — MiniGrid FullyObsWrapper возвращает grid в формате (col, row, 3). Транспонирование в (row, col, 3) обеспечивает совместимость с GridPathfinder из Stage 47.
3. **Epsilon=0.0 для детерминизма** — при полной наблюдаемости и BFS навигации random exploration не нужен. 100% success без epsilon.
4. **Отдельный MultiRoomNavigator** — не расширяем SubgoalPlanningAgent, а создаём новый легковесный класс. DoorKey-агент и MultiRoom-агент решают разные задачи разными методами.

## Веб-демо
- `demos/stage-49-multi-room.html` — Canvas с 6 раскладками, replay агента через 3 комнаты с дверями

## Файлы изменены
- `src/snks/agent/multi_room_nav.py` — NEW: MultiRoomNavigator, MultiRoomEnvWrapper, find_objects
- `tests/test_stage49_multi_room.py` — NEW: 20 тестов
- `src/snks/experiments/exp108_multi_room.py` — NEW: gate experiments
- `demos/stage-49-multi-room.html` — NEW: Canvas веб-демо
- `docs/superpowers/specs/2026-04-02-stage49-multi-room-nav-design.md` — спецификация

## Следующий этап

**M1 COMPLETE.** Переход к **M2: Языковой контроль** — Stage 50: Reconnect language pipeline (парсинг → VSA-вектор, ≥90%).
