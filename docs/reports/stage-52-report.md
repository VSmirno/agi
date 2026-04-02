# Stage 52: Integration Test — Language-Guided MultiRoom + DoorKey

## Результат: PASS

**Ветка:** `stage52-integration-test`
**Milestone:** M3 — Концепция доказана

## Что доказано
- Единый IntegrationAgent решает и MultiRoom-N3 (3 комнаты), и DoorKey-5x5 по текстовой инструкции
- 100% success на 200 random MultiRoom-N3 с "go to the goal" (gate ≥50%)
- 100% success с вариантами инструкций ("open the door then go to the goal", "toggle the door then go to the goal")
- 100% DoorKey regression — нет деградации от интеграции
- Язык → подцели → адаптивная навигация — полный pipeline работает end-to-end
- Pipeline масштабируется без изменения кода: DoorKey → MultiRoom

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 110a | MultiRoom-N3 + "go to the goal" (200 random) | 100% (200/200) | ≥50% | PASS |
| 110b | MultiRoom-N3 + variant instructions (200 random) | 100% (200/200) | ≥50% | PASS |
| 110c | DoorKey-5x5 regression (200 random) | 100% (200/200) | ≥90% | PASS |
| 110a | Mean steps (MultiRoom) | 16.7 | — | info |
| 110c | Mean steps (DoorKey) | 16.0 | — | info |
| all | Total time | 10.8s | — | info |

## Ключевые решения
- **Unified agent с env detection** — IntegrationAgent определяет тип среды (key present → DoorKey, else → MultiRoom) и выбирает стратегию. Минимальный новый код (~50 lines), максимальное переиспользование.
- **Нет нового обучения** — агент не обучается, а комбинирует proven компоненты (LanguageGrounder + SubgoalNavigator + MultiRoomNavigator). Интеграция = composition, не training.
- **CPU-only** — BFS на 25x25 grid < 1ms, GPU не нужен для этого этапа.

## Веб-демо
- `demos/stage-52-integration.html` — Canvas replay 5 эпизодов (3 MultiRoom + 2 DoorKey) с инструкциями, подцелями, trail агента

## Файлы изменены
- `src/snks/agent/integration_agent.py` — NEW: IntegrationAgent
- `tests/test_integration_agent.py` — NEW: 12 unit tests
- `src/snks/experiments/exp110_integration.py` — NEW: gate experiments
- `demos/stage-52-integration.html` — NEW: web demo
- `docs/superpowers/specs/2026-04-02-stage52-integration-test-design.md` — NEW: spec

## Следующий этап
- **Stage 53: Architecture report** (M3 финал) — документ с решениями для M4 (масштаб). R1 вердикт (осцилляторная динамика). После 53 → M3 COMPLETE.
