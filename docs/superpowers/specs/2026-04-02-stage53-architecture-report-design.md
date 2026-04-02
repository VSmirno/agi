# Stage 53: Architecture Report — Design Spec

**Дата:** 2026-04-02
**Milestone:** M3 — Концепция доказана (финальный этап)
**Тип:** Документация + анализ (не код)

---

## Цель

Написать **architecture report** — checkpoint-документ "go / no-go" перед M4 (масштабирование). Документ фиксирует:

1. Что доказано в M1-M3 (конкретные результаты и метрики)
2. R1 вердикт (осцилляторная динамика)
3. Архитектурные решения: что работает, что нет, что заменить
4. Bottlenecks и ограничения
5. Конкретный план M4 (environments, scaling, timeline)

---

## Позиция в фазе

**Фаза M3 — Концепция доказана**

Маркеры завершения:
- ✅ M1 gates (100% DoorKey-5x5, 100% MultiRoom-N3)
- ✅ M2 gates (100% language-guided DoorKey-5x5)
- ✅ Integration test (100% MultiRoom-N3 с инструкцией, gate ≥50%)
- ❌ R1 вердикт → **этот этап** (Stage 53)

После Stage 53 → **M3 COMPLETE**, переход к M4.

---

## Подходы

### A: Минимальный отчёт (2-3 страницы)
- Таблица результатов M1-M3 + вердикт R1
- Trade-off: быстро, но не даёт основу для M4

### B: Полный Architecture Decision Record (ADR)
- Каждый модуль: что доказано, ограничения, решения для M4
- R1 вердикт с конкретными числами
- Bottleneck analysis с приоритетами
- M4 plan с конкретными stages
- Trade-off: 1-2 часа, но даёт полную картину для масштабирования

### C: Формальная статья (research paper format)
- Trade-off: 4+ часа, избыточно для internal checkpoint

**Выбран: B** — полный ADR. Это checkpoint document, который определит направление следующих 10+ stages. Инвестиция в качество документа окупится.

---

## Структура Architecture Report

```
docs/architecture-report-m3.md

1. Executive Summary
   - Что СНКС, текущий статус, ключевой результат
   
2. Архитектура (текущая)
   - Модульная схема: DAF → VSA → SDM → Planner → Language
   - Каждый модуль: назначение, ключевые параметры, результат
   
3. Результаты M1-M3
   - M1: Генерализация (Stages 47-49)
   - M2: Язык (Stages 50-51)
   - M3: Интеграция (Stage 52)
   - Сводная таблица gate-критериев
   
4. R1 Вердикт: Осцилляторная динамика
   - Вопросы R1.1-R1.5, ответы, доказательства
   - Заключение: negative — excitable only, не oscillatory
   - Решение для M4
   
5. Что работает, что нет
   - Working: VSA encoding, SDM memory, BFS navigation, subgoal planning, language grounding
   - Not working: naked DAF planning, FHN oscillations, coupling propagation
   - Partial: exploration (full obs shortcut TD-006)
   
6. Bottlenecks для M4
   - Full observability (TD-006)
   - DAF role redefinition
   - Environment complexity
   
7. M4 Plan
   - Candidate environments
   - Architecture changes needed
   - Proposed stages (54-60)
   
8. Open Questions
```

---

## Gate-критерий Stage 53

- Документ `docs/architecture-report-m3.md` написан
- R1 вердикт зафиксирован (positive/negative + evidence)
- M4 plan содержит ≥3 конкретных stages с gate-критериями
- Веб-демо визуализирует архитектуру и результаты

---

## Эксперименты

Stage 53 — документационный этап. Новых экспериментов не требуется. Данные берутся из:
- Stages 44-52 отчёты
- Tech debt register
- Existing experiment results

CPU-тест: regression — 1129 tests still pass.

---

## Веб-демо

`demos/stage-53-architecture.html` — интерактивная визуализация:
- Диаграмма архитектуры (модули + связи)
- Таблица результатов M1-M3 с gate indicators
- R1 timeline (вопросы → ответы)
- M4 plan preview

---

## Risks

- R1 negative verdict → DAF role нужно переопределить для M4
- TD-006 (full obs shortcut) → генерализация может быть illusory
- M4 env selection → нет очевидного кандидата после DoorKey/MultiRoom
