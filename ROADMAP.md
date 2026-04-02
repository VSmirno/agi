# СНКС — Roadmap v2 (Milestone-Driven)

**Последнее обновление:** 2026-04-02
**Статус:** M1 IN PROGRESS
**Полная спецификация:** [docs/superpowers/specs/2026-04-02-roadmap-v2-design.md](docs/superpowers/specs/2026-04-02-roadmap-v2-design.md)

---

## Принципы

- **Модульная bio-inspired архитектура:** DAF (сенсорная кора) + VSA (ассоциативная кора) + SDM (гиппокамп) + Subgoal Planner (PFC) + Language (Брока/Вернике)
- Каждый milestone — чёткий gate-критерий (достигнут или нет)
- Каузальное планирование через subgoals, НЕ reward shaping
- Язык = интерфейс, НЕ основа мышления
- Stage 46 = реальный фундамент, Блоки 1-5 = exploration phase

---

## Milestones

### M1: Генерализация
**Gate:** ≥80% random DoorKey-5x5, ≥60% MultiRoom-N3

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 47 | Wall-aware навигация | | ≥80% DoorKey-5x5 с walls в разных позициях |
| 48 | Random layouts | | ≥80% на 200 рандомных карт |
| 49 | Multi-room | | ≥60% MultiRoom-N3 |

### M2: Языковой контроль
**Gate:** ≥70% success с языковой инструкцией
**Зависимости:** M1 Stage 48

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 50 | Reconnect language pipeline | | Парсинг → VSA-вектор, ≥90% |
| 51 | Language-guided planning | | ≥70% random DoorKey-5x5 с инструкцией |

### M3: Концепция доказана
**Gate:** M1 + M2 + интеграция ≥50% random MultiRoom-N3 с инструкцией + R1 вердикт

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 52 | Integration test | | ≥50% random MultiRoom-N3 с инструкцией |
| 53 | Architecture report | | Документ с решениями для M4 |

### M4: Масштаб (детализация после M3)
**Gate:** новый env 5+ типов объектов, GPU scaling (conditional on R1)

### M5: Автономия (vision, детализация после M4)
Self-directed goals, compositional subgoals, meta-cognition

---

## Research-трек R1: Осцилляторная динамика
Параллельно с M1-M3, на minipc. Checkpoint: R1.3 или негативный вывод до M3 Stage 53.

| ID | Вопрос | Статус |
|----|--------|--------|
| R1.1 | Oscillatory FHN (I_base > 1.0) | |
| R1.2 | Timescale fix (steps_per_cycle=5000) | |
| R1.3 | SKS quality при oscillatory params | |
| R1.4 | Downstream impact на VSA+SDM | |
| R1.5 | GPU tech debt (TD-002, TD-003) | |

---

## Exploration phase (Stages 0-46)

### Блок 1: Foundation (COMPLETE)
| Stage | Название | Статус |
|-------|----------|--------|
| 0-6 | Core DAF + Agent | COMPLETE |
| 7-9 | Language + Prediction | COMPLETE |
| 10-13 | Hierarchical Planning | COMPLETE |
| 14-15 | Embodied Agent + Debt | COMPLETE |
| 16-17 | Memory + Validation | COMPLETE |

### Блок 2: Language Pipeline (COMPLETE)
| Stage | Название | Статус |
|-------|----------|--------|
| 19-24 | Grounding, Compositional, Verbalizer, QA, Scaffold, BabyAI | COMPLETE |

### Блок 3-5: Reasoning, Self-Directed, Social (COMPLETE)
| Stage | Название | Статус |
|-------|----------|--------|
| 25-35 | Goal Composition → Integration Demo | COMPLETE (2026-03-31) |

### Блок 6: Scaling & Real Learning (COMPLETE)
| Stage | Название | Статус | Результат |
|-------|----------|--------|-----------|
| 36 | Spatial Abstraction | COMPLETE | 100% DoorKey-16x16 |
| 37 | Multi-Room Navigation | COMPLETE | 96% MultiRoom-N4 |
| 38 | Pure DAF Agent | COMPLETE | Чистый DAF pipeline, 12% DoorKey-5x5 |
| 39 | Curriculum Learning | COMPLETE | EpsilonDecay + PE exploration |
| 40 | Learnable Encoding | COMPLETE | HebbianEncoder +14% SDR |
| 41 | Temporal Credit Assignment | COMPLETE | Eligibility traces λ=0.92 |
| 42 | Spatial Representation | COMPLETE | Symbolic+CNN encoders |
| 43 | Working Memory | COMPLETE | Selective reset, sustained activation |
| 44 | Foundation Audit | COMPLETE | 26/26 PASS, naked DAF 0% — world model needed |
| 45 | VSA World Model | COMPLETE | 97% encoding, planning FAIL (detour) |
| 46 | Subgoal Planning | COMPLETE | **92.5% DoorKey-5x5** |

---

## Как обновлять этот файл

При завершении каждого stage:
1. Обновить статус в таблице milestone (COMPLETE + дата)
2. Добавить краткий результат
3. Commit с сообщением `docs: update ROADMAP — Stage N COMPLETE`
