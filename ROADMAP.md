# СНКС — Roadmap v2 (Milestone-Driven)

**Последнее обновление:** 2026-04-03
**Статус:** M1 COMPLETE → M2 COMPLETE → M3 COMPLETE → M4 IN PROGRESS (4/7)
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
| 47 | Wall-aware навигация | COMPLETE (2026-04-02) | 100% на 200 random DoorKey-5x5, mean 16 steps |
| 48 | Random layouts | COMPLETE (merged with 47) | 100% на 200 рандомных карт (covered by Stage 47) |
| 49 | Multi-room | COMPLETE (2026-04-02) | 100% на 200 random MultiRoom-N3, mean 16 steps |

### M2: Языковой контроль
**Gate:** ≥70% success с языковой инструкцией
**Зависимости:** M1 Stage 48

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 50 | Reconnect language pipeline | COMPLETE (2026-04-02) | 100% encode/decode accuracy (30 instructions) |
| 51 | Language-guided planning | COMPLETE (2026-04-02) | 100% на 200 random DoorKey-5x5 с инструкцией |

### M3: Концепция доказана
**Gate:** M1 + M2 + интеграция ≥50% random MultiRoom-N3 с инструкцией + R1 вердикт

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 52 | Integration test | COMPLETE (2026-04-02) | 100% random MultiRoom-N3 с инструкцией |
| 53 | Architecture report | COMPLETE (2026-04-02) | R1 negative, M4 plan (Stages 54-60), go for M4 |

### M4: Масштаб
**Gate:** новый env 5+ типов объектов, partial observability, subgoal chains 5+
**R1 решение:** негативный вердикт → DAF = perception only, фокус на symbolic pipeline scaling

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 54 | Partial Observability | COMPLETE (2026-04-02) | 100% DoorKey-5x5 с 7x7 view (200 seeds, mean 23.5 steps) |
| 55 | Exploration Strategy | COMPLETE (2026-04-02) | 100% MultiRoom-N3 с partial obs (200 seeds, mean 22.1 steps) |
| 56 | Complex Environment | COMPLETE (2026-04-03) | 99.5% PutNextS6N3, 18 object types, mean 14.8 steps |
| 57 | Long Subgoal Chains | COMPLETE (2026-04-03) | 40% KeyCorridorS4R3, 54% S3R3, 5-6 subgoals |
| 58 | SDM Scaling | | SDM capacity ≥1000 unique transitions |
| 59 | Transfer Learning | | ≥70% new env без re-exploration |
| 60 | M4 Integration Test | | ≥50% BabyAI BossLevel с инструкцией |

### M5: Автономия (vision, детализация после M4)
Self-directed goals, compositional subgoals, meta-cognition

---

## Research-трек R1: Осцилляторная динамика
**Вердикт: НЕГАТИВНЫЙ** (Stage 53, 2026-04-02)

FHN в текущей конфигурации — excitable regime, не oscillatory. Coupling timescale mismatch 50×. SKS формируются через SDR injection, не coupling. DAF остаётся как perception layer.

| ID | Вопрос | Статус |
|----|--------|--------|
| R1.1 | Oscillatory FHN (I_base > 1.0) | NEGATIVE — I_base=0.5 = excitable (Stage 44 audit) |
| R1.2 | Timescale fix (steps_per_cycle=5000) | NOT APPLIED — 50× slowdown, impractical |
| R1.3 | SKS quality при oscillatory params | NOT MEASURED — SKS = input-driven |
| R1.4 | Downstream impact на VSA+SDM | N/A — VSA+SDM work without DAF oscillations |
| R1.5 | GPU tech debt (TD-002, TD-003) | SUPERSEDED — modular architecture replaced Pure DAF |

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
