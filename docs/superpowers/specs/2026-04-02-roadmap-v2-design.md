# СНКС Roadmap v2 — Milestone-Driven

**Дата:** 2026-04-02
**Статус:** APPROVED

---

## Контекст

После 46 стейджей и 126+ экспериментов проект достиг точки переосмысления. Stages 44-46 выявили:

1. **DAF не может планировать** — 0% на multi-step задачах без world model
2. **VSA+SDM+Subgoals решают задачу** — 92.5% DoorKey-5x5
3. **FHN не осциллирует** — excitable regime, timescale mismatch 50x
4. **Блоки 1-5 — exploration phase** — идеи верные, реализации на toy tasks

### Принятые решения

- **Модульная bio-inspired архитектура** вместо "всё через DAF"
  - DAF = сенсорная кора (восприятие, формирование концептов)
  - VSA = ассоциативная кора (структурное кодирование)
  - SDM = гиппокамп (эпизодическая память, предсказание)
  - Subgoal Planner = префронтальная кора (планирование)
  - Language = зоны Брока/Вернике (интерфейс)
- **Сначала глубина (генерализация), потом контрольное расширение (язык)**
- **Stage 46 = реальный фундамент**, Блоки 1-5 = разведка
- **Осцилляторная динамика — параллельный research-трек**
- **Milestone-driven** вместо линейных фаз

---

## Архитектурная модель

```
┌─────────────────────────────────────────────┐
│            Субъект (Agent)                   │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ DAF      │  │ VSA+SDM  │  │ Subgoal   │ │
│  │ сенсорная│→ │ память + │→ │ планиров- │ │
│  │ кора     │  │ ассоциац.│  │ щик (PFC) │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│       ↑                           ↑         │
│  ┌──────────┐              ┌───────────┐    │
│  │ Encoder  │              │ Language   │    │
│  │ вход     │              │ интерфейс │    │
│  └──────────┘              └───────────┘    │
└─────────────────────────────────────────────┘
```

---

## Milestones

### M1: Генерализация

**Gate-критерий:**
- ≥80% success на random DoorKey-5x5 (200 эпизодов, каждый — новая карта)
- ≥60% success на multi-room (минимум 3 комнаты)
- Без хардкода позиций и layout-specific логики

**Stages:**

| Stage | Название | Gate | Что делает |
|-------|----------|------|------------|
| 47 | Wall-aware навигация | ≥80% DoorKey-5x5 с walls в разных позициях | BFS/A* через SDM world model |
| 48 | Random layouts | ≥80% на 200 рандомных карт | random_layout=True, explore на незнакомых картах |
| 49 | Multi-room | ≥60% MultiRoom-N3 | Длинные subgoal chains, SDM scaling |

**Переиспользуется:** VSA encoder, SDM, SubgoalExtractor из Stages 45-46

**Риски:**
- SDM capacity на random layouts → увеличить hard locations или добавить забывание
- BFS через SDM может быть медленным → ограничить глубину, fallback на heuristic

---

### M2: Языковой контроль

**Gate-критерий:**
- Инструкция "go to the key" → агент идёт к ключу
- ≥70% success на DoorKey-5x5 с языковой инструкцией вместо встроенного reward
- Языковая инструкция задаёт порядок subgoals

**Зависимости:** M1 Stage 47 (wall-aware навигация)

**Stages:**

| Stage | Название | Gate | Что делает |
|-------|----------|------|------------|
| 50 | Reconnect language pipeline | Парсинг "pick up the key" → VSA-вектор, ≥90% | Аудит + адаптация Stages 7-24, единый VSA codebook |
| 51 | Language-guided planning | ≥70% random DoorKey-5x5 с инструкцией | Instructed mode: язык → subgoals → навигация |

**Переиспользуется:** Language pipeline (Stages 7-24), SubgoalNavigator, PlanGraph из Stage 46

**Риски:**
- Language pipeline мог устареть → начать с аудита
- Grounding: "key" должно мапиться на тот же концепт что SubgoalExtractor → единый VSA codebook

---

### M3: Концепция доказана

**Gate-критерий:**
- M1 gates выполнены
- M2 gates выполнены
- **Интеграционный тест:** языковая инструкция + random карта + multi-room — ≥50% success
- **Осцилляторный вердикт:** документ с выводами R1

**Stages:**

| Stage | Название | Gate | Что делает |
|-------|----------|------|------------|
| 52 | Integration test | ≥50% random MultiRoom-N3 с инструкцией | Полный pipeline, integration bugs |
| 53 | Architecture report | Документ с решениями для M4 | Что доказано, bottlenecks, результаты R1 |

**Что это даёт:** checkpoint "go / no-go" перед масштабированием.

---

### M4: Масштаб (детализация после M3)

**Gate-критерий:**
- Новый env существенно сложнее DoorKey/MultiRoom (5+ типов объектов)
- 100K+ осцилляторов в DAF на GPU
- Архитектура из M3 работает без структурных изменений

**Направления:**
- MiniGrid BossLevel или кастомный env
- SDM scaling (иерархическая SDM)
- Subgoal chains длиной 10+
- GPU-оптимизация на minipc (ROCm, AMD 96GB)

---

### M5: Автономия (детализация после M4)

**Gate-критерий:**
- Агент сам ставит цели (curiosity / intrinsic motivation)
- Novel tasks без переобучения (zero-shot через комбинацию subgoals)
- Self-model: агент "знает" что умеет

**Направления:**
- Curiosity module (новая реализация)
- Compositional subgoals
- Meta-cognition: оценка уверенности в плане

---

## Research-трек R1: Осцилляторная динамика

Параллельно с M1-M3, на minipc. Не блокирует основную работу.

| ID | Вопрос | Как проверить |
|----|--------|---------------|
| R1.1 | Oscillatory FHN (I_base > 1.0)? | SKS-формирование, Stage 44 тесты |
| R1.2 | Timescale fix (steps_per_cycle=5000)? | Coupling propagation, те же тесты |
| R1.3 | SKS quality при oscillatory params? | DBSCAN stability metric |
| R1.4 | Downstream impact на VSA+SDM? | DoorKey success rate |
| R1.5 | GPU tech debt (TD-002, TD-003) | 50K nodes success rate |

**Принцип:** результаты вливаются в M3 (architecture report). Если oscillatory DAF значительно лучше — переключаемся в M4.

---

## Что убрано из старого роадмапа

| Убрано | Причина |
|--------|---------|
| Фаза 2 "Мультимодальный мир" (камера, аудио) | Преждевременно |
| Фаза 3 "Непрерывный опыт" (sleep/consolidation) | STDP работает, sleep — research |
| Фаза 4 "Языковой интерфейс" как отдельная фаза | Стал частью M2 |
| Фаза 7 "Human-level AGI" | Слишком далеко |
| "Pure DAF ≥50%" как критерий | Заменён модульной архитектурой |
| 100%-результаты Блоков 1-5 как база | Exploration phase |

## Что сохранено

| Что | Откуда | Роль |
|-----|--------|------|
| DAF-ядро (FHN, STDP, coupling) | Stages 0-6, аудит Stage 44 | Сенсорный модуль |
| Language pipeline | Stages 7-24 | M2 |
| VSA + SDM | Stage 45 | Память и ассоциации |
| Subgoal planning | Stage 46 | Планировщик |
| "Язык = интерфейс" | SPEC.md | Архитектурный принцип |
| "Каузальное планирование, не reward shaping" | Feedback | Архитектурный принцип |
| GPU pipeline на minipc | Tech debt | R1 + M4 |

---

## Нумерация stages

Stages продолжают сквозную нумерацию (47, 48, 49...) для совместимости с существующими отчётами и экспериментами.
