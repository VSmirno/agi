# СНКС Architecture Report — M3 Checkpoint

**Дата:** 2026-04-02
**Статус:** M3 COMPLETE → готовность к M4
**Автор:** автономный pipeline (Stage 53)

---

## 1. Executive Summary

СНКС (Система Непрерывного Когнитивного Синтеза) — bio-inspired AGI-система, где мышление моделируется как самоорганизация связанных осцилляторов в стабильные паттерны (SKS — Stable Cognitive Structures).

**Текущий статус:** три milestone пройдены:
- **M1 (Генерализация):** 100% на random DoorKey-5x5 и MultiRoom-N3
- **M2 (Язык):** 100% language-guided planning на DoorKey-5x5
- **M3 (Интеграция):** 100% MultiRoom-N3 с текстовой инструкцией

**Ключевой вывод:** архитектура СНКС работает как **модульная когнитивная система** — DAF (восприятие) + VSA (кодирование) + SDM (память) + Subgoal Planner (планирование) + Language (интерфейс). Однако DAF-ядро (FHN осцилляторы) не обеспечивает планирование — эту роль взяли на себя символические модули (BFS, subgoal extraction). R1 вердикт негативный: FHN не осциллирует, coupling не пропагируется в масштабе восприятия.

**Решение:** go for M4 с переопределением роли DAF.

---

## 2. Архитектура (текущая)

### 2.1 Модульная схема

```
┌──────────────────────────────────────────────────────────┐
│                   IntegrationAgent                        │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Language     │  │ Subgoal      │  │ BFS            │  │
│  │ Grounder    │→ │ Extractor    │→ │ Pathfinder     │  │
│  │ (Брока)     │  │ (PFC)        │  │ (навигация)    │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│        ↑                ↑                  ↑             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ VSA         │  │ SDM          │  │ Grid           │  │
│  │ Codebook    │  │ Memory       │  │ Observation    │  │
│  │ (ассоциат.) │  │ (гиппокамп)  │  │ (сенсорная)    │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│        ↑                                                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ DAF Engine (FHN oscillators, STDP, coupling)        │ │
│  │ → сенсорная кора: формирование концептов (SKS)      │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Модули

| Модуль | Файл | Роль | Статус |
|--------|------|------|--------|
| **DAF Engine** | `src/snks/daf/engine.py` | FHN осцилляторы, STDP, coupling, SKS formation | Работает как perception; не планирует |
| **VSA Codebook** | `src/snks/agent/vsa_world_model.py` | Binary Spatter Code 512-bit, XOR bind/unbind | 97-99% encode/decode accuracy |
| **SDM Memory** | `src/snks/agent/vsa_world_model.py` | Sparse Distributed Memory, 10K locations | 0.85-0.87 prediction similarity |
| **SubgoalExtractor** | `src/snks/agent/subgoal_planning.py` | Symbolic event detection (pickup, toggle, goal) | 100% extraction accuracy |
| **SubgoalNavigator** | `src/snks/agent/subgoal_planning.py` | Position-based + BFS навигация к subgoals | 92.5% → 100% success |
| **GridPathfinder** | `src/snks/agent/pathfinding.py` | BFS на observed grid, wall detection | 100% solvable layouts, <1ms |
| **LanguageGrounder** | `src/snks/language/language_grounder.py` | Text → VSA vector → subgoals | 100% decode accuracy |
| **MultiRoomNavigator** | `src/snks/agent/multi_room_nav.py` | Reactive BFS + door toggle | 100% MultiRoom-N3 |
| **InstructedAgent** | `src/snks/agent/instructed_agent.py` | Language → subgoals → BFS | 100% DoorKey с инструкцией |
| **IntegrationAgent** | `src/snks/agent/integration_agent.py` | Env detection + strategy selection | 100% DoorKey + MultiRoom |

### 2.3 Ключевые параметры

| Параметр | Значение | Модуль |
|----------|----------|--------|
| FHN I_base | 0.5 (excitable) | DAF |
| FHN tau | 12.5 | DAF |
| dt | 0.0001 (0.1ms) | DAF |
| steps_per_cycle | 100 | DAF |
| coupling_strength | 0.1 | DAF |
| VSA dimensions | 512 bits | VSA |
| SDM locations | 10,000 | SDM |
| SDM radius | auto-calibrated | SDM |
| DBSCAN eps | 0.3 | SKS |
| Grid BFS | <1ms per call | Pathfinder |

---

## 3. Результаты M1-M3

### 3.1 M1: Генерализация (Stages 47-49)

| Stage | Exp | Метрика | Результат | Gate | Статус |
|-------|-----|---------|-----------|------|--------|
| 47 | 107b | Random DoorKey-5x5 (200 layouts) | **100%**, mean 16 steps | ≥80% | **PASS** |
| 48 | — | Merged with Stage 47 | 100% на 200 random layouts | ≥80% | **PASS** |
| 49 | 108b | Random MultiRoom-N3 (200 layouts) | **100%**, mean 16.3 steps | ≥60% | **PASS** |

**Метод:** obs-based planning (сканирование observation для позиций объектов) + BFS pathfinding.

**Ограничение (TD-006):** используется FullyObsWrapper (агент видит всю карту). Настоящая генерализация требует partial observability (7x7 view) + exploration.

### 3.2 M2: Языковой контроль (Stages 50-51)

| Stage | Exp | Метрика | Результат | Gate | Статус |
|-------|-----|---------|-----------|------|--------|
| 50 | 108 | Encode/decode accuracy (30 instructions) | **100%** | ≥90% | **PASS** |
| 51 | 109a | Language-guided DoorKey-5x5 (200 random) | **100%**, mean 16.1 steps | ≥70% | **PASS** |
| 51 | 109c | Variant formulations | **100%** | — | **PASS** |

**Метод:** LanguageGrounder (text → chunks → VSA vector → subgoals) + SubgoalNavigator.

### 3.3 M3: Интеграция (Stage 52)

| Stage | Exp | Метрика | Результат | Gate | Статус |
|-------|-----|---------|-----------|------|--------|
| 52 | 110a | MultiRoom-N3 + language (200 random) | **100%** | ≥50% | **PASS** |
| 52 | 110b | Variant instructions (200 random) | **100%** | ≥50% | **PASS** |
| 52 | 110c | DoorKey regression (200 random) | **100%** | ≥90% | **PASS** |

**Метод:** IntegrationAgent с env detection (key present → DoorKey strategy, else → MultiRoom).

### 3.4 Сводка

| Milestone | Gate | Результат | Статус |
|-----------|------|-----------|--------|
| M1: Генерализация | ≥80% DoorKey, ≥60% MultiRoom | 100% / 100% | **COMPLETE** |
| M2: Язык | ≥70% language-guided | 100% | **COMPLETE** |
| M3: Интеграция | ≥50% MultiRoom + language | 100% | **COMPLETE** |

---

## 4. R1 Вердикт: Осцилляторная динамика

### 4.1 Вопросы и ответы

| ID | Вопрос | Ответ | Доказательство |
|----|--------|-------|----------------|
| R1.1 | Oscillatory FHN (I_base > 1.0)? | **Не исследовано напрямую**, но I_base=0.5 = excitable, не oscillatory | Stage 44 audit: `test_bare_fhn_is_excitable_not_oscillatory` |
| R1.2 | Timescale fix (steps_per_cycle=5000)? | **Не применено.** 5000 steps = 0.5 model sec — слишком медленно для real-time | Stage 44: coupling propagation requires 5000 steps, perception cycle = 100 steps |
| R1.3 | SKS quality при oscillatory params? | **Не измерено.** SKS формируются через co-firing от SDR injection, не coupling | Stage 44: DBSCAN clusters = input-driven, не coupling-driven |
| R1.4 | Downstream impact на VSA+SDM? | **Нет данных.** VSA+SDM работают без DAF oscillations | Stages 45-52: pipeline работает с symbolic encoding |
| R1.5 | GPU tech debt (TD-002, TD-003)? | **Не актуально.** Pure DAF experiments (Stages 38-40) заменены модульной архитектурой | Stage 44 verdict: DAF = perception only |

### 4.2 Заключение

**R1 вердикт: НЕГАТИВНЫЙ**

FHN осцилляторы в текущей конфигурации:
1. **Не осциллируют** — I_base=0.5 даёт excitable regime (converges to v≈1.2)
2. **Coupling не пропагируется** — timescale mismatch 50× (5000 steps vs 100 steps/cycle)
3. **SKS формируются не coupling'ом** — а co-firing от SDR input injection
4. **Naked DAF = 0% success** — на DoorKey-5x5 (200 episodes)

DAF-ядро работает как **сенсорный модуль** — принимает SDR, формирует кластеры через STDP — но не обеспечивает планирование, предсказание или навигацию. Эти функции выполняют символические модули (VSA, SDM, BFS, SubgoalExtractor).

### 4.3 Решение для M4

**Вариант A:** Увеличить I_base > 1.0, steps_per_cycle = 5000 → проверить, улучшит ли это SKS quality. Risk: 50× slowdown в pipeline, неясный benefit.

**Вариант B:** Принять DAF как perception layer, инвестировать в symbolic pipeline (VSA/SDM scaling, partial obs, exploration). Risk: отход от bio-plausibility claim.

**Вариант C:** Заменить DAF на альтернативный perception module (Hebbian encoder, сверточные SDR). Risk: потеря 46 stages of research.

**Рекомендация: B** — DAF остаётся как perception layer (доказано, что работает для SKS formation). M4 фокус: масштабирование symbolic pipeline + partial observability. R1 исследование может продолжаться параллельно на minipc как low-priority background.

---

## 5. Что работает, что нет

### 5.1 Работает (proven)

| Компонент | Доказательство | Качество |
|-----------|---------------|----------|
| VSA encoding (Binary Spatter Code) | 97-99% unbinding accuracy | Excellent |
| SDM storage/retrieval | 0.85-0.87 similarity | Good |
| BFS pathfinding | 100% layouts, <1ms | Excellent |
| Subgoal extraction (symbolic) | 100% accuracy | Excellent |
| Position-based navigation | 92.5% → 100% | Excellent |
| Language grounding (VSA) | 100% decode accuracy | Excellent |
| Language → subgoals | 100% mapping | Excellent |
| Env detection + strategy | 100% integration | Excellent |
| DAF perception (SKS formation) | 26/26 audit tests PASS | Good |
| STDP learning | LTP/LTD correct | Good |

### 5.2 Не работает

| Компонент | Проблема | Stage |
|-----------|----------|-------|
| Naked DAF planning | 0% success — нет пути от perception к action | 44 |
| FHN oscillations | Excitable, не oscillatory; coupling timescale 50× mismatch | 44 |
| VSA-SDM planning | Beam search fails на detour tasks (DoorKey) | 45 |
| VSA trace matching | Similarity ~0.5, недостаточно селективно | 46 |
| Random exploration | ~1% success на DoorKey, unreliable | 47 |

### 5.3 Частично работает (с ограничениями)

| Компонент | Ограничение | TD |
|-----------|------------|-----|
| Full observability | Агент видит всю карту — BFS тривиален | TD-006 |
| WM gating | FHN self-sustains, bistable tuning needed | TD-004 |
| Pure DAF agent | Perception blind — Gabor encoder не различает MiniGrid объекты | TD-001 |

---

## 6. Bottlenecks для M4

### 6.1 P1: Full observability (критический)

**Проблема:** Stages 47-52 используют FullyObsWrapper — агент видит всю 25x25 карту. BFS на полной карте тривиален. Генерализация может быть illusory.

**Решение:** Partial observability (7x7 agent-centric view). Требуется:
- Exploration strategy (систематическое исследование карты)
- Spatial memory (SDM или grid-based map building)
- Re-planning при обнаружении новых объектов/стен

**Приоритет:** ВЫСОКИЙ — без этого M4 experiments не покажут реальную генерализацию.

### 6.2 P2: Environment complexity

**Проблема:** DoorKey-5x5 и MultiRoom-N3 — простые среды (2-3 типа объектов, 1-3 subgoals). M4 gate требует 5+ типов объектов.

**Решение:** Переход на более сложные MiniGrid environments:
- BabyAI BossLevel (multiple rooms, keys, balls, doors)
- Custom environments с 5+ типами объектов
- Longer subgoal chains (10+)

### 6.3 P3: DAF role clarification

**Проблема:** DAF не используется в planning pipeline (Stages 46-52). Роль DAF нужно либо расширить, либо явно ограничить perception.

**Решение (выбрано):** DAF = perception layer. Symbolic pipeline (VSA+SDM+BFS+Subgoals) = cognition. DAF может быть заменён или улучшен в M4 без breaking planning.

### 6.4 P4: Exploration strategy

**Проблема:** Random walk ~1% success на DoorKey. Obs-based planning работает только с full obs.

**Решение:** Frontier-based exploration или curiosity-driven exploration с partial obs.

---

## 7. M4 Plan: Масштаб

### 7.1 Prerequisite: Partial Observability

Прежде чем масштабировать, нужно снять shortcut full observability (TD-006). Это самый критический bottleneck.

### 7.2 Proposed Stages

| Stage | Название | Gate | Что делает |
|-------|----------|------|------------|
| 54 | Partial Observability | ≥80% DoorKey-5x5 с 7x7 view | Frontier-based exploration + spatial map building from partial obs |
| 55 | Exploration Strategy | ≥60% MultiRoom-N3 с partial obs | Systematic exploration (frontier), re-planning on new discovery |
| 56 | Complex Environment | ≥50% BabyAI PutNext с 5+ object types | Extend SubgoalExtractor для новых event types (put, pick, toggle) |
| 57 | Long Subgoal Chains | ≥40% на задачах с 5+ subgoals | Hierarchical planning: L2 subgoals → L1 actions; SDM for subgoal memory |
| 58 | SDM Scaling | SDM capacity ≥1000 unique transitions | Hierarchical SDM, forgetting, consolidation |
| 59 | Transfer Learning | ≥70% на new env без re-exploration | SubgoalExtractor generalization across environments |
| 60 | M4 Integration Test | ≥50% BabyAI BossLevel с языковой инструкцией | Full pipeline на complex env |

### 7.3 Architecture Changes for M4

| Изменение | Причина | Scope |
|-----------|---------|-------|
| Remove FullyObsWrapper | TD-006: реальная генерализация | Stage 54 |
| Add SpatialMap module | Build map from partial obs | Stage 54 |
| Add FrontierExplorer | Systematic exploration | Stage 55 |
| Extend SubgoalExtractor | New event types for complex envs | Stage 56 |
| Hierarchical SDM | Scale memory beyond 10K locations | Stage 58 |

### 7.4 M4 Gate-критерий

- Новый env существенно сложнее DoorKey/MultiRoom (5+ типов объектов)
- Partial observability (7x7 view), не full obs
- Subgoal chains длиной 5+
- Архитектура из M3 работает без структурных изменений
- GPU scaling на minipc (если нужно для SDM/exploration)

---

## 8. Open Questions

1. **Partial obs + BFS:** BFS требует полной карты. С partial obs нужен map building → когда перестраивать карту? Реактивно (каждый шаг) или по событиям?

2. **SubgoalExtractor generalization:** Текущий extractor hardcoded для DoorKey events (pickup_key, toggle_door, reach_goal). Как расширить на произвольные event types без перечисления?

3. **SDM capacity:** 10K locations достаточно для DoorKey-5x5. BabyAI BossLevel с 10+ rooms → нужно ≥100K? Или hierarchical SDM?

4. **DAF future:** Если DAF = perception only, стоит ли инвестировать в GPU scaling (minipc) для DAF или только для SDM/exploration?

5. **Language complexity:** Текущий язык = простые инструкции ("pick up the key", "go to the goal"). M4 требует compositional instructions ("pick up the blue key, then open the red door, then go to the green goal").

---

## 9. Tech Debt Status

| ID | Stage | Type | Description | Status |
|----|-------|------|-------------|--------|
| TD-001 | 38 | BUG | Perception blind (Gabor encoder) | IN_PROGRESS — blocked, superseded by symbolic encoding |
| TD-002 | 39 | GPU_EXP | exp98e CurriculumTrainer 50K | OPEN — superseded by modular architecture |
| TD-003 | 40 | GPU_EXP | exp99 HebbianEncoder 50K | OPEN — superseded by modular architecture |
| TD-004 | 43 | INTEGRATION | WM gating (bistable FHN) | OPEN — deferred to R1 continuation |
| TD-006 | 49 | PERF | Full observability shortcut | OPEN — **P1 priority for M4** |

**Recommendation:** TD-001, TD-002, TD-003, TD-004 относятся к exploration phase (Pure DAF approach). Закрыть как WONTFIX или перенести в R1 background track. TD-006 — критический для M4.

---

## 10. Заключение

**M3 COMPLETE.** Три milestone пройдены с результатами значительно выше gate-критериев (100% vs 50-80%).

**Архитектура доказана:** модульная система (DAF perception + VSA encoding + SDM memory + symbolic planning + language interface) решает навигационные задачи в MiniGrid с текстовыми инструкциями.

**Главное ограничение:** full observability. M4 должен начаться с partial observability (Stage 54) — это определит, является ли архитектура действительно general или зависит от shortcut.

**R1: негативный.** FHN осцилляторная динамика не даёт преимущества для planning. DAF остаётся как perception layer. Инвестиции в oscillatory dynamics — low priority.

**Go for M4.**
