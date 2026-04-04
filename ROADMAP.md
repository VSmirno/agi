# СНКС — Roadmap v2 (Milestone-Driven)

**Последнее обновление:** 2026-04-03
**Статус:** M1 COMPLETE → M2 COMPLETE → M3 COMPLETE → M4 IN PROGRESS (8/9) — architecture pivot at Stage 59
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

### M1: Генерализация ⚠️ SYMBOLIC BASELINE
**Gate:** ≥80% random DoorKey-5x5, ≥60% MultiRoom-N3
**⚠️ Stages 47-49 = pure symbolic BFS, no learned components. Gates passed via hardcoded pathfinding.**

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 47 | Wall-aware навигация | COMPLETE ⚠️ SYMBOLIC | 100% DoorKey-5x5 — BFS pathfinding, no learning |
| 48 | Random layouts | COMPLETE ⚠️ SYMBOLIC | merged with 47 |
| 49 | Multi-room | COMPLETE ⚠️ SYMBOLIC | 100% MultiRoom-N3 — BFS + door toggle, no learning |

### M2: Языковой контроль ⚠️ SYMBOLIC BASELINE
**Gate:** ≥70% success с языковой инструкцией
**⚠️ Stages 50-51 = regex parsing + fixed random VSA vectors, no learned encoding.**

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 50 | Reconnect language pipeline | COMPLETE ⚠️ SYMBOLIC | regex parsing, fixed binary vectors (not trained) |
| 51 | Language-guided planning | COMPLETE ⚠️ SYMBOLIC | BFS + regex, no SDM/VSA learning |

### M3: Концепция доказана ⚠️ SYMBOLIC BASELINE
**Gate:** M1 + M2 + интеграция ≥50% random MultiRoom-N3 с инструкцией + R1 вердикт

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 52 | Integration test | COMPLETE ⚠️ SYMBOLIC | env detection + delegation to symbolic agents |
| 53 | Architecture report | COMPLETE (2026-04-02) | R1 negative, M4 plan |

### M4: Масштаб
**Gate:** новый env 5+ типов объектов, partial observability, subgoal chains 5+
**R1 решение:** негативный вердикт → DAF = perception only

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 54 | Partial Observability | COMPLETE ⚠️ SYMBOLIC | 100% DoorKey — SpatialMap + BFS, no learning |
| 55 | Exploration Strategy | COMPLETE ⚠️ SYMBOLIC | 100% MultiRoom — FrontierExplorer + BFS |
| 56 | Complex Environment | COMPLETE ⚠️ SYMBOLIC | 99.5% PutNext — state machine + BFS |
| 57 | Long Subgoal Chains | COMPLETE ⚠️ SYMBOLIC | 40% KeyCorridor — ChainPlanner + BFS |
| 58 | SDM Retrofit | COMPLETE ⚠️ NEGATIVE (2026-04-03) | SDM не добавляет value — heuristic=100% на DoorKey, honest ablation |
| 59 | VSA Causal Induction | COMPLETE (2026-04-03) | **100% generalization unseen colors**, bind(X,X)=identity proof |
| 60 | World Model via Demos | COMPLETE (2026-04-03) | **100% QA-A/B/C**, causal world model из 5 синтетических демо |
| 61 | Demo-Guided Agent | COMPLETE (2026-04-03) | **100% DoorKey + 100% LockedRoom**, causal planning + BFS, ablation delta=100% |
| 62 | CLS World Model | COMPLETE (2026-04-04) | **100% QA L1-L4**, neocortex+hippocampus+consolidation, 2-tier CLS |
| 63 | Abstraction + Crafter | COMPLETE (2026-04-04) | **100% Crafter QA**, 25 auto-categories, 97% MiniGrid regression |

### Roadmap v4: Scaffolding Removal (2026-04-04)
**Crafter = основной домен.** MiniGrid — только regression.

| Stage | Убираем | Gate |
|-------|---------|------|
| 64 | Синтетику → демо + exploration | COMPLETE (2026-04-04) — 93% Crafter QA, 97% MG, 9 discovered, 5 taught, 0 synthetic |
| 65 | 100% уверенность → uncertainty | Brier < 0.15, confidence~accuracy ρ>0.7 |
| 66 | Symbolic features → пиксели | ≥50% Crafter QA from pixels |

### Architecture Pivot (Stage 59, 2026-04-03)
**Stages 47-58: символический BFS + SDM = тупик.** SDM паразитирует на BFS — либо BFS решает всё (SDM не нужен), либо BFS не справляется (SDM не получает данных).

**Новый подход: обучение через демонстрации, не exploration.**
- Stage 59: VSA bind(X,X)=identity ✅ — few-shot generalization доказана (3 demos → 100% unseen colors)
- Stage 60: Causal world model ✅ — 5 правил из синтетических демо, 3 уровня QA PASS
- Stage 61: Agent использует world model — exploration только для layout, не для rule discovery
- Аналогия: младенца УЧАТ, а не он сам открывает правила за миллион попыток

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
