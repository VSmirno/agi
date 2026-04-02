# СНКС — Roadmap к AGI

**Последнее обновление:** 2026-04-01
**Статус:** Фаза 1 IN PROGRESS (Stages 0-40 COMPLETE)

---

## Принципы roadmap

- Каждый этап доказывает конкретную capability
- Каждый этап завершается экспериментами с числовыми gate-критериями
- Этапы выполняются последовательно (зависимости)
- Спецификация → тесты → реализация → эксперименты → отчёт

---

## Блок 1: Foundation (COMPLETE)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 0-6 | Core DAF + Agent | COMPLETE | Осцилляторы формируют концепты, STDP работает |
| 7-9 | Language + Prediction | COMPLETE | Текстовая модальность, предсказание в HAC |
| 10-13 | Hierarchical Planning | COMPLETE | Многоуровневое предсказание, Monte Carlo |
| 14-15 | Embodied Agent + Debt | COMPLETE | Агент в MiniGrid, техдолг закрыт |
| 16-17 | Memory + Validation | COMPLETE | Консолидация памяти, GPU scaling 50K |

## Блок 2: Language Pipeline (COMPLETE)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 19 | Zonal DAF + Grounding | COMPLETE | Cross-modal recall, complementary priming |
| 20 | Compositional Understanding | COMPLETE | Role-filler parsing, novel combinations |
| 21 | Verbalizer | COMPLETE | Концепт → текст генерация |
| 22 | Grounded QA | COMPLETE | Factual/simulation/reflective QA |
| 23 | Scaffold Removal | COMPLETE | Автономный tokenizer без sentence-transformers |
| 24 | BabyAI Execution | COMPLETE | Text → parse → plan → execute e2e |

## Блок 3: Autonomous Reasoning (IN PROGRESS)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 25 | Goal Composition | COMPLETE (2026-03-31) | Автономная декомпозиция целей, backward chaining |
| 26 | Transfer Learning | COMPLETE (2026-03-31) | Каузальные знания переносятся между средами |
| 27 | Skill Abstraction | COMPLETE (2026-03-31) | Иерархические макро-действия, 1.54x speedup, transfer 100% |
| 28 | Analogical Reasoning | COMPLETE (2026-03-31) | 100% transfer card/gate, similarity=0.75, no regression |

## Блок 4: Self-Directed Learning (COMPLETE)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 29 | Curiosity-Driven Exploration | COMPLETE (2026-03-31) | 2.48x vs random, 100% coverage, count-based r_int |
| 30 | Few-Shot Learning | COMPLETE (2026-03-31) | One-shot skill extraction 100%, cross-env transfer 100% |
| 31 | Abstract Pattern Reasoning | COMPLETE (2026-03-31) | Raven's-style 100% accuracy, HAC algebra, dual-rule, analogy |
| 32 | Meta-Learning | COMPLETE (2026-03-31) | Strategy selection 100%, 1.29x vs fixed, 5 task types |

## Блок 5: Social & Integration (COMPLETE)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 33 | Multi-Agent Communication | COMPLETE (2026-03-31) | Concept-level messaging, 1.59x speedup, 100% cooperation |
| 34 | Long-Horizon Planning | COMPLETE (2026-03-31) | 1800+ steps, 3-level hierarchy, re-planning 1.67x overhead |
| 35 | Integration Demo | COMPLETE (2026-03-31) | 10 capabilities, zero backprop, 100% integration |

## Блок 6: Scaling & Real Learning (IN PROGRESS)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 36 | Spatial Abstraction | COMPLETE (2026-03-31) | Curriculum 5→16, 100% success DoorKey-16x16, causal transfer |
| 37 | Multi-Room Navigation | COMPLETE (2026-03-31) | Closed door handling, 96% MultiRoom-N4, backward chaining scales |
| 38 | Pure DAF Agent | COMPLETE (2026-04-01) | Убран scaffolding, чистый DAF pipeline, reward-modulated STDP, env-agnostic |
| 39 | Curriculum Learning | COMPLETE (2026-04-01) | CurriculumScheduler, EpsilonDecay, PE exploration, 15 tests + 5 exps PASS |
| 40 | Learnable Encoding | COMPLETE (2026-04-01) | HebbianEncoder: Sanger's GHA, SDR discrimination +14%, 17 tests + 5 exps PASS |

### Следующие stages (Фаза 1)

| Stage | Название | Статус | Что нужно |
|-------|----------|--------|-----------|
| 41 | Temporal Credit Assignment | COMPLETE (2026-04-01) | Eligibility trace: λ=0.92, window=35 steps, 5x memory savings |
| 42 | Spatial Representation | COMPLETE (2026-04-01) | Perception fix: symbolic+CNN encoders, diagnostic confirms dual bottleneck |
| 43 | Working Memory | COMPLETE (2026-04-01) | Selective reset: WM zone preserved, sustained activation confirmed, gating needed |
| 44 | Foundation Audit | COMPLETE (2026-04-02) | 26/26 layer tests PASS, Golden Path 56.7%, Naked DAF 0% — world model needed |
| 45 | VSA World Model | COMPLETE (2026-04-02) | VSA+SDM foundation PASS (97% encoding, 0.85 prediction), planning FAIL (detour) |
| 46 | Subgoal Planning | COMPLETE (2026-04-02) | Symbolic subgoal extraction 100%, plan success 92.5% DoorKey-5x5 |

---

## Статус: Блок 6 в разработке

**Stages 0-44: COMPLETE** (126 экспериментов)
**Exp 92 GPU Scaling: DONE** (N=200K, 34/34 exps PASS on AMD ROCm)
**Stage 38: Pure DAF** — убран scaffolding, чистый DAF pipeline, 12% DoorKey-5x5

### Диагностика (Exp 92 → Block 6)
Агент не масштабируется на 12x12 из-за:
1. Взрыв пространства состояний (100 клеток vs 9 на 5x5)
2. Каузальная модель слишком разрежена за 100 эпизодов
3. Горизонт планирования (5) мал для 12x12 (нужно 15+)
4. Exploration = random walk на большом пространстве
5. Perception без пространственной абстракции

---

## Как обновлять этот файл

При завершении каждого этапа:
1. Обновить статус в таблице (COMPLETE + дата)
2. Добавить краткий результат экспериментов
3. Перенести "Текущий фокус" на следующий этап
4. Commit с сообщением `docs: update ROADMAP — Stage N COMPLETE`
