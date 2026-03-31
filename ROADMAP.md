# СНКС — Roadmap к AGI

**Последнее обновление:** 2026-03-31
**Статус:** Stages 0-35 COMPLETE, Block 6 IN PROGRESS

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
| 36 | Spatial Abstraction | IN PROGRESS | Region-based perception сжимает state space, агент обучается на 12x12 |
| 37 | Scalable Exploration | PLANNED | Frontier-based + goal-directed exploration гарантирует coverage на больших средах |
| 38 | Curriculum Learning | PLANNED | Автономный curriculum 5→8→12→16, hierarchical causal model |

---

## Статус: Блок 6 в разработке

**Stages 0-35: COMPLETE** (94 эксперимента ALL PASS)
**Exp 92 GPU Scaling: DONE** (N=200K, 34/34 exps PASS on AMD ROCm)
**Блок 6: IN PROGRESS** — решение проблемы 0% success на 12x12

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
