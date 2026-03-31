# СНКС — Roadmap к AGI

**Последнее обновление:** 2026-03-31
**Статус:** Stages 0-32 COMPLETE (85 экспериментов PASS)

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

## Блок 5: Social & Integration (PLANNED)
| Stage | Название | Статус | Что доказывает |
|-------|----------|--------|----------------|
| 33 | Multi-Agent Communication | PLANNED | Агенты обмениваются концептами (не словами) |
| 34 | Long-Horizon Planning | PLANNED | Планирование на 1000+ шагов с иерархией |
| 35 | Integration Demo | PLANNED | Все capabilities в одном когерентном агенте |

---

## Текущий фокус: Stage 33 — Multi-Agent Communication

**Спецификация:** TBD
**Ветка:** TBD
**Эксперименты:** exp83-85

### Gate-критерии:
- TBD — Агенты обмениваются концептами (не словами)

---

## Как обновлять этот файл

При завершении каждого этапа:
1. Обновить статус в таблице (COMPLETE + дата)
2. Добавить краткий результат экспериментов
3. Перенести "Текущий фокус" на следующий этап
4. Commit с сообщением `docs: update ROADMAP — Stage N COMPLETE`
