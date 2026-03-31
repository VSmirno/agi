# Stage 33: Multi-Agent Communication

**Статус:** IN PROGRESS
**Ветка:** stage33-multi-agent
**Эксперименты:** exp83, exp84, exp85
**Дата:** 2026-03-31

---

## Что доказывает

Агенты СНКС могут обмениваться **концептами** (не словами) через HAC embeddings и каузальные модели, при этом:
1. Получатель успешно использует переданные знания без повторного обучения
2. Обмен происходит на уровне концептов (SKS + CausalLinks), а не на уровне текста
3. Несколько агентов совместно решают задачу быстрее, чем один агент

---

## Философия

> "Мышление = оперирование концептами (СКС), НЕ словами"

Multi-Agent Communication в СНКС принципиально отличается от NLP-подхода (обмен текстом).
Агенты обмениваются **структурированными концептными сообщениями** — HAC-вектора + каузальные связи.
Это ближе к биологическому: обмен "смыслами", а не "словами".

---

## Архитектура

### ConceptMessage — единица обмена

```python
@dataclass
class ConceptMessage:
    sender_id: str
    receiver_id: str | None          # None = broadcast
    content_type: str                 # "causal_links" | "skill" | "warning" | "request"
    sks_context: frozenset[int]       # SKS описывающие контекст
    hac_embedding: Tensor | None      # 2048-dim HAC вектор содержания
    causal_links: list[CausalLink]    # каузальные знания
    skill: Skill | None               # навык (если передаётся)
    urgency: float                    # 0.0 (инфо) .. 1.0 (критично)
    timestamp: int                    # логический такт
```

### AgentCommunicator — протокол обмена

```
Agent A                     Agent B
   │                           │
   │ ──ConceptMessage──────►   │
   │   (causal_links)          │
   │                           │ integrate_knowledge()
   │                           │ → merge causal model
   │   ◄──ConceptMessage───   │
   │   (skill transfer)       │
   │                           │
   │ ──ConceptMessage──────►   │
   │   (warning: danger zone)  │
   │                           │ adjust_policy()
```

### MultiAgentEnv — среда для нескольких агентов

Расширение MiniGrid-подобной среды:
- N агентов в одной сетке
- Общая среда, раздельные наблюдения
- Коммуникационный канал (ConceptMessage queue)
- Совместная задача: открыть дверь + достичь цели

---

## Подходы (brainstorming)

### Подход A: Shared Causal Model (простой)
- Агенты периодически синхронизируют каузальные модели
- Merge по confidence: max(confidence_A, confidence_B)
- **Pro:** простая реализация, проверенный формат
- **Con:** O(N²) broadcast, нет избирательности

### Подход B: Concept-Level Messaging (рекомендуемый) ✓
- Агенты отправляют целевые ConceptMessage
- Получатель интегрирует через HAC similarity matching
- Типы: causal_links, skill, warning, request
- **Pro:** избирательный обмен, масштабируемый, ближе к биологии
- **Con:** нужен протокол маршрутизации

### Подход C: Emergent Communication (сложный)
- Агенты учат собственный протокол через RL
- **Pro:** потенциально оптимальный
- **Con:** нестабильный, долго обучается, не подходит для MVP

**Выбран: Подход B** — Concept-Level Messaging
- Обоснование: баланс между реализмом и простотой. Агенты обмениваются конкретными знаниями (каузальные связи, навыки), а не сырыми весами. Избирательность важна: не всё знание релевантно всем агентам.

---

## Gate-критерии

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 83 | concept_transfer_accuracy | ≥ 0.9 | Переданные CausalLinks корректно используются получателем |
| 83 | hac_alignment | ≥ 0.7 | Cosine similarity HAC embeddings между агентами |
| 84 | multi_agent_speedup | ≥ 1.3x | 2 агента решают задачу быстрее 1 агента |
| 84 | knowledge_reuse_rate | ≥ 0.5 | Доля переданных знаний, реально использованных |
| 85 | cooperative_success | ≥ 0.9 | Успешность совместного решения (2 агента, разные роли) |
| 85 | no_word_exchange | = True | Проверка: обмен только концептами, не текстом |

---

## Модули

1. `src/snks/language/concept_message.py` — ConceptMessage, MessageType
2. `src/snks/language/agent_communicator.py` — AgentCommunicator (send/receive/integrate)
3. `src/snks/language/multi_agent_env.py` — MultiAgentEnv (N агентов, общая среда)
4. `tests/test_multi_agent.py` — unit tests
5. `src/snks/experiments/exp83_concept_transfer.py`
6. `src/snks/experiments/exp84_multi_agent_speedup.py`
7. `src/snks/experiments/exp85_cooperative.py`

---

## Зависимости от предыдущих этапов

- Stage 25: GoalAgent (backward chaining) — агенты планируют к целям
- Stage 26: TransferAgent — метрики переноса знаний
- Stage 27: Skill/SkillLibrary — навыки как единица обмена
- Stage 28: AnalogicalReasoner — cross-domain mapping при обмене
- Stage 30: FewShotAgent — быстрое обучение от демонстраций = "обучение от другого агента"
- Stage 32: MetaLearner — выбор стратегии с учётом переданных знаний
