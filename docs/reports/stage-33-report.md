# Stage 33: Multi-Agent Communication

## Результат: PASS

## Что доказано
- Агенты обмениваются концептами (CausalLinks, Skills, Warnings) через ConceptMessage — без естественного языка
- Переданные каузальные связи корректно интегрируются и используются получателем (100% accuracy)
- HAC alignment между агентами = 1.0 (идентичные SKS → идентичные embeddings)
- 2 агента решают задачу в 1.59x быстрее одного (параллельная exploration + sharing)
- Cooperative success 100% — все 5 сценариев кооперации пройдены
- No text exchange verified — коммуникация исключительно на уровне концептов

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 83 | concept_transfer_accuracy | 1.000 | ≥ 0.9 | PASS |
| 83 | hac_alignment | 1.000 | ≥ 0.7 | PASS |
| 84 | multi_agent_speedup | 1.59x | ≥ 1.3x | PASS |
| 84 | knowledge_reuse_rate | 1.667 | ≥ 0.5 | PASS |
| 85 | cooperative_success | 1.000 | ≥ 0.9 | PASS |
| 85 | no_word_exchange | True | True | PASS |

## Ключевые решения
- **Concept-Level Messaging** (не shared model, не emergent communication) — баланс между реализмом и простотой. Агенты обмениваются конкретными знаниями (каузальные связи, навыки), а не сырыми весами или текстом.
- **4 типа сообщений**: causal_links, skill, warning, request — покрывают основные сценарии кооперации.
- **Confidence-weighted integration** — при merge каузальных моделей принимаются только более сильные или новые связи.
- **Request-Response protocol** — агент может запросить знания о конкретных SKS и получить релевантные CausalLinks в ответ.

## Веб-демо
- `demos/stage-33-multi-agent.html` — интерактивная симуляция кооперации двух агентов: Explorer и Solver обмениваются каузальными связями, навыками и предупреждениями для совместного решения DoorKey задачи.

## Файлы изменены
- `src/snks/language/concept_message.py` — ConceptMessage, MessageType
- `src/snks/language/agent_communicator.py` — AgentCommunicator (send/receive/integrate)
- `src/snks/language/multi_agent_env.py` — MultiAgentEnv (N агентов, общая среда)
- `src/snks/env/__init__.py` — env package init
- `src/snks/env/obs_adapter.py` — ObsAdapter (missing module fix)
- `tests/test_multi_agent.py` — 32 unit tests
- `src/snks/experiments/exp83_concept_transfer.py`
- `src/snks/experiments/exp84_multi_agent_speedup.py`
- `src/snks/experiments/exp85_cooperative.py`
- `demos/stage-33-multi-agent.html`
- `docs/stages/stage33-multi-agent-communication.md`

## Следующий этап
- Stage 34: Long-Horizon Planning — планирование на 1000+ шагов с иерархией. Ключевой вопрос: может ли TieredPlanner масштабироваться до длинных горизонтов с помощью CausalWorldModel + SkillLibrary?
