# Stage 34: Long-Horizon Planning

## Результат: PASS

## Что доказано
- HierarchicalPlanner генерирует планы на 1800+ примитивных шагов (gate ≥ 1000)
- 3-уровневая иерархия: Strategic → Tactical → Primitive с 100% coherence
- Re-planning при deviation: 100% recovery success, overhead 1.67x (gate ≤ 2.0x)
- Multi-room planning: 100% success для 3-8 комнат
- Hierarchical speedup: 16.3x throughput vs flat BFS (gate ≥ 2x)
- Plan generation sub-millisecond даже для 1800+ step plans

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 86 | plan_depth | 1802 | ≥ 1000 | PASS |
| 86 | plan_coherence | 1.000 | ≥ 0.9 | PASS |
| 87 | replan_success | 1.000 | ≥ 0.8 | PASS |
| 87 | replan_overhead | 1.67x | ≤ 2.0x | PASS |
| 88 | multi_room_success | 1.000 | ≥ 0.9 | PASS |
| 88 | hierarchical_speedup | 16.3x | ≥ 2x | PASS |

## Ключевые решения
- **Skill-Based Hierarchical Planning** (не HTN, не MCTS) — естественное расширение существующей архитектуры. Skills (Stage 27) дают абстракции, CausalWorldModel даёт forward prediction, backward chaining (Stage 25) даёт goal decomposition.
- **Loss-based deviation** — отклонение считается только при потере ожидаемых SKS, а не при получении новых. Threshold 0.5 — позволяет мелкие колебания.
- **PlanGraph с propagation** — mark_complete_up автоматически поднимает статус DONE по дереву.

## Веб-демо
- `demos/stage-34-long-horizon.html` — интерактивный plan builder: выбор размера среды, визуализация 3-уровневой иерархии, симуляция execution и re-planning.

## Файлы изменены
- `src/snks/language/plan_node.py` — PlanNode, PlanGraph, PlanStatus
- `src/snks/language/hierarchical_planner.py` — HierarchicalPlanner
- `tests/test_hierarchical_planner.py` — 23 unit tests
- `src/snks/experiments/exp86_plan_depth.py`
- `src/snks/experiments/exp87_replan.py`
- `src/snks/experiments/exp88_multi_room.py`
- `demos/stage-34-long-horizon.html`
- `docs/stages/stage34-long-horizon-planning.md`

## Следующий этап
- Stage 35: Integration Demo — все capabilities в одном когерентном агенте. Ключевой вопрос: как собрать все 34 стадии (DAF → perception → language → causal → skills → meta-learning → multi-agent → planning) в единый демонстрационный pipeline?
