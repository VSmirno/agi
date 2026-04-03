# Stage 60: Causal World Model via Demonstrations

## Результат: PASS

**Ветка:** `stage60-causal-world-model`
**Depends on:** Stage 59 (VSA bind(X,X)=identity proof)

## Что доказано
- Каузальная world model учится из 5 синтетических демонстраций (3-5 шагов каждая)
- Generalization на unseen цвета через VSA identity property: 100% accuracy
- Три уровня QA-валидации пройдены: факты, предусловия, каузальные цепочки
- Per-rule SDM архитектура (5 SDM, по одной на тип правила) работает без шума bundling

## Architectural Integrity

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| DAF | не задействован | perception не нужен для синтетических демо |
| VSA | задействован | кодирование правил, bind(X,X)=identity для generalization |
| SDM | задействован | 5 per-rule SDM для хранения каузальных правил |
| Planner | задействован | backward chaining по правилам для QA-C |
| Learning phase | есть | 170 SDM writes (90 color + 80 other rules) |
| Drift counter | RESET → 0 | Stage 59-60 = learned pipeline |

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 115a | QA-A: true/false unseen colors | 100% (9/9) | >=90% | PASS |
| 115b | QA-B: precondition lookup | 100% (8/8) | >=80% | PASS |
| 115c | QA-C: causal chains | 100% (3/3) | >=70% | PASS |

## Ключевые решения
- **Per-rule SDM вместо single bundled SDM** — избегаем шум unbinding при decode, каждое правило изолировано
- **Синтетические демонстрации** — минимальные, целенаправленные (как родитель учит ребёнка), не replay BFS-траекторий
- **Identity property как foundation** — bind(X,X)=zero_vector даёт generalization бесплатно
- **Include query color in candidates** — фикс для QA-B: unseen цвет добавляется в кандидаты при precondition lookup

## Веб-демо
- `demos/stage-60-causal-world-model.html` — граф каузальных правил + анимированный QA-лог

## Файлы изменены
- `src/snks/agent/causal_world_model.py` — CausalWorldModel, RuleEncoder (310 lines)
- `tests/test_stage60_causal_world_model.py` — 13 тестов, 7 test classes
- `src/snks/experiments/exp115_causal_world_model.py` — gate experiments
- `demos/stage-60-causal-world-model.html` — веб-демо
- `docs/superpowers/specs/2026-04-03-stage60-causal-world-model-design.md` — spec

## Следующий этап
- **Stage 61: Demo-Guided Agent** — agent получает world model из Stage 60, использует её для planning в grid среде. Exploration только для layout discovery, не для rule discovery.
