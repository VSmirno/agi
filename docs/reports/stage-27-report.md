# Stage 27: Skill Abstraction

## Результат: PASS

## Что доказано
- Примитивные навыки автоматически извлекаются из причинной модели
- Составные навыки формируются путём цепочек до глубины 3
- Навыки переносятся из DoorKey-5x5 → MultiRoomDoorKey без переобучения
- SkillAgent в 1.54x быстрее GoalAgent при наличии кэша навыков
- 0 эпизодов исследования при повторном использовании навыков

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 65 | primitives ≥ 2 | 2 | ≥ 2 | PASS |
| 65 | composites ≥ 1 | 1 | ≥ 1 | PASS |
| 65 | min success_rate | 1.00 | ≥ 0.80 | PASS |
| 66 | speedup | 1.54x | ≥ 1.5x | PASS |
| 66 | exploration_episodes | 0 | ≤ 1 | PASS |
| 67 | success rate | 1.000 | ≥ 0.90 | PASS |
| 67 | avg skills/ep | 1.0 | ≥ 1.0 | PASS |

## Ключевые решения

- **Skill.preconditions/effects как frozenset[int]**: SKS-предикаты вместо строк — совместимо с CausalWorldModel
- **find_applicable: composites first**: составные навыки пробуются первыми, примитивные — как fallback
- **Attempt==0 ограничение**: навыки применяются только 1 раз; если не помогло — GoalAgent с target_pos
- **State-change verification**: после применения навыка проверяем что SKS изменился, иначе не считаем успехом
- **_after_episode extraction**: навыки извлекаются из CausalWorldModel после каждого эпизода

## Файлы изменены

- `src/snks/language/skill.py` — NEW: Skill, SkillLibrary dataclasses
- `src/snks/language/skill_library.py` — NEW: extract_from_causal_model, compose_skills, find_applicable
- `src/snks/language/skill_agent.py` — NEW: SkillAgent extends GoalAgent; skill-first + fallback
- `src/snks/experiments/exp65_skill_extraction.py` — PASS
- `src/snks/experiments/exp66_skill_reuse.py` — PASS
- `src/snks/experiments/exp67_skill_transfer.py` — PASS

## Следующий этап

Stage 28: Analogical Reasoning — способность применять структурные паттерны из одной ситуации к другой.
Ключевой вопрос: может ли агент узнать аналогию "ключ→дверь" в новом контексте с другими объектами?
