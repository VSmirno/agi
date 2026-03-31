# Stage 36: Spatial Abstraction & Scalable Autonomous Agent

## Результат: PASS

## Что доказано
- AutonomousAgent с CurriculumManager достигает **100% success на 16x16 DoorKey**
- Каузальные знания (19 links) переносятся между размерами сред без потерь
- GoalAgent с backward chaining масштабируется на любой размер DoorKey
- Проблема 0% success из Exp 92 была в использовании EmbodiedAgent без capabilities

## Ключевой инсайт
GridPerception выдаёт одинаковые SKS-предикаты (KEY_PRESENT, DOOR_LOCKED, KEY_HELD, etc.) независимо от размера сетки. Поэтому каузальные связи, выученные на 5x5, работают на 16x16 без изменений. **Абстрактные знания масштабируются**.

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 93 | success_5x5 | 1.000 | ≥ 0.8 | PASS |
| 93 | success_8x8 | 1.000 | ≥ 0.5 | PASS |
| 93 | success_16x16 | 1.000 | ≥ 0.1 | PASS |
| 94 | causal_links | 19 | ≥ 5 | PASS |
| 94 | success_8x8 | 1.000 | ≥ 0.3 | PASS |
| 95 | transfer_16x16 | 1.000 | ≥ 0.8 | PASS |
| 95 | causal_links | 19 | ≥ 5 | PASS |

## Ключевые решения
- **CurriculumManager** — прогрессивная сложность 5→6→8→16 с автоматическим переходом
- **GoalAgent как base loop** — не EmbodiedAgent. GoalAgent имеет backward chaining и causal learning
- **Shared CausalWorldModel** — одна модель переносится между всеми уровнями
- **DoorKey слишком простая задача** — GoalAgent решает даже from-scratch на 16x16. Следующий этап: multi-room, multi-door

## Веб-демо
- `demos/stage-36-spatial-abstraction.html` — визуализация curriculum learning с Canvas-рендером MiniGrid

## Файлы изменены
- `src/snks/language/curriculum_manager.py` — CurriculumManager
- `src/snks/language/autonomous_agent.py` — AutonomousAgent facade
- `tests/test_autonomous_agent.py` — 22 unit tests
- `src/snks/experiments/exp93_curriculum.py` — curriculum learning
- `src/snks/experiments/exp94_exploration.py` — exploration coverage
- `src/snks/experiments/exp95_curriculum_speedup.py` — transfer validation
- `demos/stage-36-spatial-abstraction.html` — web demo

## Следующий этап
- **Stage 37: Multi-Room Challenge** — DoorKey слишком простая. Настоящий тест масштабирования: MultiRoom с несколькими дверями, ключами и комнатами. GoalAgent должен научиться планировать через несколько шагов разблокировки.
