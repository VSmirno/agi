# Stage 45: VSA World Model

## Результат: PARTIAL PASS (foundation PASS, planning FAIL)

**Ветка:** `stage45-vsa-world-model`
**Merge commit:** pending

## Что доказано

- VSA (Binary Spatter Code) кодирует MiniGrid наблюдения с 97-99% точностью unbinding
- SDM (Sparse Distributed Memory) хранит transitions с 0.85+ similarity при read-back
- World model корректно записывает и предсказывает state transitions
- Explore phase: random agent с записью в SDM достигает ~40% success на DoorKey-5x5

## Что НЕ работает

- Forward beam search (goal similarity): гонит агента прямо к цели → упирается в дверь. Не может планировать detour (ключ→дверь→цель)
- 1-step SDM reward lookahead: reward только у цели, промежуточные шаги имеют 0 reward → не может выучить цепочку
- Trace matching (backward chaining v1): VSA similarity ~0.5 между случайными состояниями, порог 0.6 недостаточно селективен. Matching даёт шумные результаты

## Ключевая находка

**DoorKey — это detour task.** Прямой путь к цели (forward search) блокирован дверью. Нужен каузальный план: ключ → дверь → цель. Это требует:
1. **Subgoal extraction**: выделить ключевые события из успешных traces (pickup key, toggle door)
2. **Plan graph**: start → subgoal1 → subgoal2 → goal
3. **Subgoal-conditioned navigation**: отдельный навигатор для каждого подцели

Это scope для Stage 46.

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 105a | VSA unbinding accuracy | 97-99% | ≥ 90% | PASS |
| 105b | SDM prediction similarity | 0.85-0.87 | ≥ 0.6 | PASS |
| 105c explore | DoorKey success (random) | ~40% (50 ep) | — | info |
| 105c plan (beam search) | DoorKey success | 0-20% | ≥ 15% plan | FAIL |
| 105c plan (SDM reward) | DoorKey success | 0-20% | ≥ 15% plan | FAIL |
| 105c plan (trace match) | DoorKey success | 0-10% | ≥ 15% plan | FAIL |

## Запланированные эксперименты (tech debt)

| TD | Exp | Что проверяется | Gate | Статус |
|----|-----|-----------------|------|--------|
| TD-005 | 105c+ | BackwardChain с subgoal extraction | plan rate ≥ 15% | ⏳ запланирован |

## Ключевые решения

1. **VSA = Binary Spatter Code** (XOR bind, majority bundle) — простые, обратимые операции, no backprop
2. **SDM с auto-calibration radius** — 1-5% активации обеспечивает баланс capacity/noise
3. **Explore→Plan two-phase** — заполняем world model до планирования
4. **Отказ от reward shaping** (промежуточные rewards) — это RL, не model-based planning
5. **Отказ от novelty bonus** — заливал SDM шумовыми rewards
6. **Forward beam search не работает для detour** — это архитектурное ограничение, не баг

## Веб-демо
- `demos/stage-45-vsa-world-model.html` — Canvas визуализация DoorKey, SDM метрики, VSA вектор

## Файлы изменены
- `src/snks/agent/vsa_world_model.py` — VSACodebook, VSAEncoder, SDMMemory, SDMPlanner, CausalPlanner, BackwardChainPlanner, WorldModelAgent
- `src/snks/experiments/exp105_vsa_world_model.py` — exp105a/b/c
- `tests/test_stage45_vsa.py` — 30 тестов PASS
- `demos/stage-45-vsa-world-model.html` — веб-демо
- `scripts/exp105_run.sh` — runner для minipc

## Следующий этап

**Stage 46: Subgoal Planning** — извлечение подцелей из успешных traces + plan graph + subgoal-conditioned navigation. Используя VSA+SDM foundation из Stage 45.
