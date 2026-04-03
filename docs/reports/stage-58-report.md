# Stage 58: SDM Retrofit — Learned Agent for DoorKey

## Результат: PASS (с оговорками)

**Ветка:** `stage58-sdm-retrofit`
**Тип:** RETROFIT — первый learned stage после 11-этапного symbolic drift

## Architectural Integrity

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| DAF | нет | R1 негативный вердикт |
| VSA | **задействован** | AbstractStateEncoder: abstract state → 256-dim binary vector |
| SDM | **задействован** | SDMMemory: 1000 locations, subgoal-level transitions |
| Planner | **задействован** | SDM subgoal selection (explore/key/door/goal) |
| Learning phase | **есть** | 50 exploration episodes, 1453+ transitions |
| Drift counter | **RESET → 0** | Первый learned stage после 11 symbolic (47-57) |

## Что доказано

- SDM **реально учится** subgoal transitions из exploration episodes
- Learned agent: 100% DoorKey-5x5 partial obs (200 seeds)
- SDM добавляет +11.5% поверх symbolic heuristic fallback (88.5% → 100%)
- SDM capacity: 7751 writes (gate ≥1000) PASS
- Exploration phase: 50 episodes достаточно, 0.7 секунд на CPU

## Честный анализ: что learned, что symbolic

### Learned (через SDM):
- Subgoal selection: SDM выбирает GOTO_KEY / GOTO_DOOR / GOTO_GOAL на основе reward signal
- Transition memory: SDM записывает (abstract_state, subgoal) → (next_state, reward)
- +11.5% improvement over pure heuristic

### Symbolic (hardcoded):
- Frontier exploration (навигация к неизведанным клеткам)
- BFS pathfinding (маршрутизация)
- Reflexes: toggle doors, pickup keys when facing them
- Heuristic subgoal fallback: key → door → goal priority ordering

### Вердикт
Это **hybrid agent** — честный компромисс. SDM работает на уровне subgoal selection, symbolic navigation на уровне actions. Это лучше чем pure symbolic (Stages 47-57), но далеко от fully learned agent. Следующие этапы должны заменять symbolic компоненты learned аналогами.

## Эксперименты

| Exp | Explore | Success | Baseline | SDM writes | Gate | Статус |
|-----|---------|---------|----------|------------|------|--------|
| 112a | 50 ep | 100% (200/200) | 88.5% | 7,751 | ≥30% | PASS |
| 112b | 100 ep | 100% (200/200) | 88.5% | 8,783 | ≥30% | PASS |
| — | — | — | — | — | ≥1000 writes | PASS |

## Ключевые решения

1. **Subgoal-level SDM** — v1 (raw action SDM) давал 18% vs 31% random = WORSE. v2 (subgoal SDM) = 100%. SDM не может планировать low-level actions, но эффективен для high-level subgoal ordering.
2. **Abstract state encoding** — compact features (has_key, door_state, distances) вместо raw 7×7 grid. Позволяет SDM обобщать между layouts.
3. **GPU ROCm mismatch** — torch 2.6.0+rocm6.1 segfaults на minipc GPU. Forced CPU, reduced params (dim=256, 1000 locations). TD для fix ROCm.
4. **Heuristic fallback** — когда SDM не уверен, используется symbolic heuristic. Это даёт высокий baseline (88.5%), но SDM всё равно добавляет value.

## Известные проблемы

1. **ROCm GPU не работает** — torch 2.6.0+rocm6.1 vs GPU arch mismatch. Нужен update PyTorch/ROCm.
2. **Symbolic navigation доминирует** — SDM выбирает subgoal, но BFS/frontier делает всю работу. Нужен learned navigation в будущих stages.
3. **Heuristic fallback слишком сильный** — 88.5% baseline = SDM improvement выглядит маргинальным. Нужно тестировать на задачах где heuristic не работает (non-obvious subgoal ordering).

## Файлы изменены

- `src/snks/agent/sdm_doorkey_agent.py` — NEW: AbstractStateEncoder, SDMDoorKeyAgent, SDMDoorKeyEnv
- `src/snks/agent/vsa_world_model.py` — UPDATE: GPU device support для VSACodebook, SDMMemory
- `tests/test_stage58_sdm_retrofit.py` — NEW: 14 тестов
- `src/snks/experiments/exp112_sdm_doorkey.py` — NEW: exploration + evaluation + baseline

## Следующий этап

Ретроспектива Stages 47-57: пометить как symbolic baselines в ROADMAP, затем поэтапная замена symbolic → learned компонентов.
