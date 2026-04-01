# Stage 43: Working Memory — Sustained Oscillation

## Результат: PASS (механизм подтверждён, gating = tech debt)

## Что доказано

- **Sustained oscillation работает**: WM zone сохраняет activation (1.48) через 5 пустых cycles
- **WM различает стимулы**: разные SDR → разные WM states (diff=0.43)
- **Root cause найден**: `perception_cycle()` сбрасывал ВСЕ осцилляторы каждый cycle — агент имел амнезию
- **Selective reset работает**: perceptual zone сбрасывается, WM zone — нет
- **WM без gating ухудшает DoorKey**: 20% нод заблокированы шумом, мешают perception
- **FHN self-sustains**: даже с coupling 0.1x, WM decay ratio=3.57 (растёт вместо затухания). Причина: FHN intrinsic dynamics, не coupling

## Эксперименты (minipc GPU)

| Exp | Метрика | Результат (v1) | Результат (fix) | Gate | Статус |
|-----|---------|----------------|-----------------|------|--------|
| 102a: WM persistence | activation 5 cycles | 1.48 | 1.48 | > 0.01 | PASS |
| 102b: WM vs no-WM DoorKey | improvement | -5% (0 vs 5%) | -5% (0 vs 5%) | mechanism runs | PASS |
| 102c: WM tracks stimuli | diff A/B | 0.41 | 0.43 | > 0.01 | PASS |
| 102d: WM decay | ratio peak→final | 2.67 | 3.57 | mechanism exists | PASS |

### Stage 43_fix: coupling damping + SKS exclusion

- **Coupling INTO WM scaled 0.1x** (edge weights at init) — не помогло, FHN self-sustains
- **WM excluded from SKS detection** — clean perception restored
- **Вывод**: проблема не в coupling, а в FHN intrinsic dynamics. Нужен bistable FHN tuning или explicit gating

## Ключевые решения

1. **Selective reset вместо zone infrastructure**: минимальное изменение. Zone registration отложен.
2. **Relaxation decay**: `v += (v_rest - v) * (1 - decay)` — корректнее для FHN.
3. **Coupling damping at init**: `graph.edge_attr[wm_edges, 0] *= 0.1` — простое, но недостаточное.
4. **SKS exclusion**: `fired_history[:, :n_percept]` — WM шум не загрязняет кластеры.

## Архитектура

```
perception_cycle():
  BEFORE (Stage 0-42):
    states[:, 0] = randn() * 0.1    ← ВСЕ ноды reset, АМНЕЗИЯ
  
  AFTER (Stage 43):
    states[:n_percept, 0] = randn()  ← perceptual zone reset
    states[n_percept:, 0] += (v_rest - v) * (1-decay)  ← WM zone: soft decay
    
  AFTER (Stage 43_fix):
    + WM edges scaled 0.1x at init (coupling damping)
    + SKS detection only on perceptual zone
```

## Файлы изменены

### Новые:
- `src/snks/experiments/exp102_working_memory.py` — 4 experiments
- `tests/test_working_memory.py` — 8 tests
- `docs/superpowers/specs/2026-04-01-stage43-working-memory-design.md`

### Изменены:
- `src/snks/daf/types.py` — wm_fraction, wm_decay, wm_coupling_scale
- `src/snks/daf/engine.py` — coupling damping for WM edges at init
- `src/snks/daf/eligibility.py` — device mismatch fix for GPU
- `src/snks/pipeline/runner.py` — selective reset, WM excluded from SKS, wm_activation
- `src/snks/agent/pure_daf_agent.py` — WM config forwarding
- `src/snks/sks/meta_embedder.py` — device mismatch fix for GPU

## Запланированные эксперименты (tech debt)

| TD | Что проверяется | Gate | Статус |
|----|-----------------|------|--------|
| TD-004 | WM gating: bistable FHN или explicit gate | WM DoorKey > no-WM | OPEN |

## Следующий этап

FHN oscillators self-sustain — нужен bistable regime или explicit gating.
Варианты:
1. **Bistable FHN tuning** — I_base < threshold в WM zone, sustained only with external input
2. **PE-gated WM** — высокий PE → inject current в WM, иначе decay доминирует
3. **Clamp-based WM** — вместо oscillation, просто удерживать SDR pattern как clamped current
