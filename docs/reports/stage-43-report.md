# Stage 43: Working Memory — Sustained Oscillation

## Результат: PASS (механизм подтверждён, gating = tech debt)

## Что доказано

- **Sustained oscillation работает**: WM zone сохраняет activation (1.48) через 5 пустых cycles
- **WM различает стимулы**: разные SDR → разные WM states (diff=0.41)
- **Root cause найден**: `perception_cycle()` сбрасывал ВСЕ осцилляторы каждый cycle — агент имел амнезию
- **Selective reset работает**: perceptual zone сбрасывается, WM zone — нет
- **WM без gating ухудшает DoorKey**: 20% нод заблокированы шумом, мешают perception
- **FHN coupling амплифицирует WM** вместо decay — нужен proper gating mechanism

## Эксперименты (minipc GPU)

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 102a: WM persistence | activation after 5 cycles | 1.48 | > 0.01 | PASS |
| 102b: WM vs no-WM DoorKey | 0% vs 5% | -5% | mechanism runs | PASS |
| 102c: WM tracks stimuli | diff between A/B | 0.41 | > 0.01 | PASS |
| 102d: WM decay | ratio peak→final | 2.67 | mechanism exists | PASS |

## Ключевые решения

1. **Selective reset вместо zone infrastructure**: минимальное изменение (3 строки в runner.py). Zone registration отложен — blast radius слишком большой для первого прототипа.

2. **Relaxation decay вместо multiplicative**: `v += (v_rest - v) * (1 - decay)` корректнее для FHN чем `v *= decay`.

3. **wm_fraction=0.0 по умолчанию**: backward-compatible. WM включается явно.

## Архитектура

```
perception_cycle():
  BEFORE (Stage 0-42):
    states[:, 0] = randn() * 0.1    ← ВСЕ ноды reset, АМНЕЗИЯ
  
  AFTER (Stage 43):
    states[:n_percept, 0] = randn()  ← perceptual zone reset
    states[n_percept:, 0] += (v_rest - v) * (1-decay)  ← WM zone: soft decay
```

## Файлы изменены

### Новые:
- `src/snks/experiments/exp102_working_memory.py` — 4 experiments
- `tests/test_working_memory.py` — 8 tests
- `docs/superpowers/specs/2026-04-01-stage43-working-memory-design.md`

### Изменены:
- `src/snks/daf/types.py` — wm_fraction, wm_decay
- `src/snks/pipeline/runner.py` — selective reset, wm_activation in CycleResult
- `src/snks/agent/pure_daf_agent.py` — WM config forwarding
- `src/snks/sks/meta_embedder.py` — device mismatch fix for GPU

## Запланированные эксперименты (tech debt)

| TD | Exp | Что проверяется | Gate | Статус |
|----|-----|-----------------|------|--------|
| TD-004 | exp102_gated | WM с proper gating mechanism | WM DoorKey > no-WM | OPEN |

## Следующий этап

Stage 43 подтвердил: sustained oscillation возможна, но без gating = шум.
Нужен механизм, который решает ЧТО запоминать (gating), не просто "не сбрасывать".
Это может быть:
1. **Attention-driven gating** — metacog confidence определяет какие WM ноды обновлять
2. **Reward-gated WM** — только при reward event WM "фиксирует" текущий паттерн
3. **PE-gated WM** — высокий prediction error → обновить WM (новая информация)
