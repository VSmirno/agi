# Autonomous Development Log — 2026-04-01 (Stage 43)

## Текущая фаза: 1 — Живой DAF, прогресс ~70%

Stages 0-42 COMPLETE. Stage 42 диагностика: 5% success с идеальной perception.
Root cause: 1 reward / 2000 шагов, dw ≈ 0.01, mental sim = шум.
Stage 43: fix learning signal.

## Stage 43: Working Memory / Learning Signal Amplification

### Фаза 0: Git setup
- Ветка: stage43-learning-signal от main
- Tech debt: 3 items (TD-001 blocked by perception, TD-002/003 deferred)

### Фаза 1: Спецификация
- Подход A: WM Buffer Zone (selective reset) — ВЫБРАН
- Подход B: Recurrent State Carry (no reset) — rejected, perception degradation
- Подход C: External WM buffer — rejected, not DAF-native

### Фаза 2: Реализация
- Selective reset in perception_cycle: 8 tests PASS
- DafConfig: wm_fraction, wm_decay, wm_coupling_scale
- CycleResult: wm_activation metric

### Фаза 3: Эксперименты (minipc GPU)
- Exp 102a: WM persistence — activation=1.48 PASS
- Exp 102b: WM DoorKey — 0% vs 5% (WM worse, no gating)
- Exp 102c: WM tracks stimuli — diff=0.43 PASS
- Exp 102d: WM decay — ratio=3.57 (FHN self-sustains)

### Stage 43_fix
- Coupling INTO WM scaled 0.1x — не помогло (FHN intrinsic, не coupling)
- SKS exclusion — WM nodes out of coherence matrix
- 84 tests PASS на minipc GPU

### Решения
- Root cause found: perception_cycle() сбрасывал все states → амнезия каждый cycle
- Selective reset: minimal change, backward compatible (wm_fraction=0.0 default)
- FHN self-sustain is fundamental — needs bistable tuning, not just coupling damping
- TD-004 created for proper gating mechanism
