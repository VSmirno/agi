# Stage 17: Full-Scale Validation

**Версия:** 1.0
**Дата:** 2026-03-26
**Статус:** IN PROGRESS

---

## Цель

Валидация системы в трёх измерениях, которые не были покрыты Stages 0–16:

1. **Масштаб** — N=50K осцилляторов на AMD ROCm GPU (exp38)
2. **Replay impact** — влияние Stage 16 ReplayEngine на реальное поведение агента (exp39)
3. **Сложная задача** — оригинальный DoorKey-8x8, который провалился в exp29–31 (exp40)

---

## Эксперименты

### Exp38: GPU Scaling N=50K (Stage 17)

**Контекст:** exp31 запускался на CPU, N=5K, результат 9.41 steps/sec. AMD ROCm не использовался
из-за `torch.sparse_csr_tensor` (N=50K → >10 мин инициализации). Фикс Stage 16:
`disable_csr=True` обходит эту проблему.

**Протокол:**
- N=50 000, avg_degree=30, device="cuda" (ROCm)
- 20 эпизодов, max_steps=100, DoorKey-16x16
- `disable_csr=True`, `torch.compile` отключён

**Gate:**
```
steps_per_sec >= 10          # N=50K на GPU не хуже CPU N=5K
init_elapsed_seconds < 300   # инициализация < 5 мин
```

---

### Exp39: Replay Impact on Coverage (Stage 17)

**Контекст:** exp37 доказал, что replay не ухудшает prediction error (PE). Но replay
предназначен для consolidation долгосрочной памяти — его эффект должен проявиться в
поведении агента через несколько эпизодов.

**Протокол:**
- DoorKey-5x5, N=500, 100 эпизодов × 2 варианта (no_replay / with_replay)
- Метрика: mean_coverage (доля посещённых клеток от walkable)
- Оба варианта используют одинаковый random seed

**Gate:**
```
coverage_replay >= coverage_no_replay        # replay не ухудшает
coverage_replay >= 0.25                      # абсолютный минимум качества
```

---

### Exp40: DoorKey-8x8 Solution (Stage 17)

**Контекст:** В exp29–31 агент не мог решить DoorKey-8x8 — GOAL_SEEKING никогда не
активировался (Stage 15 это исправил на EmptyRoom-5x5). Теперь тестируем полную задачу:
ключ → дверь → цель.

**Протокол:**
- **Фаза 1 (bootstrap):** 50 эп, EmptyRoom-5x5, goal_sks устанавливается из первого успеха.
  Цель: обучить каузальную модель движения и получить goal_sks.
- **Фаза 2 (transfer):** 200 эп, DoorKey-8x8, goal_sks переносится из фазы 1.
  Агент должен применить goal-seeking к новому окружению.
- N=500, max_steps=200, with_replay=True (ConsolidationConfig enabled)

**Gate:**
```
goal_seeking_activations > 0              # GOAL_SEEKING реально активировался
success_rate_phase2 >= 0.05              # >= 5% успеха на DoorKey-8x8
```

Примечание: 5% success rate на DoorKey-8x8 — жёсткий порог. Без обучения (random walk)
success_rate ≈ 0% из-за цепочки key→door→goal.

---

## Критерии завершения Stage 17

Все три эксперимента PASS → Stage 17 COMPLETE.

При FAIL любого из них:
- Exp38 FAIL (timeout/crash): диагностировать AMD ROCm bottleneck, скорректировать N
- Exp39 FAIL (coverage падает): проверить ReplayConfig параметры, увеличить top_k
- Exp40 FAIL (no success): снизить порог до 0.02 или добавить фазу предобучения

---

## Файлы

| Файл | Описание |
|------|----------|
| `src/snks/experiments/exp38_scaling_gpu.py` | N=50K AMD ROCm benchmark |
| `src/snks/experiments/exp39_replay_coverage.py` | Replay vs no-replay coverage |
| `src/snks/experiments/exp40_doorkey8x8.py` | Full DoorKey-8x8 solution |
| `scripts/run_stage17.sh` | Автономный runner для minipc |
