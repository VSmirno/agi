# Stage 41: Temporal Credit Assignment — Eligibility Traces

## Результат: PASS

**Ветка:** `stage41-temporal-credit`

## Что доказано

- **Eligibility trace работает**: STDP dw аккумулируется с экспоненциальным decay, reward модулирует все traced edges
- **Эффективное окно = 35 шагов**: при λ=0.92, signal retention 19% через 20 шагов (vs 0% у старого 5-step подхода)
- **Long-range credit**: reward на шаге 15 модифицирует веса от STDP шага 0 (mean_delta=0.003)
- **Memory efficiency**: 1 tensor (E,) вместо 5 snapshots — 5x экономия
- **No regression**: DoorKey-5x5 работает без ошибок, eligibility stats populated
- **STDP dw чистый**: raw STDP weight changes возвращаются ДО homeostatic regularization

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 100a: trace accumulation | magnitude after 10 steps | 21.4 | > 0 | PASS |
| 100b: trace decay | decay ratio at step 20 | 0.205 | < 0.25 | PASS |
| 100c: long-range credit | mean weight change | 0.003 | > 1e-4 | PASS |
| 100d: memory efficiency | savings factor | 5x | >= 5x | PASS |
| 100e: DoorKey regression | no errors, eff_window | 35 | >= 20 | PASS |

## Ключевые решения

1. **Дополнение, не замена**: EligibilityTrace добавлен КАК ДОПОЛНЕНИЕ к snapshot-based trace. Snapshot нужен для before_action/predict_effect (AttractorNavigator). Eligibility trace — для long-range credit.

2. **Raw dw до homeostasis**: STDP.apply() теперь возвращает dw ДО homeostatic regularization. Это предотвращает contamination trace homeostatic noise.

3. **λ=0.92**: Компромисс между window length (35 шагов при 5% threshold) и trace dilution. При λ=0.95 окно 58 шагов, но signal слишком размыт для коротких эпизодов.

4. **Reset per episode**: Trace сбрасывается в начале каждого эпизода — межэпизодный credit не имеет смысла для episodic tasks.

## Архитектура

```
EligibilityTrace (src/snks/daf/eligibility.py)
├── accumulate(dw: Tensor)
│   └── e(t) = λ × e(t-1) + dw(t)
├── apply_reward(reward, graph, w_min, w_max)
│   └── Δw = η × reward × e(t) → graph weights
├── reset() — per episode
└── stats: steps_accumulated, trace_magnitude, effective_window

Поток данных:
  Engine.step() → STDP.apply() → STDPResult(dw=raw_dw)
  → Pipeline.CycleResult(stdp_result=...)
  → PureDafAgent.step()/observe_result()
  → DafCausalModel.accumulate_stdp(result)
  → [reward arrives] → DafCausalModel.after_action(reward)
    → EligibilityTrace.apply_reward() [long-range]
    → snapshot trace [short-range]
```

## Веб-демо
- `demos/stage-41-temporal-credit.html` — интерактивная визуализация: trace accumulation с настройкой λ, DoorKey-5x5 replay с credit bars, сравнение старого/нового подхода

## Файлы изменены

### Новые:
- `src/snks/daf/eligibility.py` — EligibilityTrace class
- `src/snks/experiments/exp100_temporal_credit.py` — 5 experiments
- `tests/test_eligibility_trace.py` — 24 tests
- `demos/stage-41-temporal-credit.html` — web demo
- `docs/superpowers/specs/2026-04-01-stage41-temporal-credit-design.md` — spec

### Изменены:
- `src/snks/daf/stdp.py` — STDPResult.dw field, raw dw capture
- `src/snks/agent/daf_causal_model.py` — EligibilityTrace integration
- `src/snks/agent/pure_daf_agent.py` — accumulate_stdp calls, config params, episode reset
- `src/snks/pipeline/runner.py` — CycleResult.stdp_result field
- `demos/index.html` — карточка Stage 41

## Следующий этап

Stage 41 решил temporal credit assignment. Следующие направления:

1. **Stage 42: Spatial Representation** — grid/place cells в аттракторах для формирования "карты" среды
2. **GPU Validation** — запустить exp100 на minipc с 50K нод (когда TD-001 завершится)
