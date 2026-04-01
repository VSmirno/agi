# Stage 40: Learnable Encoding — Hebbian Encoder

## Результат: PASS

## Что доказано

- **Competitive Hebbian Learning работает**: Sanger's GHA с competitive selection обучает conv-фильтры без backprop
- **SDR discrimination улучшается**: overlap 0.232 → 0.199 после обучения (~14% improvement)
- **Filter diversity сохраняется**: mean dissimilarity = 0.89 после 300 обновлений
- **Сходимость**: weight delta уменьшается на 72% (ratio 0.28x early/late)
- **Drop-in compatible**: HebbianEncoder наследует VisualEncoder, PureDafAgent активирует через use_hebbian=True
- **No regression**: DoorKey runs without error с Hebbian encoder

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 99a: SDR discrimination | overlap decrease | 0.232→0.199 | improving | PASS |
| 99b: Filter diversity | dissimilarity | 0.89 | > 0.5 | PASS |
| 99c: Hebbian convergence | delta ratio | 0.28x | decreasing | PASS |
| 99d: DoorKey regression | no errors | 5 episodes OK | True | PASS |
| 99e: Learning curve | overlap trend | 0.229→0.210 | positive slope | PASS |

## Ключевые решения

1. **Sanger's GHA вместо plain Oja**: Plain Oja сходит все фильтры к PC1 (все одинаковые). Sanger's triangular decorrelation заставляет каждый фильтр извлекать свою компоненту.

2. **Competitive selection (top-25%)**: Только 25% самых активных фильтров обновляются. Остальные сохраняют текущую специализацию. Это аналог латерального торможения в коре.

3. **PE modulation из Pipeline.mean_pe**: Hebbian update вызывается внутри perception_cycle ПОСЛЕ вычисления PE (шаг 5 → шаг 8). Image и SDR кэшируются с шага 1.

4. **hebbian_update_interval=5**: На CPU обновление каждый цикл слишком дорого. Каждый 5-й цикл — amortized cost ~0.6ms.

5. **Weight clamp [-2, 2] обязателен**: PE модуляция (η_eff = η × PE_ratio) нарушает self-normalization Oja — без clamp веса уходят в бесконечность.

## Архитектура

```
HebbianEncoder (extends VisualEncoder)
├── gabor.conv.weight: Gabor init, updated in-place by Sanger's rule
├── hebbian_update(image, sdr, pe):
│   ├── Pre: F.unfold → mean input patches
│   ├── Post: gabor(x) → pool → mean activation per filter
│   ├── Top-K selection (25% most active = winners)
│   ├── Sanger's GHA: Δw_f = η_eff × post_f × (residual - post_f × w_f)
│   │   └── residual -= post_f × w_f (triangular, for each winner)
│   └── Clamp [-2, 2]
├── diversity_regularization (every 50 updates):
│   ├── Cosine similarity matrix
│   └── Decorrelate pairs with sim > 0.8
└── stats: update_count, pe_baseline, filter_similarity, weight range
```

## Веб-демо
- `demos/stage-40-learnable-encoding.html` — интерактивная визуализация: фильтры до/после, SDR comparison, overlap chart, convergence chart, обучающие образы

## Файлы изменены

### Новые:
- `src/snks/encoder/hebbian.py` — HebbianEncoder class
- `src/snks/experiments/exp99_learnable_encoding.py` — 5 experiments
- `tests/test_hebbian_encoder.py` — 17 tests PASS
- `demos/stage-40-learnable-encoding.html` — web demo
- `docs/superpowers/specs/2026-04-01-stage40-learnable-encoding-design.md` — spec

### Изменены:
- `src/snks/daf/types.py` — EncoderConfig: hebbian params
- `src/snks/pipeline/runner.py` — HebbianEncoder init + update в perception_cycle
- `src/snks/agent/pure_daf_agent.py` — use_hebbian config flag
- `demos/index.html` — карточка Stage 40

## Следующий этап

Stage 40 сделал encoder обучаемым. Следующие направления:

1. **Stage 41: Temporal Credit Assignment** — eligibility trace на десятки шагов, чтобы reward сигнал дошёл до ранних решений
2. **Stage 42: Spatial Representation** — формирование "карты" среды в аттракторах (grid cells / place cells аналог)
3. **GPU Validation** — запустить exp99 на minipc с 50K нод для проверки абсолютной производительности
