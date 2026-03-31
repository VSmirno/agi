# Этап 3: СКС + Предсказание + Эксперименты 1–4 ✅

**Статус:** Завершён (2026-03-23)
**Срок:** Неделя 5–7

## Результаты экспериментов

| # | Эксперимент | Порог | Результат | Статус |
|---|-------------|-------|-----------|--------|
| 1 | Формирование СКС | NMI > 0.7 | **0.823** | ✅ |
| 2 | Непрерывное обучение | Retention > 85% | **103.3%** | ✅ |
| 3 | Предсказание | Mean accuracy > 70% | **72.9%** (L3:64%, L5:75%, L7:79%) | ✅ |
| 4 | Устойчивость к шуму | Graceful degradation | **0% drop** (NMI=0.77 при σ=0.1–0.3) | ✅ |

## Модули (10 файлов)

| Модуль | Назначение |
|--------|-----------|
| `sks/detection.py` | phase_coherence_matrix(), cofiring_coherence_matrix(), detect_sks() |
| `sks/tracking.py` | SKSTracker (Hungarian matching, Jaccard similarity) |
| `sks/metrics.py` | compute_nmi(), sks_stability(), sks_separability() |
| `daf/prediction.py` | PredictionEngine (каузальный граф, PE, lr_modulation) |
| `pipeline/runner.py` | Pipeline, CycleResult, TrainResult |
| `data/sequences.py` | SequenceGenerator (deterministic, stochastic) |
| `experiments/exp1_sks_formation.py` | Exp 1: 10 ориентаций × 50 показов |
| `experiments/exp2_continual.py` | Exp 2: фазы A/B, retention |
| `experiments/exp3_prediction.py` | Exp 3: последовательности длины 3/5/7 |
| `experiments/exp4_noise.py` | Exp 4: шум σ=0.1/0.2/0.3 |

## Конфиг экспериментов (FHN)

```python
DafConfig(
    oscillator_model="fhn",
    coupling_strength=0.05,
    dt=0.01,
    noise_sigma=0.005,
    fhn_I_base=0.0,       # нейроны молчат без стимула
)
EncoderConfig(sdr_current_strength=1.0)
SKSConfig(
    coherence_mode="rate",  # O(N) detection
    dbscan_eps=0.3,
    dbscan_min_samples=5,
    min_cluster_size=5,
)
steps_per_cycle=200         # 200 × 0.01 = 2.0 time units
```

## Детали экспериментов

### Exp 1: Формирование СКС
- 10 ориентаций (GratingGenerator) × 50 показов = 500 изображений
- Rate-based detection → k-means на firing rate vectors → NMI
- **NMI = 0.823** (gate > 0.7)

### Exp 2: Непрерывное обучение
- Phase A: классы 0–4 (5 × 100 вариаций = 500)
- Phase B: классы 5–9 (500)
- Retest: классы 0–4 снова
- **Retention = 103.3%** — система не забывает, даже улучшает

### Exp 3: Предсказание последовательностей
- Подход: k-means кластеризация firing rate vectors → transition matrix → predict next cluster
- Training: 20 repeats, Test: 5 repeats
- L3: 64.3%, L5: 75.0%, L7: 79.4% → **Mean = 72.9%**
- Более длинные последовательности предсказываются лучше

### Exp 4: Устойчивость к шуму
- Обучение на чистых → тест с шумом σ=0.1/0.2/0.3
- k-means centroids из обучения → classify зашумлённые
- **NMI = 0.77 одинаков на всех уровнях шума**
- Gabor + SDR бинаризация полностью фильтрует гауссов шум

## Модификации существующих файлов

- `daf/stdp.py` — lr_modulation параметр в apply() для PE-driven обучения
- `daf/types.py` — SKSConfig, PredictionConfig, расширён PipelineConfig
- `device.py` — total_mem → total_memory (PyTorch 2.5 compat)

## Архитектурные решения

| Решение | Причина |
|---------|---------|
| FHN вместо Kuramoto | Kuramoto не формирует кластеры через STDP на случайном графе |
| Rate-based detection | O(N) вместо O(K²), достаточно для NMI через k-means |
| State reset между стимулами | Без него state carryover разрушает паттерны |
| k-means на rate vectors | Tracker-based SKS ID нестабилен (все стимулы → один blob) |
| Cluster-space transitions (Exp 3) | Label-space transition matrix теряет данные при many-to-one mapping |

## Тесты

63 новых тестов. Всего: **199 тестов**, все проходят (~14 сек на RTX 3070 Ti).
