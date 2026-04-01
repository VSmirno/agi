# Stage 40: Learnable Encoding — Hebbian Encoder

**Дата:** 2026-04-01
**Статус:** DESIGN
**Автор:** autonomous-dev

## Позиция в фазе

**Фаза 1: Живой DAF (~12% → 20%)**

Этот этап продвигает маркер: "Pure DAF >= 50% на DoorKey-5x5" через улучшение представления состояний. Текущий frozen Gabor encoder — главный bottleneck: агент буквально не видит разницу между ключом, дверью и пустой клеткой на уровне SDR.

Маркеры, которые продвигает Stage 40:
- [ ] Learnable encoding (прямо из ROADMAP)
- [ ] Улучшение discriminability SDR для task-relevant объектов
- [ ] Preparation для spatial representation (Stage 41+)

## Проблема

Текущий VisualEncoder использует 128 frozen Gabor фильтров (4 scales × 8 orientations × 4 phases). Проблемы:

1. **Gabor = low-level edges** — не адаптируется к task-relevant features (ключ vs дверь)
2. **SDR collision** — похожие сцены дают похожие SDR, агент не различает критические объекты
3. **No learning signal** — encoder не получает feedback от успехов/неудач агента
4. **Information loss** — AdaptiveAvgPool(4×8) теряет spatial detail

## Brainstorming

### Подход A: Oja's Hebbian Rule (ВЫБРАН)
- Gabor weights = инициализация, далее обучение через Oja's rule
- Δw = η × post × (pre - post × w) — self-normalizing Hebbian
- Модуляция prediction error: η_eff = η_base × clamp(PE, 0.1, 2.0)
- **Pros:** биологически правдоподобно, локальное правило, self-stabilizing
- **Cons:** медленная сходимость, нужна careful initialization
- **Trade-off:** скорость vs. bio-plausibility → выбираем plausibility

### Подход B: Competitive Hebbian + латеральное торможение
- Как A, но добавить anti-correlation между фильтрами
- Проигравшие фильтры (не в top-k) подавляются
- **Pros:** предотвращает коллапс фильтров, diversity guarantee
- **Cons:** сложнее, дополнительная O(n_filters²) computation
- **Trade-off:** робастность vs. простота

### Подход C: Temporal Contrastive (backprop)
- Encoder учится предсказывать SDR(t+1) из SDR(t)
- **Pros:** сильный gradient signal
- **Cons:** backprop нарушает SNKS философию (только локальные правила!)
- **Trade-off:** performance vs. paradigm compliance → ОТКЛОНЁН

### Решение: A (Oja's Hebbian) + элемент B (diversity regularization)

**Обоснование:** Oja's rule — самостабилизирующееся правило Хебба, не требует backprop. Diversity regularization (декорреляция фильтров каждые N шагов) предотвращает коллапс без постоянных O(n²) затрат.

## Архитектура

```
HebbianEncoder (extends VisualEncoder)
├── conv weights: accessed via self.gabor.conv.weight.data (in-place updates)
│   └── GaborBank.conv.weight initialized with requires_grad=False — no autograd interference
├── hebbian_update(image, sdr, prediction_error):
│   ├── Extract pre-activations (input patches via F.unfold)
│   ├── Extract post-activations (pooled features before k-WTA)
│   ├── Oja's rule: Δw = η_eff × post × (pre - post × w)
│   ├── η_eff = η_base × clamp(PE / PE_baseline, 0.1, 2.0)
│   ├── Apply update to self.gabor.conv.weight.data (in-place)
│   └── Clamp weights to [w_min, w_max] (REQUIRED — PE modulation breaks self-normalization)
├── diversity_regularization(every N cycles):
│   ├── Compute cosine similarity matrix: C = normalize(W) @ normalize(W).T
│   ├── For highly correlated pairs (>0.8): perturb one with noise
│   └── Re-normalize affected filters
└── SDR discrimination metric:
    ├── Track SDR overlap between consecutive observations
    └── Log mean discrimination (lower overlap = better)
```

## Oja's Rule — детали

Oja's rule (1982) для single neuron:
```
Δw_j = η × y × (x_j - y × w_j)
```

где x = input (pre), y = output (post), w = weight.

Свойства:
- Конвергирует к первой главной компоненте (PCA) для линейных нейронов
- Self-normalizing: ||w|| → 1 при постоянном η
- **NB:** abs() активация в GaborBank означает convergence к dominant feature direction в rectified space, не строго PCA
- **NB:** PE модуляция η нарушает self-normalization → weight clamp ОБЯЗАТЕЛЕН (не safety, а requirement)

Для конволюционного слоя:
- pre = image patches under each filter position (F.unfold → spatial average)
- post = pooled activations per filter (AdaptiveAvgPool → spatial mean)
- Update per filter: Δw_f = η_eff × <post_f> × (<pre_f> - <post_f> × w_f)
- `<>` = average across spatial positions

## Prediction Error модуляция

PE берётся из Pipeline.perception_cycle() — mean_pe (HAC prediction error):
- Доступен внутри perception_cycle после шагов DAF + SKS detection + prediction
- НЕ из DafCausalModel (тот PE доступен только в observe_result агента)
- Высокий PE = неожиданный результат → быстрее учиться
- η_eff = η_base × clamp(PE / PE_baseline, 0.1, 2.0)
- PE_baseline = EMA of recent PE values (адаптивный порог)

### Timing в perception_cycle:
1. Encode image → SDR (cache image + SDR)
2. SDR → currents → DAF engine step
3. SKS detection → HAC prediction → compute mean_pe
4. **Call encoder.hebbian_update(cached_image, cached_sdr, mean_pe)** ← ЗДЕСЬ
5. Return PerceptionResult

### Первый цикл (PE undefined):
- При первом вызове PE = 0 (нет prediction) → η_eff = η_base × 0.1 (минимум)
- Эффект: первый цикл почти не обучает encoder — корректно (нет ещё контекста)

## Diversity Regularization

Каждые `diversity_interval` шагов (default: 50):
1. Cosine similarity matrix: C = normalize(W) @ normalize(W).T (128×128)
2. Для пар с |C[i,j]| > 0.8: w_j += noise × (w_j - w_i) (decorrelate)
3. Re-normalize: w_f /= ||w_f||

### Hebbian update interval (CPU optimization):
- `hebbian_update_interval: int = 5` — обновлять weights каждые N perception_cycle
- На Ryzen 3 3200U (mini-beelink): unfold 128×19×19 по 64×64 image = ~3ms per update
- С interval=5: amortized cost ~0.6ms per cycle — приемлемо

## Gate-критерии (exp99)

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 99a | SDR discrimination | > frozen | HebbianEncoder mean SDR overlap < frozen Gabor overlap (100 random distinct obs pairs from DoorKey-5x5) |
| 99b | Filter diversity | > 0.5 | Mean pairwise filter dissimilarity после обучения |
| 99c | Hebbian convergence | monotonic | Weights стабилизируются (Δw decreasing) |
| 99d | DoorKey success | ≥ 0.10 | Не хуже frozen Gabor с учётом variance (mean over 10 episodes) |
| 99e | Learning curve | positive slope | Success rate растёт с обучением encoder |

## Интеграция

1. `HebbianEncoder` наследует `VisualEncoder` — drop-in replacement
2. `Pipeline.perception_cycle()` вызывает `encoder.hebbian_update()` после perception
3. `PureDafAgent` получает `use_hebbian=True` в config
4. Existing tests не ломаются (frozen encoder = default)

## Файлы

### Новые:
- `src/snks/encoder/hebbian.py` — HebbianEncoder class
- `src/snks/experiments/exp99_learnable_encoding.py` — experiments
- `tests/test_hebbian_encoder.py` — unit tests
- `demos/stage-40-learnable-encoding.html` — web demo

### Изменяемые:
- `src/snks/pipeline/runner.py` — вызов hebbian_update в perception_cycle
- `src/snks/agent/pure_daf_agent.py` — config flag для HebbianEncoder
- `src/snks/daf/types.py` — EncoderConfig: hebbian params

## Риски

1. **Weight instability** — mitigation: Oja's rule self-normalizing + clamp
2. **Filter collapse** — mitigation: diversity regularization
3. **Performance regression** — mitigation: gate 99d ensures >= baseline
4. **CPU too slow** — mitigation: hebbian_update каждые N циклов, не каждый
