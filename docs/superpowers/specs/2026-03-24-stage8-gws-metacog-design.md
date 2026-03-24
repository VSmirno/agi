# Stage 8: Вертикальное углубление — ГРП + Метакогниция

**Дата:** 2026-03-24
**Статус:** Design
**Зависимости:** Stage 7 ✅ (TextEncoder, multimodal pipeline)

---

## Мотивация

Этапы 1–7 создали систему, которая **реагирует**: воспринимает стимулы, формирует СКС, обучается через STDP, извлекает эпизоды через DCAM. Этап 8 добавляет два механизма, превращающих систему в **осознающую**:

1. **ГРП (Глобальное Рабочее Пространство)** — выбирает доминирующую СКС ("победителя") и делает её доступной всем модулям
2. **Метакогниция** — наблюдает за качеством текущего состояния, вычисляет `confidence` и опционально влияет на следующий цикл через pluggable policies

---

## Архитектура

### Новые модули

```
src/snks/gws/
    __init__.py
    workspace.py      — GlobalWorkspace, GWSState

src/snks/metacog/
    __init__.py
    monitor.py        — MetacogMonitor, MetacogState
    policies.py       — NullPolicy, NoisePolicy, STDPPolicy
```

### Интеграция в Pipeline

`Pipeline.__init__` инстанцирует `GlobalWorkspace` и `MetacogMonitor`. После каждого `perception_cycle` Pipeline вызывает их последовательно:

```
perception_cycle(image/text)
    → step_result = engine.step(steps_per_cycle)   # StepResult с fired_history: Tensor(T,N)
    → SKS detection → CycleResult
    → GlobalWorkspace.select_winner(sks_clusters, fired_history=step_result.fired_history) → GWSState
    → MetacogMonitor.update(gws_state, cycle_result) → MetacogState
    → MetacogMonitor.apply_policy(metacog_state, engine.config)
    → CycleResult расширяется: .gws = GWSState, .metacog = MetacogState
```

`step_result` уже существует как локальная переменная в `perception_cycle` (строка 119 runner.py); `fired_history` передаётся из него напрямую.

Pipeline вызывает `monitor.apply_policy(...)` — метод `MetacogMonitor`, который внутри делегирует активной policy. Прямой вызов `policy.apply()` снаружи Pipeline не используется.

**Примечание о runtime-мутации конфига:** `NoisePolicy` и `STDPPolicy` мутируют поля `DafConfig.noise_sigma` и `DafConfig.stdp_a_plus` напрямую после каждого цикла. `DafEngine` читает эти поля при каждом вызове `integrate_n_steps`, поэтому изменения применяются немедленно без пересоздания движка. Это намеренный паттерн; thread-safety не требуется (однопоточный pipeline).

---

## Компоненты

### `src/snks/gws/workspace.py`

```python
@dataclass
class GWSState:
    winner_id: int            # id кластера-победителя (ключ в sks_clusters)
    winner_nodes: set[int]    # узлы победителя
    winner_size: int          # len(winner_nodes)
    winner_score: float       # взвешенный score выбора
    dominance: float          # winner_size / total_active_nodes ∈ [0, 1]
                              # total_active_nodes = число уникальных узлов во всех кластерах

class GlobalWorkspace:
    """Выбирает доминирующую СКС по взвешенному score.

    score_k = w_size * size_k
             + w_coherence * coherence_k      # зарезервировано, пока 0
             + w_pred * (1 - pred_error_k)    # зарезервировано, пока 0

    Конфигурация по умолчанию: w_size=1.0, w_coherence=0.0, w_pred=0.0
    (чистый size-based, расширяется без изменения интерфейса).

    fired_history: передаётся для будущего вычисления coherence_k
    (phase/cofiring coherence). При w_coherence=0.0 игнорируется.
    Если fired_history=None — coherence_k=0.0 для всех кластеров.
    """

    def select_winner(
        self,
        sks_clusters: dict[int, set[int]],
        fired_history: torch.Tensor | None,
    ) -> GWSState | None:
        """Возвращает None если sks_clusters пуст."""
        ...
```

### `src/snks/metacog/monitor.py`

```python
@dataclass
class MetacogState:
    confidence: float    # ∈ [0, 1]
    dominance: float     # из GWSState.dominance
    stability: float     # |winner_now ∩ winner_prev| / winner_size ∈ [0, 1]
    pred_error: float    # CycleResult.mean_prediction_error (глобальный по всем узлам)

class MetacogMonitor:
    """Наблюдает за состоянием системы, вычисляет confidence.

    confidence = α * dominance + β * stability + γ * (1 - pred_error_norm)

    pred_error_norm = pred_error / max_observed_pred_error
    где max_observed_pred_error — скользящий максимум за всё время работы.
    Инициализируется 1.0 (до первого наблюдения pred_error_norm = pred_error).

    stability = |winner_now.nodes ∩ winner_prev.nodes| / winner_now.size
    На первом цикле (нет предыдущего победителя) stability = 0.0.
    Если gws_state is None — confidence = 0.0, все компоненты = 0.0.

    pred_error берётся из CycleResult.mean_prediction_error —
    агрегат по всем узлам DAF (не per-winner). Победитель-специфичная
    ошибка недоступна без расширения CycleResult; отложено на Stage 9+.
    """

    def update(
        self,
        gws_state: GWSState | None,
        cycle_result: CycleResult,
    ) -> MetacogState:
        """Обновляет prev_winner, max_pred_error; возвращает MetacogState."""
        ...

    def apply_policy(self, state: MetacogState, config: DafConfig) -> None:
        """Вызывает self._policy.apply(state, config)."""
        ...
```

### `src/snks/metacog/policies.py`

```python
class NullPolicy:
    """Только наблюдение. Ничего не меняет. Дефолт."""
    def apply(self, state: MetacogState, config: DafConfig) -> None:
        pass

class NoisePolicy:
    """Адаптирует noise_sigma по confidence.

    noise_sigma = base_sigma * (1 + strength * (1 - confidence))

    confidence=1.0 → noise = base_sigma        (стабилизация паттерна)
    confidence=0.0 → noise = base_sigma * (1 + strength)  (исследование)

    Параметр strength: по умолчанию 1.0.
    base_sigma фиксируется при первом вызове apply() из текущего config.noise_sigma.
    """

class STDPPolicy:
    """Адаптирует stdp_a_plus по confidence.

    a_plus = base_a_plus * (1 + strength * confidence)

    При высокой уверенности — усиливаем обучение (закрепляем паттерн).
    При низкой — возвращаемся к базовому значению.
    base_a_plus фиксируется при первом вызове apply().
    """
```

---

## Конфигурация

Добавляем два новых датакласса в `src/snks/daf/types.py` и поля в `PipelineConfig`:

```python
@dataclass
class GWSConfig:
    enabled: bool = True
    w_size: float = 1.0
    w_coherence: float = 0.0    # зарезервировано
    w_pred: float = 0.0         # зарезервировано

@dataclass
class MetacogConfig:
    enabled: bool = True
    alpha: float = 1/3          # вес dominance
    beta: float = 1/3           # вес stability
    gamma: float = 1/3          # вес (1 - pred_error_norm)
    policy: str = "null"        # "null" | "noise" | "stdp"
    policy_strength: float = 1.0

@dataclass
class PipelineConfig:
    # ... существующие поля ...
    gws: GWSConfig = field(default_factory=GWSConfig)
    metacog: MetacogConfig = field(default_factory=MetacogConfig)
```

`load_config` в `src/snks/pipeline/config.py` явно перечисляет секции (`daf`, `encoder`, `dcam`, `pipeline`) и должен быть расширен для секций `gws` и `metacog` — иначе YAML-конфиги молча проигнорируют новые поля. Это обязательное изменение, не опциональное.

---

## Эксперименты и Gate-критерии

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 15 | ГРП-победитель | mean_stability (повторный ввод) | > 0.7 |
| 16 | Калибровка confidence | confidence(focused) / confidence(noise) | > 1.5 |
| 17 | NoisePolicy | NMI(policy) ≥ NMI(null)−0.05 AND std_confidence↓ | см. ниже |

### Exp 15: ГРП-победитель

5 синтетических категорий × 10 повторов одного изображения.
Изображения генерируются через `make_synthetic_image(cat_idx, variation=0)` из той же функции, что и в Exp 12 (категории 0–4: bright_circle, dark_square, bright_triangle, dark_circle, bright_square).

Для каждого повтора t ≥ 2:
```
stability_t = |winner_t.nodes ∩ winner_{t-1}.nodes| / winner_t.size
```
`mean_stability = mean(stability_t)` по всем категориям и повторам t ≥ 2.
Gate: `mean_stability > 0.7`

### Exp 16: Калибровка confidence

- **Focused:** 10 изображений — `[(cat_idx, variation) for cat_idx in range(5) for variation in [0, 1]]`, т.е. категории 0–4 × вариации 0 и 1. Генерируются через `make_synthetic_image(cat_idx, variation)`
- **Noise:** 10 тензоров `torch.rand(32, 32).clamp(0, 1)` с seed 0–9

Каждый стимул — 3 цикла pipeline, берём среднее confidence за 3 цикла.
Gate: `mean_confidence(focused) / mean_confidence(noise) > 1.5`

### Exp 17: NoisePolicy

Два pipeline с одинаковым seed:
- `policy="null"`: базовый (NullPolicy)
- `policy="noise"`: NoisePolicy(strength=1.0)

**Данные:** 3 категории × 20 предложений (животные, еда, техника — как Exp 10), 100 циклов обучения.

**Метрики после обучения:**
- `NMI(noise_policy)` и `NMI(null)` — качество СКС
- `std_confidence(noise_policy)` и `std_confidence(null)` — стандартное отклонение confidence за все 100 циклов

**Gate:** `NMI(noise_policy) ≥ NMI(null) − 0.05` AND `std_confidence(noise_policy) < std_confidence(null)`

**Обоснование gate std↓:** NoisePolicy повышает noise когда confidence низкая → система активнее исследует → быстрее находит стабильный паттерн → в итоге confidence стабилизируется раньше и std за весь прогон оказывается ниже, чем у NullPolicy, которая продолжает колебаться на постоянном noise.

---

## Тесты

```
tests/gws/
    test_global_workspace.py
        - test_winner_is_largest_cluster           — size-based selection, w_size=1
        - test_dominance_formula                   — dominance = winner / total_active
        - test_returns_none_when_no_clusters       — пустой dict
        - test_winner_score_respects_weights       — изменение w_size влияет на выбор
        - test_fired_history_none_with_clusters    — fired_history=None, w_coherence=0 → работает нормально

tests/metacog/
    test_monitor.py
        - test_confidence_formula                  — проверка весов α, β, γ
        - test_stability_first_cycle               — stability=0.0 на первом цикле
        - test_stability_identical_winner          — stability=1.0 при совпадении узлов
        - test_stability_disjoint_winner           — stability=0.0 при полном несовпадении
        - test_pred_error_normalization            — pred_error_norm ∈ [0, 1]
        - test_gws_none_returns_zero_confidence    — gws_state=None → confidence=0.0

    test_policies.py
        - test_null_policy_no_change               — config не изменяется
        - test_noise_policy_high_confidence        — noise_sigma ≈ base при confidence=1.0
        - test_noise_policy_low_confidence         — noise_sigma > base при confidence=0.0
        - test_stdp_policy_high_confidence         — a_plus > base при confidence=1.0
        - test_stdp_policy_low_confidence          — a_plus ≈ base при confidence=0.0
```

---

## Порядок реализации

1. `GWSConfig`, `MetacogConfig` → добавить в `src/snks/daf/types.py`; добавить поля в `PipelineConfig`; расширить `load_config` в `src/snks/pipeline/config.py` для секций `gws` и `metacog`
2. `src/snks/gws/workspace.py` + тесты
3. `src/snks/metacog/policies.py` + тесты
4. `src/snks/metacog/monitor.py` + тесты
5. Интеграция в `Pipeline.perception_cycle`: вызов GWS → MetacogMonitor, расширение CycleResult
6. Эксперименты `exp15_gws.py`, `exp16_confidence.py`, `exp17_noise_policy.py`

---

## Отложено (Stage 9+)

- **Exp 18: BroadcastPolicy** — узлы победителя ГРП получают усиленный ток на следующем цикле. Gate: cross_activation_ratio(with) / cross_activation_ratio(without) > 1.2
- **per-node prediction error в CycleResult** — для winner-специфичной составляющей confidence (сейчас используется глобальный `mean_prediction_error`)
- **w_coherence, w_pred в GWSConfig** — активировать после добавления per-node PE и coherence данных в CycleResult
