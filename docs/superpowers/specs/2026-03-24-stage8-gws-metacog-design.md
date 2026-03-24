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
    → DafEngine.run() → SKS detection → CycleResult
    → GlobalWorkspace.select_winner(sks_clusters, fired_history) → GWSState
    → MetacogMonitor.update(gws_state, cycle_result) → MetacogState
    → policy.apply(metacog_state, engine.config)
    → CycleResult расширяется: .gws, .metacog
```

`GWSState` и `MetacogState` добавляются в `CycleResult` как опциональные поля (None если модули отключены).

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

class GlobalWorkspace:
    """Выбирает доминирующую СКС по взвешенному score.

    score_k = w_size * size_k
             + w_coherence * coherence_k
             + w_pred * (1 - pred_error_norm_k)

    Конфигурация по умолчанию: w_size=1.0, w_coherence=0.0, w_pred=0.0
    (чистый size-based, расширяется без изменения интерфейса).
    """

    def select_winner(
        self,
        sks_clusters: dict[int, set[int]],
        fired_history: torch.Tensor | None,
    ) -> GWSState | None:
        """Возвращает None если нет ни одного кластера."""
        ...
```

`total_active_nodes` = число уникальных узлов во всех кластерах (не `num_nodes`).

### `src/snks/metacog/monitor.py`

```python
@dataclass
class MetacogState:
    confidence: float    # ∈ [0, 1], взвешенная комбинация компонент
    dominance: float     # из GWSState.dominance
    stability: float     # |winner_now ∩ winner_prev| / winner_size ∈ [0, 1]
    pred_error: float    # mean prediction error на узлах победителя

class MetacogMonitor:
    """Наблюдает за состоянием системы, вычисляет confidence.

    confidence = (α * dominance + β * stability + γ * (1 - pred_error_norm)) / (α + β + γ)

    По умолчанию α = β = γ = 1/3 (равновесная комбинация).

    pred_error_norm = pred_error / max_observed_pred_error (running max для нормировки).
    """

    def update(
        self,
        gws_state: GWSState | None,
        cycle_result: CycleResult,
    ) -> MetacogState:
        """Обновляет историю и возвращает текущее MetacogState."""
        ...

    def apply_policy(self, state: MetacogState, config: DafConfig) -> None:
        """Применяет активную policy к конфигурации движка."""
        ...
```

`stability` на первом цикле = 0.0 (нет предыдущего победителя).

### `src/snks/metacog/policies.py`

```python
class NullPolicy:
    """Только наблюдение. Ничего не меняет. Дефолт."""
    def apply(self, state: MetacogState, config: DafConfig) -> None:
        pass

class NoisePolicy:
    """Адаптирует noise_sigma по confidence.

    noise_sigma = base_sigma * (1 + strength * (1 - confidence))

    При confidence=1.0 → noise = base_sigma (минимум, стабилизация)
    При confidence=0.0 → noise = base_sigma * (1 + strength) (максимум, исследование)

    Параметр strength: по умолчанию 1.0 (удвоение при нулевой уверенности).
    """

class STDPPolicy:
    """Адаптирует a_plus по confidence.

    a_plus = base_a_plus * (1 + strength * confidence)

    При высокой уверенности — усиливаем обучение (закрепляем паттерн).
    При низкой — возвращаемся к базовому значению.
    """
```

---

## Конфигурация

Добавляем в `DafConfig` (или отдельный `GWSConfig` / `MetacogConfig`):

```python
@dataclass
class GWSConfig:
    enabled: bool = True
    w_size: float = 1.0
    w_coherence: float = 0.0
    w_pred: float = 0.0

@dataclass
class MetacogConfig:
    enabled: bool = True
    alpha: float = 1/3      # вес dominance
    beta: float = 1/3       # вес stability
    gamma: float = 1/3      # вес (1 - pred_error)
    policy: str = "null"    # "null" | "noise" | "stdp"
    policy_strength: float = 1.0
```

Добавляются в `PipelineConfig`.

---

## Эксперименты и Gate-критерии

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 15 | ГРП-победитель | mean_stability (повторный ввод) | > 0.7 |
| 16 | Калибровка confidence | confidence(focused) / confidence(noise) | > 1.5 |
| 17 | NoisePolicy | NMI(policy) vs NMI(null), std(confidence) | NMI ≥ NMI(null) − 0.05 AND std↓ |

### Exp 15: ГРП-победитель

5 категорий × 10 повторов одного изображения. Для каждого повтора t ≥ 2:
```
stability_t = |winner_t.nodes ∩ winner_{t-1}.nodes| / winner_t.size
mean_stability = mean(stability_t) по всем категориям и повторам
```
Gate: `mean_stability > 0.7`

### Exp 16: Калибровка confidence

- Focused: 10 чётких изображений (5 категорий × 2 вариации, из Exp 12)
- Noise: 10 случайных тензоров `torch.rand(32, 32).clamp(0, 1)`
- Каждый — 3 цикла, берём среднее confidence
- Gate: `mean_confidence(focused) / mean_confidence(noise) > 1.5`

### Exp 17: NoisePolicy

Два pipeline с одинаковым seed:
- `policy="null"`: базовый
- `policy="noise"`: NoisePolicy(strength=1.0)

5 категорий × 20 предложений (как Exp 10), 100 циклов обучения.

Метрики после обучения:
- `NMI(noise_policy)` и `NMI(null)` — качество СКС
- `std_confidence(noise_policy)` и `std_confidence(null)` — стабильность уверенности

Gate: `NMI(noise_policy) ≥ NMI(null) − 0.05` AND `std_confidence(noise) < std_confidence(null)`

---

## Тесты

```
tests/gws/
    test_global_workspace.py
        - test_winner_is_largest_cluster      — size-based selection
        - test_dominance_formula              — dominance = winner / total
        - test_returns_none_when_no_clusters  — пустой CycleResult
        - test_winner_score_respects_weights  — изменение w_size влияет на score

tests/metacog/
    test_monitor.py
        - test_confidence_formula             — проверка весов α, β, γ
        - test_stability_first_cycle          — stability=0.0 на первом цикле
        - test_stability_identical_winner     — stability=1.0 при совпадении
        - test_stability_disjoint_winner      — stability=0.0 при полном несовпадении
        - test_pred_error_normalization       — pred_error_norm ∈ [0, 1]

    test_policies.py
        - test_null_policy_no_change          — config не изменяется
        - test_noise_policy_high_confidence   — noise_sigma близко к base при confidence=1
        - test_noise_policy_low_confidence    — noise_sigma увеличен при confidence=0
        - test_stdp_policy_high_confidence    — a_plus увеличен при confidence=1
```

---

## Порядок реализации

1. `GWSConfig`, `MetacogConfig` → в `DafConfig` types
2. `src/snks/gws/workspace.py` + тесты
3. `src/snks/metacog/policies.py` + тесты
4. `src/snks/metacog/monitor.py` + тесты
5. Интеграция в `Pipeline.perception_cycle` (CycleResult расширяется)
6. Эксперименты Exp 15, 16, 17

---

## Отложено (Stage 9+)

- **Exp 18: BroadcastPolicy** — узлы победителя ГРП получают усиленный ток на следующем цикле, метрика: cross_activation_ratio(with) / cross_activation_ratio(without) > 1.2
- **w_coherence, w_pred** в GWSConfig — активировать после появления соответствующих данных в CycleResult
