# Stages 10–13: Архитектурная спецификация (Фаза 1)

**Дата:** 2026-03-25
**Версия:** 1.1
**Статус:** Approved (после review-fixes)
**Контекст:** JEPA-inspired extensions. Основан на [`research/lecun_jepa_research.md`](../../../research/lecun_jepa_research.md)
**Зависимость:** Этапы 0–9 завершены (SPEC v0.4.0)

---

## Содержание

1. [Принципы](#1-принципы)
2. [Зависимости и поток данных](#2-зависимости-и-поток-данных)
3. [Stage 10: Hierarchical Prediction](#3-stage-10-hierarchical-prediction)
4. [Stage 11: Multi-Future Simulation](#4-stage-11-multi-future-simulation)
5. [Stage 12: Intrinsic Cost Module](#5-stage-12-intrinsic-cost-module)
6. [Stage 13: Configurator](#6-stage-13-configurator)
7. [Эксперименты 21–28](#7-эксперименты-2128)
8. [Интерфейсы и обратная совместимость](#8-интерфейсы-и-обратная-совместимость)
9. [Риски](#9-риски)
10. [Альтернативы (отложены)](#10-альтернативы-отложены)

---

## 1. Принципы

**Перенос из JEPA:** архитектурные идеи (иерархия, множественные будущие, unified cost, мета-контроль) — да. Механизмы обучения (backprop, EMA, VICReg) — нет.

**СНКС-способ реализации:**
- Иерархия → разные временные константы EWA поверх одного HAC-механизма
- Множественные будущие → стохастическая выборка из уже существующего CausalWorldModel
- Cost module → weighted sum существующих сигналов (confidence, PE, firing rate)
- Configurator → детерминированная конечный автомат (не мета-обучение)

**Инкрементальность:** каждый этап расширяет, не заменяет предыдущий. HAC предиктор Stage 9 остаётся работать, Stage 10 добавляет второй уровень поверх него.

**Backward compatibility:** все новые поля в `CycleResult` и `MetacogState` опциональны со значением по умолчанию. `PipelineConfig` расширяется через `field(default_factory=...)`.

---

## 2. Зависимости и поток данных

```
Stage 9 (done)
  SKSEmbedder → embeddings: dict[int, Tensor]  ← L1 input
  HACPredictionEngine L1 → winner_pe: float
  BroadcastPolicy → broadcast_currents

Stage 10: Hierarchical Prediction
  MetaEmbedder(embeddings) → meta_embed: Tensor     ← новый
  HACPredictionEngine L2(meta_embed) → meta_pe: float  ← новый
  MetacogState.meta_pe = meta_pe

Stage 11: Multi-Future Simulation
  StochasticSimulator(CausalWorldModel) → sample trajectories  ← новый
  Используется ТОЛЬКО в agent.find_plan(); pipeline не затрагивается

Stage 12: Intrinsic Cost Module
  IntrinsicCostModule(MetacogState + firing_rate) → CostState  ← новый
  MetacogState.cost = CostState

Stage 13: Configurator
  Configurator(MetacogState) → ConfiguratorAction  ← новый
  Modifies DafConfig/MetacogConfig in-place (bounded)
  CycleResult.configurator_action = action
```

**Ключевой инвариант:** Stage 11 (StochasticSimulator) работает полностью в слое агента, не в pipeline. Он не затрагивает `perception_cycle()`. Это изолирует риски Stage 11.

---

## 3. Stage 10: Hierarchical Prediction

### 3.1 Цель

Добавить второй временной масштаб предсказания. L1 (Stage 9) предсказывает цикл-к-циклу. L2 (Stage 10) предсказывает на горизонте ~4–8 циклов, аккумулируя контекст через EWA.

### 3.2 Ключевое решение: MetaEmbedder

**Выбранный подход: непрерывная EWA поверх HAC.**

```python
# Каждый цикл:
cycle_embed_t = hac.bundle(list(embeddings.values()))  # агрегат L1
meta_embed_t = normalize(decay * meta_embed_{t-1} + (1-decay) * cycle_embed_t)
```

`meta_embed` — это "медленный" аккумулятор, интегрирующий ~1/(1-decay) циклов.
При decay=0.7: эффективное окно ≈ 3.3 цикла (быстрое L2).
При decay=0.9: эффективное окно ≈ 10 циклов (медленное L2).

**Дефолт: decay=0.8** (окно ≈ 5 циклов).

**Почему EWA, не явное окно:** не требует буфера, легко масштабируется, биологически реалистичен (аналог slow cortical potentials).

### 3.3 L2 Predictor

Тот же `HACPredictionEngine`, но получает meta_embed вместо per-SKS embeddings:

```python
# L2 получает dict с одним ключом "meta"
l2_input = {"meta": meta_embed_t}
l2_predicted = l2_predictor.predict_next(l2_input)   # predict перед observe
l2_predictor.observe(l2_input)                        # update memory
if l2_predicted is not None and prev_l2_predicted is not None:
    meta_pe = l2_predictor.compute_winner_pe(prev_l2_predicted, meta_embed_t)
else:
    meta_pe = 0.0
```

**Временная логика:** `meta_pe_t` = расстояние между `l2_predicted_{t-1}` и `meta_embed_t`. Это честное forward-looking предсказание (предыдущая предсказала → сравниваем с реальностью).

### 3.4 Влияние на confidence (top-down)

Добавляем `delta` в формулу уверенности:

```
confidence = α*dominance + β*stability + γ*(1-pe_L1) + δ*(1-meta_pe)
```

**Backward compatibility:** если `HierarchicalConfig.enabled = False`, то `meta_pe = 0` и `delta = 0` → формула идентична Stage 9.

Дефолт при включённом Stage 10: `alpha=0.25, beta=0.25, gamma=0.25, delta=0.25`.

**Guard на warmup (fix CRITICAL-1):** `meta_pe` включается только если `meta_pe > 0.0` (аналог Stage 9 guard на `winner_pe`). Это предотвращает завышение confidence пока L2 predictor не накопил наблюдения. Нормировка перевзвешивает оставшиеся компоненты:

```python
# В MetacogMonitor.update():
if meta_pe > 0.0:
    pe_L2_term = delta * (1.0 - meta_pe)
    norm = alpha + beta + gamma + delta
else:
    pe_L2_term = 0.0
    norm = alpha + beta + gamma  # нормировка без delta

confidence = (alpha * dominance + beta * stability
              + gamma * (1.0 - pe_for_confidence)
              + pe_L2_term) / max(norm, 1e-8)
confidence = max(0.0, min(1.0, confidence))
```

### 3.5 Новые модули

**`src/snks/sks/meta_embedder.py`** — MetaEmbedder:

```python
class MetaEmbedder:
    def __init__(self, hac: HACEngine, config: HierarchicalConfig) -> None:
        self._decay: float = config.meta_decay  # 0.8
        self._meta: Tensor | None = None

    def update(self, embeddings: dict[int, Tensor]) -> Tensor | None:
        """Compute cycle aggregate, update EWA meta-embedding.

        cycle_embed = bundle(embeddings.values())
        meta = normalize(decay * meta_prev + (1-decay) * cycle_embed)

        Returns current meta_embed or None if embeddings empty.
        """

    def get_meta_embed(self) -> Tensor | None:
        """Current meta-embedding, unit norm."""

    def reset(self) -> None:
        """Reset meta-embedding to None.

        Кто вызывает: вызывающая сторона (Agent/experiment) после конца эпизода.
        Pipeline НЕ знает о границах эпизодов и НЕ вызывает reset автоматически.
        Пример использования:
            for episode in episodes:
                pipeline.meta_embedder.reset()
                pipeline.l2_predictor.reset()  # HACPredictionEngine._memory = None
                for image in episode:
                    result = pipeline.perception_cycle(image)
        При отсутствии reset(): meta_embed несёт контекст предыдущего эпизода.
        Это допустимо в continuous control, но не в episodic experiments.
        """
```

**Изменённые модули:**
- `src/snks/daf/types.py` → добавить `HierarchicalConfig`, расширить `MetacogConfig` (delta), `PipelineConfig`
- `src/snks/metacog/monitor.py` → `MetacogState` добавить `meta_pe: float = 0.0`; формула confidence учитывает delta
- `src/snks/pipeline/runner.py` → `CycleResult` добавить `meta_embedding: Tensor | None = None`, `meta_pe: float = 0.0`; расширить `perception_cycle()` шагами 7c–7f

**Расширение `perception_cycle()` (после шага 7b Stage 9):**
```
7c. [NEW Stage 10] MetaEmbedder.update(embeddings) → meta_embed_t
7d. [NEW] L2 predictor: l2_predicted = hac_l2.predict_next({"meta": meta_embed_t})
7e. [NEW] meta_pe_t = compute_meta_pe(prev_l2_predicted, meta_embed_t)
7f. [NEW] hac_l2.observe({"meta": meta_embed_t})
7g. [NEW] Store l2_predicted для следующего цикла
```

**Новые конфиги:**
```python
@dataclass
class HierarchicalConfig:
    enabled: bool = True
    meta_decay: float = 0.8          # EWA decay ≈ 5-cycle window
    memory_decay: float = 0.95       # L2 predictor memory decay
```

```python
# MetacogConfig расширяется:
delta: float = 0.0   # вес meta_pe в confidence (0 = backward compatible)
```

---

## 4. Stage 11: Multi-Future Simulation

### 4.1 Цель

Расширить `MentalSimulator.find_plan()` стохастическим вариантом: вместо детерминистического BFS — N случайных траекторий с оценкой среднего исхода.

### 4.2 Ключевое решение: StochasticSimulator

**Источник стохастичности:** `CausalWorldModel.get_all_effects_for_action()` уже возвращает ВСЕ известные эффекты с confidence. Сэмплируем по Softmax(confidence/T).

```python
class StochasticSimulator:
    def __init__(
        self, causal_model: CausalWorldModel, seed: int | None = None
    ) -> None:
        self._causal = causal_model
        self._rng = np.random.RandomState(seed)

    def sample_effect(
        self, context: set[int], action: int, temperature: float = 1.0
    ) -> tuple[set[int], float]:
        """Стохастически выбрать один эффект из распределения.

        P(effect_i) ∝ softmax(confidence_i / temperature)
        При temperature→0: детерминистично (argmax).
        При temperature→∞: равномерно.
        Если effects пуст → (set(), 0.0).
        """

    def rollout(
        self,
        initial_sks: set[int],
        action_sequence: list[int],
        temperature: float = 1.0,
    ) -> tuple[list[tuple[set[int], float]], float]:
        """Один стохастический rollout.

        Returns: (trajectory[(sks, confidence)], total_cost)
        total_cost = sum(1 - confidence_i for each step)
        """

    def find_plan_stochastic(
        self,
        current_sks: set[int],
        goal_sks: set[int],
        n_actions: int = 5,
        n_samples: int = 8,
        temperature: float = 1.0,
        max_depth: int = 10,
        min_confidence: float = 0.3,
    ) -> tuple[list[int] | None, float]:
        """Monte Carlo планирование (fix CRITICAL-2 — явный pseudocode).

        Алгоритм — жадный greedy-step с N-sample scoring:

        plan = []
        state = current_sks
        for step in range(max_depth):
            if goal_sks ⊆ state:
                return plan, success_rate

            # Для каждого действия: оценить через N rollouts глубины max_depth-step
            best_action, best_score = None, -inf
            for a in range(n_actions):
                scores = []
                for _ in range(n_samples):
                    effect, conf = sample_effect(state, a, temperature)
                    if conf < min_confidence:
                        scores.append(0.0)
                        continue
                    next_s = state | effect
                    # Rollout: от next_s запустить rollout(next_s, [], depth-1)
                    # Score = 1 если goal достигнута в rollout, иначе 0
                    traj, _ = rollout(next_s, random_actions, temperature)
                    scores.append(1.0 if goal_sks ⊆ traj[-1][0] else 0.0)
                score_a = mean(scores)
                if score_a > best_score:
                    best_score, best_action = score_a, a

            if best_action is None or best_score == 0.0:
                break  # нет прогресса

            # Зафиксировать действие детерминистически (argmax, не sample):
            effect, conf = predict_effect(state, best_action)  # детерм.
            if conf < min_confidence:
                break
            state = state | effect
            plan.append(best_action)

        return (plan if goal_sks ⊆ state else None), best_score

        Ключевое: выбор best_action детерминистический (не стохастический).
        Стохастика используется ТОЛЬКО при оценке (scoring phase).
        Переход после выбора — через детерминистический predict_effect.
        Это избегает двусмысленности "какое next_state взять".

        Returns: (plan или None, ожидаемая success_rate)
        """
```

**Стоимость step**: `cost_per_step = 1 - confidence`. Stage 12 заменит это на ICM.

### 4.3 Интеграция с агентом

`CausalAgent` (или потребитель) выбирает планировщик через конфиг:

```python
# CausalAgentConfig расширяется:
@dataclass
class StochasticPlanConfig:
    enabled: bool = False        # включается явно
    n_samples: int = 8
    temperature: float = 1.0
```

`MentalSimulator.find_plan()` остаётся без изменений (backward compatible). `StochasticSimulator` — отдельный класс.

**Почему не модифицировать MentalSimulator:** MentalSimulator простой и хорошо протестированный (Exp 8 PASS). Лучше добавить новый класс, чем рисковать регрессией.

### 4.4 Новые файлы

- `src/snks/agent/stochastic_simulator.py` — StochasticSimulator
- `src/snks/daf/types.py` → добавить `StochasticPlanConfig` в `CausalAgentConfig`

---

## 5. Stage 12: Intrinsic Cost Module

### 5.1 Цель

Унифицировать гетерогенные сигналы (confidence, curiosity, homeostasis) в единую energy function по аналогии с Cost Module ЛеКуна.

### 5.2 Ключевое решение: CostState

```python
@dataclass
class CostState:
    total: float           # итоговая стоимость ∈ [0, 1]. Высокая = плохо.
    homeostatic: float     # отклонение от target firing rate ∈ [0, 1]
    epistemic_value: float # информационная ценность (PE) ∈ [0, 1]. Высокая = любопытно.
    goal: float            # задачная стоимость ∈ [0, 1]. Внешняя установка.
```

### 5.3 Формула

```
comfort        = 1 - homeostatic_cost     ∈ [0, 1]  # гомеостатический комфорт
curiosity      = epistemic_value           ∈ [0, 1]  # информационная ценность
goal_progress  = 1 - goal_cost             ∈ [0, 1]  # прогресс к цели

value          = w_h * comfort + w_e * curiosity + w_g * goal_progress   ∈ [0, 1]
total_cost     = 1 - value                                                 ∈ [0, 1]
```

Интуиция:
- Гомеостатическое нарушение → ↓comfort → ↑cost
- Высокая PE (удивление) → ↑curiosity → ↓cost (система стремится к новому)
- Достижение цели → ↑goal_progress → ↓cost

Дефолты: `w_h=0.3, w_e=0.4, w_g=0.3`.

### 5.4 Вычисление компонентов

```python
homeostatic_cost = min(abs(mean_firing_rate - target) / max(target, 1e-8), 1.0)

# epistemic_value = max доступных PE-сигналов (fix MAJOR-1).
# Используем max(), а не mean(): если хотя бы один источник сигнализирует
# удивление — это уже информативно. Mean подавлял бы сильный сигнал
# при отсутствии второго. Max консервативнее и устойчивее к cold-start L2.
available = [x for x in [winner_pe, meta_pe] if x > 0.0]
epistemic_value = max(available) if available else 0.0

goal_cost = self._goal_cost   # устанавливается извне через .set_goal_cost()
```

### 5.5 IntrinsicCostModule

```python
class IntrinsicCostModule:
    def __init__(self, config: CostModuleConfig) -> None:
        self.config = config
        self._goal_cost: float = 0.0

    def set_goal_cost(self, cost: float) -> None:
        """Установить задачную стоимость [0, 1]."""
        self._goal_cost = float(cost)

    def compute(
        self, metacog_state: MetacogState, mean_firing_rate: float
    ) -> CostState:
        """Вычислить intrinsic cost из текущего состояния."""
```

### 5.6 Интеграция в pipeline

**После шага 8 MetacogMonitor.update():**
```
8c. [NEW Stage 12] IntrinsicCostModule.compute(metacog_state, mean_firing_rate) → cost_state
    metacog_state.cost = cost_state
```

**mean_firing_rate** берётся из DafEngine: среднее `states[:, 0].mean()` (компонент v FHN-осциллятора). Добавить как `CycleResult.mean_firing_rate: float = 0.0`.

### 5.7 Отношение к confidence

ICM **не заменяет** confidence. Confidence остаётся для GWS/BroadcastPolicy. ICM — отдельный сигнал, использующийся в:
- StochasticSimulator (оценка траекторий, Stage 11 upgrade)
- Configurator (Stage 13)
- Exp 25–26

### 5.8 Новые файлы и изменения

- NEW: `src/snks/metacog/cost_module.py` — IntrinsicCostModule
- `src/snks/daf/types.py` → добавить `CostModuleConfig`, `CostState`
- `src/snks/metacog/monitor.py` → `MetacogState` добавить `cost: CostState | None = None`
- `src/snks/pipeline/runner.py` → `CycleResult` добавить `mean_firing_rate: float = 0.0`
- `src/snks/daf/types.py` → `PipelineConfig` добавить `cost_module: CostModuleConfig`

**Новый конфиг:**
```python
@dataclass
class CostModuleConfig:
    enabled: bool = True
    w_homeostatic: float = 0.3
    w_epistemic: float = 0.4
    w_goal: float = 0.3
    # firing_rate_target: None означает "взять из DafConfig.homeostasis_target".
    # Это предотвращает рассинхронизацию при изменении DafConfig (fix MINOR-1).
    firing_rate_target: float | None = None
```

При создании `IntrinsicCostModule` в Pipeline:
```python
if config.cost_module.firing_rate_target is None:
    config.cost_module.firing_rate_target = config.daf.homeostasis_target
```

---

## 6. Stage 13: Configurator

### 6.1 Цель

Реализовать мета-контроль: динамически адаптировать параметры системы на основе наблюдаемого состояния (CostState + MetacogState), без backprop.

### 6.2 Ключевое решение: детерминированный конечный автомат

**Не мета-обучение** (backprop запрещён). **Не Hebbian meta-control** (слишком сложно без чёткого обоснования). Вместо этого — явные правила переключения режимов с гистерезисом.

Это надёжно, прозрачно и не добавляет learning-зависимостей.

### 6.3 Режимы

```python
class ConfiguratorMode(str, Enum):
    NEUTRAL      = "neutral"       # дефолт
    EXPLORE      = "explore"       # высокий cost + высокая epistemic_value → увеличить пластичность
    CONSOLIDATE  = "consolidate"   # низкий cost + высокая stability → снизить пластичность
    GOAL_SEEKING = "goal_seeking"  # goal_cost > 0 → переориентировать confidence
```

**Правила перехода (после гистерезиса K=8 циклов):**
```
EXPLORE:      total_cost > 0.65  AND  epistemic_value > 0.45
CONSOLIDATE:  total_cost < 0.35  AND  stability > 0.70
GOAL_SEEKING: goal_cost > 0.1   (приоритет над EXPLORE/CONSOLIDATE)
NEUTRAL:      иначе
```

### 6.4 Действия по режимам

```
EXPLORE:
  stdp_a_plus  = min(original * 1.15, stdp_a_plus_max)   # ↑ пластичность
  stdp_a_minus = min(original * 1.10, stdp_a_minus_max)
  hac_pred.memory_decay = 0.98                             # медленнее забывать

  # Divergence safeguard (fix MAJOR-2):
  # Если cycles_in_mode > max_explore_cycles (=32) → принудительно NEUTRAL.
  # Предотвращает "застревание" в EXPLORE при divergence, когда cost
  # не снижается несмотря на увеличение пластичности.
  if cycles_in_mode > max_explore_cycles:
      force_mode = NEUTRAL  # сбросить и дать системе успокоиться

CONSOLIDATE:
  stdp_a_plus  = max(original * 0.85, stdp_a_plus_min)   # ↓ пластичность
  stdp_a_minus = max(original * 0.90, stdp_a_minus_min)
  hac_pred.memory_decay = 0.90                             # быстрее обновлять

GOAL_SEEKING:
  metacog.gamma = 0.0                                      # игнорировать pred_error
  metacog.delta = 0.0                                      # игнорировать meta_pe
  metacog.alpha = 0.5, metacog.beta = 0.5                  # dominance + stability

NEUTRAL:
  Восстановить все параметры до оригинальных значений из config
```

**Ограничения параметров (защита от разгона):**
```
stdp_a_plus:  [original * 0.5, original * 2.0]
stdp_a_minus: [original * 0.5, original * 2.0]
memory_decay: [0.85, 0.99]
metacog weights: [0.0, 1.0], сумма должна оставаться ≤ 1.0
```

### 6.5 Configurator

```python
@dataclass
class ConfiguratorAction:
    mode: str                             # текущий режим
    changed: dict[str, tuple[float, float]]  # param → (old, new)
    cycles_in_mode: int                   # сколько циклов в режиме

class Configurator:
    def __init__(
        self,
        config: ConfiguratorConfig,
        original_daf: DafConfig,
        original_metacog: MetacogConfig,
        original_hac_pred: HACPredictionConfig,
    ) -> None:
        """Сохраняет оригинальные значения для восстановления."""

    def update(
        self,
        metacog_state: MetacogState,
    ) -> ConfiguratorAction | None:
        """Определить режим, применить действия если режим подтверждён.

        Returns ConfiguratorAction если что-то изменилось, иначе None.
        Изменяет daf_config/metacog_config in-place (bounded).
        """
```

**Важно:** Configurator хранит ссылки на конфиги и модифицирует их напрямую. Это не thread-safe, но СНКС однопоточный.

### 6.6 Интеграция в pipeline

```
8d. [NEW Stage 13] Configurator.update(metacog_state) → action | None
    CycleResult.configurator_action = action
```

**Требует:** `metacog_state.cost` (от Stage 12). Stage 13 без Stage 12 работает в NEUTRAL режиме (cost = None → все пороги не срабатывают).

### 6.7 Новые файлы и изменения

- NEW: `src/snks/metacog/configurator.py` — Configurator, ConfiguratorMode
- `src/snks/daf/types.py` → добавить `ConfiguratorConfig`, `ConfiguratorAction`
- `src/snks/pipeline/runner.py` → `CycleResult` добавить `configurator_action: ConfiguratorAction | None = None`
- `src/snks/daf/types.py` → `PipelineConfig` добавить `configurator: ConfiguratorConfig`

**Новый конфиг:**
```python
@dataclass
class ConfiguratorConfig:
    enabled: bool = True
    hysteresis_cycles: int = 8           # минимум циклов для смены режима
    max_explore_cycles: int = 32         # принудительный выход из EXPLORE (divergence guard)
    explore_cost_threshold: float = 0.65
    explore_epistemic_threshold: float = 0.45
    consolidate_cost_threshold: float = 0.35
    consolidate_stability_threshold: float = 0.70
    goal_cost_threshold: float = 0.10
```

---

## 7. Эксперименты 21–28

### Stage 10: Hierarchical Prediction

| # | Название | Метрика | Порог | Описание |
|---|---------|---------|-------|---------|
| 21 | Meta-embedding stability | cosine_similarity(meta_embed_A, meta_embed_B) | > 0.7 для одного класса, < 0.3 для разных | Подаём последовательности одного класса → meta_embed должны быть похожи; разных классов → различаться |
| 22 | Hierarchical prediction accuracy | L2 accuracy vs L1 accuracy на горизонте K=5 | L2 > L1 + 0.05 (5 pp) | Обучаем на shapes последовательностях. L2 предсказывает meta_embed через 5 циклов; L1 предсказывает прямым extrapolation. Метрика: cosine similarity с фактическим. |

**Exp 21 design (fix MINOR-2):** 10 классов shapes, 50 последовательностей каждого, по 20 циклов каждая.
- Вычислить `intra_sim[c]` = mean cosine_similarity всех пар meta_embed внутри класса c (взять финальный meta_embed через 20 циклов)
- Вычислить `inter_sim[c1,c2]` = mean cosine_similarity финальных meta_embed разных классов
- Пороги выбираются из percentiles: порог intra > P10 выборки inter, порог inter < P90 выборки intra
- Конкретные значения 0.7/0.3 — стартовые гипотезы; если не достигаются, ablation по `meta_decay ∈ {0.5, 0.7, 0.8, 0.9}` выбирает оптимальное.

**Exp 22 design:** Shapes последовательности A→B→C→A→... L1 prediction horizon=1 (встроен). L2 horizon=5. Точность = cosine(predicted, actual) > 0.6 как "правильно".

### Stage 11: Multi-Future Simulation

| # | Название | Метрика | Порог | Описание |
|---|---------|---------|-------|---------|
| 23 | Multi-trajectory planning | success_rate: stochastic / deterministic | > 1.2× | CausalGrid с добавленным шумом (p=0.2 случайный эффект). Сравниваем find_plan_stochastic (N=8) vs find_plan. |
| 24 | Stochastic robustness | success_rate при разных N | Монотонный рост N=1..16 | Фиксируем задачу, варьируем N={1,2,4,8,16}. Ожидаем: N=1 ≈ детерминистическому, N=16 ≈ оптимальному. |

**Exp 23 design:** CausalGrid 8×8, goal = угол, 100 эпизодов на метод. Шум: с вероятностью 0.2 `predict_effect` возвращает случайный эффект вместо известного. Метрика: доля эпизодов, где цель достигнута за ≤ max_depth шагов.

### Stage 12: Intrinsic Cost Module

| # | Название | Метрика | Порог | Описание |
|---|---------|---------|-------|---------|
| 25 | Cost-driven exploration | coverage_ratio ICM / curiosity-only (Exp 9) | > 1.0× (не хуже) | Заменяем IntrinsicMotivation.select_action() cost-based выбором. ICM должен не уступать Exp 9 baseline. |
| 26 | Goal-directed behavior | goal_success_rate | > 0.7 | Задаём goal через set_goal_cost(). Агент должен достигать цели в ≥ 70% эпизодов. Baseline: без goal_cost → 50% (случайно). |

**Exp 25 design:** Та же среда что Exp 9 (CausalGrid). Агент использует ICM.total_cost как novelty сигнал для action selection. Coverage_ratio ≥ Exp9.baseline (1.258).

**Exp 26 design:** CausalGrid, цель = конкретная клетка. set_goal_cost: высокий для всех клеток, 0 для целевой. Через эпизоды агент должен научиться идти к цели.

### Stage 13: Configurator

| # | Название | Метрика | Порог | Описание |
|---|---------|---------|-------|---------|
| 27 | Adaptive vs fixed config | task_score: adaptive / fixed | > 1.1× | Mixed benchmark: сначала новая среда (нужен EXPLORE), потом известная (нужен CONSOLIDATE). Configurator должен адаптироваться. |
| 28 | Context switching speed | cycles_to_switch | < 20 | Смена задачи: CONSOLIDATE → EXPLORE. Метрика: сколько циклов до переключения режима. Ожидаем: ≤ hysteresis_cycles × 2 = 16. |

**Exp 27 design:** Двухфазный эксперимент. Фаза 1: новая CausalGrid (20 × 20, незнакомая). Фаза 2: знакомая Grid (обученная ранее). Метрика: средняя успешность задач (по 50 эпизодов).

**Exp 28 design:** Переключение среды на шаге T=100. Регистрируем `configurator_action.mode` каждый цикл. Время переключения = от T до первого stable EXPLORE.

---

## 8. Интерфейсы и обратная совместимость

### CycleResult (полный список дополнений)

```python
@dataclass
class CycleResult:
    # --- Stage 0–9: без изменений ---
    sks_clusters: dict[int, set[int]]
    n_sks: int
    mean_prediction_error: float
    n_spikes: int
    cycle_time_ms: float
    gws: GWSState | None = None
    metacog: MetacogState | None = None
    winner_pe: float = 0.0
    winner_embedding: Tensor | None = None
    hac_predicted: Tensor | None = None
    # --- Stage 10 ---
    meta_embedding: Tensor | None = None
    meta_pe: float = 0.0
    # --- Stage 12 ---
    mean_firing_rate: float = 0.0
    # --- Stage 13 ---
    configurator_action: ConfiguratorAction | None = None
```

### MetacogState (полный список дополнений)

```python
@dataclass
class MetacogState:
    # --- Stage 0–9: без изменений ---
    confidence: float
    dominance: float
    stability: float
    pred_error: float
    winner_pe: float = 0.0
    winner_nodes: set[int] = field(default_factory=set)
    # --- Stage 10 ---
    meta_pe: float = 0.0
    # --- Stage 12 ---
    cost: CostState | None = None
```

### PipelineConfig (полный список дополнений)

```python
@dataclass
class PipelineConfig:
    # --- Stage 0–9: без изменений ---
    daf: DafConfig = field(default_factory=DafConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dcam: DcamConfig = field(default_factory=DcamConfig)
    sks: SKSConfig = field(default_factory=SKSConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    gws: GWSConfig = field(default_factory=GWSConfig)
    metacog: MetacogConfig = field(default_factory=MetacogConfig)
    sks_embed: SKSEmbedConfig = field(default_factory=SKSEmbedConfig)
    hac_prediction: HACPredictionConfig = field(default_factory=HACPredictionConfig)
    steps_per_cycle: int = 100
    device: str = "auto"
    # --- Stage 10 ---
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    # --- Stage 12 ---
    cost_module: CostModuleConfig = field(default_factory=CostModuleConfig)
    # --- Stage 13 ---
    configurator: ConfiguratorConfig = field(default_factory=ConfiguratorConfig)
```

### Карта файлов (все новые и изменённые)

```
NEW:
  src/snks/sks/meta_embedder.py           # Stage 10
  src/snks/agent/stochastic_simulator.py  # Stage 11
  src/snks/metacog/cost_module.py         # Stage 12
  src/snks/metacog/configurator.py        # Stage 13

  src/snks/experiments/exp21_meta_embedding.py
  src/snks/experiments/exp22_hierarchical_prediction.py
  src/snks/experiments/exp23_multifuture_planning.py
  src/snks/experiments/exp24_stochastic_robustness.py
  src/snks/experiments/exp25_cost_exploration.py
  src/snks/experiments/exp26_goal_navigation.py
  src/snks/experiments/exp27_adaptive_config.py
  src/snks/experiments/exp28_context_switching.py

  tests/test_stage10.py
  tests/test_stage11.py
  tests/test_stage12.py
  tests/test_stage13.py

MODIFIED:
  src/snks/daf/types.py          # новые конфиги и датаклассы
  src/snks/metacog/monitor.py    # MetacogState, формула confidence
  src/snks/pipeline/runner.py    # CycleResult, perception_cycle()
```

---

## 9. Риски

### R1: Meta-embedding collapse (Stage 10)
**Проблема:** При decay=0.8 и монотонном вводе meta_embed может коллапсировать в один вектор.
**Митигация:** Exp 21 явно тестирует это. Если collapse → снизить decay (0.5–0.6) или добавить noise injection в EWA.
**Альтернатива:** Explicit K-window (см. §10).

### R2: Стохастическая выборка не даёт разнообразия (Stage 11)
**Проблема:** Если CausalWorldModel знает только один эффект для большинства (action, context), все N samples идентичны → нет выгоды от стохастики.
**Митигация:** Exp 24 тестирует это. Если N>1 не помогает → добавить ε-greedy noise: с вероятностью ε брать random effect.
**Альтернатива:** Perturbation в SKS space (заменять случайные SKS в state).

### R3: ICM дестабилизирует обучение (Stage 12)
**Проблема:** Если epistemic_value (PE) всегда высокий → total_cost всегда низкий → нет сигнала для Configurator.
**Митигация:** Нормализовать PE относительно running mean (аналог pred_error_norm в MetacogMonitor).
**Альтернатива:** Уменьшить w_epistemic до 0.2 (более консервативная мотивация).

### R4: Configurator oscillation (Stage 13)
**Проблема:** Режимы переключаются слишком часто → система нестабильна.
**Митигация:** Гистерезис K=8 уже предусмотрен. Если мало → увеличить до K=16.
**Альтернатива:** Добавить cooldown: после смены режима запретить повторную смену на 2K циклов.

### R5: mean_firing_rate как сигнал (Stage 12)
**Проблема:** `states[:, 0].mean()` — это потенциал FHN, не firing rate. Может быть нестабильным.
**Митигация:** Альтернативно: `fired_history.float().mean()` если доступно. Проверить корреляцию с homeostasis.py target в Exp 25.

---

## 10. Альтернативы (отложены)

### A1: Explicit K-window Meta-SKS (вместо EWA)
Формировать мета-СКС явной кластеризацией за окно K=16 циклов.
- Плюсы: более чёткие эпизоды, лучше для длинных горизонтов
- Минусы: требует буфера, дискретные обновления, сложнее
- Когда вернуться: если EWA (Stage 10) не даёт хорошего Exp 22

### A2: MCTS вместо жадного поиска (Stage 11)
Monte Carlo Tree Search с UCB для выбора действий.
- Плюсы: оптимальнее для длинных горизонтов
- Минусы: O(n_actions^depth) → медленно без GPU
- Когда вернуться: если жадный поиск не проходит Exp 23

### A3: Hebbian meta-learning для Configurator (Stage 13)
Configurator обучается через Hebbian rule: ассоциировать (cost_pattern → config_change) через накопленный опыт.
- Плюсы: адаптируется к задаче, не только к пороговым правилам
- Минусы: требует тщательного дизайна Hebbian rule без backprop, высокий риск
- Когда вернуться: после Stage 14+ если rule-based Configurator окажется недостаточным

### A4: Отдельный L3 уровень (Stage 10+)
Добавить третий уровень с decay=0.95 (десятки циклов).
- Плюсы: более полная H-JEPA иерархия
- Минусы: сложность растёт, нет явного тестового требования
- Когда вернуться: после Stage 10 если L2 не хватает для долгосрочного планирования

### A5: ICM как replacement для confidence (Stage 12)
Полностью заменить confidence на ICM total_cost как основной сигнал GWS.
- Плюсы: единый унифицированный сигнал
- Минусы: риск регрессии на Exp 15–17 (confidence-based gates); большой рефакторинг
- Когда вернуться: Stage 14+ после доказательства надёжности ICM

---

## 11. Роадмап реализации

| Этап | Зависит от | Новых файлов | Модифицированных | Экспериментов |
|------|-----------|-------------|-----------------|--------------|
| Stage 10 | Stage 9 (done) | 3 (meta_embedder, exp21, exp22) | 4 (types, monitor, runner, tests) | Exp 21, 22 |
| Stage 11 | Stage 6 CausalModel | 3 (stoch_sim, exp23, exp24) | 2 (types, tests) | Exp 23, 24 |
| Stage 12 | Stage 10 (meta_pe) | 3 (cost_module, exp25, exp26) | 4 (types, monitor, runner, tests) | Exp 25, 26 |
| Stage 13 | Stage 12 (CostState) | 3 (configurator, exp27, exp28) | 3 (types, runner, tests) | Exp 27, 28 |

**Этапы 10 и 11 можно реализовывать параллельно** — они не зависят друг от друга (Stage 11 работает в agent layer, Stage 10 в pipeline layer).

**Этап 12 требует Stage 10** (meta_pe нужен для epistemic_value).
**Этап 13 требует Stage 12** (CostState нужен для mode detection).
