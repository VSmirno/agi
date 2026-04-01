# Stage 41: Temporal Credit Assignment — Eligibility Traces

**Дата:** 2026-04-01
**Статус:** DESIGN
**Автор:** autonomous-dev

## Позиция в фазе

**Фаза 1: Живой DAF (~60% → 70%)**

Этот этап продвигает маркер: "Reward signal достигает ранних решений через eligibility trace". Текущий DafCausalModel хранит только 5 weight-снимков — reward с шага 10 не может повлиять на решение с шага 1. Это критический bottleneck для любых задач длиннее 5 шагов (DoorKey-5x5 требует ~15-25 шагов).

Маркеры, которые продвигает Stage 41:
- [ ] Eligibility trace на 20+ шагов (вместо 5)
- [ ] Reward modulation через trace вместо weight snapshots
- [ ] Память-эффективное решение (O(E) вместо O(E × T))
- [ ] Интеграция с STDP: dw → trace → reward → weight update

## Проблема

Текущий `DafCausalModel` (daf_causal_model.py):
1. **before_action()**: снимает snapshot ВСЕХ edge weights (O(E) per step)
2. **after_action(reward)**: проходит по 5 snapshots, вычисляет delta, модулирует
3. **Проблема**: trace_length=5, decay=0.8^i — шаг 6+ уже не получает credit
4. **Проблема**: хранит 5 полных копий edge_attr — O(5E) memory
5. **Проблема**: delta_w вычисляется как разница текущих и прошлых весов, что включает шум от homeostatic regularization и STDP других пар — credit contamination

### Конкретный пример (DoorKey-5x5)
```
Шаг 1:  Агент видит ключ, STDP усиливает связи key→approach
Шаг 5:  Подбирает ключ (reward=0 в MiniGrid)
Шаг 10: Открывает дверь (reward=0)
Шаг 18: Достигает цели (reward > 0)
→ Текущий trace (5 шагов): credit получают шаги 14-18, но не шаг 1 (key→approach)
```

## Brainstorming

### Подход A: Edge-level Eligibility Traces (ВЫБРАН)

Классический подход из нейронауки (Florian 2007, Izhikevich 2007):

- STDP.apply() уже вычисляет dw per edge — просто возвращаем его
- Trace: `e(t) = λ × e(t-1) + dw(t)` где λ ∈ [0.9, 0.99]
- На reward: `Δw_reward = η × r × e(t)` — credit пропорционален trace
- Один tensor (E,) вместо 5 snapshots

**Pros:** O(E) memory (одна trace), стандартный нейронаучный подход, естественная интеграция с STDP, плавный decay без жёсткого window
**Cons:** λ-decay может быть слишком быстрым или медленным, trace "размывается" для очень длинных эпизодов
**Trade-off:** простота vs. precision → выбираем простоту (можно улучшить позже)

### Подход B: Расширенные Weight Snapshots
- Увеличить trace_length до 30
- **Rejected:** O(30E) memory = 30× текущего потребления, contamination проблема остаётся

### Подход C: Sparse Trace (top-K edges)
- Хранить trace только для top-K изменённых edges
- **Rejected for now:** усложняет реализацию, premature optimization

### Обоснование выбора A
1. Memory: O(E) вместо O(5E), при этом покрываем 20+ шагов
2. Чистота credit: trace аккумулирует только STDP dw, без homeostatic noise
3. Естественный decay: λ^20 = 0.12 при λ=0.9 (ещё значимый credit через 20 шагов)
4. Модульность: EligibilityTrace — отдельный класс, минимальные изменения в STDP

## Дизайн

### Новые компоненты

#### 1. `EligibilityTrace` (src/snks/daf/eligibility.py)

```python
class EligibilityTrace:
    """Edge-level eligibility trace for reward-modulated STDP.
    
    Accumulates: e(t) = λ × e(t-1) + dw(t)
    On reward:   Δw = η × reward × e(t)
    """
    def __init__(self, decay: float = 0.92, reward_lr: float = 0.5):
        self.decay = decay        # λ — trace decay per step
        self.reward_lr = reward_lr # η — reward learning rate
        self._trace: Tensor | None = None  # (E,) accumulated trace
        self._steps_since_reward: int = 0
        
    def accumulate(self, dw: Tensor) -> None:
        """Add STDP weight changes to trace."""
        if self._trace is None or self._trace.shape != dw.shape:
            self._trace = dw.clone()
        else:
            self._trace.mul_(self.decay).add_(dw)
        self._steps_since_reward += 1
        
    def apply_reward(self, reward: float, graph: SparseDafGraph, 
                     w_min: float, w_max: float) -> int:
        """Apply reward-modulated credit to all traced edges.
        
        Returns: number of edges modulated
        """
        if self._trace is None or abs(reward) < 1e-8:
            return 0
        w = graph.get_strength()
        delta = self.reward_lr * reward * self._trace
        w_new = (w + delta).clamp_(w_min, w_max)
        graph.set_strength(w_new)
        modulated = int((delta.abs() > 1e-8).sum())
        self._steps_since_reward = 0
        return modulated
    
    def reset(self) -> None:
        """Reset trace (e.g., at episode start)."""
        if self._trace is not None:
            self._trace.zero_()
        self._steps_since_reward = 0
```

#### 2. Изменения в STDP

`STDP.apply()` → `STDPResult` уже содержит статистику, но не возвращает dw.
Добавить `dw` в STDPResult:

```python
@dataclass
class STDPResult:
    edges_potentiated: int
    edges_depressed: int
    mean_weight_change: float
    dw: Tensor | None = None  # NEW: per-edge weight change vector
```

В `_apply_rate()` и `_apply_timing()` — сохранить **чистый STDP dw** (до homeostatic regularization), вернуть в result. Homeostatic regularization добавляется ПОСЛЕ snapshot dw.

#### 3. Изменения в DafCausalModel

**НЕ удалять** snapshot-based trace и predict_effect() — они нужны для AttractorNavigator.
Добавить EligibilityTrace КАК ДОПОЛНЕНИЕ к существующему механизму:

```python
class DafCausalModel:
    def __init__(self, engine, reward_scale=2.0, trace_length=5,
                 negative_scale=0.5, trace_decay=0.92, trace_reward_lr=0.5):
        # ... существующий __init__ сохраняется ...
        # NEW: eligibility trace
        self._eligibility = EligibilityTrace(decay=trace_decay, reward_lr=trace_reward_lr)
        
    def accumulate_stdp(self, stdp_result: STDPResult) -> None:
        """Called after each STDP step to accumulate eligibility trace."""
        if stdp_result.dw is not None:
            self._eligibility.accumulate(stdp_result.dw)
    
    def after_action(self, reward: float) -> None:
        """Apply reward: eligibility trace (long-range) + snapshot (short-range)."""
        if abs(reward) < 1e-8 or not self._trace:
            return
        self._total_reward_received += abs(reward)
        self._total_modulations += 1
        
        # 1. Eligibility trace — long-range credit (20+ steps)
        effective_reward = reward * self.reward_scale if reward > 0 \
                          else reward * self.negative_scale
        self._eligibility.apply_reward(
            effective_reward, self.engine.graph,
            self.engine.stdp.w_min, self.engine.stdp.w_max)
    
    # predict_effect(), before_action(), _trace — СОХРАНЯЮТСЯ без изменений
    # before_action() по-прежнему нужен для last_action tracking
```

**Важно:** `before_action()` сохраняется как есть — PureDafAgent использует `_trace[-1].action` для PE explorer. `predict_effect()` сохраняется — AttractorNavigator вызывает его.

#### 4. Изменения в DafEngine

engine.step() → после STDP.apply(), вернуть dw в step_result:

```python
# В engine.step():
stdp_result = self.stdp.apply(self.graph, fired_history)
# step_result включает stdp_result для внешнего потребления
```

#### 5. Изменения в PureDafAgent

В `step()` метод — после pipeline.perception_cycle():
```python
# Accumulate STDP trace
if hasattr(result, 'stdp_result') and result.stdp_result is not None:
    self._causal.accumulate_stdp(result.stdp_result)
```

В `observe_result()`:
```python
# before_action() и snapshot-based trace — УДАЛИТЬ
# after_action(reward) — теперь использует eligibility trace
```

### Конфигурация

Добавить в `CausalAgentConfig`:
```python
trace_decay: float = 0.92        # λ для eligibility trace
trace_reward_lr: float = 0.5     # η для reward modulation
```

### Параметры и обоснование

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| decay (λ) | 0.92 | λ^20 = 0.19 (19% credit через 20 шагов), λ^30 = 0.08 |
| reward_lr (η) | 0.5 | Умеренный reward signal, не перезаписывает STDP |
| negative_scale | 0.5 | Отрицательный reward слабее (не разрушать выученное) |

### Поток данных (новый)

```
1. Engine.step() → STDP.apply() → STDPResult (с dw)
2. PureDafAgent получает STDPResult из perception_cycle
3. DafCausalModel.accumulate_stdp(result) → trace += dw
4. Env.step(action) → reward
5. DafCausalModel.after_action(reward) → Δw = η * r * trace → graph weights
6. Trace *= λ (decay продолжается)
```

## Gate-критерии

| Test | Метрика | Gate |
|------|---------|------|
| Trace accumulation | trace non-zero after 10 steps | True |
| Trace decay | trace[t+20] < 0.2 × trace[t] | True |
| Reward modulation | edges_modulated > 0 on reward | True |
| Long-range credit | dw at step 1 changes from reward at step 15 | True |
| No regression | DoorKey-5x5 runs without error | True |
| Memory efficiency | peak memory <= 1.5× baseline | True |
| Credit reach | effective_window >= 20 steps | True |

## Файлы

### Новые
- `src/snks/daf/eligibility.py` — EligibilityTrace
- `src/snks/experiments/exp100_temporal_credit.py` — experiments
- `tests/test_eligibility_trace.py` — unit tests

### Изменяемые
- `src/snks/daf/stdp.py` — STDPResult.dw, return dw из apply()
- `src/snks/agent/daf_causal_model.py` — заменить snapshot на EligibilityTrace
- `src/snks/agent/pure_daf_agent.py` — передача stdp_result в causal model
- `src/snks/daf/engine.py` — expose stdp_result
- `src/snks/daf/types.py` — trace_decay, trace_reward_lr в CausalAgentConfig
- `demos/index.html` — карточка Stage 41
