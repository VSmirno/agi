# Этап 6: Каузальный агент в среде

**Статус:** Реализовано — 49 тестов PASS, 287 total PASS (без регрессии)
**Зависимости:** Этапы 3 ✅ (Pipeline, PredictionEngine), 4 ✅ (DCAM)

## Цель

Перевести СНКС из пассивного наблюдателя в **активного агента**: система действует в среде, наблюдает последствия, формирует каузальные модели и использует ментальную симуляцию для планирования. Это ключевой шаг от перцепции к когниции.

### Что доказывает этот этап

1. **Каузальное обучение** — система отличает причинность от корреляции через интервенцию
2. **Ментальная симуляция** — предсказание результатов действий без их выполнения
3. **Intrinsic motivation** — curiosity-driven исследование среды без внешнего reward
4. **Замкнутый цикл** — восприятие → решение → действие → наблюдение последствий

---

## Среда: MiniGrid + расширения

**Движок:** [MiniGrid](https://minigrid.farama.org/) (Farama Foundation)
- `pip install minigrid`
- Gymnasium-совместимый API
- RGB-выход через `RGBImgObsWrapper` → resize до 64×64 → grayscale → совместимо с `VisualEncoder`

### Кастомная среда: `CausalGridWorld`

Расширение MiniGrid с объектами для каузальных экспериментов:

| Объект | Тип | Свойства |
|--------|-----|----------|
| Agent | — | Позиция, ориентация, 5 действий |
| Wall | Immovable | Блокирует движение |
| Box | Pushable | Двигается при толчке агентом |
| Ball | Rollable | Двигается при толчке, катится до стены |
| Key | Pickable | Агент может подобрать |
| Door | Toggleable | Открывается ключом |
| Goal | Static | Маркер цели |
| Lava | Static | "Опасная" зона (отрицательная валентность) |

### Действия агента

| # | Действие | Эффект |
|---|----------|--------|
| 0 | turn_left | Поворот на 90° |
| 1 | turn_right | Поворот на 90° |
| 2 | forward | Движение на 1 клетку вперёд |
| 3 | interact | Толкнуть/подобрать/переключить объект перед агентом |
| 4 | noop | Ничего не делать |

### Сценарии (уровни)

| Уровень | Название | Сетка | Описание |
|---------|----------|-------|----------|
| L1 | EmptyExplore | 8×8 | Пустая комната — базовое исследование |
| L2 | PushBox | 8×8 | 1 коробка — выучить push-каузальность |
| L3 | PushChain | 10×10 | 3 коробки в ряд — транзитивная каузальность |
| L4 | BallRoll | 8×8 | 1 мяч — отложенный эффект (катится дальше) |
| L5 | DoorKey | 8×8 | Ключ + дверь — инструментальная каузальность |
| L6 | MultiRoom | 12×12 | 2 комнаты, дверь, коробки — комбинация |

---

## Новые модули

| Модуль | Назначение |
|--------|-----------|
| `env/causal_grid.py` | CausalGridWorld: MiniGrid-сценарии, Gymnasium API |
| `env/obs_adapter.py` | ObsAdapter: RGB→grayscale 64×64, Gymnasium obs→Tensor |
| `agent/motor.py` | MotorEncoder: действие (int) → SDR currents для ДАП |
| `agent/causal_model.py` | CausalWorldModel: каузальный граф действие→СКС→эффект |
| `agent/simulation.py` | MentalSimulator: прогон каузальных цепочек без среды |
| `agent/motivation.py` | IntrinsicMotivation: curiosity на базе prediction error |
| `agent/agent.py` | CausalAgent facade: восприятие→решение→действие |
| `experiments/exp7_causal.py` | Exp 7: каузальное обучение |
| `experiments/exp8_simulation.py` | Exp 8: ментальная симуляция |
| `experiments/exp9_curiosity.py` | Exp 9: curiosity-driven exploration |

---

## Контракты

### ObsAdapter

```python
class ObsAdapter:
    """Converts MiniGrid RGB observations to DAF-compatible tensors."""

    def __init__(self, target_size: int = 64):
        ...

    def convert(self, obs: np.ndarray) -> torch.Tensor:
        """RGB (H,W,3) → grayscale (64,64) float32 [0,1]."""
        ...
```

### MotorEncoder

```python
class MotorEncoder:
    """Encodes discrete actions as SDR current patterns for DAF injection."""

    def __init__(self, n_actions: int, num_nodes: int, sdr_size: int = 512):
        ...

    def encode(self, action: int) -> torch.Tensor:
        """Action index → (num_nodes, state_dim) current injection.

        Каждое действие активирует фиксированную группу нод (моторная зона).
        Группы не пересекаются для разных действий.
        """
        ...

    def decode(self, firing_rates: torch.Tensor) -> int:
        """Firing rate vector → most likely action (winner-take-all)."""
        ...
```

### CausalWorldModel

```python
@dataclass
class CausalLink:
    """Directed causal relationship: action in context → effect."""
    action: int                 # какое действие
    context_sks: frozenset[int] # при каких активных СКС (что видим)
    effect_sks: frozenset[int]  # какие СКС активируются в результате
    strength: float             # уверенность (0..1)
    count: int                  # сколько раз наблюдалось

class CausalWorldModel:
    """Learns causal relationships: (context, action) → effect.

    Ключевое отличие от PredictionEngine (этап 3):
    - PredictionEngine: СКС_A → СКС_B (пассивные последовательности)
    - CausalWorldModel: (контекст + действие) → эффект (интервенционная каузальность)
    """

    def __init__(self, config: CausalAgentConfig):
        ...

    def observe_transition(
        self,
        pre_sks: set[int],     # СКС до действия
        action: int,            # выполненное действие
        post_sks: set[int],     # СКС после действия
    ) -> None:
        """Записать наблюдение: (контекст, действие) → эффект."""
        ...

    def predict_effect(
        self,
        context_sks: set[int],
        action: int,
    ) -> tuple[set[int], float]:
        """Предсказать эффект действия в контексте.

        Returns:
            (predicted_sks, confidence)
        """
        ...

    def get_causal_links(self, min_confidence: float = 0.3) -> list[CausalLink]:
        """Извлечь все каузальные связи выше порога."""
        ...
```

### MentalSimulator

```python
class MentalSimulator:
    """Simulates action sequences using CausalWorldModel without real environment.

    'Что будет, если я сделаю A, потом B, потом C?'
    Прогоняет каузальную цепочку, возвращает предсказанную траекторию СКС.
    """

    def __init__(self, causal_model: CausalWorldModel):
        ...

    def simulate(
        self,
        initial_sks: set[int],
        action_sequence: list[int],
    ) -> list[tuple[set[int], float]]:
        """Simulate action sequence, return (predicted_sks, confidence) per step."""
        ...

    def find_plan(
        self,
        current_sks: set[int],
        goal_sks: set[int],
        max_depth: int = 10,
    ) -> list[int] | None:
        """BFS/DFS через каузальную модель для поиска плана достижения цели.

        Returns:
            Sequence of actions to reach goal, or None if not found.
        """
        ...
```

### IntrinsicMotivation

```python
class IntrinsicMotivation:
    """Curiosity-driven action selection based on prediction error.

    Принцип: выбирай действие, которое максимизирует ожидаемый prediction error
    (информационный gain). Но с затуханием — уже изученные переходы скучны.
    """

    def __init__(self, config: CausalAgentConfig):
        ...

    def select_action(
        self,
        current_sks: set[int],
        causal_model: CausalWorldModel,
        n_actions: int,
    ) -> int:
        """Select action that maximizes expected information gain.

        Формула:
            interest(a) = novelty(context, a) × uncertainty(context, a)
            novelty = 1 / (1 + visit_count(context, a))
            uncertainty = 1 - confidence(context, a)

        Epsilon-greedy: с вероятностью epsilon выбираем случайное действие.
        """
        ...

    def update(self, context_sks: set[int], action: int, prediction_error: float) -> None:
        """Update visit counts and novelty estimates."""
        ...
```

### CausalAgent

```python
class CausalAgent:
    """Top-level agent: perceive → decide → act → learn.

    Интегрирует Pipeline (этап 3) с каузальным обучением (этап 6).
    """

    def __init__(self, config: CausalAgentConfig):
        self.pipeline: Pipeline           # из этапа 3
        self.obs_adapter: ObsAdapter
        self.motor: MotorEncoder
        self.causal_model: CausalWorldModel
        self.simulator: MentalSimulator
        self.motivation: IntrinsicMotivation

    def step(self, obs: np.ndarray) -> int:
        """Full agent cycle:

        1. obs → grayscale 64×64 (ObsAdapter)
        2. image → Pipeline.perception_cycle() → CycleResult (СКС)
        3. IntrinsicMotivation.select_action() → action
        4. MotorEncoder.encode(action) → inject motor currents into DAF
        5. Return action for environment
        """
        ...

    def observe_result(self, obs: np.ndarray) -> None:
        """After env.step(action), observe the consequence:

        1. obs → perception_cycle → new СКС
        2. CausalWorldModel.observe_transition(pre_sks, action, post_sks)
        3. Compute prediction error
        4. IntrinsicMotivation.update()
        """
        ...

    def plan_to_goal(self, goal_obs: np.ndarray) -> list[int] | None:
        """Plan action sequence to reach a goal state using mental simulation."""
        ...
```

---

## Конфигурация

```python
@dataclass
class CausalAgentConfig:
    """Configuration for causal agent (Stage 6)."""
    # Pipeline (reuse from stage 3-5)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Environment
    grid_size: int = 8
    max_steps_per_episode: int = 200

    # Motor encoder
    motor_sdr_size: int = 512           # SDR bits per action
    motor_zone_start: int = 0           # start node index for motor zone
    # motor zone: nodes [motor_zone_start : motor_zone_start + n_actions * motor_sdr_size]

    # Causal model
    causal_min_observations: int = 3    # min obs before confident
    causal_confidence_threshold: float = 0.5
    causal_decay: float = 0.99
    causal_context_hash_bits: int = 64  # для быстрого поиска по контексту

    # Motivation
    curiosity_epsilon: float = 0.2      # random exploration rate
    curiosity_decay: float = 0.995      # novelty decay per visit

    # Mental simulation
    simulation_max_depth: int = 10
    simulation_min_confidence: float = 0.3
```

---

## Эксперименты

### Exp 7: Каузальное обучение (push-каузальность)

**Среда:** L2 (PushBox) — 8×8, 1 коробка

**Протокол:**
1. Агент исследует среду 500 шагов (curiosity-driven)
2. Фиксируем каузальные связи в CausalWorldModel
3. Тест: предъявляем 50 ситуаций "агент перед коробкой"
   - Спрашиваем: predict_effect(context, action=interact)
   - Ground truth: коробка сдвигается на 1 клетку

**Контрольный тест (каузальность vs корреляция):**
- В среде есть "декоративный" объект (Ball), который двигается одновременно с Box, но НЕ из-за действия агента (скрипт двигает его)
- Агент должен НЕ формировать каузальную связь action→Ball
- Каузальная связь action→Box должна быть сильной

**Метрики:**
- **Causal precision:** доля верных каузальных связей среди всех выученных
- **Causal recall:** доля реальных каузальных связей, которые выучены

**Gate:** Precision > 0.8, Recall > 0.7

### Exp 8: Ментальная симуляция

**Среда:** L3 (PushChain) — 10×10, 3 коробки в ряд

**Протокол:**
1. Обучение: агент изучает среду 1000 шагов, учит каузальную модель
2. Тест: предъявляем начальное состояние (агент → box1 → box2 → box3)
   - Задача: MentalSimulator.simulate([interact, forward, interact])
   - Ожидание: предсказать, что все 3 коробки сдвинутся

**Метрики:**
- **Simulation accuracy:** доля верно предсказанных состояний в цепочке
- **Planning success:** для 20 случайных начальных позиций, может ли find_plan() найти путь к цели

**Gate:** Simulation accuracy > 0.7, Planning success > 0.5

### Exp 9: Curiosity-driven exploration

**Среда:** L6 (MultiRoom) — 12×12, 2 комнаты, дверь, коробки

**Протокол:**
1. Запускаем CausalAgent с IntrinsicMotivation на 2000 шагов
2. Параллельно запускаем random agent (uniform random actions)
3. Сравниваем:
   - Покрытие среды (% посещённых клеток)
   - Количество обнаруженных каузальных связей
   - Скорость обнаружения "интересных" объектов (дверь, ключ)

**Метрики:**
- **Coverage ratio:** curious_coverage / random_coverage
- **Discovery speed:** шаги до обнаружения всех типов каузальных связей

**Gate:** Coverage ratio > 1.5 (любопытство покрывает в 1.5x больше за то же время)

---

## Архитектурные решения

### 1. Моторная зона в ДАП

Ноды ДАП разделяются на зоны:
- **Сенсорная зона** (0 .. N-motor): получает SDR от VisualEncoder (как раньше)
- **Моторная зона** (N-motor .. N): получает SDR от MotorEncoder

Моторная и сенсорная зоны связаны через STDP — это формирует сенсомоторные ассоциации. Когда агент видит коробку и толкает — STDP усиливает связи между "вижу коробку" (сенсорная СКС) и "толкаю" (моторная СКС).

### 2. Каузальность через интервенцию

Ключевое отличие от PredictionEngine (этап 3):
- PredictionEngine учит: "после A обычно B" (корреляция)
- CausalWorldModel учит: "когда я ДЕЛАЮ X в контексте C, происходит Y" (каузальность)

Формально: P(effect | do(action), context) ≠ P(effect | action, context)

Каузальная связь укрепляется ТОЛЬКО когда:
1. Агент сам выполнил действие (интервенция)
2. Эффект наблюдался ПОСЛЕ действия
3. Эффект НЕ наблюдался без действия (контрфактуальный контроль)

### 3. Двойной injection в ДАП

В каждом цикле Pipeline получает два входа:
1. Визуальный SDR → сенсорная зона (как раньше)
2. Моторный SDR → моторная зона (новое)

ДАП эволюционирует с обоими входами, формируя сенсомоторные СКС.

### 4. Контекстный хеш для каузальной модели

Для быстрого lookup в CausalWorldModel:
- context = frozenset активных СКС → хешируем в 64-бит ключ
- (context_hash, action) → effect — основная таблица каузальной модели
- Коллизии хешей не критичны: confidence сглаживает

---

## Модификации существующих модулей

| Файл | Изменение |
|------|----------|
| `daf/types.py` | Добавить `CausalAgentConfig` |
| `pipeline/runner.py` | Добавить `inject_motor_currents()` — второй вход в ДАП |
| `daf/engine.py` | Поддержка dual injection (сенсорный + моторный) |

---

## Зависимости (pip)

```
minigrid>=2.3.0
gymnasium>=0.29.0
```

---

## Структура файлов после этапа 6

```
src/snks/
├── env/                        # NEW: среда
│   ├── __init__.py
│   ├── causal_grid.py          # CausalGridWorld (MiniGrid scenarios)
│   └── obs_adapter.py          # ObsAdapter: RGB → grayscale 64×64
├── agent/                      # NEW: каузальный агент
│   ├── __init__.py
│   ├── motor.py                # MotorEncoder
│   ├── causal_model.py         # CausalWorldModel, CausalLink
│   ├── simulation.py           # MentalSimulator
│   ├── motivation.py           # IntrinsicMotivation
│   └── agent.py                # CausalAgent facade
├── experiments/
│   ├── exp7_causal.py          # NEW
│   ├── exp8_simulation.py      # NEW
│   └── exp9_curiosity.py       # NEW
└── ... (существующие модули без изменений)
```

---

## Gate (критерии прохождения этапа)

| # | Критерий | Порог |
|---|----------|-------|
| 1 | Exp 7: Causal precision | > 0.8 |
| 2 | Exp 7: Causal recall | > 0.7 |
| 3 | Exp 8: Simulation accuracy | > 0.7 |
| 4 | Exp 8: Planning success | > 0.5 |
| 5 | Exp 9: Coverage ratio vs random | > 1.5 |
| 6 | Unit tests | 100% pass |
| 7 | Интеграция с DCAM | Каузальные связи сохраняются в SSG causal layer |

---

## Порядок реализации

1. **Среда** — `env/causal_grid.py`, `env/obs_adapter.py` + тесты
2. **Моторный кодировщик** — `agent/motor.py` + тесты
3. **Dual injection в Pipeline** — модификация `engine.py`, `runner.py`
4. **Каузальная модель** — `agent/causal_model.py` + тесты
5. **Мотивация** — `agent/motivation.py` + тесты
6. **Ментальная симуляция** — `agent/simulation.py` + тесты
7. **Агент** — `agent/agent.py` + integration tests
8. **Exp 7** — каузальное обучение
9. **Exp 8** — ментальная симуляция
10. **Exp 9** — curiosity exploration
