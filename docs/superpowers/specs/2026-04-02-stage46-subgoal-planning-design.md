# Stage 46: Subgoal Planning — Design Spec

**Date:** 2026-04-02
**Phase:** 1 — Живой DAF
**Type:** Planning mechanism — subgoal extraction + chained navigation
**Depends on:** Stage 45 (VSA+SDM foundation)

## Мотивация

Stage 45 доказал: VSA+SDM foundation работает (97% encoding, 0.85 prediction). Но все три стратегии планирования провалились на DoorKey — detour task, где прямой путь к цели блокирован дверью.

**Корневая причина:** планирование оптимизирует единую метрику (goal similarity / reward) без понимания **каузальной структуры** задачи. DoorKey требует цепочку: key → door → goal. Forward beam search гонит агента прямо к цели → упирается в дверь.

**Решение:** извлечь subgoals из успешных traces и навигировать к ним последовательно.

## Позиция в фазе

**Фаза 1 маркер:** Pure DAF ≥ 50% DoorKey-5x5.
Stage 46 адресует TD-005 (planning FAIL). Если subgoal planning достигнет ≥ 15% plan phase success, это подтвердит архитектурную жизнеспособность model-based planning через VSA+SDM.

## Что НЕ входит в scope

- Изменения в VSACodebook/VSAEncoder/SDMMemory (Stage 45, замороженные)
- Автоматическое обнаружение subgoals без prior experience (needs more traces)
- Multi-room / multi-key environments (future stages)
- Интеграция с DAF oscillators

---

## Анализ подходов

### Подход A: Symbolic Event Detection
Обнаружить key events сравнивая последовательные observations: key исчезает (picked up), door state меняется (opened).
- **Pro:** детерминированный, интерпретируемый, zero-shot для known object types
- **Con:** хрупкий к новым object types, не масштабируется на unknown tasks
- **Trade-off:** работает для DoorKey, но нужна расширяемость

### Подход B: VSA State Diff
XOR текущего и следующего VSA-вектора — "событие" = большое изменение (low similarity).
- **Pro:** generic, не зависит от типов объектов
- **Con:** "большое изменение" может быть просто ходьба в новую позицию; нужен threshold
- **Trade-off:** потенциально масштабируемый, но нужна калибровка

### Подход C: Trace Segmentation + Landmark States
Разбить successful trace на сегменты по ключевым состояниям (landmarks). Landmark = state перед значительным изменением в trajectory (bottleneck в state space).
- **Pro:** data-driven, generic
- **Con:** нужно много traces для статистики, сложнее реализация
- **Trade-off:** лучшая масштабируемость, но больше данных

### **Выбран: A + B гибрид**

**Обоснование:**
1. Symbolic detection (A) для known events (pickup, toggle) — гарантирует работу на DoorKey
2. VSA diff (B) как fallback для unknown events — обеспечивает расширяемость
3. Оба метода дополняют друг друга: symbolic = precision, VSA diff = recall
4. Не используем reward shaping (против философии СНКС, memory: feedback_planning_approach)

---

## Архитектура

```
Successful Traces → SubgoalExtractor → ordered [Subgoal]
                                              ↓
                                         PlanGraph
                                              ↓
                    current_obs → SubgoalNavigator → action
                                    ↓ (uses SDM for prediction)
                                    ↓ (advances subgoal on achievement)
```

### Компонент 1: SubgoalExtractor

Извлекает subgoals из successful episode traces.

```python
@dataclass
class Subgoal:
    name: str                    # human-readable: "pickup_key", "open_door", "reach_goal"
    target_state: torch.Tensor   # VSA vector of the target state (after achievement)
    precondition_state: torch.Tensor  # state just before achievement
    detection_type: str          # "symbolic" | "vsa_diff"
    
class SubgoalExtractor:
    def extract(self, trace: list[TraceStep]) -> list[Subgoal]:
        """Extract ordered subgoals from a successful episode trace.
        
        TraceStep = (obs_before, action, obs_after, reward)
        
        Strategy:
        1. Symbolic scan: detect pickup (key disappears), toggle (door opens)
        2. VSA diff scan: detect states with similarity < threshold (big changes)
        3. Merge & order by trace position
        4. Always append "reach_goal" as final subgoal
        """
```

**Symbolic detection rules:**
- `pickup_key`: key object (type=5) present in obs_before, absent in obs_after
- `open_door`: door state changes from locked (state=2) to open (state=0)
- `reach_goal`: agent position matches goal position

**VSA diff detection:**
- Encode obs_before and obs_after with VSAEncoder
- If similarity < 0.7 AND not already caught by symbolic detection → new subgoal

### Компонент 2: PlanGraph

Ordered chain of subgoals.

```python
class PlanGraph:
    def __init__(self, subgoals: list[Subgoal]):
        self.subgoals = subgoals
        self.current_idx = 0
    
    def current_subgoal(self) -> Subgoal | None:
        if self.current_idx >= len(self.subgoals):
            return None
        return self.subgoals[self.current_idx]
    
    def advance(self) -> bool:
        """Advance to next subgoal. Returns True if plan complete."""
        self.current_idx += 1
        return self.current_idx >= len(self.subgoals)
    
    def reset(self):
        self.current_idx = 0
```

### Компонент 3: SubgoalNavigator

Навигирует к текущему subgoal используя SDM world model.

```python
class SubgoalNavigator:
    def select(self, current_state: Tensor, target_subgoal: Subgoal) -> int:
        """Select action that moves toward subgoal.
        
        Strategy:
        1. For each action, predict next_state via SDM
        2. Compute similarity of predicted next_state to subgoal.target_state
        3. Pick action with highest similarity (+ epsilon exploration)
        
        Fallback: if no SDM predictions confident → random action
        """
```

**Subgoal achievement detection:**
```python
def is_achieved(self, current_obs: ndarray, subgoal: Subgoal) -> bool:
    """Check if subgoal is achieved in current observation."""
    if subgoal.name == "pickup_key":
        return not self._key_visible(current_obs)
    elif subgoal.name == "open_door":
        return self._door_open(current_obs)
    elif subgoal.name == "reach_goal":
        return False  # detected by env reward
    # VSA diff fallback
    current_vsa = self.encoder.encode(current_obs)
    return self.codebook.similarity(current_vsa, subgoal.target_state) > 0.75
```

### Компонент 4: SubgoalPlanningAgent (extends WorldModelAgent)

```python
class SubgoalPlanningAgent(WorldModelAgent):
    """Extended agent with subgoal extraction and chained navigation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.extractor = SubgoalExtractor(self.codebook, self.encoder)
        self.navigator = SubgoalNavigator(self.sdm, self.codebook, self.encoder)
        self.plan: PlanGraph | None = None
        self._traces: list = []  # successful traces
    
    def run_episode(self, env, max_steps=200):
        # Explore phase: random + collect traces (same as Stage 45)
        # Plan phase: if we have traces → extract subgoals → navigate
        
        if self._exploring:
            return super().run_episode(env, max_steps)
        
        if not self.plan and self._traces:
            subgoals = self.extractor.extract(self._traces[0])  # best trace
            self.plan = PlanGraph(subgoals)
        
        if not self.plan:
            return super().run_episode(env, max_steps)
        
        # Subgoal-directed navigation
        self.plan.reset()
        obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            current_subgoal = self.plan.current_subgoal()
            if current_subgoal is None:
                break  # plan complete
            
            state = self.encoder.encode(obs)
            action = self.navigator.select(state, current_subgoal)
            obs, reward, term, trunc, _ = env.step(action)
            self.observe(obs, reward)
            total_reward += reward
            
            # Check subgoal achievement
            if self.navigator.is_achieved(obs, current_subgoal):
                self.plan.advance()
            
            if term or trunc:
                break
        
        return total_reward > 0, step + 1, total_reward
```

---

## Конфигурация

```python
@dataclass
class SubgoalConfig(WorldModelConfig):
    vsa_diff_threshold: float = 0.7   # below this = significant state change
    achievement_threshold: float = 0.75  # VSA sim for subgoal achievement
    max_subgoals: int = 5             # max extracted subgoals per trace
    use_best_trace: bool = True       # use shortest successful trace for planning
```

---

## Эксперименты

### Exp 106a: Subgoal Extraction Accuracy
- Run 50 explore episodes on DoorKey-5x5
- Extract subgoals from each successful trace
- Gate: extracted subgoals include "pickup_key" AND "open_door" in ≥ 80% of successful traces

### Exp 106b: Plan Graph Construction
- From extracted subgoals, verify ordering: pickup_key before open_door before reach_goal
- Gate: correct ordering in 100% of cases

### Exp 106c: SubgoalPlanningAgent on DoorKey-5x5
- 100 episodes (50 explore + 50 plan)
- Gate (primary): plan phase success_rate ≥ 15% (TD-005 gate)
- Gate (stretch): plan phase success_rate ≥ 30%
- Gate (vs random): plan phase > explore phase success rate

### Exp 106d: Subgoal Navigation Quality
- Measure steps-to-subgoal for each subgoal independently
- Gate: mean steps to pickup_key ≤ 50 (from start)
- Gate: mean steps to open_door ≤ 30 (from having key)

---

## Порядок выполнения

| Шаг | Что | Где | Зависимости |
|-----|-----|-----|-------------|
| 1 | SubgoalExtractor + tests | CPU | — |
| 2 | PlanGraph + tests | CPU | — |
| 3 | SubgoalNavigator + tests | CPU | Шаг 1, 2 |
| 4 | SubgoalPlanningAgent + tests | CPU | Шаг 1, 2, 3 |
| 5 | Exp 106a (extraction accuracy) | CPU | Шаг 1 |
| 6 | Exp 106b (plan graph) | CPU | Шаг 2 |
| 7 | Exp 106c (DoorKey-5x5) | CPU/minipc | Шаг 4 |
| 8 | Exp 106d (navigation quality) | CPU | Шаг 3, 4 |

---

## Что дальше (не в scope)

- **Hierarchical subgoals:** subgoal decomposition (subgoal of subgoal)
- **Automatic subgoal discovery** without prior success traces
- **Multi-room planning:** chain of rooms with per-room subgoals
- **Transfer:** subgoal patterns across different environments
