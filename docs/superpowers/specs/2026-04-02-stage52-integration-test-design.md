# Stage 52: Integration Test — Language-Guided MultiRoom Navigation

**Milestone:** M3 — Концепция доказана
**Gate:** ≥50% random MultiRoom-N3 с языковой инструкцией
**Dependencies:** Stage 49 (MultiRoomNavigator), Stage 51 (InstructedAgent)

---

## Проблема

Stages 49 и 51 доказали отдельные capabilities:
- Stage 49: 100% MultiRoom-N3 (BFS + reactive door toggle)
- Stage 51: 100% DoorKey-5x5 с инструкцией (language → subgoals → BFS)

Но они работают изолированно. Stage 52 интегрирует оба: агент получает текстовую инструкцию и решает MultiRoom-N3 с произвольной раскладкой.

## Подходы

### A: Unified IntegrationAgent (ВЫБРАН)
Единый агент с pipeline:
1. `LanguageGrounder.to_subgoals(instruction)` → subgoal names
2. Анализ среды: сканировать obs → определить env type (has_key? multiple_doors?)
3. Для reach_goal в multi-room: BFS + reactive door toggle (MultiRoomNavigator strategy)
4. Для DoorKey subgoals: SubgoalNavigator chain (InstructedAgent strategy)

**Trade-offs:** +единый interface, +масштабируемый, -чуть больше кода чем B

### B: Dispatcher
Диспетчер по env type. Минимальный код, но не настоящая интеграция — не доказывает, что pipeline масштабируется.

### C: Retrain InstructedAgent
Переписать InstructedAgent для поддержки MultiRoom. Максимальная чистота, но избыточный scope для M3 gate.

## Решение: Подход A

### Архитектура IntegrationAgent

```
instruction → LanguageGrounder → subgoals
                                    ↓
                           env observation scan
                                    ↓
                        ┌───────────┴───────────┐
                    has key?                  no key?
                        ↓                        ↓
                InstructedAgent           MultiRoomNavigator
                (SubgoalNavigator)        (BFS + reactive toggle)
```

### Класс IntegrationAgent

```python
class IntegrationAgent:
    def __init__(self, codebook=None):
        self.grounder = LanguageGrounder(codebook)
        self.instructed = InstructedAgent(codebook)  # DoorKey strategy
        self.multi_room = MultiRoomNavigator(epsilon=0.0)  # MultiRoom strategy
        self._strategy = None  # "doorkey" | "multiroom"

    def run_episode(self, env, instruction, max_steps=500):
        subgoals = self.grounder.to_subgoals(instruction)
        obs = env.reset()
        
        # Detect environment type from obs
        has_key = self._has_object(obs, OBJ_KEY)
        
        if has_key and "pickup_key" in subgoals:
            self._strategy = "doorkey"
            return self._run_doorkey(env, obs, instruction, max_steps)
        else:
            self._strategy = "multiroom"
            return self._run_multiroom(env, obs, subgoals, max_steps)
```

### Инструкции для MultiRoom-N3

```
"go to the goal"           → ["reach_goal"]
"open the doors then go to the goal" → ["open_door", "reach_goal"]
"navigate to the goal"     → ["reach_goal"]  (via goto → reach)
```

Для "reach_goal" в MultiRoom: BFS to goal с allow_door=True + reactive toggle.
Для "open_door" в MultiRoom: reactive — не нужен explicit subgoal, BFS решает.

### Gate criteria

| Exp | Описание | Gate |
|-----|----------|------|
| 110a | 200 random MultiRoom-N3 + "go to the goal" | ≥50% |
| 110b | 200 random MultiRoom-N3 + variant instructions | ≥50% |
| 110c | 200 random DoorKey-5x5 + full instruction (regression) | ≥90% |

### Позиция в фазе

**Фаза M3** — Концепция доказана. Маркеры:
- [x] M1 (генерализация) COMPLETE
- [x] M2 (языковой контроль) COMPLETE
- [ ] Интеграция ≥50% MultiRoom-N3 с инструкцией ← **этот этап**
- [ ] R1 вердикт (Stage 53)

Stage 52 закрывает третий маркер M3. После 52+53 → M3 COMPLETE.

## Файлы

- `src/snks/agent/integration_agent.py` — NEW: IntegrationAgent
- `tests/test_integration_agent.py` — NEW: unit tests
- `src/snks/experiments/exp110_integration.py` — NEW: gate experiments
- `demos/stage-52-integration.html` — NEW: web demo
