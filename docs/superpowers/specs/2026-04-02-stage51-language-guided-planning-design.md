# Stage 51: Language-Guided Planning — Design Spec

**Дата:** 2026-04-02
**Milestone:** M2 — Языковой контроль
**Gate:** ≥70% success на random DoorKey-5x5 с текстовой инструкцией
**Зависимости:** Stage 50 (LanguageGrounder), Stage 47 (SubgoalNavigator + BFS)

---

## Контекст

Stage 50 создал мост text → VSA → subgoals. Теперь нужно замкнуть цикл: агент получает текстовую инструкцию и выполняет её в среде.

### Что есть

1. **LanguageGrounder** (Stage 50) — `to_subgoals("pick up the key then open the door then go to the goal") → ["pickup_key", "open_door", "reach_goal"]`
2. **SubgoalPlanningAgent** (Stage 46-47) — `build_plan_from_obs(obs) → True`, создаёт PlanGraph + target positions, BFS-навигация, 100% на random DoorKey-5x5
3. **SubgoalNavigator** (Stage 46-47) — BFS pathfinding к target position, subgoal achievement detection

### Что нужно

**InstructedAgent** — агент, который:
1. Получает текстовую инструкцию вместо hardcoded reward
2. Парсит инструкцию → subgoals через LanguageGrounder
3. Строит plan из observation (target positions)
4. Выполняет ТОЛЬКО subgoals из инструкции (не полный DoorKey sequence)

---

## Позиция в фазе

**Фаза:** M2 — Языковой контроль
**Маркеры M2:**
- [x] Парсинг инструкции → VSA-вектор (≥90%) — Stage 50
- [ ] Языковая инструкция → subgoals → навигация (≥70%) — **этот этап**

---

## Подходы

### A: Thin wrapper over SubgoalPlanningAgent (РЕКОМЕНДУЕТСЯ)

InstructedAgent = SubgoalPlanningAgent + LanguageGrounder. Instruction определяет КАКИЕ subgoals выполнять (вместо hardcoded "pickup_key → open_door → reach_goal").

```
instruction → LanguageGrounder.to_subgoals() → subgoal_names
obs → build_plan_from_obs() → target_positions (для тех subgoals что в instruction)
SubgoalNavigator → BFS → actions
```

**Trade-offs:**
- ✅ Переиспользует 100% proven infrastructure (BFS, SubgoalNavigator, PlanGraph)
- ✅ Минимальный новый код — фокус на интеграции
- ✅ Gate 70% реалистичен (Stage 47 уже 100% с hardcoded subgoals)
- ❌ Не добавляет "понимания" языка — маппинг чисто символический

### B: LanguageGrounder + VSA state matching

Использовать VSA-вектор инструкции для прямого matching с world state.

**Trade-offs:**
- ✅ Более "когнитивный" подход
- ❌ VSA state similarity ~0.5-0.7 для разных контекстов (Stage 45) — ненадёжно
- ❌ Не решает проблему навигации (всё равно нужен BFS)

### Выбор: A — Thin wrapper

Обоснование: Stage 51 проверяет ИНТЕГРАЦИЮ language → planning, не новые cognitive mechanisms. Proven infrastructure + language bridge = M2 gate.

---

## Дизайн

### InstructedAgent

Файл: `src/snks/agent/instructed_agent.py`

```python
class InstructedAgent:
    """Agent that follows natural language instructions in MiniGrid environments.
    
    Pipeline: instruction → LanguageGrounder → subgoals → SubgoalNavigator → actions
    """
    
    def __init__(self, codebook: VSACodebook):
        self.grounder = LanguageGrounder(codebook)
        self.navigator = SubgoalNavigator(...)
        self.plan: PlanGraph | None = None
    
    def set_instruction(self, instruction: str) -> list[str]:
        """Parse instruction → subgoal names. Returns subgoal list."""
    
    def build_plan(self, obs: np.ndarray) -> bool:
        """Build navigation plan from observation for current subgoals."""
    
    def step(self, obs: np.ndarray) -> int:
        """Select action based on current plan and observation."""
    
    def run_episode(self, env, instruction: str, max_steps: int = 200) -> tuple[bool, int]:
        """Run one episode with given instruction."""
```

### Instruction variants for testing

```python
DOORKEY_INSTRUCTIONS = [
    "pick up the key then open the door then go to the goal",
    "pick up the key and then open the door and then go to the goal",
    # Partial instructions (test that agent follows ONLY what's asked):
    "pick up the key",          # → agent picks up key, stops
    "go to the goal",           # → impossible without key+door, should fail
]
```

### Success criteria

- Full instruction ("key → door → goal"): ≥70% success на 200 random layouts
- Partial instruction ("pick up the key"): agent reaches key position ≥70%
- Wrong order instruction: agent attempts but may fail (no gate)

---

## Тестовый план

### Unit tests

1. **Instruction parsing** — correct subgoals extracted
2. **Plan building** — target positions match subgoal objects  
3. **Single step** — correct action toward current subgoal
4. **Subgoal advancement** — plan advances when subgoal achieved
5. **Episode completion** — full DoorKey episode with instruction

### Experiment (exp109)

- 200 random DoorKey-5x5, full instruction → success rate
- 50 random DoorKey-5x5, partial instruction "pick up the key" → key pickup rate

---

## Файлы

| Файл | Действие |
|------|----------|
| `src/snks/agent/instructed_agent.py` | NEW — InstructedAgent |
| `tests/test_instructed_agent.py` | NEW — unit tests |
| `src/snks/experiments/exp109_instructed_agent.py` | NEW — experiment |
| `demos/stage-51-language-guided.html` | NEW — web demo |
| `docs/reports/stage-51-report.md` | NEW — report |
