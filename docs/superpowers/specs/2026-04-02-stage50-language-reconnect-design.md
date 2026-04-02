# Stage 50: Reconnect Language Pipeline — Design Spec

**Дата:** 2026-04-02
**Milestone:** M2 — Языковой контроль
**Gate:** Парсинг "pick up the key" → VSA-вектор, ≥90% accuracy (decode action+object)
**Зависимости:** Stage 46 (SubgoalExtractor), Stage 47 (BFS navigation)

---

## Контекст

M1 завершён: 100% DoorKey + MultiRoom. Начинаем M2 — языковой контроль.

### Проблема

Языковой пайплайн (Stages 7-24) использует HAC (2048-dim FFT) для кодирования предложений.
Planning pipeline (Stages 45-46) использует VSA (512-dim binary) для кодирования состояний.
Это два РАЗНЫХ векторных пространства. Нет моста между языком и планированием.

### Что есть

1. **RuleBasedChunker** (Stage 24a) — парсит текст в chunks: `[ACTION, ATTR, OBJECT, LOCATION, SEQ_BREAK]`
2. **RoleFillerParser** (Stage 20) — кодирует chunks в HAC вектор (2048-dim FFT)
3. **GroundingMap** (Stage 19) — `word ↔ SKS_id`, но SKS_id не используется в VSA pipeline
4. **VSACodebook** (Stage 45) — Binary Spatter Code, `role(name)` и `filler(name)`, XOR bind + majority bundle
5. **SubgoalExtractor** (Stage 46) — `extract(trace) → [Subgoal("pickup_key"), Subgoal("open_door"), Subgoal("reach_goal")]`
6. **SubgoalNavigator** (Stage 46) — навигация к subgoal по имени, target positions, BFS pathfinding

### Что нужно

Единый путь: **text → VSA vector → subgoal sequence**.

---

## Позиция в фазе

**Фаза:** M2 — Языковой контроль
**Маркеры M2:**
- [x] Парсинг инструкции → VSA-вектор (≥90%) — **этот этап**
- [ ] Языковая инструкция → subgoals → навигация (≥70%) — Stage 51

---

## Подходы

### A: Direct VSA encoding (РЕКОМЕНДУЕТСЯ)

Кодировать текстовые инструкции напрямую в VSA-пространство через VSACodebook.

Пайплайн:
```
text → RuleBasedChunker.chunk() → chunks
chunks → LanguageGrounder.encode() → VSA vector (512-dim binary)
VSA vector → LanguageGrounder.decode() → (action, object, attr?)
chunks → LanguageGrounder.to_subgoals() → [Subgoal, ...]
```

Concept vocabulary (в VSACodebook):
- Roles: `"instr_action"`, `"instr_object"`, `"instr_attr"`
- Action fillers: `"action_pickup"`, `"action_open"`, `"action_goto"`, `"action_toggle"`, `"action_drop"`, `"action_put"`
- Object fillers: `"object_key"`, `"object_door"`, `"object_goal"`, `"object_ball"`, `"object_box"`
- Attr fillers: `"color_red"`, `"color_green"`, `"color_blue"`, `"color_purple"`, `"color_yellow"`, `"color_grey"`

Subgoal mapping:
- `("action_pickup", "object_key") → "pickup_key"`
- `("action_open", "object_door") → "open_door"`
- `("action_toggle", "object_door") → "open_door"`
- `("action_goto", "object_goal") → "reach_goal"`

**Trade-offs:**
- ✅ Единое пространство с world model (VSACodebook)
- ✅ Простая архитектура, тестируемая
- ✅ Прямая совместимость с SubgoalNavigator
- ❌ Bypasses HAC/DAF perception (но для M2 это OK — язык = интерфейс, не основа мышления)
- ❌ Не использует learned grounding (GroundingMap)

### B: HAC encoding + conversion

Сохранить HAC парсинг, добавить конвертер HAC → VSA.

**Trade-offs:**
- ✅ Сохраняет инвестиции в HAC pipeline
- ❌ Два разных пространства, конвертация нетривиальна (2048-dim FFT ≠ 512-dim binary)
- ❌ Нет гарантии сохранения семантики при конвертации
- ❌ Overengineered для текущего gate

### C: Dual encoding (HAC + VSA)

Кодировать в оба пространства параллельно.

**Trade-offs:**
- ✅ Максимальная гибкость
- ❌ Двойная сложность без двойной пользы
- ❌ Какое пространство использовать для planning?

### Выбор: A — Direct VSA encoding

Обоснование: единое пространство = нет проблемы "мост между пространствами". HAC pipeline (Stages 7-24) остаётся доступным для будущего использования, но для M2 VSA достаточно. Принцип СНКС: "язык = интерфейс, не основа мышления" — encoding формат не принципиален, важна семантическая корректность.

---

## Дизайн

### LanguageGrounder

Файл: `src/snks/language/language_grounder.py`

```python
class LanguageGrounder:
    """Maps natural language instructions to VSA vectors and subgoal sequences.
    
    Uses RuleBasedChunker for parsing and VSACodebook for encoding.
    Единое VSA-пространство с world model.
    """
    
    def __init__(self, codebook: VSACodebook):
        self.cb = codebook
        self.chunker = RuleBasedChunker()
        # Word → VSA filler mapping
        self._action_map: dict[str, str]  # "pick up" → "action_pickup"
        self._object_map: dict[str, str]  # "key" → "object_key"
        self._attr_map: dict[str, str]    # "red" → "color_red"
        # Subgoal mapping
        self._subgoal_map: dict[tuple[str, str], str]  # (action, object) → subgoal_name
    
    def encode(self, instruction: str) -> torch.Tensor:
        """Encode instruction → VSA vector (512-dim binary).
        
        Returns bundled vector: bind(role_action, filler) ⊕ bind(role_object, filler) [⊕ bind(role_attr, filler)]
        """
    
    def decode(self, vsa_vector: torch.Tensor) -> dict[str, str]:
        """Decode VSA vector → {action, object, attr?}.
        
        Unbind each role, find closest filler by similarity.
        """
    
    def to_subgoals(self, instruction: str) -> list[str]:
        """Convert instruction → ordered list of subgoal names.
        
        "pick up the key then open the door" → ["pickup_key", "open_door"]
        """
    
    def encode_sequence(self, instruction: str) -> list[torch.Tensor]:
        """For sequential instructions: list of VSA vectors (one per sub-instruction)."""
```

### Word → VSA Filler Mapping

```python
ACTION_TO_VSA = {
    "pick up": "action_pickup",
    "go to": "action_goto", 
    "open": "action_open",
    "toggle": "action_toggle",
    "put": "action_put",
    "drop": "action_drop",
}

OBJECT_TO_VSA = {
    "key": "object_key",
    "door": "object_door",
    "goal": "object_goal",
    "ball": "object_ball",
    "box": "object_box",
}

ATTR_TO_VSA = {
    "red": "color_red",
    "green": "color_green",
    "blue": "color_blue",
    "purple": "color_purple",
    "yellow": "color_yellow",
    "grey": "color_grey",
    "gray": "color_grey",
}

SUBGOAL_MAP = {
    ("action_pickup", "object_key"): "pickup_key",
    ("action_open", "object_door"): "open_door",
    ("action_toggle", "object_door"): "open_door",
    ("action_goto", "object_goal"): "reach_goal",
    ("action_goto", "object_door"): "open_door",  # "go to the door" → go near door
    ("action_goto", "object_key"): "pickup_key",   # "go to the key" → go near key
}
```

### Decode Strategy

Для decode VSA → action/object:
1. Unbind `role("instr_action")` from vector → candidate
2. Compare candidate to all action fillers: `similarity(candidate, filler("action_X"))`
3. Return action with highest similarity (≥0.6 threshold)
4. Same for object and attr

### Gate Test

```python
def test_gate_90():
    """Gate: ≥90% of instructions correctly encode→decode."""
    instructions = [
        "pick up the key",
        "pick up the red key",
        "open the door",
        "go to the goal",
        "toggle the door",
        "pick up the blue key",
        "pick up the green key",
        "go to the door",
        "drop the key",
        "put the ball",
        # ... 20+ instructions
    ]
    correct = 0
    for instr in instructions:
        vsa = grounder.encode(instr)
        decoded = grounder.decode(vsa)
        if matches_original(decoded, instr):
            correct += 1
    assert correct / len(instructions) >= 0.90
```

---

## Тестовый план

### Unit tests (`tests/test_language_grounder.py`)

1. **Chunker compatibility** — RuleBasedChunker produces expected chunks for MiniGrid instructions
2. **Word mapping** — all expected words map to VSA fillers
3. **Single instruction encode/decode** — "pick up the key" roundtrip
4. **Attributed instruction** — "pick up the red key" preserves color
5. **Sequential instruction** — "pick up the key then open the door" produces 2 VSA vectors
6. **Subgoal mapping** — all (action, object) pairs map to correct subgoal names
7. **Unknown words** — graceful handling of unmapped words
8. **Gate test** — ≥90% accuracy on 20+ instructions

### Experiment (`src/snks/experiments/exp108_language_vsa.py`)

- Encode→decode accuracy on 30+ varied instructions
- Similarity matrix between encoded instructions (should be distinct)
- Subgoal mapping accuracy

---

## Файлы

| Файл | Действие |
|------|----------|
| `src/snks/language/language_grounder.py` | NEW — LanguageGrounder class |
| `tests/test_language_grounder.py` | NEW — unit tests |
| `src/snks/experiments/exp108_language_vsa.py` | NEW — experiment |
| `demos/stage-50-language-reconnect.html` | NEW — web demo |
| `docs/reports/stage-50-report.md` | NEW — report |
