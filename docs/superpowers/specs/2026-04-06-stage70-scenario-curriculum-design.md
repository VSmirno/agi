# Stage 70: Scenario Curriculum — Целенаправленное Обучение

**Дата:** 2026-04-06  
**Статус:** Дизайн  
**Предпосылка:** Stage 69 (outcome labeling) не завершён — random walk даёт 33-650 сэмплов, smoke FAIL

---

## Проблема

Все предыдущие этапы обучают энкодер на **случайных траекториях**:

```
phase1_collect(n_trajectories=50, steps_per_traj=200)
→ 50 × случайный walk → JEPA/SupCon на том, что встретили
```

Это работает для простых объектов (дерево — часто встречается), но принципиально не работает для сложных:

| Объект | Встречается при random walk | Требует |
|--------|----------------------------|---------|
| tree | ~40% траекторий | ничего |
| stone | ~9% | ничего (но do-направление!) |
| coal | ~1% | wood_pickaxe |
| iron | ~0% | stone_pickaxe |
| diamond | ~0% | iron_pickaxe |
| table | 0% | 2 wood + place |

**Итог:** Энкодер не видит coal/iron/diamond никогда. NearDetector не может их детектировать.

---

## Принцип

> Человек не учится "случайно тыкаясь" — он учится через **целенаправленное действие**.

Ребёнок, изучающий инструменты:
1. Сначала делает простое (взять палку)
2. Потом среднее (сделать рукоятку)
3. Потом сложное (сделать топор)

Каждый шаг **предполагает** предыдущий. Обучение идёт по **цепочке предпосылок**, не случайно.

Аналог в нашем проекте:
- Знаем что в инвентаре (проприоцепция: `info["inventory"]`)
- Знаем что видим рядом (перцепция: NearDetector)
- Можем **планировать** последовательность действий для гарантированного получения ситуации

---

## Архитектура: ScenarioCurriculum

### Компоненты

```
ScenarioLibrary          — библиотека сценариев (порядок + предпосылки)
    │
    ▼
ScenarioRunner           — FSM исполнитель на основе inventory + NearDetector
    │
    ▼
CurriculumDataCollector  — заменяет phase1_collect() везде
    │
    ▼
(pixels, near_label) × N — обучающие данные без info["semantic"]
```

### ScenarioStep (шаг FSM)

```python
@dataclass
class ScenarioStep:
    navigate_to: str         # что найти через NearDetector ("tree", "stone", ...)
    action: str              # действие ("do", "place_table", "make_wood_pickaxe")
    success_item: str | None # item в инвентаре, по которому судим об успехе
    success_delta: int       # +N (появился) или -N (потрачен)
    near_label: str          # что лейблировать при успехе
    prerequisite_inv: dict   # {"wood": 2} — что нужно в инвентаре перед шагом
```

### Библиотека сценариев для Crafter

```
S1: harvest_wood
    navigate "tree" → do → if wood+1 → label "tree"
    requires: ничего
    
S2: place_table  
    requires: wood≥2
    navigate "empty" → place_table → if wood-2 → label "empty"

S3: craft_wood_pickaxe
    requires: table в мире (S2 выполнен)
    navigate "table" → make_wood_pickaxe → if wood_pickaxe+1 → label "table"

S4: harvest_stone
    requires: wood_pickaxe в инвентаре (S3)
    navigate "stone" → do → if stone+1 → label "stone"

S5: craft_stone_pickaxe
    requires: stone≥3 (S4 × 3) + table
    navigate "table" → make_stone_pickaxe → if stone_pickaxe+1 → label "table"

S6: harvest_coal
    requires: wood_pickaxe (S3)
    navigate "coal" → do → if coal+1 → label "coal"

S7: harvest_iron
    requires: stone_pickaxe (S5) + furnace
    navigate "iron" → do → if iron+1 → label "iron"

S8: harvest_diamond
    requires: iron_pickaxe
    navigate "diamond" → do → if diamond+1 → label "diamond"
```

**Цепочка зависимостей:**
```
S1(tree) → S1×2 → S2(table/empty) → S3(table) → S4(stone) → S5(table) → S6(coal) → S7(iron) → S8(diamond)
```

### ScenarioRunner (FSM)

```python
class ScenarioRunner:
    def run(self, env, detector, scenario_chain, labeler, rng):
        """Выполнить цепочку сценариев. Вернуть (pixels, near_label) пары."""
        labeled = []
        inv = {}  # текущий инвентарь (из info["inventory"])
        
        for step in scenario_chain:
            # Проверить предпосылки через инвентарь
            if not self._prereqs_met(inv, step.prerequisite_inv):
                break  # нет смысла идти дальше
            
            # Навигация к цели через NearDetector + spatial map
            pixels, info, found = find_target_with_map(
                env, detector, smap, step.navigate_to, ...
            )
            if not found:
                break
            
            # Направленный опрос: попробовать все 4 направления
            success, frames = self._directional_probe(
                env, info, step.action, step.success_item, step.success_delta
            )
            if success:
                # Ретроспективное оконное лейблирование
                labeled.extend([(f, step.near_label) for f in frames[-W:]])
                inv = updated_inventory  # обновить состояние
        
        return labeled
```

---

## Покрытие объектов (матрица)

| near_label | Сценарий | Этапов до достижения |
|------------|----------|----------------------|
| tree | S1 | 1 |
| empty | S2 | 2 |
| table | S3, S5 | 3 |
| stone | S4 | 4 |
| coal | S6 | 4 |
| iron | S7 | 6 |
| diamond | S8 | 8 |
| water | специальный (navigate→water + drink) | 2 |
| cow | специальный (navigate→cow + do) | 2 |
| grass | специальный (navigate→grass + do) | 1 |

**Итог:** ВСЕ классы объектов покрыты через цепочку сценариев.

---

## Влияние на все фазы обучения

### Phase 0: Nav encoder (было: random trajectories)

Сейчас:
```python
dataset = phase1_collect(n_trajectories=50, steps_per_traj=200)
# 50 × случайный walk → mostly tree/grass/empty
```

Будет:
```python
dataset = curriculum_collect(
    scenarios=[S1, S4, S6],  # дерево + камень + уголь
    n_runs=50,
    steps_per_run=500,
)
# 50 × целевые траектории → balanced distribution
```

Nav encoder увидит все классы в правильном контексте — лучше детектирует coal/iron.

### Phase 1: Outcome labeling (было: random walk → do)

Сейчас: `_scenario_harvest("coal", 200 seeds) → 0/200` (нет кирки)

Будет: `run_chain(S1→S2→S3→S6, 50 seeds)` → 50 × гарантированный уголь

### Prototype collection для CLSWorldModel

Тоже выиграет: сейчас `do near coal: 0/50 prototypes`, с цепочкой — полное покрытие.

---

## Stage 70 vs Stage 69b

**Вариант A:** Stage 69b = ScenarioCurriculum для outcome labeling
- Закрыть Stage 69 правильно
- Nav encoder всё ещё обучается на random (circular dep. остаётся там)

**Вариант B:** Stage 70 = ScenarioCurriculum везде  
- Nav encoder тоже получает curriculum trajectories
- Полный разрыв circular dependency на всех уровнях
- Stage 69 объявить "принципиально доказанным" (OutcomeLabeler работает), Stage 70 — реализацией

**Рекомендация:** Вариант B.  
Stage 69 доказал принцип (OutcomeLabeler выводит near_label из инвентаря, без info["semantic"]). Stage 70 реализует этот принцип правильно через ScenarioCurriculum на всех уровнях.

---

## Ограничения и открытые вопросы

1. **Nav encoder для coal/iron/diamond** — чтобы навигировать к углю, нужно сначала его увидеть. Если nav encoder никогда не видел уголь, он не будет детектировать его. Решение: сначала обучить nav encoder на curriculum данных (S1-S3), потом запустить S6 с этим лучшим encoder.

2. **Итеративное обучение** — возможно нужно 2-3 итерации: 
   - Итерация 1: nav encoder на S1-S3 → умеет дерево/камень
   - Итерация 2: nav encoder на S1-S6 с улучшенным detector → умеет уголь
   - Итерация 3: полный coverage

3. **Ретроспективное лейблирование** — окно W=5 кадров до успеха. Качество меток: кадры в окне "скорее всего" у целевого объекта, но не гарантированно (игрок мог отойти). Можно добавить фильтрацию: лейблировать только кадры, где NearDetector детектирует target.

4. **Баланс классов** — с curriculum легко контролировать: запускать S1 × 100, S6 × 100, итд.

---

## Критерии успеха Stage 70

| Метрика | Порог |
|---------|-------|
| Классов в обучении | ≥8 из 13 (tree/stone/coal/iron/table/empty/water/cow) |
| Smoke (fair, на обученных классах) | ≥60% |
| QA gate | ≥85% |
| Regression (exp123) | ≥90% |
