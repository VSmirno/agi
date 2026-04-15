# Roadmap SNKS AGI

**Версия:** 1
**Дата:** 2026-04-14
**Статус:** Living document. Обновлять при закрытии stage-а.

> Этот roadmap выводится из `IDEOLOGY.md`, не из wishlist-а.
> Каждый stage решает конкретный идеологический долг.
> Если stage не решает ничего из идеологии — он неправильный.

---

## Текущая позиция

**Stage 84 — COMPLETE (2026-04-15, partial PASS)**

avg_survival=178.9 (+16 vs Stage 82), wood=0%, sleep%=0%. Gates: 2/3.
Vital fix + StimuliLayer реализованы. Wood стена → Stage 85.

---

## Roadmap

### Stage 84 — Real Stimuli Infrastructure ✓ COMPLETE

**Результат:** avg_survival=178.9, gates 2/3 (survival ✓, sleep ✓, wood ✗ pre-existing).
**Идеологический долг:** Категория 4 (Stimuli). `score_trajectory` захардкожен
в mechanism layer. `state.body` всегда 9.0 — реальные виталы не доходят до планировщика.

**Что делаем:**

1. Подключить `HomeostaticTracker` к `VectorState.body` — реальные
   значения food/drink/health/energy на каждом шаге вместо дефолтных 9.0.

2. Выделить `StimuliLayer` из `score_trajectory`:
   ```python
   # Было (в механизме):
   score = (survived, known, total_gain, min_vital, -steps)

   # Стало (в отдельном слое):
   score = stimuli.evaluate(simulated_state)
   # где stimuli = [HomeostasisStimulus, SurvivalAversion]
   ```

3. Sleep работает правильно без хаков: выигрывает только когда
   реальные виталы низкие, проигрывает когда полные.

**Gate:** sleep выбирается агентом при `energy < 3`, не выбирается при
`energy = 9`. Homeostatic recovery статистически выше baseline.

---

### Stage 85 — Curiosity as Primary Driver

**Идеологический долг:** Категория 4, информационный стимул.
`total_gain` знает про wood — это Crafter-специфично. Должен быть
универсальный движок исследования.

**Что делаем:**

Заменить `total_gain` на `expected_surprise`:
```python
# Было (Crafter-специфично):
U_wood(s) = s.inventory["wood"] - initial.inventory["wood"]

# Стало (универсально):
U_curiosity(s) = world_model.expected_surprise(concept, action)
               = 1 - cosine_similarity(predicted, actual)
```

Агент собирает дерево не потому что `wood` захардкожен в scoring,
а потому что `do` рядом с деревом — предсказуемо сюрпризный исход
(до тех пор пока модель не выучила правило).

**Gate:** агент собирает ресурсы без явного `total_gain` в scoring.
Knowledge flow улучшается: gen2 любопытен именно к тому что gen1
не выучил.

---

### Stage 86 — Post-Mortem Learning

**Идеологический долг:** Принцип 6 (Система, не агент). Смерть сейчас
= выброшенная информация. Должна быть обучающим сигналом.

**Что делаем:**

После каждого эпизода:
```python
death_context = {
    "cause": "health",
    "steps": 173,
    "last_vitals": {"health": 0, "food": 2, "drink": 7},
    "nearby_entities": ["zombie"],
    "last_plan": "single:tree:do",
}
post_mortem_learn(death_context, world_model, stimuli)
```

Конкретные обновления:
- Умер рядом с zombie → усилить `aversion(zombie)` stimulus
- Умер при `food=0` → понизить threshold в `HomeostasisStimulus(food)`
- Умер в шаге 35 (сразу) → что изменилось в первые 20 шагов?

**Gate:** `cause=zombie deaths` снижается от gen1 к gen3. `cause=starvation
deaths` снижается при включённом post-mortem vs выключенном.

---

### Stage 87 — Curiosity About Death

**Идеологический долг:** Принцип 6, переформулировка curiosity.
После Stage 85 любопытство = "исследуй новое".
После Stage 87 любопытство = "уменьши неопределённость о причинах смерти".

**Что делаем:**

Система формирует явные гипотезы между эпизодами:
```
"Я умираю от zombies при food < 3 чаще чем при food > 6.
Гипотеза: low food → distracted navigation → zombie exposure.
Следующий эпизод: тест этой гипотезы."
```

Curiosity stimulus теперь взвешен по death-relevance:
```python
U_curiosity(s) = expected_surprise(s) * death_relevance(s)
# death_relevance: насколько этот исход мог повлиять на причины смерти
```

**Gate:** система формулирует минимум 1 проверяемую гипотезу о смерти
за 20 эпизодов. Гипотеза верифицируется в следующих эпизодах.

---

### Stage 88 — Knowledge Flow: Textbook Promotion

**Идеологический долг:** Принцип 5 (Knowledge flow). Learned rules
умирают с runtime — не передаются следующему поколению явно.

**Что делаем:**

Реализовать стрелку `learned_rules → teacher YAML` из диаграммы в IDEOLOGY.md:

```python
# После N стабильных наблюдений:
if rule.confidence > threshold and rule.n_observations > 50:
    textbook.promote(rule)  # записать в crafter_textbook.yaml
    # Следующее поколение стартует с этим правилом как фактом
```

Тогда:
- Gen 1 discover'ит "zombie + proximity → health -0.5"
- Gen 2 стартует с этим как textbook fact — не тратит эпизоды на discovery
- Gen 10: textbook стабилизирован, каждое новое поколение начинает
  с накопленной мудростью всех предыдущих

**Gate:** gen5 `avg_len` > gen1 `avg_len` на ≥20%. Разница объясняется
promoted rules, не случайностью карт.

---

## Зависимости

```
Stage 83 (fix bugs)
    │
    ▼
Stage 84 (real stimuli)
    │
    ├──▶ Stage 85 (curiosity)
    │         │
    │         ▼
    │    Stage 87 (death curiosity)
    │
    └──▶ Stage 86 (post-mortem)
              │
              ▼
         Stage 88 (textbook promotion)
```

84 разблокирует 85 и 86 параллельно.
85 + 86 вместе разблокируют 87 и 88.

---

## Что НЕ в roadmap-е (сознательно)

- **Crafter-специфичные улучшения** (лучший pathfinding, больше entity types):
  это tactics, не ideology. Не расширяем textbook чтобы покрыть eval gap.

- **Смена env на более сложный** (Minecraft, NetHack): только после
  Stage 88 — когда knowledge flow доказан в Crafter.

- **Нейронные компоненты вместо SDM**: SDM не замена нейросети,
  это другая парадигма. Менять только если SDM доказуемо не справляется.

---

## Как читать прогресс

Каждый stage закрывается когда:
1. Eval gate пройден на minipc
2. `docs/ASSUMPTIONS.md` обновлён
3. Memory запись обновлена

Roadmap пересматривается после каждых 2-3 закрытых stage-ов.
