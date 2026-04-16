# Roadmap SNKS AGI

**Версия:** 1
**Дата:** 2026-04-14
**Статус:** Living document. Обновлять при закрытии stage-а.

> Этот roadmap выводится из `IDEOLOGY.md`, не из wishlist-а.
> Каждый stage решает конкретный идеологический долг.
> Если stage не решает ничего из идеологии — он неправильный.

---

## Текущая позиция

**Stage 88 — CLOSED (2026-04-16, 1/2 gates)**

gen1=189.4, gen5=179.7, ratio=0.949. Secondary PASS (n_promoted=2 ✓), Primary FAIL.
Knowledge flow механически работает (гипотезы формируются и промоутируются).
Structural wall: vital threshold adjustments не улучшают zombie-боевую выживаемость.
Arrow attribution добавлен (exp136 + entity-specific ranges). Next: Stage 89.

**Stage 87 — COMPLETE (2026-04-15, PASS)**

avg_survival=186.85, n_verifiable=4, curiosity_active=17/20. Gates: 3/3.
DeathHypothesis + HypothesisTracker + CuriosityStimulus death-relevance weighting.

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

### Stage 85 — Goal Selector Design ✓ COMPLETE

**Результат (2026-04-15):** avg_survival=197.0, wood_ge3_pct=10%, no_total_gain=✓. Gates: 3/3 PASS.
**Что сделано:** GoalSelector (textbook-derived threats), Goal.progress() (vital_delta/inventory_delta/item_gained/explore), proactive crafting chain (wood chain_cost threshold), confidences в VectorTrajectory, total_gain убран из score_trajectory.

---

### Stage 85 — Curiosity as Primary Driver (АРХИВ — заменён Goal Selector)

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

### Stage 86 — Post-Mortem Learning ✓ COMPLETE

**Результат (2026-04-15):** avg_survival=179.7, zombie_deaths early=6→late=3, starvation with_pm=0 < without_pm=1. Gates: 3/3 PASS.
**Что сделано:** DamageEvent log, PostMortemAnalyzer (temporal decay + multi-source), PostMortemLearner (thresholds + health_weight), HomeostasisStimulus deficit-based scoring.

### Stage 86 — Post-Mortem Learning (АРХИВ)

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

### Stage 89 — Arrow Trajectory Modeling

**Идеологический долг:** Принцип 3 (World Model completeness). Стрела —
движущийся объект с предсказуемой траекторией, но VectorWorldModel не
моделирует её как динамическую сущность. ~25% смертей агента — от стрел
(диагностика diag_unknown_deaths, 2026-04-16).

**Ключевой инсайт:** Стрела летит по прямой 1 тайл/шаг, Player тоже
движется 1 тайл/шаг → dodge механически возможен. Это _уклоняемая угроза_
в отличие от zombie (гонится) и skeleton (стреляет издали).

**Что делаем:**

1. Расширить entity tracker: фиксировать позицию + направление arrow между
   шагами → добавить arrow как динамическую сущность с вектором движения.

2. Расширить VectorWorldModel / симулятор MPC: предсказывать позицию стрелы
   через N шагов по линейной экстраполяции.

3. MPC тогда сам обнаружит: "уйти перпендикулярно = стрела не попадёт" —
   dodge _эмерджирует_ из планирования, без hardcoded рефлексов.

**Gate:** доля смертей от arrow снижается ≥50% относительно baseline
(без arrow modeling). Avg survival растёт.

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
