# Stage 69: Outcome-Supervised Near Labeling — закрываем circular dependency

**Дата:** 2026-04-06  
**Статус:** Design  
**Зависимости:** Stage 68 (CrafterSpatialMap, NearDetector, `checkpoints/stage67_encoder.pt`)

---

## Проблема

Все предыдущие этапы (66-68) обучали CNN near_head с near_labels из `_detect_near_from_info(info["semantic"])`.  
Это **circular dependency**: убираем символику — обучаясь на ней.

```
info["semantic"] → near_label → обучаем NearDetector → NearDetector заменяет info["semantic"]
          ↑_______________________________________________|
```

---

## Решение: Outcome-Supervised Labeling

Источник near_label — **результат действия** (что изменилось в инвентаре), не ground truth мира.

```
pixels + action "do" → inventory_before vs inventory_after
                             ↓
                 wood +1 → near_label = "tree"
                 stone +1 → near_label = "stone"
                 wood_pickaxe +1 → near_label = "table"
                 wood -2 (place_table) → near_label = "empty"
                        ↓
                 (pixels, near_label) → train NearDetector
```

Используется только проприоцепция:
- `info["inventory"]` — агент знает что у него есть
- `info["player_pos"]` — агент знает где он

`info["semantic"]` **не используется нигде**.

---

## Outcome → Near маппинг

```python
# inventory GAIN после "do" → near был этот объект
DO_GAIN_TO_NEAR = {
    "wood": "tree",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    # "drink"/"food" — не inventory items, skip
}

# inventory GAIN после make_* → near был "table"
MAKE_GAIN_TO_NEAR = {
    "wood_pickaxe": "table",
    "stone_pickaxe": "table",
    "iron_pickaxe": "table",
    "wood_sword": "table",
    "stone_sword": "table",
    "iron_sword": "table",
}

# inventory LOSS после place_* → near было "empty"
PLACE_LOSS_TO_NEAR = {
    "place_table": {"wood": 2},     # wood -2 → placed table near empty
    "place_furnace": {"stone": 4},
    "place_stone": {"stone": 1},
    "place_plant": {"sapling": 1},
}
```

**Покрытие:** 15/17 правил (water/cow пропускаются — дают buff, не inventory item).

---

## Архитектура

### До (Stage 66-68)
```
Phase 1 collection:
  pixels → info["semantic"] → near_label → NearDetector обучение
```

### После (Stage 69)
```
Phase 1 collection:
  pixels + action + inventory_before/after
      ↓
  inventory_diff → near_label (только для labelled frames)
      ↓
  NearDetector обучение на outcome-supervised labels
```

---

## Компоненты

### 1. `OutcomeLabeler` (`src/snks/agent/outcome_labeler.py`)

```python
class OutcomeLabeler:
    """Infers near_label from inventory changes — no info["semantic"] needed.

    Uses only info["inventory"] (proprioception) + action name.
    """

    def label(
        self, action: str, inv_before: dict[str, int], inv_after: dict[str, int]
    ) -> str | None:
        """Infer near_label from action outcome.

        Returns near_str or None if outcome is unrecognizable.
        """
```

### 2. `exp125_outcome_near.py` (новый эксперимент)

**Phase 0:** Загрузить Stage 68 encoder + NearDetector (для навигации к объектам).

**Phase 1: Outcome-supervised collection**
- Для каждого объекта из `DO_GAIN_TO_NEAR.keys()`:
  - Использовать `find_target_with_map()` (Stage 68) для навигации к объекту
  - Взять "do" action
  - Сравнить inventory до/после
  - Если inventory изменился → записать (pixels_before_do, near_label)
- Для make_* (table):
  - Навигация к "table"
  - Попробовать make_wood_pickaxe (при наличии wood в инвентаре)
  - Inventory gain → (pixels, "table")
- Для place_*:
  - Навигация к "empty" позиции
  - place_table (при наличии wood≥2)
  - Inventory loss → (pixels, "empty")

**Phase 2: Train new encoder** — тот же pipeline JEPA+SupCon, но `near_labels` из outcome, не из symbols.

**Phase 3: Smoke** — новый NearDetector vs ground truth near: ≥60% (ниже bar чем Stage 67, т.к. меньше labels).

**Phase 4: QA gate** — полный QA L1-L4: ≥85%.

**Phase 5: Regression** — exp123 pipeline ≥90%.

---

## Gate-критерий Stage 69

```
Phase 3 smoke:      новый NearDetector vs GT ≥60% (без символьных labels)
Phase 4 QA gate:    Crafter QA L1-L4 avg ≥85%
Phase 5 regression: exp123 gate ≥90%
```

---

## Что остаётся после Stage 69

```
Символьные данные в обучении: NONE
Символьные данные в агенте: NONE (только проприоцепция)
```

**Следующий рубеж:** make_* coverage — 0 прототипов для craft actions. Stage 70.
