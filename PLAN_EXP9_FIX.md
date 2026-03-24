# План исправления Exp 9: Curiosity-driven exploration

**Дата:** 2026-03-24
**Проблема:** coverage_ratio = 0.69 (gate > 1.5) — curiosity хуже random
**Корень:** «noisy TV» — повороты создают максимальную перцептуальную новизну при нулевом пространственном прогрессе

---

## Диагноз

1. `_perceptual_hash()` в `agent.py` — бинарная карта 8×8 ячеек (mean > 0.5). Поворот на 90° перетасовывает ~50-70% ячеек, forward — только ~10-25%
2. `select_action()` в `motivation.py` — формула `(0.6 × state_novelty + 0.4 × action_novelty) × uncertainty` всегда выбирает поворот
3. Context coarsening (16 bins) — разные позиции коллапсируют в один контекст
4. `info["agent_pos"]` отбрасывается — агент не знает (x,y)

---

## Шаг 1: Direction-invariant perceptual hash

**Файл:** `src/snks/agent/agent.py` → функция `_perceptual_hash()`

**Сейчас:**
```python
def _perceptual_hash(image, n_bins=8):
    for i in range(n_bins):
        for j in range(n_bins):
            cell = image[..., i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            if cell.mean() > 0.5:
                ids.add(offset + i * n_bins + j)  # позиционно-зависимый!
    return ids
```

**Исправление:** Заменить на rotation-invariant представление — отсортированная гистограмма интенсивностей:
```python
def _perceptual_hash(image, n_bins=8):
    # Собрать средние интенсивности всех ячеек
    intensities = []
    for i in range(n_bins):
        for j in range(n_bins):
            cell = image[..., i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            intensities.append(cell.mean().item())

    # Отсортировать → rotation-invariant
    intensities.sort()

    # Квантизовать в бины и создать pseudo-SKS IDs
    ids = set()
    offset = 10000
    for idx, val in enumerate(intensities):
        bin_id = int(val * 8)  # 8 уровней яркости
        ids.add(offset + idx * 8 + bin_id)
    return ids
```

**Эффект:** Одна и та же физическая позиция → одинаковый хеш независимо от ориентации агента. Повороты перестают создавать «новые» состояния.

---

## Шаг 2: Learning progress вместо raw prediction error

**Файл:** `src/snks/agent/motivation.py`

### 2a. Изменить `select_action()` — использовать learning progress

**Сейчас:**
```python
interest = (0.6 * state_novelty + 0.4 * action_novelty) * uncertainty
```

**Исправление:** Добавить learning progress как дополнительный сигнал:
```python
# Получить learning progress для этого (context, action)
lp = self._learning_progress.get((ctx_hash, a), 1.0)  # default = высокий (неизвестно)

# Заменить uncertainty на learning_progress
interest = (0.6 * state_novelty + 0.4 * action_novelty) * lp
```

### 2b. Изменить `update()` — трекать delta prediction error

**Сейчас:**
```python
self._prediction_errors[key] = (
    self._prediction_errors[key] * self.decay + prediction_error
)
```

**Исправление:**
```python
prev_error = self._prediction_errors.get(key, 1.0)  # first time = max
learning_progress = max(0.0, prev_error - prediction_error)

# EMA для сглаживания
self._learning_progress[key] = (
    self._learning_progress.get(key, 1.0) * self.decay
    + learning_progress * (1 - self.decay)
)

# Обновить текущую ошибку
self._prediction_errors[key] = prediction_error
```

**Эффект:** Повороты выучиваются за ~5 наблюдений → learning_progress → 0 → interest → 0. Forward в новые клетки — непредсказуемо → learning_progress > 0 → interest > 0.

---

## Шаг 3: Обновить тесты

**Файл:** `tests/test_motivation.py` (или где тесты на IntrinsicMotivation)

- Проверить что direction-invariant hash одинаков при повороте
- Проверить что learning_progress затухает для повторных действий
- Проверить что forward в новую клетку имеет больший interest чем turn

---

## Шаг 4: Запустить Exp 9

```bash
cd D:\Projects\AGI
python -m snks.experiments.exp9_curiosity
```

**Ожидаемый результат:** coverage_ratio > 1.5

---

## Шаг 5: Обновить stage6_data.json (real mode)

Если Exp 9 PASS — обновить визуализацию реальными данными.

---

## Научная база

- [Beyond Noisy-TVs: Learning Progress Monitoring](https://arxiv.org/abs/2509.25438) (2025)
- [Humans monitor learning progress in curiosity](https://www.nature.com/articles/s41467-021-26196-w) (2021)
- [Impact of intrinsic rewards on exploration](https://link.springer.com/article/10.1007/s00521-025-11340-0) (2025) — count-based лучше для grid worlds
- [Novelty is not Surprise](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009070) — теоретическое обоснование разницы

## Fallback

Если Шаги 1+2 не дают >1.5:
- **Rank 3:** Count-based на rotation-invariant hash (чистые визит-каунты, без prediction error)
- **Rank 4:** Go-Explore archive (использовать MentalSimulator.find_plan() для возврата к фронтиру)
