# Stage 67: Pixel Agent — убираем символьный near

**Дата:** 2026-04-06  
**Статус:** Design approved  
**Зависимости:** Stage 66 (CNN encoder + PrototypeMemory, чекпоинт `checkpoints/stage66_encoder.pt`)

---

## Цель

Убрать символьный near из наблюдений агента. Сейчас `CrafterPixelEnv._to_symbolic()` читает `info["semantic"]` и конвертирует в строку (`"tree"`, `"stone"`, ...). После Stage 67 `near` приходит из CNN encoder через `NearDetector`. Inventory остаётся проприоцептивным (агент помнит сам через `info["inventory"]`). Навигация — маркер для Stage 68, не трогаем.

---

## Скоуп

| Что меняем | Что не трогаем |
|---|---|
| `CrafterPixelEnv` — убираем `_to_symbolic()` | `CLSWorldModel` |
| Новый `NearDetector` | `PrototypeMemory` |
| Агент — новый `_build_situation()` | `SubgoalPlanning` / BFS |
| Новый `exp123_pixel_agent.py` | `exp122_pixels.py` (регрессия) |

**Маркер Stage 68:** `info["inventory"]` → visual inventory head (CNN).

---

## Архитектура

### До (Stage 66)
```
env.step() → pixels + _to_symbolic(info["semantic"]) → situation dict → CLSWorldModel.query()
```

### После (Stage 67)
```
env.step() → pixels + info
                ↓
         NearDetector.detect(pixels) → near_str
                ↓
         _build_situation(near_str, info["inventory"]) → situation dict → CLSWorldModel.query()
```

---

## Компоненты

### 1. `CrafterPixelEnv` (изменения в `src/snks/agent/crafter_pixel_env.py`)

**Новые сигнатуры:**
```python
def reset(self) -> tuple[np.ndarray, dict]:
    """Returns (pixels (3,64,64) float32 [0,1], native Crafter info dict).
    
    Noop step сохраняется — Crafter не возвращает info на reset(),
    поэтому один шаг noop по-прежнему нужен для получения info["inventory"].
    """

def step(self, action: int | str) -> tuple[np.ndarray, float, bool, dict]:
    """Returns (pixels, reward, done, info).
    
    Было: (pixels, symbolic_obs: dict[str, str], reward, done)
    Стало: (pixels, reward, done, info: dict)  — нативный Crafter info dict.
    info["inventory"] всегда присутствует (dict[str, int]).
    """

def observe(self) -> tuple[np.ndarray, dict]:
    """Returns (pixels, info) without stepping."""
```

**Удаляем:** `_to_symbolic()`, `_detect_nearby()`.  
**Оставляем:** `INVENTORY_ITEMS` — используется также в `decode_head.py`, не трогаем.

**Сохраняем:** `_to_pixels()`, `SEMANTIC_NAMES`, `NEAR_OBJECTS`, `ACTION_NAMES`, `ACTION_TO_IDX`.

**Комментарий-маркер в `step()`:**
```python
# Stage 67: near detection moved to NearDetector (CNN-based)
# Stage 68 TODO: replace info["inventory"] with visual inventory head
```

`info["inventory"]` нативно присутствует в Crafter info dict — дополнительных изменений не требуется.

---

### 2. `NearDetector` (новый файл `src/snks/encoder/near_detector.py`)

```python
from snks.agent.decode_head import NEAR_CLASSES  # ["empty", "tree", "stone", ...]

class NearDetector:
    def __init__(self, encoder: CNNEncoder):
        """
        Args:
            encoder: обученный CNNEncoder (Stage 66 checkpoint).
        """
        assert encoder.near_head[-1].out_features == len(NEAR_CLASSES), (
            f"Encoder n_near_classes={encoder.near_head[-1].out_features} "
            f"!= len(NEAR_CLASSES)={len(NEAR_CLASSES)}"
        )
        self._encoder = encoder

    def detect(self, pixels: torch.Tensor) -> str:
        """Определить ближайший объект из пикселей.

        Args:
            pixels: (3, 64, 64) float32 [0, 1].

        Returns:
            near_str из NEAR_CLASSES, или "empty" если idx вне диапазона.
        """
        with torch.no_grad():
            out = self._encoder(pixels)  # encoder сам обрабатывает single input
        # single input → near_logits shape: (n_classes,)
        idx = int(out.near_logits.argmax().item())
        if idx < len(NEAR_CLASSES):
            return NEAR_CLASSES[idx]
        return "empty"
```

**`NEAR_CLASSES`:** уже определён в `decode_head.py` как `["empty"] + NEAR_OBJECTS` (12 классов). Никакого JSON-файла не нужно — это константа.

**Загрузка:** `NearDetector` принимает уже загруженный encoder — не загружает сам (разделение ответственности).

---

### 3. `_build_situation()` (добавляется в агента / хелпер)

```python
def _build_situation(self, pixels: torch.Tensor, info: dict) -> dict[str, str]:
    """Построить situation dict из пикселей + info.

    Args:
        pixels: (3, 64, 64) float32 [0, 1].
        info: info dict из CrafterPixelEnv.step().

    Returns:
        situation dict совместимый с CLSWorldModel.query().
    """
    near_str = self._near_detector.detect(pixels)
    situation: dict[str, str] = {"domain": "crafter", "near": near_str}
    for item, count in info.get("inventory", {}).items():
        if count > 0:
            situation[f"has_{item}"] = str(count)
    return situation
```

Дальше `situation` идёт в `cls.query(situation, action)` без изменений.

---

### 4. `exp123_pixel_agent.py` (новый эксперимент)

**Фазы:**

**Phase 0: Load** — загрузить чекпоинт encoder Stage 66, построить `idx_to_near` из `label_to_idx` (сохранённого из exp122), создать `NearDetector`.

**Phase 1: Smoke test** — сравнить `NearDetector.detect()` с ground truth `_to_symbolic()["near"]` на 500 кадрах из случайных траекторий. Ожидаем ≥ 70% совпадений (smoke, не gate).

**Phase 2: QA gate** — тот же Crafter QA L1-L4 что в exp122, но `near` берётся из CNN, не из символики.

```
Gate: ≥ 90% accuracy (L1-L4 avg)
```

**Phase 3: Regression** — вызвать exp122 gate-тест и убедиться что ≥ 90% (Stage 66 достиг 100%, регрессионный порог = 90%).

---

## Тестирование

### Unit (`tests/test_near_detector.py`)
- `detect()` возвращает строку из `NEAR_OBJECTS + ["empty"]`
- На random-пиксельном тензоре не падает, возвращает строку
- `idx` вне `idx_to_near` → `"empty"` (graceful fallback)

### Integration (`tests/test_crafter_pixel_env_67.py`)
- `step()` возвращает `(np.ndarray, float, bool, dict)` — правильные типы и шейп `(3, 64, 64)`
- `info["inventory"]` присутствует и является `dict`
- Атрибуты `_to_symbolic` и `_detect_nearby` не существуют

### End-to-end (`exp123_pixel_agent.py`)
- Phase 1 smoke: ≥ 70% совпадение CNN near vs ground truth
- Phase 2 gate: ≥ 90% QA accuracy
- Phase 3 regression: exp122 ≥ 90%

---

## Файлы

| Действие | Файл |
|---|---|
| Изменить | `src/snks/agent/crafter_pixel_env.py` |
| Создать | `src/snks/encoder/near_detector.py` |
| Создать | `experiments/exp123_pixel_agent.py` |
| Создать | `tests/test_near_detector.py` |
| Создать | `tests/test_crafter_pixel_env_67.py` |
| Сохранить | `checkpoints/stage67_idx_to_near.json` (в ходе exp123 Phase 0) |

---

## Gate-критерий Stage 67

```
Phase 1 smoke:  CNN near accuracy vs ground truth ≥ 70%
Phase 2 gate:   Crafter QA L1-L4 avg ≥ 90%
Phase 3 reg:    exp122 gate ≥ 90% (Stage 66 достиг 100%, порог регрессии)
```

При прохождении всех трёх: Stage 67 COMPLETE.

---

## Маркеры будущей работы

```python
# Stage 68 TODO: replace info["inventory"] with visual inventory head (CNN)
# Stage 68 TODO: replace symbolic navigation map with pixel-based spatial memory
```
