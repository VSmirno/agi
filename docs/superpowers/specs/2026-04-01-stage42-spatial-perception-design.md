# Stage 42: Spatial Representation — Perception Fix

**Дата:** 2026-04-01
**Статус:** DESIGN
**Автор:** autonomous-dev

## Позиция в фазе

**Фаза 1: Живой DAF (~65% → 75%)**

Perception — первопричина провала exp97 (0/14 episodes reward на GPU). Агент буквально слеп: Gabor+grayscale не различает key/door/goal. Без фикса все дальнейшие этапы бессмысленны.

Маркеры, которые продвигает Stage 42:
- [ ] Агент различает объекты MiniGrid (key, door, goal) на уровне SDR
- [ ] Пространственная информация (где объект) сохранена в SDR
- [ ] Диагностика: если symbolic encoder даёт >30% — bottleneck был perception

## Проблема

Текущий perception pipeline:
```
MiniGrid obs (56×56 RGB via RGBImgPartialObsWrapper)
  → ObsAdapter: grayscale + resize to 64×64
  → GaborBank: 128 edge filters (4 scales × 8 orient × 4 phases)
  → AdaptiveAvgPool(4,8) → flatten(4096) → kWTA(k=164) → SDR
```

**3 фатальных проблемы:**
1. **Grayscale** — цвет уничтожен. Ключ (жёлтый), дверь (коричневая), цель (зелёная) неразличимы.
2. **Gabor** — детектор рёбер. Маленькие цветные квадраты MiniGrid ≠ края.
3. **AvgPool(4,8)** — позиция потеряна. "Ключ слева" = "ключ справа" для SDR.

## Brainstorming

### Подход A: Symbolic Encoder (ДИАГНОСТИКА) — ВЫБРАН для P0

Обойти пиксели: взять symbolic observation из MiniGrid (`env.unwrapped.gen_obs()` → 7×7×3 int: type, color, state) и закодировать напрямую в SDR.

- Каждая клетка (i,j) активирует блок SDR бит по формуле: `base = (i*7+j) * bits_per_cell`
- В блоке: one-hot object_type (11 типов) + one-hot color (6) + one-hot state (3)
- Итого: 49 клеток × 20 бит/клетку = 980 бит из ~4096 SDR (24% спарсность)
- **Идеальная информация**, ноль вычислений

**Pros:** мгновенная диагностика — если DoorKey >30%, проблема точно в encoder
**Cons:** не visual encoding, только для MiniGrid
**Trade-off:** не bio-plausible, но нужен как ДИАГНОСТИКА перед вложением в CNN

### Подход B: RGB Conv Encoder — ВЫБРАН для P1

3-слойная CNN на RGB input. Заменяет GaborBank.

```python
Conv2d(3, 32, 3, stride=2, padding=1)  # 64→32, видит цвет
Conv2d(32, 64, 3, stride=2, padding=1)  # 32→16
Conv2d(64, 128, 3, stride=2, padding=1) # 16→8
→ flatten(128*8*8=8192) → Linear(8192, sdr_size) → kWTA → SDR
```

- RGB вход (цвет сохранён)
- Spatial feature map 8×8 (позиция сохранена на уровне ~8 пикселей)
- Веса: Xavier init (не Gabor). Без backprop — фиксированные random проекции.
- **Random projection на RGB всё равно лучше Gabor на grayscale**, потому что сохраняет цветовой контраст.

**Pros:** visual encoder, работает для любых сред
**Cons:** random init может быть слабым; потенциально потребует pretraining
**Trade-off:** простота vs. quality → начинаем с random, если >20% — отлично

### Подход C: VQ-VAE Tokenizer — отложен до P2

Дискретные токены, natural SDR. Но требует pretraining. Отложен.

### Обоснование двухступенчатого подхода

P0 (Symbolic) отвечает на вопрос: "Может ли DAF вообще решить DoorKey при идеальной perception?"
- Если ДА → P1 (CNN) нужен, чтобы perception работала на пикселях
- Если НЕТ → проблема в DAF/STDP/planning, encoder не поможет

## Дизайн

### 1. SymbolicEncoder (src/snks/encoder/symbolic.py)

```python
class SymbolicEncoder:
    """Encode MiniGrid symbolic observation directly to SDR.
    
    Input: (7, 7, 3) int array from env.unwrapped.gen_obs()
           channel 0: object_type (0-10)
           channel 1: color (0-5)  
           channel 2: state (0-2)
    Output: (sdr_size,) binary SDR tensor
    """
    def __init__(self, sdr_size=4096, bits_per_cell=20):
        ...
    
    def encode(self, symbolic_obs: np.ndarray) -> Tensor:
        """Encode symbolic grid to SDR."""
        ...
    
    def sdr_to_currents(self, sdr, num_nodes, zone=None) -> Tensor:
        """Map SDR to DAF external currents (same API as VisualEncoder)."""
        ...
```

### 2. RGBConvEncoder (src/snks/encoder/rgb_conv.py)

```python
class RGBConvEncoder(nn.Module):
    """RGB CNN encoder — replaces Gabor+grayscale pipeline.
    
    Input: (3, H, W) RGB float32 in [0, 1]
    Output: (sdr_size,) binary SDR
    """
    def __init__(self, config: EncoderConfig):
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(128 * 8 * 8, sdr_size)
        # Freeze — no backprop, random projection
        for p in self.parameters():
            p.requires_grad_(False)
    
    def encode(self, images: Tensor) -> Tensor:
        """Encode RGB image to SDR."""
        ...
```

### 3. Изменения в ObsAdapter

Добавить режим RGB: `ObsAdapter(mode="rgb")` возвращает (3, H, W) tensor вместо (H, W) grayscale.

### 4. Изменения в MiniGridAdapter

Добавить метод `symbolic_obs()` или режим, возвращающий raw symbolic grid.
Альтернатива: `SymbolicMiniGridAdapter` — отдельный адаптер для diagnostic.

### 5. Изменения в PureDafConfig

```python
encoder_type: str = "gabor"  # "gabor" | "symbolic" | "cnn"
```

### 6. Изменения в PureDafAgent

В `__init__`: выбор encoder по config.encoder_type.
В `step()`/`observe_result()`: если symbolic — bypass ObsAdapter, использовать SymbolicEncoder напрямую.

### 7. Изменения в Pipeline

`perception_cycle()` уже принимает image tensor. Для CNN — принимает (3,H,W) RGB.
Для symbolic — новый метод `perception_cycle_from_sdr(sdr)` или передача pre-computed SDR.

## Gate-критерии

| Test | Метрика | Gate |
|------|---------|------|
| Symbolic SDR discrimination | unique SDRs for key/door/goal | > 3 distinct |
| Symbolic DoorKey-5x5 | success_rate (CPU, 2K nodes, 20 eps) | >= 0.15 |
| CNN SDR discrimination | overlap(key_sdr, door_sdr) | < 0.5 |
| CNN DoorKey-5x5 | success_rate (CPU, 2K nodes, 20 eps) | >= 0.05 |
| Gabor baseline (reference) | success_rate | measured (expect ~0%) |
| No regression | existing tests pass | True |

## Файлы

### Новые
- `src/snks/encoder/symbolic.py` — SymbolicEncoder
- `src/snks/encoder/rgb_conv.py` — RGBConvEncoder  
- `src/snks/experiments/exp101_perception.py` — experiments
- `tests/test_perception_encoders.py` — unit tests

### Изменяемые
- `src/snks/env/obs_adapter.py` — RGB mode
- `src/snks/env/adapter.py` — symbolic obs support
- `src/snks/agent/pure_daf_agent.py` — encoder_type switch
- `src/snks/pipeline/runner.py` — accept RGB / pre-computed SDR
- `src/snks/daf/types.py` — encoder_type in config
- `demos/index.html` — карточка Stage 42
