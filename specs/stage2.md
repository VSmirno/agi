# Этап 2: Визуальный кодировщик ✅

**Статус:** Завершён (2026-03-22)
**Срок:** Неделя 4–5

## Модули

| Модуль | Назначение |
|--------|-----------|
| `encoder/gabor.py` | GaborBank: 128 фильтров (8 ориентаций × 4 масштаба × 4 фазы), замороженный Conv2d |
| `encoder/sdr.py` | kwta(), sdr_overlap(), batch_overlap_matrix() |
| `encoder/encoder.py` | VisualEncoder: encode() + sdr_to_currents() |
| `data/shapes.py` | ShapeGenerator: 10 геометрических фигур |
| `data/stimuli.py` | GratingGenerator: 10 ориентаций с вариациями |

## Поток данных

```
image (64,64) float32 [0,1]
  → GaborBank: Conv2d(1,128,19,19,padding=9) + abs() → (128,64,64)
  → AdaptiveAvgPool2d((4,8)) → (128,4,8)
  → flatten → (4096,)
  → kwta(k=164) → (4096,) binary SDR
  → sdr_to_currents(N) → (N,8) external currents
```

## Параметры

**Габор-фильтры:**
- 4 масштаба: σ=(1,2,3,4), λ=(4,8,12,16)
- 8 ориентаций: θ ∈ [0, π)
- 4 фазы: offset ∈ [0, 2π)
- Ядра: pad до 19×19, zero-mean + L2 normalize
- Активация: torch.abs() (complex-cell V1)

**SDR:**
- k = round(4096 × 0.04) = 164 active bits
- Overlap: |A ∩ B| / k

**SDR→currents:**
- Hash-based mapping: каждый SDR-бит → группа ceil(N/4096) узлов
- currents[:, 0] = sdr[node_sdr_idx] × current_strength (канал v)

**EncoderConfig:**
```python
gabor_kernel_size: int = 19
sdr_current_strength: float = 1.0
pool_h: int = 4
pool_w: int = 8
```

## Gate

- SDR within-class overlap: 0.43 (порог > 0.3) ✅
- SDR between-class overlap: 0.074 (порог < 0.1) ✅
- Gate-тест использует GratingGenerator, не ShapeGenerator

## Тесты

65 тестов (gabor:13, sdr:14, encoder:14, shapes:12, stimuli:10+2 gate).

## Отклонения

- GratingGenerator для gate вместо ShapeGenerator (Gabor — ориентационные детекторы V1)
- ShapeGenerator сохранён для будущих экспериментов
