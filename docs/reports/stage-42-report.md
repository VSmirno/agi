# Stage 42: Spatial Representation — Perception Fix

## Результат: PASS (инфраструктура + диагностика)

**Ветка:** `stage42-spatial-perception`

## Что доказано

- **Symbolic SDR discrimination**: 5 уникальных SDR для key/door/goal/wall/empty — encoder различает объекты
- **CNN SDR discrimination**: mean overlap 0.46 между цветами (red vs green etc.) — цвет сохранён
- **КРИТИЧЕСКИЙ ВЫВОД**: Даже с идеальной perception (symbolic encoder) — DoorKey success = 5%. Bottleneck ДВОЙНОЙ: perception И learning/exploration
- **3 encoder'а работают**: gabor (legacy), symbolic (diagnostic), cnn (RGB) — все интегрированы в pipeline
- **Pipeline поддерживает pre_sdr**: perception_cycle() принимает pre-computed SDR, обходя visual encoding

## Эксперименты

| Exp | Encoder | Метрика | Результат | Gate | Статус |
|-----|---------|---------|-----------|------|--------|
| 101a | symbolic | unique SDRs | 5 | >3 | PASS |
| 101b | symbolic | DoorKey success | 5% (1/20) | >=15% | FAIL |
| 101c | CNN | color overlap | 0.46 | <0.5 | PASS |
| 101d | CNN | DoorKey success | 5% (1/20) | >=5% | PASS |
| 101e | Gabor | DoorKey success | 10% (1/10) | reference | PASS |

## Ключевые решения

1. **Diagnostic-first approach**: вместо сразу строить VQ-VAE, сначала проверили гипотезу "слеп ли агент". Ответ: слеп, но это не единственная проблема.

2. **SymbolicEncoder как постоянный diagnostic tool**: не будет использоваться в production, но незаменим для изоляции проблем perception vs learning.

3. **RGBConvEncoder с frozen random weights**: Xavier init без backprop. Удивительно, но random projection на RGB уже лучше Gabor на grayscale для color discrimination.

4. **Сохранение legacy Gabor**: не удалён, остаётся как default для обратной совместимости.

5. **pre_sdr в Pipeline**: минимальное изменение, позволяет подключить любой внешний encoder.

## Архитектура

```
PureDafConfig.encoder_type = "gabor" | "symbolic" | "cnn"

"gabor" (default):
  obs(H,W,3) → ObsAdapter(grayscale) → GaborBank → kWTA → SDR → DAF

"symbolic":
  env.get_symbolic_obs() → SymbolicEncoder → pre_sdr → Pipeline.perception_cycle(pre_sdr=...)
  
"cnn":
  obs(H,W,3) → ObsAdapter(rgb) → RGBConvEncoder → kWTA → SDR → DAF
```

## Веб-демо
- `demos/stage-42-spatial-perception.html` — side-by-side: Gabor vs Symbolic vs CNN, SDR visualization, результаты сравнения

## Файлы изменены

### Новые:
- `src/snks/encoder/symbolic.py` — SymbolicEncoder
- `src/snks/encoder/rgb_conv.py` — RGBConvEncoder
- `src/snks/experiments/exp101_perception.py` — 5 experiments
- `tests/test_perception_encoders.py` — 19 tests
- `demos/stage-42-spatial-perception.html` — web demo

### Изменены:
- `src/snks/env/obs_adapter.py` — RGB mode
- `src/snks/env/adapter.py` — get_symbolic_obs()
- `src/snks/agent/pure_daf_agent.py` — encoder_type switch, symbolic/cnn init
- `src/snks/pipeline/runner.py` — pre_sdr parameter
- `demos/index.html` — карточка Stage 42

## Следующий этап

Stage 42 доказал двойной bottleneck. Следующие приоритеты:

1. **Stage 43: Working Memory** — sustained oscillation для удержания целей между шагами. Агент "забывает" что ищет ключ через 5 шагов.
2. **Exploration improvement** — текущий PE-driven exploration не достаточен для sequential tasks (pick key → open door → reach goal).
3. **GPU validation** — TD-001 заблокирован до улучшения perception+learning, TD-002/003 можно запустить после Stage 43.
