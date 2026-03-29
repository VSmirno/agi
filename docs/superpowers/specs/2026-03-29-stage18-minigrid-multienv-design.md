# Stage 18: Multi-Env Validation + Transfer Learning

**Дата:** 2026-03-29
**Статус:** Design Approved
**Предпосылка:** Stage 17 COMPLETE — MVP валидирован на N=50K (AMD ROCm, exp38/39/40 PASS)

---

## Цель

Валидация системы СНКС на всём спектре MiniGrid-окружений (кроме BabyAI) и исследование трёх сценариев переноса знаний:
- **A) Zero-shot transfer** — агент обучен на source env, запускается без дообучения в target env
- **B) Continual learning** — sequential A→B→C, проверка отсутствия catastrophic forgetting
- **C) Multitask** — один агент обучается на всех env одновременно (random switching)

---

## Окружения (15 штук)

| # | Env ID | Сложность | Что тестирует |
|---|--------|-----------|--------------|
| 1 | `MiniGrid-Empty-5x5-v0` | ★☆☆ | baseline навигация |
| 2 | `MiniGrid-Empty-8x8-v0` | ★☆☆ | масштаб пространства |
| 3 | `MiniGrid-FourRooms-v0` | ★★☆ | структурные переходы |
| 4 | `MiniGrid-DoorKey-5x5-v0` | ★★☆ | объект→действие |
| 5 | `MiniGrid-DoorKey-8x8-v0` | ★★★ | сложная цепочка |
| 6 | `MiniGrid-MultiRoom-N2-S4-v0` | ★★☆ | последовательные комнаты |
| 7 | `MiniGrid-MultiRoom-N4-S5-v0` | ★★★ | длинная цепочка |
| 8 | `MiniGrid-LavaCrossing-S9N1-v0` | ★★☆ | избегание опасности |
| 9 | `MiniGrid-LavaCrossing-S9N2-v0` | ★★★ | плотная лава |
| 10 | `MiniGrid-SimpleCrossing-S9N1-v0` | ★★☆ | обход стен |
| 11 | `MiniGrid-KeyCorridor-S3R1-v0` | ★★☆ | ключ в соседней комнате |
| 12 | `MiniGrid-Unlock-v0` | ★★☆ | только дверь+ключ |
| 13 | `MiniGrid-UnlockPickup-v0` | ★★★ | ключ→дверь→объект |
| 14 | `MiniGrid-MemoryS7-v0` | ★★★ | рабочая память |
| 15 | `MiniGrid-ObstructedMaze-1Dlhb-v0` | ★★★ | лабиринт с препятствиями |

---

## Конфигурация агента

```yaml
N: 50_000                    # осцилляторов (как exp38)
disable_csr: true            # ROCm fix (torch.sparse_csr_tensor медленный)
replay_mode: uniform         # biological sleep consolidation (exp39 key finding)
replay_buffer_size: 10_000   # увеличен с 5K
steps_per_cycle: 20          # scatter_add atomic fix (exp38)
checkpoint_every: 10_000     # для восстановления при падении
```

Шаги обучения по сложности:
- ★☆☆ (Empty-5x5, Empty-8x8): 30K шагов
- ★★☆ (FourRooms, DoorKey-5x5, MultiRoom-N2, LavaCrossing-S9N1, SimpleCrossing, KeyCorridor, Unlock): 50K шагов
- ★★★ (DoorKey-8x8, MultiRoom-N4, LavaCrossing-S9N2, UnlockPickup, MemoryS7, ObstructedMaze): 80K шагов

**Итого:** ~850K шагов. При 16 steps/sec — ~15 часов на AMD minipc.

---

## Эксперименты

### exp41 — `multi_env_baseline.py`

**Цель:** обучить агента на каждом из 15 env независимо.

**Процесс:**
1. Для каждого env инициализировать EmbodiedAgent (N=50K, fresh weights)
2. Обучать заданное число шагов с checkpoint каждые 10K
3. Записывать метрики на каждом шаге

**Метрики на env:**
- `coverage_ratio` — доля уникальных позиций / total_cells
- `goal_seeking_steps` — шаги в режиме GOAL_SEEKING
- `mean_nmi` — средняя NMI за последние 1K шагов
- `steps_per_sec` — производительность

**Выходные файлы:**
- `checkpoints/exp41/{env_id}/final.safetensors` — веса агента
- `checkpoints/exp41/{env_id}/step_{N}.safetensors` — промежуточные
- `results/exp41_baseline.json` — все метрики
- `reports/exp41_baseline.html` — HTML-отчёт с кривыми

**Критерий PASS:** coverage_ratio > 0.3 для всех ★☆☆ и ★★☆ env за отведённые шаги.

---

### exp42 — `transfer_matrix.py`

**Цель:** zero-shot transfer — загрузить веса source env, запустить в target env без дообучения.

**Процесс:**
1. Загрузить `checkpoints/exp41/{source_env}/final.safetensors`
2. Запустить агента в target_env на 10K шагов (без обновления весов STDP)
3. Параллельно запустить random baseline (необученный агент) в том же target_env
4. Повторить для всех пар (15×15 = 225 запусков, исключая диагональ)

**Метрики на пару (source→target):**
- `adaptation_auc` — площадь под кривой coverage за 10K шагов
- `baseline_auc` — то же для random агента
- `transfer_score = adaptation_auc / baseline_auc` — >1.0 означает позитивный трансфер
- `final_coverage_ratio` — coverage в конце 10K шагов

**Выходные файлы:**
- `results/exp42_transfer.json` — матрица 15×15
- `reports/exp42_transfer.html` — тепловая карта transfer_score + топ-10 лучших пар

**Критерий PASS:** среднее transfer_score > 1.0 (хотя бы базовый положительный трансфер).

---

### exp43 — `continual_multitask.py`

Два под-эксперимента в одном файле.

#### 43a — Sequential Continual Learning

**Цель:** проверить, что обучение на новых env не разрушает знания о предыдущих.

**Процесс:**
- 5 цепочек по 3 env (от простых к сложным):
  - Chain 1: Empty-5x5 → FourRooms → DoorKey-5x5
  - Chain 2: Empty-8x8 → MultiRoom-N2 → MultiRoom-N4
  - Chain 3: SimpleCrossing → LavaCrossing-S9N1 → LavaCrossing-S9N2
  - Chain 4: Unlock → DoorKey-5x5 → UnlockPickup
  - Chain 5: KeyCorridor → DoorKey-8x8 → ObstructedMaze
- После каждого перехода A→B и B→C: замерить NMI/coverage на предыдущих env

**Метрика:** `retention_ratio = metric_after / metric_before >= 0.8` (паттерн из exp34)

**Критерий PASS:** retention_ratio >= 0.8 для >= 80% переходов.

#### 43b — Multitask Training

**Цель:** один агент обучается на всех 15 env, каждый шаг — случайный env.

**Процесс:**
- Инициализировать один EmbodiedAgent (N=50K)
- 200K шагов суммарно, равномерный рандомный выбор env
- Каждые 10K шагов: оценочный прогон на каждом env (1K шагов, без обучения)

**Метрика:** `multitask_coverage[env] / exp41_baseline_coverage[env]` — насколько мультизадачный агент хуже/лучше специализированного

**Критерий PASS:** среднее отношение >= 0.5 (мультизадачный агент достигает хотя бы 50% от специализированного).

**Выходные файлы:**
- `results/exp43_continual.json`, `results/exp43_multitask.json`
- `reports/exp43_continual.html`, `reports/exp43_multitask.html`

---

## Визуализация

### Ночью на minipc: HTML-отчёты

Каждый experiment генерирует `reports/expNN_*.html` с:
- Кривыми coverage/goal_seeking по шагам (plotly)
- Финальными метриками в таблице
- Тепловой картой для exp42

### Утром на локальном RTX 3070: live inference viewer

**Файл:** `src/snks/viz/inference_viewer.py`
**Сервер:** FastAPI на `localhost:8765`
**UI:** обновлённый `static/inference.html`

**Функционал:**
- Выбор checkpoint (dropdown: env + шаг)
- Live рендер MiniGrid в canvas (агент двигается в реальном времени)
- График coverage в реальном времени (WebSocket обновление)
- Индикатор состояния Configurator FSM (EXPLORE / GOAL_SEEKING / NEUTRAL)
- Heatmap активных SKS кластеров
- Кнопки: Play / Pause / Step / Reset

**Запуск инференса на CUDA (локально):**
```bash
python -m snks.viz.inference_viewer --checkpoint checkpoints/exp41/MiniGrid-FourRooms-v0/final.safetensors --device cuda
```

---

## Структура файлов Stage 18

```
src/snks/experiments/
├── exp41_multi_env_baseline.py
├── exp42_transfer_matrix.py
└── exp43_continual_multitask.py

src/snks/viz/
└── inference_viewer.py          # новый файл

src/snks/static/
└── inference.html               # новый файл

checkpoints/
└── exp41/
    └── {env_id}/
        ├── step_10000.safetensors
        ├── step_20000.safetensors
        └── final.safetensors

results/
├── exp41_baseline.json
├── exp42_transfer.json
├── exp43_continual.json
└── exp43_multitask.json

reports/
├── exp41_baseline.html
├── exp42_transfer.html
├── exp43_continual.html
└── exp43_multitask.html
```

---

## Ночной план запуска (на minipc)

```bash
# На minipc (ssh gem@10.253.0.179 -p 2244)
cd /opt/agi && git pull
HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/pytest src/snks/experiments/exp41_multi_env_baseline.py -s -v   # ~8h
HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/pytest src/snks/experiments/exp42_transfer_matrix.py -s -v      # ~4h
HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/pytest src/snks/experiments/exp43_continual_multitask.py -s -v  # ~3h
```

После завершения: `git add results/ reports/ checkpoints/ && git commit -m "Stage 18 results" && git push`

---

## Критерии успеха Stage 18

| Exp | Критерий PASS |
|-----|--------------|
| exp41 | coverage_ratio > 0.3 для всех ★☆☆ и ★★☆ env |
| exp42 | mean transfer_score > 1.0 хотя бы для 20% пар |
| exp43a | retention_ratio >= 0.8 для >= 80% переходов |
| exp43b | mean multitask_coverage >= 0.5 × baseline |
