# Stage 18: Multi-Env Validation + Transfer Learning

**Дата:** 2026-03-29
**Статус:** Design Approved (v2 — после spec review)
**Предпосылка:** Stage 17 COMPLETE — MVP валидирован на N=50K (AMD ROCm, exp38/39/40 PASS)

---

## Цель

Валидация системы СНКС на всём спектре MiniGrid-окружений (кроме BabyAI) и исследование трёх сценариев переноса знаний:
- **A) Zero-shot transfer** — агент обучен на source env, запускается без дообучения в target env
- **B) Continual learning** — sequential A→B→C, проверка отсутствия catastrophic forgetting
- **C) Multitask** — один агент обучается на всех env одновременно (random switching)

---

## Окружения (15 штук)

Env ID верифицированы по gymnasium registry (без лишних дефисов в названиях типа LavaCrossing).

| # | Env ID | Сложность | Что тестирует |
|---|--------|-----------|--------------|
| 1 | `MiniGrid-Empty-5x5-v0` | ★☆☆ | baseline навигация |
| 2 | `MiniGrid-Empty-8x8-v0` | ★☆☆ | масштаб пространства |
| 3 | `MiniGrid-FourRooms-v0` | ★★☆ | структурные переходы |
| 4 | `MiniGrid-DoorKey-5x5-v0` | ★★☆ | объект→действие |
| 5 | `MiniGrid-DoorKey-8x8-v0` | ★★★ | сложная цепочка |
| 6 | `MiniGrid-MultiRoom-N2-S4-v0` | ★★☆ | последовательные комнаты |
| 7 | `MiniGrid-MultiRoom-N4-S5-v0` | ★★★ | длинная цепочка |
| 8 | `MiniGrid-LavaCrossingS9N1-v0` | ★★☆ | избегание опасности |
| 9 | `MiniGrid-LavaCrossingS9N2-v0` | ★★★ | плотная лава |
| 10 | `MiniGrid-SimpleCrossingS9N1-v0` | ★★☆ | обход стен |
| 11 | `MiniGrid-KeyCorridorS3R1-v0` | ★★☆ | ключ в соседней комнате |
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
- ★★☆ (FourRooms, DoorKey-5x5, MultiRoom-N2, LavaCrossingS9N1, SimpleCrossing, KeyCorridor, Unlock): 50K шагов
- ★★★ (DoorKey-8x8, MultiRoom-N4, LavaCrossingS9N2, UnlockPickup, MemoryS7, ObstructedMaze): 80K шагов

**Итого:** ~890K шагов. При 5 параллельных воркерах × 16 steps/sec — **~3.1 часа** для exp41.

---

## Параллелизм

AMD minipc имеет 96GB unified VRAM. Каждый агент N=50K занимает ~150MB. Можно запускать **5 агентов одновременно** через `multiprocessing.Pool`.

Стратегия:
- exp41: 5 параллельных воркеров, каждый обучает 3 env последовательно (15 env / 5 = 3 env/worker)
- exp42: 5 параллельных воркеров по парам трансфера
- exp43a: 5 цепочек параллельно (по одной на воркер)
- exp43b: один агент (нельзя параллелить — единое состояние)

**Итоговое время:** exp41 ~3.1h + exp42 ~0.5h + exp43a ~1.5h + exp43b ~3.5h = **~8.5 часов**. Влезает в ночь.

---

## Сериализация агента (новое требование)

Существующий `persistence.py` сохраняет только DcamWorldModel. Для exp42 (transfer learning) нужно восстановить **полное состояние агента**, включая обученные синаптические веса DAF.

В exp41 каждый checkpoint сохраняет два артефакта:
1. `{path}.safetensors` + `{path}.json` — DcamWorldModel через существующий persistence API
2. `{path}_daf.safetensors` — DAF веса: `{"edge_attr": tensor, "edge_index": tensor, "node_state": tensor}`

В exp42 загружаются оба артефакта для восстановления полного состояния агента.

---

## Эксперименты

### exp41 — `multi_env_baseline.py`

**Цель:** обучить агента на каждом из 15 env независимо, сохранить checkpoint для exp42.

**Процесс:**
1. Разделить 15 env на 5 групп по 3 env (от простой к сложной в каждой группе)
2. Запустить 5 параллельных воркеров через `multiprocessing.Pool`
3. Каждый воркер: инициализирует EmbodiedAgent (N=50K), обучает 3 env последовательно
4. Checkpoint каждые 10K шагов: сохранить DcamWorldModel + DAF веса

**Метрики на env:**
- `coverage_ratio` — доля уникальных позиций / total_cells
- `goal_seeking_steps` — шаги в режиме GOAL_SEEKING
- `steps_per_sec` — производительность

**Выходные файлы:**
- `checkpoints/exp41/{env_id}/final.safetensors` + `final.json` — DcamWorldModel
- `checkpoints/exp41/{env_id}/final_daf.safetensors` — DAF веса
- `checkpoints/exp41/{env_id}/step_{N}.safetensors` + `step_{N}.json` — промежуточные
- `results/exp41_baseline.json` — все метрики
- `reports/exp41_baseline.html` — HTML-отчёт с кривыми (plotly)

**Критерий PASS:**
- coverage_ratio > 0.3 для всех ★☆☆ и ★★☆ env
- ★★★ env: информационный (порогового критерия нет — они заведомо сложнее)

---

### exp42 — `transfer_matrix.py`

**Цель:** zero-shot transfer — загрузить состояние агента из exp41, запустить в target env без дообучения.

**Объём:** 30 кураторных пар (не 210) — 6 source envs × 5 target envs. Source envs выбраны как представители каждого класса сложности и типа задачи:

| Source | Targets |
|--------|---------|
| Empty-5x5 | FourRooms, DoorKey-5x5, LavaCrossingS9N1, MultiRoom-N2, SimpleCrossing |
| FourRooms | Empty-8x8, DoorKey-5x5, MultiRoom-N2, LavaCrossingS9N1, KeyCorridorS3R1 |
| DoorKey-5x5 | DoorKey-8x8, Unlock, UnlockPickup, KeyCorridorS3R1, FourRooms |
| LavaCrossingS9N1 | LavaCrossingS9N2, SimpleCrossing, Empty-5x5, FourRooms, ObstructedMaze |
| MultiRoom-N2 | MultiRoom-N4, FourRooms, DoorKey-5x5, KeyCorridorS3R1, MemoryS7 |
| KeyCorridorS3R1 | Unlock, DoorKey-5x5, UnlockPickup, DoorKey-8x8, MultiRoom-N2 |

**Процесс:**
1. Предварительно вычислить `baseline_auc` для каждого target env (random агент, 2K шагов, однократно)
2. Для каждой из 30 пар: загрузить DcamWorldModel + DAF веса source, запустить в target 2K шагов без STDP-обновления
3. Запустить 5 параллельных воркеров по 6 пар каждый

**Метрики на пару (source→target):**
- `adaptation_auc` — площадь под кривой coverage за 2K шагов
- `baseline_auc` — то же для random агента (предвычислено)
- `transfer_score = adaptation_auc / baseline_auc` — >1.0 = позитивный трансфер
- `final_coverage_ratio` — coverage в конце 2K шагов

**Выходные файлы:**
- `results/exp42_transfer.json` — матрица 6×5
- `reports/exp42_transfer.html` — тепловая карта transfer_score + топ-10 лучших пар

**Критерий PASS:** transfer_score > 1.0 для >= 20% пар (6 из 30).

---

### exp43 — `continual_multitask.py`

Два под-эксперимента в одном файле.

#### 43a — Sequential Continual Learning

**Цель:** проверить, что обучение на новых env не разрушает знания о предыдущих.

**Процесс:**
- 5 цепочек по 3 env (от простых к сложным), запускаются параллельно:
  - Chain 1: Empty-5x5 → FourRooms → DoorKey-5x5
  - Chain 2: Empty-8x8 → MultiRoom-N2 → MultiRoom-N4
  - Chain 3: SimpleCrossingS9N1 → LavaCrossingS9N1 → LavaCrossingS9N2
  - Chain 4: Unlock → DoorKey-5x5 → UnlockPickup
  - Chain 5: KeyCorridorS3R1 → DoorKey-8x8 → ObstructedMaze
- После каждого перехода A→B и B→C: замерить coverage на предыдущих env (1K шагов, без обучения)

**Метрика:** `retention_ratio = coverage_after / coverage_before >= 0.8` (паттерн из exp34)

**Критерий PASS:** retention_ratio >= 0.8 для >= 80% переходов.

#### 43b — Multitask Training

**Цель:** один агент обучается на всех 15 env, каждый шаг — случайный env.

**Процесс:**
- Инициализировать один EmbodiedAgent (N=50K)
- 200K шагов суммарно, равномерный рандомный выбор env на каждый шаг
- Каждые 10K шагов: оценочный прогон на каждом env (500 шагов, без STDP-обновления)

**Метрика:** `multitask_coverage[env] / exp41_baseline_coverage[env]` — ratio к специализированному

**Критерий PASS:** mean ratio >= 0.5 по всем 15 env.

**Выходные файлы:**
- `results/exp43_continual.json`, `results/exp43_multitask.json`
- `reports/exp43_continual.html`, `reports/exp43_multitask.html`

---

## Визуализация

### Ночью на minipc: HTML-отчёты

Каждый experiment генерирует `reports/expNN_*.html` с:
- Кривыми coverage/goal_seeking по шагам (plotly)
- Финальными метриками в таблице
- Тепловой картой для exp42 (transfer_score matrix)
- Retention bars для exp43a

### Утром на локальном RTX 3070: live inference viewer

**Файл:** `src/snks/viz/inference_viewer.py`
**Сервер:** FastAPI на `localhost:8765`
**UI:** `src/snks/viz/static/inference.html`

**Функционал:**
- Выбор checkpoint (dropdown: env + шаг)
- Live рендер MiniGrid в canvas (агент двигается в реальном времени)
- График coverage в реальном времени (WebSocket обновление)
- Индикатор состояния Configurator FSM (EXPLORE / GOAL_SEEKING / NEUTRAL)
- Heatmap активных SKS кластеров
- Кнопки: Play / Pause / Step / Reset

**Запуск инференса на CUDA (локально):**
```bash
python -m snks.viz.inference_viewer \
  --checkpoint checkpoints/exp41/MiniGrid-FourRooms-v0/final.safetensors \
  --device cuda
```

---

## Структура файлов Stage 18

```
src/snks/experiments/
├── exp41_multi_env_baseline.py
├── exp42_transfer_matrix.py
└── exp43_continual_multitask.py

src/snks/viz/
├── inference_viewer.py          # новый файл
└── static/
    └── inference.html           # новый файл

checkpoints/
└── exp41/
    └── {env_id}/
        ├── step_10000.safetensors + step_10000.json
        ├── step_20000.safetensors + step_20000.json
        ├── final.safetensors + final.json      # DcamWorldModel
        └── final_daf.safetensors               # DAF edge_attr + edge_index + node_state

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
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Запускаем все три последовательно через nohup
nohup bash -c "
  venv/bin/pytest src/snks/experiments/exp41_multi_env_baseline.py -s -v &&
  venv/bin/pytest src/snks/experiments/exp42_transfer_matrix.py -s -v &&
  venv/bin/pytest src/snks/experiments/exp43_continual_multitask.py -s -v &&
  git add results/ reports/ checkpoints/ &&
  git commit -m 'Stage 18 results: exp41/42/43' &&
  git push
" > /tmp/stage18.log 2>&1 &

echo "PID: $!"
tail -f /tmp/stage18.log
```

Оценка времени:
- exp41: ~3.1h (5 воркеров × 890K/5 шагов / 16 steps/sec)
- exp42: ~0.5h (30 пар × 2K шагов / 5 воркеров / 16 steps/sec)
- exp43a: ~1.5h (5 цепочек параллельно × 150K шагов / 16 steps/sec)
- exp43b: ~3.5h (200K шагов, 1 агент)
- **Итого: ~8.5 часов**

---

## Критерии успеха Stage 18

| Exp | Критерий PASS |
|-----|--------------|
| exp41 | coverage_ratio > 0.3 для всех ★☆☆ и ★★☆ env (★★★ — информационный) |
| exp42 | transfer_score > 1.0 для >= 20% пар (6 из 30) |
| exp43a | retention_ratio >= 0.8 для >= 80% переходов (8 из 10) |
| exp43b | mean multitask_coverage >= 0.5 × baseline |
