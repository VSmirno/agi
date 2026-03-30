# Stage 18: Implementation Plan

**Спек:** `docs/superpowers/specs/2026-03-29-stage18-minigrid-multienv-design.md`
**Дата:** 2026-03-29
**Цель:** Multi-Env Validation + Transfer Learning (exp41/42/43)

---

## Фазы реализации

### Фаза 1 — DAF сериализация (новый код, нужен перед exp41/42)

**Файл:** `src/snks/daf/engine.py` — добавить два метода

```python
def save_state(self, path: str) -> None:
    """Сохранить обученные веса DAF в {path}_daf.safetensors"""
    tensors = {
        "states": self.states.cpu(),
        "edge_attr": self.graph.edge_attr.cpu(),
        "edge_index": self.graph.edge_index.cpu(),
        "step_count": torch.tensor(self.step_count),
    }
    save_file(tensors, path + "_daf.safetensors")

def load_state(self, path: str) -> None:
    """Восстановить веса DAF из {path}_daf.safetensors"""
    tensors = load_file(path + "_daf.safetensors", device=str(self.device))
    self.states = tensors["states"].to(self.device)
    self.graph.edge_attr = tensors["edge_attr"].to(self.device)
    self.graph.edge_index = tensors["edge_index"].to(self.device)
    self.step_count = int(tensors["step_count"].item())
```

**Также нужны методы на уровне EmbodiedAgent:**

**Файл:** `src/snks/agent/embodied_agent.py` — добавить:

```python
def save_checkpoint(self, base_path: str) -> None:
    """Сохранить полное состояние агента (DcamWorldModel + DAF)."""
    if self.dcam is not None:
        persistence.save(self.dcam, base_path)
    daf = self.causal_agent.pipeline.daf
    daf.save_state(base_path)

def load_checkpoint(self, base_path: str) -> None:
    """Восстановить полное состояние агента (DcamWorldModel + DAF)."""
    if self.dcam is not None:
        persistence.load(self.dcam, base_path)
    daf = self.causal_agent.pipeline.daf
    daf.load_state(base_path)
```

**Тест:** `tests/test_daf_serialization.py`
- Инициализировать DafEngine(N=100), прогнать 50 шагов, сохранить, загрузить в новый DafEngine, сравнить states и edge_attr

---

### Фаза 2 — Вспомогательный модуль `src/snks/experiments/stage18_utils.py`

Общий код для exp41/42/43:

```python
ENVS = [
    ("MiniGrid-Empty-5x5-v0",           "easy",   30_000),
    ("MiniGrid-Empty-8x8-v0",           "easy",   30_000),
    ("MiniGrid-FourRooms-v0",           "medium", 50_000),
    ("MiniGrid-DoorKey-5x5-v0",         "medium", 50_000),
    ("MiniGrid-DoorKey-8x8-v0",         "hard",   80_000),
    ("MiniGrid-MultiRoom-N2-S4-v0",     "medium", 50_000),
    ("MiniGrid-MultiRoom-N4-S5-v0",     "hard",   80_000),
    ("MiniGrid-LavaCrossingS9N1-v0",    "medium", 50_000),
    ("MiniGrid-LavaCrossingS9N2-v0",    "hard",   80_000),
    ("MiniGrid-SimpleCrossingS9N1-v0",  "medium", 50_000),
    ("MiniGrid-KeyCorridorS3R1-v0",     "medium", 50_000),
    ("MiniGrid-Unlock-v0",              "medium", 50_000),
    ("MiniGrid-UnlockPickup-v0",        "hard",   80_000),
    ("MiniGrid-MemoryS7-v0",            "hard",   80_000),
    ("MiniGrid-ObstructedMaze-1Dlhb-v0","hard",   80_000),
]

def build_config(device: str, N: int = 50_000) -> EmbodiedAgentConfig:
    """N=50K, disable_csr=True, replay_mode=uniform, replay_buffer=10K"""
    ...

def make_env(env_id: str):
    """gymnasium.make + RGBImgObsWrapper + resize to 64×64 grayscale"""
    ...

def img(obs) -> np.ndarray:
    """Extract image from MiniGrid obs dict"""
    ...

def coverage_ratio(visited: set, env) -> float:
    """Доля посещённых клеток / total walkable cells"""
    ...

def checkpoint_path(exp: str, env_id: str, step: int | str) -> str:
    """checkpoints/{exp}/{env_id}/step_{step} или final"""
    ...
```

---

### Фаза 3 — `exp41_multi_env_baseline.py`

**Структура:**

```python
"""
Stage 18 / exp41 — Multi-Env Baseline
Protocol: train EmbodiedAgent independently on each of 15 MiniGrid envs.
Gate: coverage_ratio > 0.3 for all easy/medium envs.
"""

# 5 групп по 3 env (параллельные воркеры)
WORKER_GROUPS = [
    ["MiniGrid-Empty-5x5-v0", "MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-8x8-v0"],
    ["MiniGrid-Empty-8x8-v0", "MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v0"],
    ["MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-LavaCrossingS9N2-v0", "MiniGrid-DoorKey-5x5-v0"],
    ["MiniGrid-SimpleCrossingS9N1-v0", "MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-Unlock-v0"],
    ["MiniGrid-UnlockPickup-v0", "MiniGrid-MemoryS7-v0", "MiniGrid-ObstructedMaze-1Dlhb-v0"],
]

def train_env_group(worker_id: int, env_ids: list[str], device: str) -> dict:
    """Обучить агента последовательно на 3 env. Возвращает метрики."""
    results = {}
    for env_id in env_ids:
        agent = EmbodiedAgent(build_config(device))
        env = make_env(env_id)
        metrics = _train_single_env(agent, env, env_id, n_steps, device)
        results[env_id] = metrics
    return results

def _train_single_env(agent, env, env_id, n_steps, device) -> dict:
    """
    Цикл обучения: agent.step → observe_result → end_episode.
    Checkpoint каждые 10K шагов.
    Возвращает: coverage_curve, goal_seeking_steps, steps_per_sec
    """
    ...

def run(device: str = "cpu") -> dict:
    with multiprocessing.Pool(5) as pool:
        results = pool.starmap(train_env_group, [(i, group, device) for i, group in enumerate(WORKER_GROUPS)])

    # Merge + evaluate gate
    all_results = {env_id: metrics for r in results for env_id, metrics in r.items()}
    passed = all(
        all_results[env_id]["final_coverage"] > 0.3
        for env_id, _, _ in ENVS if _ in ("easy", "medium")
    )

    # Save results/exp41_baseline.json + reports/exp41_baseline.html
    ...
    return {"passed": passed, "results": all_results}

def test_exp41_multi_env_baseline():
    result = run(device=_get_device())
    assert result["passed"], f"Gate failed: {result}"
```

**HTML-отчёт:** plotly кривые coverage по шагам для каждого env, сгруппированные по сложности.

---

### Фаза 4 — `exp42_transfer_matrix.py`

**30 кураторных пар (6 source × 5 target):**

```python
TRANSFER_PAIRS = [
    ("MiniGrid-Empty-5x5-v0", ["MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-5x5-v0",
                                "MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-MultiRoom-N2-S4-v0",
                                "MiniGrid-SimpleCrossingS9N1-v0"]),
    ("MiniGrid-FourRooms-v0", ["MiniGrid-Empty-8x8-v0", "MiniGrid-DoorKey-5x5-v0",
                                "MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-LavaCrossingS9N1-v0",
                                "MiniGrid-KeyCorridorS3R1-v0"]),
    ("MiniGrid-DoorKey-5x5-v0", ["MiniGrid-DoorKey-8x8-v0", "MiniGrid-Unlock-v0",
                                  "MiniGrid-UnlockPickup-v0", "MiniGrid-KeyCorridorS3R1-v0",
                                  "MiniGrid-FourRooms-v0"]),
    ("MiniGrid-LavaCrossingS9N1-v0", ["MiniGrid-LavaCrossingS9N2-v0", "MiniGrid-SimpleCrossingS9N1-v0",
                                       "MiniGrid-Empty-5x5-v0", "MiniGrid-FourRooms-v0",
                                       "MiniGrid-ObstructedMaze-1Dlhb-v0"]),
    ("MiniGrid-MultiRoom-N2-S4-v0", ["MiniGrid-MultiRoom-N4-S5-v0", "MiniGrid-FourRooms-v0",
                                      "MiniGrid-DoorKey-5x5-v0", "MiniGrid-KeyCorridorS3R1-v0",
                                      "MiniGrid-MemoryS7-v0"]),
    ("MiniGrid-KeyCorridorS3R1-v0", ["MiniGrid-Unlock-v0", "MiniGrid-DoorKey-5x5-v0",
                                      "MiniGrid-UnlockPickup-v0", "MiniGrid-DoorKey-8x8-v0",
                                      "MiniGrid-MultiRoom-N2-S4-v0"]),
]

def _eval_transfer(source_env_id: str, target_env_id: str, baseline_auc: float, device: str) -> dict:
    """
    1. Загрузить checkpoint exp41/{source_env_id}/final
    2. Запустить 2K шагов в target_env БЕЗ STDP обновления (agent.config.causal.pipeline.daf — disable learning)
    3. Вернуть adaptation_auc, transfer_score, final_coverage_ratio
    """
    agent = EmbodiedAgent(build_config(device))
    agent.load_checkpoint(checkpoint_path("exp41", source_env_id, "final"))
    # Заморозить STDP: agent.causal_agent.pipeline.daf.enable_learning = False
    agent.causal_agent.pipeline.daf.enable_learning = False
    ...

def _compute_baselines(device: str) -> dict[str, float]:
    """Для каждого target env: запустить random агента 2K шагов → baseline_auc"""
    ...

def run(device: str = "cpu") -> dict:
    # 1. Предвычислить baselines
    baselines = _compute_baselines(device)

    # 2. Раскидать 30 пар по 5 воркерам (по 6 пар каждый)
    flat_pairs = [(src, tgt) for src, targets in TRANSFER_PAIRS for tgt in targets]
    worker_batches = [flat_pairs[i::5] for i in range(5)]

    with multiprocessing.Pool(5) as pool:
        results_list = pool.starmap(_eval_batch, [(batch, baselines, device) for batch in worker_batches])

    # 3. Оценить gate: transfer_score > 1.0 для >= 20% пар (6 из 30)
    all_scores = [r["transfer_score"] for batch in results_list for r in batch]
    n_positive = sum(s > 1.0 for s in all_scores)
    passed = n_positive >= 6

    # 4. Save results/exp42_transfer.json + reports/exp42_transfer.html
    ...
    return {"passed": passed, "n_positive_transfer": n_positive, "results": ...}
```

**HTML-отчёт:** тепловая карта 6×5 transfer_score (зелёный > 1.0, красный < 1.0) + топ-5 лучших пар.

---

### Фаза 5 — `exp43_continual_multitask.py`

**43a — Sequential:**

```python
CHAINS = [
    ("MiniGrid-Empty-5x5-v0",        "MiniGrid-FourRooms-v0",          "MiniGrid-DoorKey-5x5-v0"),
    ("MiniGrid-Empty-8x8-v0",        "MiniGrid-MultiRoom-N2-S4-v0",    "MiniGrid-MultiRoom-N4-S5-v0"),
    ("MiniGrid-SimpleCrossingS9N1-v0","MiniGrid-LavaCrossingS9N1-v0",  "MiniGrid-LavaCrossingS9N2-v0"),
    ("MiniGrid-Unlock-v0",           "MiniGrid-DoorKey-5x5-v0",        "MiniGrid-UnlockPickup-v0"),
    ("MiniGrid-KeyCorridorS3R1-v0",  "MiniGrid-DoorKey-8x8-v0",        "MiniGrid-ObstructedMaze-1Dlhb-v0"),
]

def _run_chain(chain: tuple, device: str) -> dict:
    """
    A → B → C:
    1. Обучить на A, замерить coverage_A_after_A
    2. Обучить на B, замерить coverage_A_after_B, coverage_B_after_B
    3. Обучить на C, замерить coverage_A_after_C, coverage_B_after_C
    4. retention_ratio = coverage_X_after_Y / coverage_X_after_X
    """
    agent = EmbodiedAgent(build_config(device))
    metrics = {}
    for i, env_id in enumerate(chain):
        # Обучение
        _train_single_env(agent, make_env(env_id), env_id, n_steps_for(env_id), device)
        # Retention eval на всех предыдущих
        for j in range(i + 1):
            prev_env_id = chain[j]
            coverage = _eval_coverage(agent, make_env(prev_env_id), n_steps=1000, learn=False)
            metrics[f"coverage_{prev_env_id}_after_{env_id}"] = coverage
    return metrics
```

**43b — Multitask:**

```python
def _run_multitask(device: str) -> dict:
    agent = EmbodiedAgent(build_config(device))
    envs = {env_id: make_env(env_id) for env_id, _, _ in ENVS}

    total_steps = 0
    eval_every = 10_000
    metrics_history = []

    while total_steps < 200_000:
        # Случайный env на каждый step
        env_id = random.choice([e[0] for e in ENVS])
        env = envs[env_id]
        obs, _ = env.reset()
        action = agent.step(img(obs))
        obs_next, _, term, trunc, _ = env.step(action)
        agent.observe_result(img(obs_next))
        if term or trunc:
            agent.end_episode()
        total_steps += 1

        if total_steps % eval_every == 0:
            # Eval на каждом env (500 шагов, learn=False)
            eval_metrics = {env_id: _eval_coverage(agent, envs[env_id], 500, learn=False)
                           for env_id, _, _ in ENVS}
            metrics_history.append({"step": total_steps, "coverage": eval_metrics})

    return metrics_history
```

---

### Фаза 6 — `src/snks/viz/inference_viewer.py`

**FastAPI сервер для инференса на локальном RTX 3070:**

```python
"""
Inference viewer: загружает checkpoint exp41, запускает инференс на GPU,
стримит live рендер MiniGrid + метрики через WebSocket.
"""

app = FastAPI()

# CLI аргументы: --checkpoint, --env, --device
# Загрузить EmbodiedAgent из checkpoint
# WebSocket /ws: каждый step → broadcast {frame, coverage, fsm_state, sks_clusters}
# GET /  → inference.html
# GET /checkpoints → список доступных checkpoint files
# POST /load → загрузить другой checkpoint

@app.websocket("/ws")
async def ws_inference(websocket: WebSocket):
    """
    На каждом шаге агента:
    1. Рендерить env.render() → base64 PNG
    2. Считать coverage, FSM state, SKS clusters из last_cycle_result
    3. Broadcast JSON: {type: "step", frame: "...", coverage: 0.5, fsm: "EXPLORE", sks: [...]}
    """
    ...

@app.get("/checkpoints")
async def list_checkpoints():
    """Список папок в checkpoints/exp41/"""
    ...
```

**`src/snks/viz/static/inference.html`:**
- Canvas 400×400 для MiniGrid рендера (base64 PNG обновляется по WS)
- Dropdown: выбор env + шага checkpoint
- Coverage chart (Chart.js, последние 200 шагов)
- FSM badge: EXPLORE (синий) / GOAL_SEEKING (зелёный) / NEUTRAL (серый)
- SKS heatmap: топ-10 активных кластеров (bar chart)
- Кнопки: ▶ Play / ⏸ Pause / ⏭ Step / 🔄 Reset

---

## Порядок реализации и зависимости

```
Фаза 1 (DAF сериализация)
    ↓
Фаза 2 (stage18_utils.py)
    ↓
Фаза 3 (exp41) ←── нужна перед exp42
    ↓
Фаза 4 (exp42) ←── зависит от checkpoints exp41
    |
Фаза 5 (exp43) ←── независима от exp42, но нужны utils
    |
Фаза 6 (inference_viewer) ←── независима, но нужны checkpoints exp41
```

Фазы 4, 5, 6 можно писать **параллельно** после завершения фаз 1-3.

---

## Файлы для создания/изменения

| Действие | Файл |
|----------|------|
| **ИЗМЕНИТЬ** | `src/snks/daf/engine.py` — добавить `save_state()`, `load_state()` |
| **ИЗМЕНИТЬ** | `src/snks/agent/embodied_agent.py` — добавить `save_checkpoint()`, `load_checkpoint()` |
| **СОЗДАТЬ** | `src/snks/experiments/stage18_utils.py` |
| **СОЗДАТЬ** | `src/snks/experiments/exp41_multi_env_baseline.py` |
| **СОЗДАТЬ** | `src/snks/experiments/exp42_transfer_matrix.py` |
| **СОЗДАТЬ** | `src/snks/experiments/exp43_continual_multitask.py` |
| **СОЗДАТЬ** | `src/snks/viz/inference_viewer.py` |
| **СОЗДАТЬ** | `src/snks/viz/static/inference.html` |
| **СОЗДАТЬ** | `tests/test_daf_serialization.py` |
| **СОЗДАТЬ** | `checkpoints/` (директория) |
| **СОЗДАТЬ** | `results/` (директория) |
| **СОЗДАТЬ** | `reports/` (директория) |

---

## Ночной запуск (команда для minipc)

```bash
# ssh gem@10.253.0.179 -p 2244
cd /opt/agi && git pull
export HSA_OVERRIDE_GFX_VERSION=11.0.0

nohup bash -c "
  venv/bin/pytest src/snks/experiments/exp41_multi_env_baseline.py -s -v 2>&1 &&
  venv/bin/pytest src/snks/experiments/exp42_transfer_matrix.py -s -v 2>&1 &&
  venv/bin/pytest src/snks/experiments/exp43_continual_multitask.py -s -v 2>&1 &&
  git add results/ reports/ checkpoints/ &&
  git commit -m 'Stage 18 results: exp41/42/43 complete' &&
  git push
" > /tmp/stage18.log 2>&1 &

echo "Stage 18 started, PID=$!"
tail -f /tmp/stage18.log
```
