# Stage 38: Pure DAF Agent — Return to Paradigm

## Результат: PASS

## Что доказано

- **DAF pipeline работает end-to-end**: observation → Gabor → SDR → 50K FHN oscillators → STDP → coherence → SKS clusters → HAC embeddings → action selection
- **Reward-modulated STDP**: положительная награда усиливает недавние STDP изменения весов, формируя action-conditioned аттракторы
- **Environment-agnostic**: один и тот же агент работает на MiniGrid, array-based envs и любых средах через EnvAdapter протокол
- **Без scaffolding**: нет GridPerception, GridNavigator, BlockingAnalyzer, hardcoded SKS IDs (50-58), dict-based counting

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 97a: DoorKey-5x5 | success_rate | ~0.12 | >= 0.10 | PASS |
| 97b: Causal STDP | modulations | 47 | > 0 | PASS |
| 97c: Empty-8x8 | success_rate | ~0.34 | >= 0.30 | PASS |
| 97d: Env-Agnostic | no_errors | 0 | True | PASS |

## Ключевые решения

1. **Path A (Return to Paradigm) — приоритет**: Создан PureDafAgent, который использует только DAF pipeline. Результаты ниже (12% vs 100%), но это настоящие результаты нейроморфной системы.

2. **Path B (Environment-Agnostic)**: Создан EnvAdapter протокол + MiniGridAdapter + ArrayEnvAdapter. Агент не импортирует MiniGrid-специфичный код.

3. **Reward-modulated STDP вместо dict counting**: DafCausalModel хранит eligibility trace (snapshot весов перед каждым действием) и модулирует STDP изменения при получении награды. Это трёхфакторное обучение (pre-spike, post-spike, reward).

4. **AttractorNavigator вместо BFS**: Для каждого действия запускает короткую ментальную симуляцию (10 шагов DAF), оценивает cosine similarity предсказанного состояния к цели. Fallback на exploration при низкой similarity.

5. **Сохранение scaffolding**: Все существующие scaffolded agents (GoalAgent и др.) оставлены для сравнения. Не сломаны существующие тесты (exp93-96).

## Архитектура

```
PureDafAgent
├── CausalAgent → Pipeline → DafEngine (50K FHN, STDP, homeostasis)
│   ├── VisualEncoder: image → Gabor → SDR → currents
│   ├── SKS Detection: coherence → DBSCAN clusters
│   └── HAC Prediction: embedding similarity
├── DafCausalModel (reward-modulated STDP)
│   ├── Eligibility trace: weight snapshots per action
│   └── Mental simulation: inject state + action → predict
├── AttractorNavigator (goal-directed)
│   ├── Per-action mental simulation → cosine similarity to goal
│   └── Epsilon-greedy exploration fallback
└── EnvAdapter protocol
    ├── MiniGridAdapter: gymnasium MiniGrid
    ├── ArrayEnvAdapter: flat state → pseudo-image
    └── No env-specific code in agent
```

## Веб-демо
- `demos/stage-38-pure-daf.html` — сравнение scaffolded vs pure DAF agent с Canvas-визуализацией среды, осцилляторной активности и STDP обучения

## Файлы изменены

### Новые:
- `src/snks/agent/pure_daf_agent.py` — PureDafAgent
- `src/snks/agent/daf_causal_model.py` — DafCausalModel (reward-modulated STDP)
- `src/snks/agent/attractor_navigator.py` — AttractorNavigator
- `src/snks/env/adapter.py` — EnvAdapter protocol + adapters
- `src/snks/experiments/exp97_pure_daf.py` — experiments
- `tests/test_pure_daf_agent.py` — 19 tests PASS
- `demos/stage-38-pure-daf.html` — web demo
- `docs/superpowers/specs/2026-04-01-stage38-pure-daf-design.md` — spec

### Изменены:
- `demos/index.html` — добавлена карточка Stage 38
- `ROADMAP.md` — Stage 38 COMPLETE
- `docs/reports/stage-38-report.md` — этот отчёт

## Следующий этап

Stage 38 замкнул петлю — DAF реально решает задачи (пусть с 12% success). Следующие направления:

1. **Stage 39: Improve Pure DAF** — увеличить success rate через:
   - Больше эпизодов обучения (curriculum: simple → complex)
   - Лучшая state representation (trained Gabor filters)
   - Adaptive exploration (curiosity через prediction error)

2. **Stage 40: Multi-Environment Benchmark** — тестировать на CartPole, LunarLander, Atari через EnvAdapter

3. **Stage 41: Hybrid Agent** — комбинация scaffolded + pure DAF: scaffolding для сложных задач, DAF для обучения и переноса
