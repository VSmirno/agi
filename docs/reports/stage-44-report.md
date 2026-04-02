# Stage 44: Foundation Audit

## Результат: PASS (audit) / FAIL (naked DAF)

**Ветка:** `stage44-foundation-audit`
**Тип:** Audit / Verification (не новая фича)

## Что доказано

- Все 6 слоёв DAF-ядра работают корректно по отдельности (26/26 тестов PASS)
- FHN dynamics: excitable при default params, stable numerical integration
- STDP: LTP/LTD работают, reward modulation работает, homeostasis не убивает signal
- Coupling: spike propagation работает (нужно 0.5 model sec)
- SKS: DBSCAN находит группы, воспроизводимые, различимые
- Pipeline E2E: SDR → FHN → SKS chain работает, нет ghost signal
- Action selection: epsilon-greedy работает, range valid

## Ключевые находки

### Находка 1: FHN excitable, не oscillatory
- Одиночный FHN при I_base=0.5, tau=12.5 НЕ осциллирует
- Converges к stable fixed point ~v=1.2
- Система полагается на noise + coupling для spiking activity
- **Импликация:** oscillator dynamics ≠ oscillation. СНКС фактически работает как noise-driven excitable network.

### Находка 2: Coupling timescale mismatch
- Coupling propagation: 5000 steps = 0.5 model seconds
- Pipeline steps_per_cycle: 100 = 0.01 model seconds
- **50× разрыв:** за один perception cycle coupling НЕ успевает передать активность
- **Импликация:** SKS формируются НЕ через coupling, а через общий input (co-firing от SDR injection). Coupling contribution минимальный.

### Находка 3: Structural pruning
- За 30 эпизодов graph topology меняется на ~1% (edges removed)
- Штатное поведение, но нужно учитывать при метриках

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| Audit (26 тестов) | all pass | 26/26 PASS | all pass | PASS |
| 103 Golden Path | success rate | 56.7% (17/30) | > 50% | PASS |
| 103 weight delta | mean delta | 0.098 | > 0 | PASS |
| 104 Naked DAF | success rate | 0/200 = 0% | ≥ 15% | FAIL |

## Запланированные эксперименты (tech debt)

| TD | Exp | Что проверяется | Gate | Статус |
|----|-----|-----------------|------|--------|
| — | 104 | Naked DAF DoorKey-5x5, 50K нод, 200 ep | success ≥ 15% | FAIL (0/200) |

## Ключевые решения

- Спецификация написана ДО реализации, spec review через subagent
- Все тесты воспроизводимы (фиксированные seed-ы)
- Никаких изменений в production code — только тестовый код
- Golden Path использует SimpleGridEnv для изоляции от MiniGrid dependencies
- FHN excitability documented как наблюдение, не баг (engine relies on noise)

## Веб-демо
- `demos/stage-44-foundation-audit.html` — результаты аудита, golden path grid, key findings

## Файлы изменены
- `tests/test_stage44_audit.py` — 26 audit tests
- `src/snks/experiments/exp103_golden_path.py` — Golden Path experiment
- `src/snks/experiments/exp104_naked_daf.py` — Naked DAF experiment
- `scripts/exp103_104_run.sh` — GPU runner
- `scripts/exp104_run.sh` — dedicated exp104 runner
- `demos/stage-44-foundation-audit.html` — web demo
- `docs/superpowers/specs/2026-04-02-stage44-foundation-audit-design.md` — spec

## Следующий этап

exp104 FAIL (0/200 = 0%) подтверждает: Naked DAF не может решить DoorKey-5x5.
Причина не в timescale mismatch, а в **отсутствии world model** — STDP меняет coupling weights, но нет пути от reward к policy.

→ **Stage 45: VSA World Model** — добавить отдельный world model на основе VSA + SDM.

---

## Post-merge обновления

### 2026-04-02 — exp104 завершён (FAIL)

**Контекст:** Naked DAF DoorKey-5x5 на 50K нод, 200 эпизодов, GPU minipc.

**Результаты:**
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 104 | success_rate | 0/200 = 0% | ≥ 15% | FAIL |
| 104 | reward | 0.0 (все эпизоды) | > 0 | FAIL |
| 104 | PE range | 0.066-0.168 | — | info |

**Вывод:** DAF-ядро корректно как perception engine, но не может решать задачи без world model. Нужен отдельный модуль для запоминания transitions и action selection на основе predicted reward. Stage 45 (VSA World Model) адресует это.
