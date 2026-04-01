# Technical Debt Register

Structured tracking of deferred work across all stages.
Maintained automatically by the `autonomous-dev` skill.

**Types:** `GPU_EXP` | `INTEGRATION` | `PERF` | `BUG`
**Status:** `OPEN` | `IN_PROGRESS` | `CLOSED`

---

## Open Items

| ID | Stage | Type | Description | Gate Criteria | Status | Opened | Closed |
|----|-------|------|-------------|---------------|--------|--------|--------|
| TD-001 | 38 | GPU_EXP | exp97 Pure DAF DoorKey-5x5 на 50K нод. Partial: exp97a=0% (FAIL), exp97b=0 modulations (FAIL), exp97c=INCOMPLETE (torch.compile hang, fixed), exp97d=PASS. Нужен перезапуск с fix compile + анализ почему 0 modulations. | success_rate ≥ 0.10, modulations > 0 | IN_PROGRESS | 2026-04-01 | — |
| TD-002 | 39 | GPU_EXP | exp98e CurriculumTrainer DoorKey-5x5 на 50K нод. CPU-гейты механизмов PASS, абсолютная производительность не проверена. | success_rate ≥ 0.25 | OPEN | 2026-04-01 | — |
| TD-003 | 40 | GPU_EXP | exp99 HebbianEncoder + PureDafAgent на 50K нод. CPU показал SDR overlap improvement 14%, реальное влияние на success_rate неизвестно. | success_rate ≥ 0.15 (лучше Stage 38 baseline) | OPEN | 2026-04-01 | — |

---

## Closed Items

_Нет закрытых позиций_

---

## Notes

- GPU_EXP items close when run on minipc (evo-x2, AMD 96GB, ROCm 7.2) via `scripts/minipc-run.sh`
- When closing an item: update Status + Closed date in this table, then append results to the original stage report under `## Post-merge обновления`
- Partially resolved items: update Status to IN_PROGRESS, note partial results in the stage report section
