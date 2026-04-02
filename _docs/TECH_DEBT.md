# Technical Debt Register

Structured tracking of deferred work across all stages.
Maintained automatically by the `autonomous-dev` skill.

**Types:** `GPU_EXP` | `INTEGRATION` | `PERF` | `BUG`
**Status:** `OPEN` | `IN_PROGRESS` | `CLOSED`

---

## Open Items

| ID | Stage | Type | Description | Gate Criteria | Git ref | Status | Opened | Closed |
|----|-------|------|-------------|---------------|---------|--------|--------|--------|
| TD-001 | 38 | BUG | exp97 GPU: 0% success after 14/20 episodes on 50K nodes (killed). Root cause identified: **perception blind** — Gabor encoder + grayscale conversion cannot distinguish key/door/goal in MiniGrid. Agent is effectively random-walking. Stage 38_fix (curiosity/PE) verified mechanisms but cannot help if agent cannot see. **Blocked by Stage 42 (perception fix).** | success_rate ≥ 0.10, modulations > 0 | `main` | IN_PROGRESS | 2026-04-01 | — |
| TD-002 | 39 | GPU_EXP | exp98e CurriculumTrainer DoorKey-5x5 на 50K нод. CPU-гейты механизмов PASS, абсолютная производительность не проверена. | success_rate ≥ 0.25 | `main` @ `c875a82` | OPEN | 2026-04-01 | — |
| TD-003 | 40 | GPU_EXP | exp99 HebbianEncoder + PureDafAgent на 50K нод. CPU показал SDR overlap improvement 14%, реальное влияние на success_rate неизвестно. | success_rate ≥ 0.15 (лучше Stage 38 baseline) | `main` @ `c875a82` | OPEN | 2026-04-01 | — |
| TD-004 | 43 | INTEGRATION | WM gating: sustained oscillation confirmed, coupling damping 0.1x applied, SKS excluded. But FHN **self-sustains** (intrinsic dynamics, not coupling). Need bistable FHN tuning or explicit gating for WM to actually help. | WM DoorKey success > no-WM success | `main` | OPEN | 2026-04-01 | — |
| TD-005 | 45 | INTEGRATION | VSA World Model planning: forward beam search fails on detour tasks (DoorKey). Need subgoal extraction from successful traces + plan graph + subgoal-conditioned navigation. VSA+SDM foundation is solid (encoding 97%, prediction 0.85). | plan phase success_rate ≥ 15% DoorKey-5x5 | `main` | OPEN | 2026-04-02 | — |

---

## Closed Items

_Нет закрытых позиций_

---

## Notes

- GPU_EXP items close when run on minipc (evo-x2, AMD 96GB, ROCm 7.2)
- **Деплой на minipc ТОЛЬКО через git:** `git push origin main` → `ssh minipc "cd /opt/agi && git pull origin main"`. НИКОГДА rsync.
- **Git ref** указывает ветку и коммит, на котором запускать эксперимент
- When closing an item: update Status + Closed date in this table, then append results to the original stage report under `## Post-merge обновления`
- Partially resolved items: update Status to IN_PROGRESS, note partial results in the stage report section
