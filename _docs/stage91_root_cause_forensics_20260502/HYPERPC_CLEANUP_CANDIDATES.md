# HyperPC cleanup candidates — Stage 91 session

Snapshot: 2026-05-10, после закрытия Stage 91.
Хост: `cuda@192.168.98.56`
Диск: `/dev/nvme0n1p2` — 766G/916G использовано (89%).

Все директории ниже — рабочие checkout'ы и артефакт-дампы, созданные
в этой сессии. **Этот файл — список кандидатов, не план уничтожения.**
Решение по каждой группе принимаешь ты; команды удаления приведены готовыми.

---

## Что НИ В КОЕМ СЛУЧАЕ не трогать

Эти три каталога — каноничные референсы, на них завязаны все артефакты
и runbook'и:

- `/opt/cuda/agi/` — основной репо
- `/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/` (404K) —
  каноничный feasibility checkout. Используется как источник `cp -al`
  hardlink-клонов, и в нём лежит `_docs/stage90r_seed7_actor_selection_probe3.pt`
  (нужен для eval). **Удалять его сломает всё**.
- `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/` (1.3M) —
  каноничный verify checkout. На него ссылаются все ранние findings.
- `/opt/cuda/agi-stage91-strict-determinism-kthvalue-cpu-20260509T145726Z/` (15M) —
  источник `deterministic_wrapper.py`, который нужен для любого strict-eval.

---

## Группа 1: рабочие checkout'ы провалившихся попыток

Все четыре итерации хирургических правок откачены и зафиксированы как
net-negative. Их рабочие checkout'ы больше не нужны: код в git, eval JSON
выгружены локально в `_docs/stage91_root_cause_forensics_20260502/transfer/`.

| путь | размер | что было |
|---|---:|---|
| `agi-stage91-adjacent-penalty-gate-20260509T195137Z` | 2.0G | попытка #1 (gate adjacent_penalty) |
| `agi-stage91-null-target-do-fix-20260509T202758Z` | 18M | попытка #2 (drop continue + backstop) |
| `agi-stage91-blocked-h-broaden-20260509T211630Z` | 23M | попытка #3 (broaden blocked_h) |
| `agi-stage91-emergency-no-op-20260510T014154Z` | 18M | попытка #4 (emergency-gated no-op) |
| `agi-stage91-emergency-no-op-20260509T222253Z` | 832K | первая stale попытка #4 |

**Освобождает: ~2.1G**

```bash
# на HyperPC
ssh cuda@192.168.98.56 'rm -rf \
  /opt/cuda/agi-stage91-adjacent-penalty-gate-20260509T195137Z \
  /opt/cuda/agi-stage91-null-target-do-fix-20260509T202758Z \
  /opt/cuda/agi-stage91-blocked-h-broaden-20260509T211630Z \
  /opt/cuda/agi-stage91-emergency-no-op-20260510T014154Z \
  /opt/cuda/agi-stage91-emergency-no-op-20260509T222253Z'
```

---

## Группа 2: deterministic re-baseline checkout'ы (Phase A/B/C)

Использовались для перемера всех направлений под детерминизмом. Findings
коммитнуты (`deterministic_rebaseline_findings.md`), eval JSON выгружены
локально для seed 7/17/47 (`phaseA_seed_7_eval.json`, `phaseA_seed_17_eval.json`).
Если понадобится seed 27/37/47 Phase A — re-pull одной командой scp.

| путь | размер | что было |
|---|---:|---|
| `agi-stage91-determ-rebase-B-20260509T180000Z` | 2.3G | Phase B (no Stage91 fixes) |
| `agi-stage91-determ-rebase-A-20260509T180000Z` | 25M | Phase A (HEAD) — но в нём всего 25M. Наверное hardlink. |
| `agi-stage91-determ-rebase-C-extractor-20260509T180000Z` | 12M | Phase C extractor v1 |
| `agi-stage91-determ-rebase-C-do_facing-20260509T180000Z` | 10M | Phase C do_facing |
| `agi-stage91-determ-rebase-C-threat_ranking-20260509T180000Z` | 9.3M | Phase C threat_ranking |

**Внимание**: B содержит `_docs/stage90r_seed7_actor_selection_probe3.pt` (2.3M),
который мы использовали как `EVAL_PT` для eval-команд. Перед удалением убедиться
что этот .pt есть либо в feasibility-label-fix (его **там не было** при последнем
запуске), либо скопировать в feasibility-label-fix или в strict-determinism
checkout. Иначе будущие eval-запуски сломаются.

**Освобождает: ~2.36G** (если решён вопрос с .pt)

```bash
# Сначала перенести .pt в безопасное место:
ssh cuda@192.168.98.56 'cp /opt/cuda/agi-stage91-determ-rebase-B-20260509T180000Z/_docs/stage90r_seed7_actor_selection_probe3.pt /opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/_docs/'

# Затем удалить:
ssh cuda@192.168.98.56 'rm -rf \
  /opt/cuda/agi-stage91-determ-rebase-A-20260509T180000Z \
  /opt/cuda/agi-stage91-determ-rebase-B-20260509T180000Z \
  /opt/cuda/agi-stage91-determ-rebase-C-extractor-20260509T180000Z \
  /opt/cuda/agi-stage91-determ-rebase-C-do_facing-20260509T180000Z \
  /opt/cuda/agi-stage91-determ-rebase-C-threat_ranking-20260509T180000Z'
```

---

## Группа 3: seed-17 attribution layers (M1/M2/M2b/M3)

Использовались чтобы декомпозировать −9 на seed 17 по слоям фиксов.
Findings коммитнуты (`seed17_attribution_findings.md`). Сами checkout'ы
больше не нужны — все интересные seed 17 JSON выгружены локально.

| путь | размер |
|---|---:|
| `agi-stage91-seed17-attribution-M1-20260509T220000Z` | 5.3M |
| `agi-stage91-seed17-attribution-M2-20260509T220000Z` | 5.1M |
| `agi-stage91-seed17-attribution-M2b-20260509T220000Z` | 5.4M |
| `agi-stage91-seed17-attribution-M3-20260509T220000Z` | 4.9M |
| `agi-stage91-seed17-attribution-extractorP2-20260509T220000Z` | 18M |

**Освобождает: ~39M** (мало, потому что hardlink-клоны)

```bash
ssh cuda@192.168.98.56 'rm -rf /opt/cuda/agi-stage91-seed17-attribution-*-20260509T220000Z'
```

---

## Группа 4: ранние determinism investigation checkout'ы

`agi-stage91-determinism-bisect-...` был рабочим каталогом задачи `4ecaf360`,
которая нашла crafter `_balance_chunk` баг. Findings коммитнуты
(`determinism_root_cause_and_fix_findings.md`), бисект-инструментация
больше не нужна.

| путь | размер |
|---|---:|
| `agi-stage91-determinism-bisect-20260509T155656Z` | 2.9M |
| `agi-stage91-determinism-validation-20260509T160313Z` | 33M |

**Освобождает: ~36M**

```bash
ssh cuda@192.168.98.56 'rm -rf \
  /opt/cuda/agi-stage91-determinism-bisect-20260509T155656Z \
  /opt/cuda/agi-stage91-determinism-validation-20260509T160313Z'
```

---

## Сводка

Если всё снести из групп 1–4:

- **Освобождается**: ~4.55G
- **Остаётся**: 5 защищённых каталогов + `agi/` (основной репо)
- **Диск после**: ~109G свободно (был 104G)

Не сильно много, потому что почти все checkout'ы — hardlink-клоны
(`cp -al`) и реально занимают только уникальные артефакты в `_docs/`.
Большую часть хранилища (≥4G из 4.55G) дают всего две директории:
`adjacent-penalty-gate` и `determ-rebase-B`.

---

## Для запуска новых экспериментов в будущем

Если возвращаемся к Stage 91 (или начинаем Stage 92), потребуется
hardlink-clone'ить `agi-stage91-feasibility-label-fix` как стартовую
точку. Чтобы это работало:

1. `agi-stage91-feasibility-label-fix-20260507T181856Z/` должна остаться нетронутой.
2. `_docs/stage90r_seed7_actor_selection_probe3.pt` должен быть либо в ней, либо
   в `agi-stage91-strict-determinism-kthvalue-cpu-20260509T145726Z/_docs/`
   (она тоже остаётся).
3. `commands/deterministic_wrapper.py` лежит в `agi-stage91-strict-determinism-kthvalue-cpu-...`
   — этот checkout не трогаем.

Если все три условия соблюдены, любой будущий experiment начинается
одной командой:
```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
ssh cuda@192.168.98.56 "cp -al /opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z /opt/cuda/agi-stage92-<NAME>-$TS"
```
