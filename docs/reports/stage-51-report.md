# Stage 51: Language-Guided Planning

## Результат: PASS

**Ветка:** `stage51-language-guided-planning`
**Milestone:** M2 — Языковой контроль

## Что доказано
- Агент управляется текстовой инструкцией: "pick up the key then open the door then go to the goal"
- 100% success на 200 random DoorKey-5x5 с языковой инструкцией (gate ≥70%)
- Частичные инструкции работают: "pick up the key" → агент подбирает ключ и останавливается
- Варианты формулировок ("toggle the door", "red key") парсятся корректно
- InstructedAgent = LanguageGrounder (Stage 50) + SubgoalNavigator (Stage 46-47) — чистая интеграция

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 109a | Full instruction (200 random) | 100% (200/200) | ≥70% | PASS |
| 109b | Partial "pick up key" (50 random) | 100% (50/50) | ≥70% | PASS |
| 109c | Variant formulations (50 each) | 100% | — | PASS |
| 109a | Mean steps | 16.1 | — | info |
| 109a | Total time (200 eps) | 1.1s | — | info |

## Ключевые решения
- **Thin wrapper** — InstructedAgent не дублирует логику SubgoalNavigator, а переиспользует её
- Инструкция ЗАМЕНЯЕТ hardcoded subgoals, а не дополняет — агент выполняет ТОЛЬКО то, что указано
- CPU-only experiment (BFS на 7x7 grid) — GPU не нужен

## Веб-демо
- `demos/stage-51-language-guided.html` — Canvas DoorKey-5x5 с выбором инструкции, BFS навигация, подцели с прогрессом, trail агента

## Файлы изменены
- `src/snks/agent/instructed_agent.py` — NEW: InstructedAgent class
- `tests/test_instructed_agent.py` — NEW: 11 unit tests
- `src/snks/experiments/exp109_instructed_agent.py` — NEW: gate experiment
- `demos/stage-51-language-guided.html` — NEW: web demo
- `docs/superpowers/specs/2026-04-02-stage51-language-guided-planning-design.md` — NEW: spec

## Следующий этап
- **Stage 52: Integration test** (M3) — ≥50% random MultiRoom-N3 с языковой инструкцией. Полный пайплайн: язык + random карта + multi-room.
