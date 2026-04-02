# Stage 50: Reconnect Language Pipeline

## Результат: PASS

**Ветка:** `stage50-language-reconnect`
**Milestone:** M2 — Языковой контроль

## Что доказано
- Языковые инструкции кодируются в VSA-пространство (512-dim binary) с 100% точностью decode
- Единый VSACodebook связывает текстовые концепты и планировщик subgoals
- Sequential инструкции ("pick up the key then open the door") корректно разбиваются на цепочки subgoals
- RuleBasedChunker (Stages 7-24) полностью совместим с VSA encoding

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 108 | Encode→decode accuracy (30 instructions) | 100% (30/30) | ≥90% | PASS |
| 108 | Subgoal mapping accuracy | 100% (30/30) | ≥90% | PASS |
| 108 | Max off-diagonal VSA similarity | 0.799 | <0.85 | PASS |
| 108 | Encode time (30 instructions) | 16ms | — | info |

## Ключевые решения
- **Direct VSA encoding** вместо HAC→VSA conversion — единое пространство, нет проблемы конвертации между 2048-dim FFT и 512-dim binary
- **RuleBasedChunker сохранён** — уже корректно парсит MiniGrid инструкции (action, object, attr, sequential)
- HAC pipeline (Stages 7-24) остаётся доступным, но для M2 VSA достаточно
- Принцип СНКС "язык = интерфейс" — encoding формат не принципиален, важна семантическая корректность

## Веб-демо
- `demos/stage-50-language-reconnect.html` — интерактивная визуализация пайплайна: ввод инструкции → chunks → VSA вектор (битовая карта) → decode → subgoals. Матрица схожести 8 инструкций.

## Файлы изменены
- `src/snks/language/language_grounder.py` — NEW: LanguageGrounder class
- `tests/test_language_grounder.py` — NEW: 30 unit tests
- `src/snks/experiments/exp108_language_vsa.py` — NEW: gate experiment
- `demos/stage-50-language-reconnect.html` — NEW: web demo
- `docs/superpowers/specs/2026-04-02-stage50-language-reconnect-design.md` — NEW: spec

## Следующий этап
- **Stage 51: Language-guided planning** — инструкция → subgoals → SubgoalNavigator → действия в среде. Gate: ≥70% random DoorKey-5x5 с текстовой инструкцией вместо встроенного reward.
