# Настройка Claude Code для работы над СНКС

Этот документ описывает, как перенести полное окружение Claude Code на новую машину.

---

## 1. Установить Claude Code

```bash
npm install -g @anthropic/claude-code
# или скачать десктопное приложение: https://claude.ai/download
```

---

## 2. Клонировать репозиторий

```bash
git clone <repo-url> ~/Projects/AGI
cd ~/Projects/AGI
```

---

## 3. Скопировать конфигурацию Claude

Скопировать с текущей машины три папки:

| Что | Откуда (Windows) | Куда (Linux/Mac) |
|-----|-----------------|-----------------|
| Агенты | `C:\Users\vl\.claude\agents\` | `~/.claude/agents/` |
| Команды | `C:\Users\vl\.claude\commands\` | `~/.claude/commands/` |
| Настройки | `C:\Users\vl\.claude\settings.json` | `~/.claude/settings.json` |

### Быстро через git (рекомендуется)

Папки `agents/` и `commands/` удобно держать в отдельном git-репо:

```bash
# На исходной машине — заархивировать
cd C:/Users/vl/.claude
tar czf claude-config.tar.gz agents/ commands/ settings.json

# На новой машине — распаковать
mkdir -p ~/.claude
cd ~/.claude
tar xzf claude-config.tar.gz
```

### Что содержится в agents/ (кастомные агенты)

| Файл | Назначение |
|------|-----------|
| `research-orchestrator.md` | Мульти-агентное исследование (opus) |
| `python-pro.md` | Production Python код |
| `backend-architect.md` | Архитектура бэкенда и API |
| `frontend-developer.md` | React/Vue/Angular разработка |
| `prompt-engineer.md` | Оптимизация промптов |
| `business-analyst.md` | Бизнес-анализ и требования |
| `ui-ux-designer.md` | UX/UI дизайн |

### Что содержится в commands/ (кастомные команды /slash)

| Файл | Команда | Назначение |
|------|---------|-----------|
| `autonomous-dev.md` | `/autonomous-dev` | Автономная разработка этапа СНКС |
| `silent-mode.md` | `/silent-mode` | Работа без участия пользователя |
| `analyze-chat.md` | `/analyze-chat` | Анализ Telegram-чата |
| `test.md` | `/test` | Кастомная тест-команда |

---

## 4. Установить плагины

```bash
# В любой директории запустить claude и выполнить:
/plugin install postgres-best-practices@supabase-agent-skills
/plugin install frontend-design@claude-code-plugins
/plugin install telegram@claude-plugins-official
```

---

## 5. Установить skills из Superpowers Marketplace

```bash
# Сначала добавить marketplace (если не в settings.json)
# Затем установить каждый skill:
/skill install brainstorming
/skill install data-analysis
/skill install executing-plans
/skill install fullstack-developer
/skill install project-planner
/skill install python-expert
/skill install qa-test-planner
/skill install senior-devops
/skill install senior-qa
/skill install systematic-debugging
/skill install using-superpowers
/skill install ux-designer
```

Если marketplace не найден — добавить его вручную в `~/.claude/settings.json`:

```json
"extraKnownMarketplaces": {
  "superpowers-marketplace": {
    "source": {
      "source": "github",
      "repo": "obra/superpowers-marketplace"
    }
  }
}
```

### Проектный skill (multi-agent-brainstorming)

Устанавливается из `skills-lock.json` в корне репозитория:

```bash
cd ~/Projects/AGI
/skill install   # claude прочитает skills-lock.json автоматически
```

Или вручную:

```bash
/skill install multi-agent-brainstorming@sickn33/antigravity-awesome-skills
```

---

## 6. Перенести память (Memory)

Память хранится в `~/.claude/projects/<encoded-path>/memory/`.

Имя папки — это путь к проекту, где `/` и `\` заменены на `-`, а `:` удалён.

Примеры:
- `D:\Projects\AGI` → `D--Projects-AGI`
- `/home/gem/Projects/AGI` → `home-gem-Projects-AGI`

### Шаги:

**На исходной машине — скопировать:**
```bash
cp -r C:/Users/vl/.claude/projects/D--Projects-AGI/memory/ /tmp/snks-memory/
```

**На новой машине — определить путь к проекту и создать папку:**
```bash
# Если проект в /home/gem/Projects/AGI:
PROJECT_PATH="home-gem-Projects-AGI"
mkdir -p ~/.claude/projects/$PROJECT_PATH/memory/
cp /tmp/snks-memory/* ~/.claude/projects/$PROJECT_PATH/memory/
```

> Память будет работать корректно, как только Claude Code откроет проект по новому пути.

---

## 7. Настроить minipc (если нужен)

Данные доступа к minipc (AMD GPU) хранятся в памяти:

- **Host:** `10.253.0.179:2244`
- **User:** `gem`
- **Путь:** `/opt/agi`
- **Ключ:** SSH-ключ должен быть скопирован отдельно

```bash
ssh-copy-id -p 2244 gem@10.253.0.179
```

---

## 8. Проверка окружения

```bash
# Убедиться, что Python окружение работает
cd ~/Projects/AGI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -x -q --tb=short
```

---

## Итоговый чеклист

- [ ] Claude Code установлен
- [ ] Репозиторий клонирован
- [ ] `agents/` скопированы в `~/.claude/agents/`
- [ ] `commands/` скопированы в `~/.claude/commands/`
- [ ] `settings.json` скопирован в `~/.claude/settings.json`
- [ ] Плагины установлены (4 штуки)
- [ ] Skills установлены (12 штук из superpowers + 1 проектный)
- [ ] Память скопирована в правильную папку
- [ ] SSH-ключ для minipc настроен
- [ ] Python venv создан, тесты проходят
