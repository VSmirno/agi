#!/bin/bash
# Импортирует конфиг Claude Code из архива на новую машину
# Использование: bash scripts/import_claude_config.sh claude-config-ДАТА.tar.gz /путь/к/проекту
# Пример: bash scripts/import_claude_config.sh claude-config-2026-03-31.tar.gz /home/gem/Projects/AGI

set -e

ARCHIVE="$1"
PROJECT_PATH="$2"

if [ -z "$ARCHIVE" ] || [ -z "$PROJECT_PATH" ]; then
    echo "Использование: $0 <архив.tar.gz> </путь/к/проекту/agi>"
    exit 1
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "Архив не найден: $ARCHIVE"
    exit 1
fi

# Определяем домашнюю папку
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    CLAUDE_HOME=$(cygpath "$USERPROFILE")/.claude
else
    CLAUDE_HOME="$HOME/.claude"
fi

echo "Claude home: $CLAUDE_HOME"
mkdir -p "$CLAUDE_HOME"

TMPDIR=$(mktemp -d)
tar xzf "$ARCHIVE" -C "$TMPDIR"

# Агенты
if [ -d "$TMPDIR/claude/agents" ]; then
    mkdir -p "$CLAUDE_HOME/agents"
    cp -r "$TMPDIR/claude/agents/." "$CLAUDE_HOME/agents/"
    echo "  ✓ agents/ установлены"
fi

# Команды
if [ -d "$TMPDIR/claude/commands" ]; then
    mkdir -p "$CLAUDE_HOME/commands"
    cp -r "$TMPDIR/claude/commands/." "$CLAUDE_HOME/commands/"
    echo "  ✓ commands/ установлены"
fi

# Настройки (мержим, не перезаписываем полностью)
if [ -f "$TMPDIR/claude/settings.json" ]; then
    if [ -f "$CLAUDE_HOME/settings.json" ]; then
        echo "  ! settings.json уже существует, сохранён как settings.json.bak"
        cp "$CLAUDE_HOME/settings.json" "$CLAUDE_HOME/settings.json.bak"
    fi
    cp "$TMPDIR/claude/settings.json" "$CLAUDE_HOME/settings.json"
    echo "  ✓ settings.json установлен"
fi

# Память
if [ -d "$TMPDIR/memory" ]; then
    # Кодируем путь к проекту
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        ENCODED=$(echo "$PROJECT_PATH" | sed 's|^/\([a-z]\)/|\U\1--|' | tr '/' '-')
    else
        ENCODED=$(echo "$PROJECT_PATH" | sed 's|^/||' | tr '/' '-')
    fi

    MEMORY_DIR="$CLAUDE_HOME/projects/$ENCODED/memory"
    mkdir -p "$MEMORY_DIR"
    cp -r "$TMPDIR/memory/." "$MEMORY_DIR/"
    echo "  ✓ память установлена в: $MEMORY_DIR"
fi

rm -rf "$TMPDIR"

echo ""
echo "Базовый конфиг установлен."
echo ""
echo "Осталось установить skills и плагины в Claude Code:"
echo ""
echo "  Плагины (выполнить в claude):"
echo "    /plugin install postgres-best-practices@supabase-agent-skills"
echo "    /plugin install frontend-design@claude-code-plugins"
echo "    /plugin install telegram@claude-plugins-official"
echo ""
echo "  Skills (выполнить в claude):"
echo "    /skill install brainstorming"
echo "    /skill install systematic-debugging"
echo "    /skill install python-expert"
echo "    /skill install using-superpowers"
echo "    /skill install executing-plans"
echo "    /skill install data-analysis"
echo "    /skill install fullstack-developer"
echo "    /skill install project-planner"
echo "    /skill install senior-devops"
echo "    /skill install senior-qa"
echo "    /skill install qa-test-planner"
echo "    /skill install ux-designer"
echo ""
echo "  Проектный skill (в папке проекта):"
echo "    /skill install   # читает skills-lock.json"
