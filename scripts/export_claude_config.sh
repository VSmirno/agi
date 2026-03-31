#!/bin/bash
# Экспортирует конфиг Claude Code и память проекта СНКС в один архив
# Использование: bash scripts/export_claude_config.sh
# Результат: claude-config-ДАТА.tar.gz в текущей папке

set -e

DATE=$(date +%Y-%m-%d)
ARCHIVE="claude-config-$DATE.tar.gz"
TMPDIR=$(mktemp -d)

echo "Собираем конфиг Claude Code..."

# Определяем домашнюю папку пользователя (работает на Windows/Linux/Mac)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    CLAUDE_HOME=$(cygpath "$USERPROFILE")/.claude
else
    CLAUDE_HOME="$HOME/.claude"
fi

echo "Claude home: $CLAUDE_HOME"

mkdir -p "$TMPDIR/claude"

# Агенты
if [ -d "$CLAUDE_HOME/agents" ]; then
    cp -r "$CLAUDE_HOME/agents" "$TMPDIR/claude/"
    echo "  ✓ agents/ ($(ls "$CLAUDE_HOME/agents" | wc -l) файлов)"
fi

# Команды
if [ -d "$CLAUDE_HOME/commands" ]; then
    cp -r "$CLAUDE_HOME/commands" "$TMPDIR/claude/"
    echo "  ✓ commands/ ($(ls "$CLAUDE_HOME/commands" | wc -l) файлов)"
fi

# Настройки
if [ -f "$CLAUDE_HOME/settings.json" ]; then
    cp "$CLAUDE_HOME/settings.json" "$TMPDIR/claude/"
    echo "  ✓ settings.json"
fi

# Память проекта СНКС
# Находим папку памяти по текущему пути проекта
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows: преобразуем /d/Projects/AGI → D--Projects-AGI
    ENCODED=$(echo "$SCRIPT_DIR" | sed 's|^/\([a-z]\)/|\U\1--|' | tr '/' '-')
else
    # Linux/Mac: /home/user/Projects/AGI → home-user-Projects-AGI
    ENCODED=$(echo "$SCRIPT_DIR" | sed 's|^/||' | tr '/' '-')
fi

MEMORY_DIR="$CLAUDE_HOME/projects/$ENCODED/memory"
if [ -d "$MEMORY_DIR" ]; then
    mkdir -p "$TMPDIR/memory"
    cp -r "$MEMORY_DIR/." "$TMPDIR/memory/"
    echo "  ✓ memory/ ($(ls "$MEMORY_DIR" | wc -l) файлов) [проект: $ENCODED]"
else
    echo "  ! память не найдена: $MEMORY_DIR"
fi

# Создаём архив
tar czf "$ARCHIVE" -C "$TMPDIR" .
rm -rf "$TMPDIR"

echo ""
echo "Готово: $ARCHIVE"
echo ""
echo "Для переноса на новую машину:"
echo "  1. Скопируй $ARCHIVE на новую машину"
echo "  2. Запусти: bash scripts/import_claude_config.sh $ARCHIVE /путь/к/проекту"
