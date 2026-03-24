#!/bin/bash
# fetch_demo_results.sh — скачать результаты демо с minipc
#
# Создаёт папку demo_output/ локально с HTML-отчётами

REMOTE="gem@10.253.0.179"
SSH_PORT=2244
SSH="ssh -p $SSH_PORT $REMOTE"
REMOTE_DIR="/opt/agi/demo_output"
LOCAL_DIR="demo_output"

echo "=== Скачивание результатов с minipc ==="

mkdir -p "$LOCAL_DIR"

# Проверить что файлы существуют
echo "[CHECK] Файлы на minipc:"
$SSH "ls -lh $REMOTE_DIR/ 2>/dev/null || echo 'demo_output не найден'"

# Скачать всё
echo ""
echo "[COPY] Копирование..."
scp -P $SSH_PORT -r "${REMOTE}:${REMOTE_DIR}/" ./ 2>/dev/null || \
  echo "WARN: scp не сработал, пробую rsync..."
  # rsync -avz -e "ssh -p $SSH_PORT" "${REMOTE}:${REMOTE_DIR}/" "$LOCAL_DIR/"

echo ""
echo "=== Готово ==="
echo "Открой в браузере:"
ls demo_output/*.html 2>/dev/null | while read f; do echo "  file://$(pwd)/$f"; done
