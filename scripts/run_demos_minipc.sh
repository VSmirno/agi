#!/bin/bash
# run_demos_minipc.sh — запустить демо-эксперименты на minipc в параллель
#
# Использование (с локальной машины):
#   bash scripts/run_demos_minipc.sh
#
# Что делает:
#   1. git push + git pull на minipc
#   2. pip install matplotlib (если нет)
#   3. Запускает shapes demo (быстрый, ~20 мин) и MNIST demo (основной, ~90 мин) параллельно
#   4. Логи: /opt/agi/demo_shapes.log и /opt/agi/demo_mnist.log

set -e

REMOTE="gem@10.253.0.179"
SSH_PORT=2244
SSH="ssh -p $SSH_PORT $REMOTE"
REMOTE_DIR="/opt/agi"

echo "=== СНКС Demo Runner ==="
echo ""

# --- 1. Push + pull ---
echo "[1/4] git push..."
git push origin main

echo "[1/4] git pull на minipc..."
$SSH "cd $REMOTE_DIR && git pull origin main"

# --- 2. Зависимости ---
echo "[2/4] Проверка зависимостей..."
$SSH "cd $REMOTE_DIR && source venv/bin/activate && \
  python -c 'import matplotlib' 2>/dev/null || \
  pip install matplotlib --quiet"

# --- 3. Запуск экспериментов ---
echo "[3/4] Запуск экспериментов..."

# Shapes demo (быстрый — 20K узлов, 80/class, 5 эпох)
$SSH "cd $REMOTE_DIR && source venv/bin/activate && \
  nohup bash -c '\
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    python -m snks.experiments.exp_demo_shapes \
      --nodes 20000 --per-class 80 --epochs 5 --output demo_output \
    > demo_shapes.log 2>&1 \
  ' &
  echo \"Shapes PID: \$!\""

echo "[INFO] Shapes demo запущен"

# MNIST demo (основной — 50K узлов, 200/class, 5 эпох)
$SSH "cd $REMOTE_DIR && source venv/bin/activate && \
  nohup bash -c '\
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    python -m snks.experiments.exp_demo_mnist \
      --nodes 50000 --per-class 200 --epochs 5 --output demo_output \
    > demo_mnist.log 2>&1 \
  ' &
  echo \"MNIST PID: \$!\""

echo "[INFO] MNIST demo запущен"
echo ""

# --- 4. Статус ---
echo "[4/4] Проверка запуска через 10 секунд..."
sleep 10

echo ""
echo "=== Shapes log (последние 5 строк) ==="
$SSH "tail -5 $REMOTE_DIR/demo_shapes.log 2>/dev/null || echo '(пусто)'"

echo ""
echo "=== MNIST log (последние 5 строк) ==="
$SSH "tail -5 $REMOTE_DIR/demo_mnist.log 2>/dev/null || echo '(пусто)'"

echo ""
echo "=== Команды для мониторинга ==="
echo "  Shapes: ssh -p $SSH_PORT $REMOTE 'tail -f $REMOTE_DIR/demo_shapes.log'"
echo "  MNIST:  ssh -p $SSH_PORT $REMOTE 'tail -f $REMOTE_DIR/demo_mnist.log'"
echo ""
echo "=== Когда готово — скачать результаты ==="
echo "  bash scripts/fetch_demo_results.sh"
