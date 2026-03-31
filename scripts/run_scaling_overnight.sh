#!/usr/bin/env bash
# run_scaling_overnight.sh — GPU Scaling Suite on minipc (overnight)
#
# Usage (from local machine):
#   bash scripts/run_scaling_overnight.sh
#
# What it does:
#   1. git push local → remote
#   2. git pull on minipc
#   3. Run exp92_gpu_scaling_suite.py via nohup (~4-6 hours)
#   4. Show monitoring commands
#
# Results will be in /opt/agi/results/scaling/*.json

set -euo pipefail

SSH="ssh minipc"
REMOTE_DIR="/opt/agi"
LOG_FILE="scaling_overnight.log"

echo "=== СНКС GPU Scaling Suite — Overnight Runner ==="
echo ""

# --- 1. Push local changes ---
echo "[1/4] git push ..."
git add src/snks/experiments/exp92_gpu_scaling_suite.py scripts/run_scaling_overnight.sh
git commit -m "$(cat <<'EOF'
feat: Exp 92 — GPU Scaling Suite for overnight run on minipc

Parts: A) DAF throughput sweep N=50K/100K/200K, B) Large grids 12x12/16x16,
C) All exp58-91 on GPU, D) IntegratedAgent scaling.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" || echo "(nothing to commit)"
git push origin main

# --- 2. Pull on minipc ---
echo "[2/4] git pull on minipc ..."
$SSH "cd $REMOTE_DIR && git pull origin main"

# --- 3. Launch scaling suite ---
echo "[3/4] Launching scaling suite ..."
$SSH "cd $REMOTE_DIR && source venv/bin/activate && \
  nohup bash -c '\
    export HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    && export PYTHONPATH=$REMOTE_DIR/src \
    && export HF_HUB_OFFLINE=1 \
    && export TRANSFORMERS_OFFLINE=1 \
    && export PYTHONUNBUFFERED=1 \
    && python3 -m snks.experiments.exp92_gpu_scaling_suite cuda \
  ' > $REMOTE_DIR/$LOG_FILE 2>&1 &
  echo \"PID: \$!\""

echo ""
echo "[4/4] Scaling suite launched!"
echo ""

# Wait briefly and check it's running
sleep 5
echo "=== First log lines ==="
$SSH "head -20 $REMOTE_DIR/$LOG_FILE 2>/dev/null || echo '(waiting for output...)'"

echo ""
echo "=== Monitoring commands ==="
echo "  Live log:    ssh minipc 'tail -f $REMOTE_DIR/$LOG_FILE'"
echo "  Check running: ssh minipc 'pgrep -f exp92'"
echo "  Results:     ssh minipc 'ls -la $REMOTE_DIR/results/scaling/'"
echo "  Last lines:  ssh minipc 'tail -30 $REMOTE_DIR/$LOG_FILE'"
echo ""
echo "=== When done — fetch results ==="
echo "  scp minipc:$REMOTE_DIR/results/scaling/*.json results/scaling/"
echo ""
echo "Good night! Results will be ready in ~4-6 hours."
