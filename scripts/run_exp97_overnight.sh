#!/bin/bash
# Run Experiment 97 overnight on minipc GPU
# Estimated runtime: ~2-3 hours with reduced episodes
# Usage: ./scripts/run_exp97_overnight.sh

set -euo pipefail

REMOTE="minipc"
REMOTE_DIR="/opt/agi"

echo "=== Stage 38: Pure DAF Agent — GPU Overnight Run ==="
echo "Estimated runtime: ~2-3 hours"
echo ""

# Sync code
echo ">>> Syncing code..."
rsync -avz --delete \
    --exclude='.venv' --exclude='venv' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.pyc' --exclude='.env' \
    /home/yorick/agi/ ${REMOTE}:${REMOTE_DIR}/ 2>&1 | tail -2

echo ""
echo ">>> Launching on GPU (nohup)..."
ssh ${REMOTE} "cd ${REMOTE_DIR} && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    nohup python3 -u -c '
import sys
try:
    import torchvision
except ImportError:
    sys.modules[\"torchvision\"] = type(sys)(\"torchvision\")
    sys.modules[\"torchvision.datasets\"] = type(sys)(\"torchvision.datasets\")

from snks.experiments.exp97_pure_daf import main
main()
' > ${REMOTE_DIR}/_docs/exp97_gpu_results.txt 2>&1 &
echo \"PID=\$!\"
"

echo ""
echo ">>> Experiment launched! Check results:"
echo "    ssh minipc 'tail -30 /opt/agi/_docs/exp97_gpu_results.txt'"
echo ""
echo ">>> Kill if needed:"
echo "    ssh minipc 'pkill -f pure_daf'"
