#!/bin/bash
# Run Experiment 97 (Pure DAF Agent) on minipc GPU server
# Usage: ./scripts/run_exp97_gpu.sh
#
# Prerequisites: sync repo to minipc first:
#   rsync -avz --exclude='.venv' --exclude='__pycache__' . minipc:/opt/agi/

set -euo pipefail

REMOTE="minipc"
REMOTE_DIR="/opt/agi"
ENV_VARS="HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=${REMOTE_DIR}/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

echo "=== Stage 38: Pure DAF Agent — GPU Experiment ==="
echo "Server: $REMOTE ($REMOTE_DIR)"
echo ""

# Step 1: Sync code
echo ">>> Syncing code to minipc..."
rsync -avz --delete \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.env' \
    /home/yorick/agi/ ${REMOTE}:${REMOTE_DIR}/

echo ""
echo ">>> Installing minigrid if needed..."
ssh ${REMOTE} "cd ${REMOTE_DIR} && source venv/bin/activate && pip install minigrid -q 2>&1 | tail -2"

echo ""
echo ">>> Running exp97 on GPU (50K nodes)..."
ssh ${REMOTE} "cd ${REMOTE_DIR} && source venv/bin/activate && ${ENV_VARS} python3 -c '
import sys
# Bypass torchvision if not installed
try:
    import torchvision
except ImportError:
    sys.modules[\"torchvision\"] = type(sys)(\"torchvision\")
    sys.modules[\"torchvision.datasets\"] = type(sys)(\"torchvision.datasets\")

from snks.experiments.exp97_pure_daf import main
main()
'" 2>&1 | tee /home/yorick/agi/_docs/exp97_gpu_results.txt

echo ""
echo "=== Done. Results saved to _docs/exp97_gpu_results.txt ==="
