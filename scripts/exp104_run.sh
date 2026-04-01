#!/bin/bash
# Stage 44 Phase 2: Naked DAF on DoorKey-5x5 (GPU)
set -e

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export SNKS_NO_COMPILE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Exp 104: Naked DAF on DoorKey-5x5 ==="
echo "Started: $(date)"

cd /opt/agi
source venv/bin/activate

python src/snks/experiments/exp104_naked_daf.py 2>&1

echo ""
echo "=== Experiment complete ==="
echo "Finished: $(date)"
