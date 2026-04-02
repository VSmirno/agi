#!/bin/bash
# Exp 105: VSA World Model — Stage 45
# Run on minipc (evo-x2, AMD GPU 96GB, ROCm 7.2)

set -euo pipefail

cd /opt/agi
source venv/bin/activate

export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "=== Exp 105: VSA World Model — $(date) ==="
echo "Device: CPU (VSA is pure tensor ops, no GPU needed)"
echo ""

python src/snks/experiments/exp105_vsa_world_model.py 2>&1 | tee _docs/exp105_output.txt

echo ""
echo "=== Done: $(date) ==="
