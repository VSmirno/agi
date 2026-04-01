#!/bin/bash
# Stage 44 Foundation Audit — GPU experiments on minipc
# Phase 0: Golden Path (exp103) + Phase 2: Naked DAF (exp104)

set -e

echo "=== Stage 44: Foundation Audit — GPU Experiments ==="
echo "Started: $(date)"
echo ""

cd /opt/agi
source venv/bin/activate

# Phase 0: Golden Path (fast, ~5-10 min)
echo "--- Phase 0: Golden Path (exp103) ---"
SNKS_NO_COMPILE=1 python src/snks/experiments/exp103_golden_path.py 2>&1 | tee _docs/exp103_output.txt
echo ""

# Phase 1: Audit tests on GPU
echo "--- Phase 1: Audit Tests on GPU ---"
SNKS_NO_COMPILE=1 python -m pytest tests/test_stage44_audit.py -v --tb=short 2>&1 | tee _docs/exp_audit_gpu.txt
echo ""

# Phase 2: Naked DAF on DoorKey (long, ~1-2 hours)
echo "--- Phase 2: Naked DAF (exp104) ---"
SNKS_NO_COMPILE=1 python src/snks/experiments/exp104_naked_daf.py 2>&1 | tee _docs/exp104_output.txt
echo ""

echo "=== All experiments complete ==="
echo "Finished: $(date)"
