#!/bin/bash
# Exp97 fix: run with curiosity-driven action selection on GPU (50K nodes)
# Stage 38_fix: TD-001 verification
cd /opt/agi
source venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH=/opt/agi/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -u -c '
import sys
try:
    import torchvision
except ImportError:
    sys.modules["torchvision"] = type(sys)("torchvision")
    sys.modules["torchvision.datasets"] = type(sys)("torchvision.datasets")

from snks.experiments.exp97_pure_daf import main
main()
' 2>&1 | tee /opt/agi/_docs/exp97_fix_gpu_results.txt
