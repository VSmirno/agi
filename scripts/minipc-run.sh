#!/bin/bash
# Universal launcher for running Python scripts on minipc via tmux.
#
# Usage:
#   ./scripts/minipc-run.sh <session_name> <python_command>
#
# Examples:
#   ./scripts/minipc-run.sh exp97 "from snks.experiments.exp97_pure_daf import main; main()"
#   ./scripts/minipc-run.sh exp98 "from snks.experiments.exp98_foo import main; main()"
#   ./scripts/minipc-run.sh test  "import pytest; pytest.main(['-x', '-q', 'tests/'])"
#
# Features:
#   - Syncs code to minipc first
#   - Kills existing tmux session with same name
#   - Runs in tmux (survives SSH disconnect)
#   - Output tee'd to _docs/<session>_results.txt
#   - Handles torchvision stub, ROCm env vars
#
# Monitor:
#   ssh minipc "tmux attach -t <session>"
#   ssh minipc "tail -f /opt/agi/_docs/<session>_results.txt"

set -euo pipefail

SESSION="${1:?Usage: $0 <session_name> <python_command>}"
PYCMD="${2:?Usage: $0 <session_name> <python_command>}"

REMOTE="minipc"
REMOTE_DIR="/opt/agi"
RESULTS="${REMOTE_DIR}/_docs/${SESSION}_results.txt"

echo "=== minipc-run: ${SESSION} ==="

# 1. Sync code
echo ">>> Syncing code..."
rsync -az --delete \
    --exclude='.venv' --exclude='venv' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.pyc' --exclude='.env' \
    /home/yorick/agi/ ${REMOTE}:${REMOTE_DIR}/ 2>&1 | tail -2

# 2. Kill old session if exists
ssh ${REMOTE} "tmux kill-session -t ${SESSION} 2>/dev/null || true"

# 3. Create runner script on remote
ssh ${REMOTE} "cat > ${REMOTE_DIR}/_docs/_run_${SESSION}.sh << 'SCRIPT'
#!/bin/bash
cd ${REMOTE_DIR}
source venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH=${REMOTE_DIR}/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -u -c '
import sys
try:
    import torchvision
except ImportError:
    sys.modules[\"torchvision\"] = type(sys)(\"torchvision\")
    sys.modules[\"torchvision.datasets\"] = type(sys)(\"torchvision.datasets\")

${PYCMD}
' 2>&1 | tee ${RESULTS}

echo ''
echo '=== DONE ==='
SCRIPT
chmod +x ${REMOTE_DIR}/_docs/_run_${SESSION}.sh"

# 4. Launch in tmux
ssh ${REMOTE} "tmux new-session -d -s ${SESSION} 'bash ${REMOTE_DIR}/_docs/_run_${SESSION}.sh'"

echo ""
echo ">>> Launched in tmux session: ${SESSION}"
echo ""
echo "Monitor:"
echo "  ssh minipc \"tmux attach -t ${SESSION}\""
echo "  ssh minipc \"tail -f ${RESULTS}\""
echo ""
echo "Kill:"
echo "  ssh minipc \"tmux kill-session -t ${SESSION}\""
