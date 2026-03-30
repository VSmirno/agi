#!/usr/bin/env bash
# run_stage18.sh — Autonomous Stage 18 runner for minipc (AMD ROCm)
#
# Usage:
#   bash scripts/run_stage18.sh [device]
#
# Default device: cuda (AMD ROCm via HIP)
# Runs exp41, exp42, exp43 in sequence, saves results, commits and pushes.
#
# Prerequisites (minipc):
#   cd /opt/agi && git pull origin main

set -euo pipefail

DEVICE="${1:-cuda}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results/stage18"
LOG_FILE="${RESULTS_DIR}/run_stage18.log"
VENV="${REPO_DIR}/venv"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH="${REPO_DIR}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "${VENV}/bin/activate"

mkdir -p "${RESULTS_DIR}"

echo "========================================"   | tee "${LOG_FILE}"
echo "Stage 18 runner — $(date)"                  | tee -a "${LOG_FILE}"
echo "device: ${DEVICE}"                          | tee -a "${LOG_FILE}"
echo "repo:   ${REPO_DIR}"                        | tee -a "${LOG_FILE}"
echo "========================================"   | tee -a "${LOG_FILE}"

cd "${REPO_DIR}"
git pull origin main 2>&1 | tee -a "${LOG_FILE}"

# ---------------------------------------------------------------------------
# Helper: run one experiment
# spawn-safe: write runner to a real .py file (not stdin) so multiprocessing
# spawn can re-import it in worker processes.
# ---------------------------------------------------------------------------
run_exp() {
    local name="$1"
    local module="$2"
    local runner="/tmp/run_${name}.py"

    cat > "${runner}" <<PYEOF
import os, json, sys, importlib
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "${REPO_DIR}/src")

if __name__ == "__main__":
    mod = importlib.import_module("${module}")
    result = mod.run(device="${DEVICE}")

    out_path = "${RESULTS_DIR}/${name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    status = "PASS" if result.get("passed") else "FAIL"
    print(f"\\n{status}")
    sys.exit(0 if result.get("passed") else 1)
PYEOF

    echo "" | tee -a "${LOG_FILE}"
    echo "--- ${name} start: $(date) ---" | tee -a "${LOG_FILE}"

    python "${runner}" 2>&1 | tee -a "${LOG_FILE}"; local ec=${PIPESTATUS[0]}

    echo "--- ${name} end: $(date) exit=${ec} ---" | tee -a "${LOG_FILE}"
    return ${ec}
}

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
EXP41_STATUS=0
EXP42_STATUS=0
EXP43_STATUS=0

run_exp "exp41" "snks.experiments.exp41_multi_env_baseline" || EXP41_STATUS=$?
run_exp "exp42" "snks.experiments.exp42_transfer_matrix"    || EXP42_STATUS=$?
run_exp "exp43" "snks.experiments.exp43_continual_multitask" || EXP43_STATUS=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG_FILE}"
echo "========================================"   | tee -a "${LOG_FILE}"
echo "Stage 18 Summary — $(date)"                 | tee -a "${LOG_FILE}"
echo "  exp41: $([ $EXP41_STATUS -eq 0 ] && echo PASS || echo FAIL)" | tee -a "${LOG_FILE}"
echo "  exp42: $([ $EXP42_STATUS -eq 0 ] && echo PASS || echo FAIL)" | tee -a "${LOG_FILE}"
echo "  exp43: $([ $EXP43_STATUS -eq 0 ] && echo PASS || echo FAIL)" | tee -a "${LOG_FILE}"
echo "========================================"   | tee -a "${LOG_FILE}"

ALL_PASSED=$(( EXP41_STATUS + EXP42_STATUS + EXP43_STATUS ))

python - <<EOF
import json, os
results = {}
for name in ["exp41", "exp42", "exp43"]:
    path = "${RESULTS_DIR}/" + name + ".json"
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)
    else:
        results[name] = {"passed": False, "error": "file not found"}

summary = {
    "stage": 18,
    "all_passed": all(r.get("passed") for r in results.values()),
    "experiments": results,
}
with open("${RESULTS_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary written to ${RESULTS_DIR}/summary.json")
EOF

# ---------------------------------------------------------------------------
# Commit and push results
# ---------------------------------------------------------------------------
git add "${RESULTS_DIR}/" results/ reports/ checkpoints/ 2>/dev/null || true
git commit -m "$(cat <<'COMMITMSG'
Stage 18 results: exp41/exp42/exp43 on AMD minipc

Auto-committed by run_stage18.sh

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
COMMITMSG
)"
git push origin main

echo "" | tee -a "${LOG_FILE}"
echo "Results committed and pushed." | tee -a "${LOG_FILE}"

exit ${ALL_PASSED}
