#!/usr/bin/env bash
# run_stage17.sh — Autonomous Stage 17 runner for minipc (AMD ROCm)
#
# Usage:
#   bash scripts/run_stage17.sh [device]
#
# Default device: cuda (AMD ROCm via HIP)
# Runs exp38, exp39, exp40 in sequence, saves JSON results, commits and pushes.
#
# Prerequisites (minipc):
#   cd /opt/agi && git pull origin main
#   source venv/bin/activate

set -euo pipefail

DEVICE="${1:-cuda}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results/stage17"
LOG_FILE="${RESULTS_DIR}/run_stage17.log"
VENV="${REPO_DIR}/venv"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH="${REPO_DIR}/src"

source "${VENV}/bin/activate"

mkdir -p "${RESULTS_DIR}"

echo "========================================"   | tee "${LOG_FILE}"
echo "Stage 17 runner — $(date)"                  | tee -a "${LOG_FILE}"
echo "device: ${DEVICE}"                          | tee -a "${LOG_FILE}"
echo "repo:   ${REPO_DIR}"                        | tee -a "${LOG_FILE}"
echo "========================================"   | tee -a "${LOG_FILE}"

# Ensure we're up to date
cd "${REPO_DIR}"
git pull origin main 2>&1 | tee -a "${LOG_FILE}"

# ---------------------------------------------------------------------------
# Helper: run one experiment, save JSON result
# ---------------------------------------------------------------------------
run_exp() {
    local name="$1"
    local module="$2"
    local out_file="${RESULTS_DIR}/${name}.json"

    echo "" | tee -a "${LOG_FILE}"
    echo "--- ${name} start: $(date) ---" | tee -a "${LOG_FILE}"

    python - <<EOF 2>&1 | tee -a "${LOG_FILE}"
import json, sys, importlib
sys.path.insert(0, "${REPO_DIR}/src")

# Disable torch.compile before import
import snks.daf.compiled_step as cs
from snks.daf.compiled_step import _fhn_step_inner as _raw
cs._compiled_cache.clear()
cs._compiled_cache["fn"] = _raw

mod = importlib.import_module("${module}")
result = mod.run(device="${DEVICE}")

with open("${out_file}", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
status = "PASS" if result.get("passed") else "FAIL"
print(f"\\n{status}")
sys.exit(0 if result.get("passed") else 1)
EOF
    local ec=$?
    echo "--- ${name} end: $(date) exit=${ec} ---" | tee -a "${LOG_FILE}"
    return $ec
}

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
EXP38_STATUS=0
EXP39_STATUS=0
EXP40_STATUS=0

run_exp "exp38" "snks.experiments.exp38_scaling_gpu"   || EXP38_STATUS=$?
run_exp "exp39" "snks.experiments.exp39_replay_coverage" || EXP39_STATUS=$?
run_exp "exp40" "snks.experiments.exp40_doorkey8x8"    || EXP40_STATUS=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG_FILE}"
echo "========================================"   | tee -a "${LOG_FILE}"
echo "Stage 17 Summary — $(date)"                 | tee -a "${LOG_FILE}"
echo "  exp38: $([ $EXP38_STATUS -eq 0 ] && echo PASS || echo FAIL)" | tee -a "${LOG_FILE}"
echo "  exp39: $([ $EXP39_STATUS -eq 0 ] && echo PASS || echo FAIL)" | tee -a "${LOG_FILE}"
echo "  exp40: $([ $EXP40_STATUS -eq 0 ] && echo PASS || echo FAIL)" | tee -a "${LOG_FILE}"
echo "========================================"   | tee -a "${LOG_FILE}"

ALL_PASSED=$(( EXP38_STATUS + EXP39_STATUS + EXP40_STATUS ))

# ---------------------------------------------------------------------------
# Write aggregate summary JSON
# ---------------------------------------------------------------------------
python - <<EOF
import json, os
results = {}
for name in ["exp38", "exp39", "exp40"]:
    path = "${RESULTS_DIR}/" + name + ".json"
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)
    else:
        results[name] = {"passed": False, "error": "file not found"}

summary = {
    "stage": 17,
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
git add "${RESULTS_DIR}/"
git commit -m "$(cat <<'COMMITMSG'
Stage 17 results: exp38/exp39/exp40 on AMD minipc

Auto-committed by run_stage17.sh

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
COMMITMSG
)"
git push origin main

echo "" | tee -a "${LOG_FILE}"
echo "Results committed and pushed." | tee -a "${LOG_FILE}"

exit ${ALL_PASSED}
