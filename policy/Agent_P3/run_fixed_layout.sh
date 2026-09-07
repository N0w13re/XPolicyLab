#!/usr/bin/env bash
# Run P3 on one layout range. No policy GPU: P3 serves no VLA, so the only
# inference is the API call and the simulator gets the machine to itself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

layout=${1:-0}
env_gpu=${2:-0}
task_name=${3:-arrange_largest_number}
eval_env=${4:-uv}
run_id=${ROBODOJO_RUN_ID:-p3-${task_name}-layout${layout//,/_}}

if [[ -z "${P3_MODEL:-}" ]]; then
    echo "[P3][ERROR] set P3_MODEL, e.g. anthropic/claude-sonnet-4-20250514" >&2
    exit 1
fi

export ROBODOJO_RUN_ID="${run_id}"
export ROBODOJO_UNTILED_CAMERAS="${ROBODOJO_UNTILED_CAMERAS:-0}"
export ROBODOJO_PATH_TRACING="${ROBODOJO_PATH_TRACING:-1}"
export ROBODOJO_NUM_ENVS=1
export ROBODOJO_ACTION_TYPE="${P3_ACTION_TYPE:-joint}"
export ROBODOJO_SIM_ENV="${ROBODOJO_ROOT:-${XPL_ROOT}/../RoboDojo-eval}/.venv"
export P3_TRACE_DIR="${P3_TRACE_DIR:-/tmp/xpolicylab-p3/${run_id}}"
export P3_MAX_CHUNK="${P3_MAX_CHUNK:-1}"
mkdir -p "${P3_TRACE_DIR}"

echo "[P3] model=${P3_MODEL} chunk=${P3_MAX_CHUNK} trace=${P3_TRACE_DIR}"

bash "${XPL_ROOT}/scripts/run_robodojo_layout_range.sh" \
    Agent_P3 "${task_name}" "${layout}" \
    --policy-gpu "${env_gpu}" --env-gpu "${env_gpu}" \
    --eval-env "${eval_env}"
