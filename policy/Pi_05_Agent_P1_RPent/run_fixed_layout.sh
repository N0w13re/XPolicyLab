#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

layout=${1:-16}
policy_gpu=${2:-0}
env_gpu=${3:-1}
eval_env=${4:-uv}
task_name=${5:-classify_objects_by_language}
run_id=${ROBODOJO_RUN_ID:-rpent-${task_name}-layout${layout}}

export ROBODOJO_RUN_ID="${run_id}"
export ROBODOJO_UNTILED_CAMERAS="${ROBODOJO_UNTILED_CAMERAS:-0}"
export ROBODOJO_PATH_TRACING="${ROBODOJO_PATH_TRACING:-1}"
export ROBODOJO_NUM_ENVS=1
export RPENT_TRACE_DIR="${RPENT_TRACE_DIR:-/tmp/xpolicylab-rpent/${run_id}}"
export ROBODOJO_ACTION_TYPE=joint
export ROBODOJO_POLICY_ENV=uv
export ROBODOJO_SIM_ENV="${ROBODOJO_ROOT:-${XPL_ROOT}/../RoboDojo-eval}/.venv"
if [[ "${RPENT_TRACE_EPISODE_DIRS:-0}" == "1" ]]; then
    mkdir -p "${RPENT_TRACE_DIR}"
    python3 - "${layout}" "${RPENT_TRACE_DIR}/episodes.json" <<'PY'
import json
import sys
from pathlib import Path

spec, output = sys.argv[1:3]
first, separator, last = spec.partition("-")
layout_ids = list(range(int(first), int(last) + 1)) if separator else [int(first)]
Path(output).write_text(json.dumps({"layout_ids": layout_ids}, indent=2))
PY
fi

cleanup() {
    if [[ -n "${QWEN_PID:-}" ]]; then
        kill -TERM -- -"${QWEN_PID}" 2>/dev/null \
            || kill "${QWEN_PID}" 2>/dev/null \
            || true
    fi
}
trap cleanup EXIT

if [[ -z "${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-}}" ]]; then
    # shellcheck source=start_local_qwen.sh
    source "${SCRIPT_DIR}/start_local_qwen.sh"
fi

bash "${XPL_ROOT}/scripts/run_robodojo_layout_range.sh" \
    Pi_05_Agent_P1_RPent "${task_name}" "${layout}" \
    --policy-gpu "${policy_gpu}" --env-gpu "${env_gpu}" \
    --eval-env "${eval_env}"
