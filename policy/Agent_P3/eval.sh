#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_uv_env=${9:-uv}
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
export P3_TASK_NAME="${task_name}"
export P3_LAYOUT_ID="${P3_LAYOUT_ID:-${seed}}"
export P3_ACTION_TYPE="${action_type}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export P3_TRACE_DIR="${P3_TRACE_DIR:-${XPL_ROOT}/experiments/l5-inspect-agent/${task_name}/seed-${seed}/${ROBODOJO_RUN_ID}}"
additional_info="condition=L5-inspect-agent-v0.26.0,model=${P3_MODEL:-},wire=${P3_WIRE:-auto},images=${P3_IMAGES:-always},depth=${P3_DEPTH:-render},image_horizon=${P3_IMAGE_HORIZON:-2},max_speed_frac=${P3_MAX_SPEED_FRAC:-0.1},max_llm_calls=${P3_MAX_LLM_CALLS:-100},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM -- -"${SERVER_PID}" 2>/dev/null \
            || kill "${SERVER_PID}" 2>/dev/null \
            || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start L5 inspect-agent policy server, port=${policy_server_port}"
setsid bash "${SERVER_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_uv_env}" \
    "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" \
    "Policy server" 1200

echo "[MAIN] start L5 client, server=${policy_server_ip}:${policy_server_port}"
bash "${CLIENT_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] L5 eval finished"
