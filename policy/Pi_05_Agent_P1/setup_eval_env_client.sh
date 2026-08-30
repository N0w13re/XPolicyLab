#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"

if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" && -z "${P1_LOCATOR_URL:-}" ]]; then
    # shellcheck source=start_locator.sh
    source "${SCRIPT_DIR}/start_locator.sh"
    cleanup_locator() {
        if [[ -n "${LOCATOR_PID:-}" ]]; then
            kill -TERM -- -"${LOCATOR_PID}" 2>/dev/null || kill "${LOCATOR_PID}" 2>/dev/null || true
        fi
    }
    trap cleanup_locator EXIT
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${SCRIPT_DIR}/deploy.yml" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
