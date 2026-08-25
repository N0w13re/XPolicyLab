#!/usr/bin/env bash
# After the in-flight Pi_05 seed-0 sweep: retry non-Traj failures, start G05 on
# GPUs that Traj is not using, wait for Traj, then Xiaomi on all 8 cards.
set -euo pipefail

XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${LOGDIR:-${XPL_ROOT}/experiments/robodojo-official-2026-08-25/logs}"
PI05_PID="${PI05_PID:-216753}"
TRAJ_PID_FILE="${TRAJ_PID_FILE:-/tmp/pi05-traj-all.pid}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TRAJ_GPUS="${TRAJ_GPUS:-0,1}"
TRAJ_TASKS="imitate_sorting_sequence,make_kong,play_tic_tac_toe"

wait_pid() {
  local pid="$1" label="$2"
  pid="$(echo "${pid}" | awk '{print $1}')"
  [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[chain] ${label} pid ${pid} already exited"
    return 0
  fi
  echo "[chain] waiting for ${label} (pid ${pid})"
  while kill -0 "${pid}" 2>/dev/null; do sleep 60; done
  echo "[chain] ${label} finished at $(date -Is)"
}

traj_alive() {
  [[ -f "${TRAJ_PID_FILE}" ]] || return 1
  local pid
  pid="$(head -1 "${TRAJ_PID_FILE}" | awk '{print $1}')"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

free_gpus_while_traj() {
  if traj_alive; then
    echo "${GPU_IDS}" | tr ',' '\n' | grep -vxE "$(echo "${TRAJ_GPUS}" | tr ',' '|')" | paste -sd,
  else
    echo "${GPU_IDS}"
  fi
}

failed_tasks_from_log() {
  local log="$1" summary
  summary="$(sed -n 's/^\[smoke_all_tasks\] markdown=//p' "${log}" | tail -1)"
  [[ -n "${summary}" && -f "${summary}" ]] || return 0
  awk -F'|' '$2 ~ /FAIL/ {gsub(/[ `]/, "", $3); print $3}' "${summary}" | paste -sd,
}

strip_traj_tasks() {
  local failed="$1"
  python3 - "${failed}" "${TRAJ_TASKS}" <<'PY'
import sys
failed = [t for t in sys.argv[1].split(",") if t]
skip = set(sys.argv[2].split(","))
print(",".join(t for t in failed if t not in skip))
PY
}

run_benchmark() {
  local policy="$1" gpus="$2" logfile="$3"
  shift 3
  echo "[chain] ${policy} gpus=${gpus} extra=$* log=${logfile}"
  set +e
  bash "${XPL_ROOT}/scripts/run_robodojo_sim_eval.sh" benchmark "${policy}" \
    --eval-num native --seed 0 \
    --policy-gpu-ids "${gpus}" --env-gpu-ids "${gpus}" \
    "$@" > "${logfile}" 2>&1
  echo "[chain] ${policy} rc=$? gpus=${gpus} at $(date -Is)"
  set -e
}

wait_pid "${PI05_PID}" "Pi_05 seed0 sweep"

RETRY="$(failed_tasks_from_log "${LOGDIR}/Pi_05-seed0.log" || true)"
RETRY="$(strip_traj_tasks "${RETRY:-}")"
echo "[chain] Pi_05 FAIL excluding Traj trio: ${RETRY:-none}"
if [[ -n "${RETRY}" ]]; then
  run_benchmark Pi_05 "$(free_gpus_while_traj)" "${LOGDIR}/Pi_05-seed0-retry.log" --only "${RETRY}"
fi

G05_GPUS="$(free_gpus_while_traj)"
echo "[chain] starting G05 on ${G05_GPUS} (Traj still using ${TRAJ_GPUS} if coordinator is alive)"
run_benchmark G05 "${G05_GPUS}" "${LOGDIR}/G05-seed0.log"
G05_RETRY="$(failed_tasks_from_log "${LOGDIR}/G05-seed0.log" || true)"
if [[ -n "${G05_RETRY}" ]]; then
  run_benchmark G05 "$(free_gpus_while_traj)" "${LOGDIR}/G05-seed0-retry.log" --only "${G05_RETRY}"
fi

wait_pid "$(head -1 "${TRAJ_PID_FILE}" 2>/dev/null || true)" "Pi_05 Traj retry"

run_benchmark Xiaomi_Robotics_1 "${GPU_IDS}" "${LOGDIR}/Xiaomi_Robotics_1-seed0.log"

echo "[chain] aggregating"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)/RoboDojo-eval}"
( cd "${ROBODOJO_ROOT}" && python3 scripts/internal/summarize_result.py ) || true
python3 "${XPL_ROOT}/scripts/compare_robodojo_to_official.py" \
  --eval-root "${ROBODOJO_ROOT}" --seed 0 \
  --json-out "${LOGDIR}/../results/compare-seed0.json"
echo "[chain] all done at $(date -Is)"
