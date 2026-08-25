#!/usr/bin/env bash
# Wait for the in-flight Pi_05 seed-0 sweep (and optional Traj retry) then continue
# the official protocol for G05 and Xiaomi_Robotics_1 on the same GPUs.
#
# The first Pi_05 sweep started before Assets/Traj finished downloading. Those three
# tasks are retried separately on an idle GPU; this sequencer waits for that PID
# file so it does not launch a second Isaac Sim on the same card.
set -euo pipefail

XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${LOGDIR:-${XPL_ROOT}/experiments/robodojo-official-2026-08-25/logs}"
PI05_PID="${PI05_PID:-216753}"
TRAJ_PID_FILE="${TRAJ_PID_FILE:-/tmp/pi05-traj-all.pid}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
POLICIES="${POLICIES:-G05,Xiaomi_Robotics_1}"

wait_pid() {
  local pid="$1" label="$2"
  [[ -n "${pid}" ]] || return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[chain] ${label} pid ${pid} already exited"
    return 0
  fi
  echo "[chain] waiting for ${label} (pid ${pid})"
  while kill -0 "${pid}" 2>/dev/null; do sleep 60; done
  echo "[chain] ${label} finished at $(date -Is)"
}

wait_pid "${PI05_PID}" "Pi_05 seed0 sweep"

if [[ -f "${TRAJ_PID_FILE}" ]]; then
  wait_pid "$(cat "${TRAJ_PID_FILE}")" "Pi_05 Traj retry"
fi

failed_tasks_from_log() {
  local log="$1" summary
  summary="$(sed -n 's/^\[smoke_all_tasks\] markdown=//p' "${log}" | tail -1)"
  [[ -n "${summary}" && -f "${summary}" ]] || return 0
  awk -F'|' '$2 ~ /FAIL/ {gsub(/[ `]/, "", $3); print $3}' "${summary}" | paste -sd,
}

RETRY="$(failed_tasks_from_log "${LOGDIR}/Pi_05-seed0.log" || true)"
# Traj trio is handled by the GPU-1 retry; skip them if they already have a later result.
echo "[chain] Pi_05 markdown FAIL tasks: ${RETRY:-none}"
if [[ -n "${RETRY}" ]]; then
  set +e
  bash "${XPL_ROOT}/scripts/run_robodojo_sim_eval.sh" benchmark Pi_05 \
    --eval-num native --seed 0 --only "${RETRY}" \
    --policy-gpu-ids "${GPU_IDS}" --env-gpu-ids "${GPU_IDS}" \
    > "${LOGDIR}/Pi_05-seed0-retry.log" 2>&1
  echo "[chain] Pi_05 retry rc=$?"
  set -e
fi

bash "${XPL_ROOT}/scripts/run_robodojo_official_protocol.sh" \
  --seeds 0 --policies "${POLICIES}" --gpu-ids "${GPU_IDS}" \
  --log-dir "${LOGDIR}"
echo "[chain] all done at $(date -Is)"
