#!/usr/bin/env bash
# Finish the two remaining Traj-backed Pi_05 tasks without colliding with the
# in-flight imitate_sorting_sequence job on GPU 1.
#
# GPU 0 is idle (its smoke_all_tasks shard already finished). make_kong runs
# there immediately; play_tic_tac_toe follows on GPU 0 so it overlaps imitate
# (horizon 1600 on GPU 1) instead of waiting for it.
set -euo pipefail

XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${LOGDIR:-${XPL_ROOT}/experiments/robodojo-official-2026-08-25/logs}"
PIDFILE="${TRAJ_PID_FILE:-/tmp/pi05-traj-all.pid}"
IMITATE_PID="${IMITATE_PID:-1419659}"
RETRY_PARENT="${RETRY_PARENT:-1419654}"
mkdir -p "${LOGDIR}"
echo $$ > "${PIDFILE}"
echo "[traj-coord] pid=$$ imitate=${IMITATE_PID} retry_parent=${RETRY_PARENT} at $(date -Is)"

run_task() {
  local task="$1" gpu="$2"
  echo "[traj-coord] start ${task} gpu=${gpu} at $(date -Is)"
  set +e
  bash "${XPL_ROOT}/scripts/run_robodojo_sim_eval.sh" eval Pi_05 \
    --task "${task}" --eval-num native --seed 0 \
    --policy-gpu "${gpu}" --env-gpu "${gpu}" \
    >> "${LOGDIR}/Pi_05-seed0-traj-${task}.log" 2>&1
  echo "[traj-coord] ${task} rc=$? at $(date -Is)"
  set -e
}

# Stop the sequential retry loop so it cannot launch make_kong on GPU 1.
if kill -0 "${RETRY_PARENT}" 2>/dev/null; then
  kill -STOP "${RETRY_PARENT}" 2>/dev/null || true
  echo "[traj-coord] SIGSTOP retry parent ${RETRY_PARENT}"
fi

run_task make_kong 0 &
MAKE_PID=$!
wait "${MAKE_PID}" || true

run_task play_tic_tac_toe 0 &
PLAY_PID=$!

echo "[traj-coord] waiting for imitate pid ${IMITATE_PID}"
while kill -0 "${IMITATE_PID}" 2>/dev/null; do sleep 30; done
echo "[traj-coord] imitate exited at $(date -Is)"

wait "${PLAY_PID}" || true

# Drop the stopped sequential loop; imitate is already finished.
kill -KILL "${RETRY_PARENT}" 2>/dev/null || true
echo "[traj-coord] done at $(date -Is)"
