#!/usr/bin/env bash
# Re-run the three Pi_05 tasks that died because Assets/Traj was still an LFS pointer.
# Bind to one idle GPU so they do not collide with the rest of the seed-0 sweep.
set -euo pipefail

XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${LOGDIR:-${XPL_ROOT}/experiments/robodojo-official-2026-08-25/logs}"
GPU="${GPU:-1}"
mkdir -p "${LOGDIR}"

for task in imitate_sorting_sequence make_kong play_tic_tac_toe; do
  echo "[traj-retry] ${task} gpu=${GPU} at $(date -Is)"
  set +e
  bash "${XPL_ROOT}/scripts/run_robodojo_sim_eval.sh" eval Pi_05 \
    --task "${task}" --eval-num native --seed 0 \
    --policy-gpu "${GPU}" --env-gpu "${GPU}"
  rc=$?
  set -e
  echo "[traj-retry] ${task} rc=${rc} at $(date -Is)"
done
