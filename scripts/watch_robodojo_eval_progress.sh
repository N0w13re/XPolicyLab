#!/usr/bin/env bash
# Dump official-table compare JSON while a live Isaac sweep is running.
# Does not launch or stop eval clients.
set -euo pipefail
XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${XPL_ROOT}/experiments/robodojo-official-2026-08-25/results/pi05-seed0-untiled-compare.json"
LOG="${XPL_ROOT}/experiments/robodojo-official-2026-08-25/logs/progress-watch.log"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-/mnt/bn/robotics-data-mx/wenbo/RoboDojo-eval}"
INTERVAL="${PROGRESS_WATCH_SECONDS:-600}"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
while true; do
  python3 "${XPL_ROOT}/scripts/compare_robodojo_to_official.py" \
    --eval-root "${ROBODOJO_ROOT}" --seed 0 --json-out "$OUT" \
    >>"$LOG" 2>&1 || true
  sleep "$INTERVAL"
done
