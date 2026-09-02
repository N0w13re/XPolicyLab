#!/usr/bin/env bash
# When a general_pickup shard GPU goes free, fill layouts the live shards
# already skipped or will never cover. Does not steal in-flight layouts.
set -euo pipefail

XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_DIR="${XPL_ROOT}/policy/Pi_05_Agent_P1_RPent"
PLAN="${XPL_ROOT}/scripts/plan_robodojo_layout_fill.py"
SESSION="${FILL_SESSION:-rpent-general-pick-fill}"
INTERVAL="${FILL_POLL_SECONDS:-30}"

copy_gpt_key_from_live_eval() {
  python3 - <<'PY'
from pathlib import Path
for pid_dir in Path("/proc").iterdir():
    if not pid_dir.name.isdigit():
        continue
    try:
        cmd = (pid_dir / "cmdline").read_bytes()
    except OSError:
        continue
    if b"eval_client/main.py" not in cmd:
        continue
    try:
        raw = (pid_dir / "environ").read_bytes()
    except OSError:
        continue
    for item in raw.split(b"\0"):
        if item.startswith(b"RPENT_GPT_API_KEY="):
            print(item.split(b"=", 1)[1].decode(), end="")
            raise SystemExit(0)
raise SystemExit(1)
PY
}

launch_fill() {
  local gpu="$1" layout="$2"
  local run_id="2026-09-02_gpick-fill-${layout}"
  local window="g${gpu}-fill-${layout}"
  local key
  key="$(copy_gpt_key_from_live_eval || true)"
  if [[ -z "${key}" && -z "${RPENT_GPT_API_KEY:-}" ]]; then
    echo "[fill] no GPT key available; cannot launch layout ${layout}" >&2
    return 1
  fi
  tmux -f /exec-daemon/tmux.portal.conf has-session -t "${SESSION}" 2>/dev/null \
    || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "${SESSION}" -n "${window}" -- bash -l
  if ! tmux -f /exec-daemon/tmux.portal.conf list-windows -t "${SESSION}" -F '#{window_name}' \
      | grep -qx "${window}"; then
    tmux -f /exec-daemon/tmux.portal.conf new-window -t "${SESSION}" -n "${window}" -- bash -l
  fi
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "${SESSION}:${window}" C-c
  sleep 1
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "${SESSION}:${window}" \
    "cd ${XPL_ROOT} && export ROBODOJO_RUN_ID=${run_id} EVAL_SEED=0 ROBODOJO_UNTILED_CAMERAS=0 ROBODOJO_PATH_TRACING=0 ROBODOJO_NUM_ENVS=1 RPENT_TRACE_EPISODE_DIRS=1 RPENT_TRACE_DIR=/tmp/xpolicylab-rpent/${run_id} RPENT_LLM_BACKEND=gpt RPENT_GPT_MAX_RETRIES=200 RPENT_GPT_RETRY_BUDGET_S=14400 RPENT_GPT_RETRY_CAP_S=120 RPENT_GPT_MAX_TOKENS=4096 ROBODOJO_SIM_ENV=/mnt/bn/robotics-data-mx/wenbo/RoboDojo-eval/.venv ROBODOJO_POLICY_ENV=uv && export RPENT_GPT_API_KEY='${key:-${RPENT_GPT_API_KEY}}' && bash ${POLICY_DIR}/run_fixed_layout.sh ${layout} ${gpu} ${gpu} uv general_pickup" \
    C-m
  echo "[fill] launched layout ${layout} on GPU ${gpu} run_id=${run_id}"
}

echo "[fill] watching for free GPUs every ${INTERVAL}s"
while true; do
  plan_json="$(python3 "${PLAN}" --json)"
  echo "${plan_json}" | python3 -c "import json,sys; d=json.load(sys.stdin); print('[fill]', 'evaluated', len(d['evaluated']), 'missing', d['missing'], 'fillable', d['fillable'], 'free', d['free_gpus'], 'assign', d['assignments'], flush=True)"
  python3 - "${plan_json}" <<'PY' | while read -r gpu layout; do
import json, sys
for item in json.loads(sys.argv[1]).get("assignments", []):
    print(item["gpu"], item["layout_id"])
PY
    launch_fill "${gpu}" "${layout}"
  done
  evaluated="$(echo "${plan_json}" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['evaluated']))")"
  missing="$(echo "${plan_json}" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['missing']))")"
  occupied="$(echo "${plan_json}" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['occupied_gpus']))")"
  if [[ "${evaluated}" -ge 50 && "${missing}" -eq 0 && "${occupied}" -eq 0 ]]; then
    echo "[fill] 0-49 covered and no eval clients remain; exiting"
    break
  fi
  sleep "${INTERVAL}"
done
