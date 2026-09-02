#!/usr/bin/env bash
# Evaluate a chosen RoboDojo layout, or a chosen range, instead of the first N.
#
# RoboDojo selects layouts by index from the front, which is limiting in two
# ways: iterating needs the layout closest to succeeding rather than layout 0,
# and covering a later range needs to re-run everything before it. Its resume
# manifest can exclude layouts, so a manifest that abandons everything outside
# the request leaves exactly the wanted layouts. Excluded layouts are recorded
# as abandoned rather than completed so they stay out of the result details.
#
# Usage:
#   scripts/run_robodojo_single_layout.sh <POLICY> <TASK> <LAYOUTS> [extra eval args...]
#
# LAYOUTS is a single index (`16`), an inclusive range (`20-39`), or a
# comma-separated mix of those (`1-3,5,12-14`).
#
# Environment:
#   ROBODOJO_ROOT   RoboDojo checkout (default: ../RoboDojo-eval)
#   ROBODOJO_RUN_ID run identifier; required, since the manifest is keyed by it
#   EVAL_SEED       evaluation seed selecting the layout set (default: 0)
#   ENV_CFG         env config name (default: arx_x5)
#   ACTION_TYPE     policy action type (default: joint)
#   CKPT_NAME       checkpoint name (default: sim)

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <POLICY> <TASK> <LAYOUTS> [extra eval args...]" >&2
  exit 2
fi

policy_name="$1"
task_name="$2"
layout_spec="$3"
shift 3

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robodojo_root="${ROBODOJO_ROOT:-$(cd "${repo_root}/.." && pwd)/RoboDojo-eval}"
eval_seed="${EVAL_SEED:-0}"
env_cfg="${ENV_CFG:-arx_x5}"
action_type="${ACTION_TYPE:-joint}"
ckpt_name="${CKPT_NAME:-sim}"

if [[ -z "${ROBODOJO_RUN_ID:-}" ]]; then
  echo "ROBODOJO_RUN_ID must be set; the resume manifest is keyed by it." >&2
  exit 2
fi

layout_dir="${robodojo_root}/Assets/Eval_Layout/RoboDojo/${env_cfg}/${eval_seed}"
if [[ ! -d "${layout_dir}" ]]; then
  echo "No layout directory at ${layout_dir}" >&2
  exit 1
fi

result_dir="${robodojo_root}/eval_result/RoboDojo/${task_name}/${policy_name}/${env_cfg}/${eval_seed}_ckpt_name=${ckpt_name},action_type=${action_type}"
manifest="${result_dir}/_resume_${ROBODOJO_RUN_ID}.json"

mkdir -p "${result_dir}"
python3 - "${layout_dir}" "${task_name}" "${layout_spec}" "${manifest}" \
  "${ROBODOJO_RUN_ID}" "${result_dir}" "${policy_name}" "${env_cfg}" \
  "${eval_seed}" "${ckpt_name}" "${action_type}" <<'PY'
import json
import re
import sys
from pathlib import Path

(
    layout_dir,
    task_name,
    layout_spec,
    manifest_path,
    run_id,
    result_dir,
    policy_name,
    env_cfg,
    eval_seed,
    ckpt_name,
    action_type,
) = sys.argv[1:12]

pattern = re.compile(rf"{re.escape(task_name)}_\d+\.json")
count = sum(1 for p in Path(layout_dir).iterdir() if pattern.fullmatch(p.name))
if count == 0:
    raise SystemExit(f"No layouts for {task_name} under {layout_dir}")

wanted_list = []
seen = set()
for part in layout_spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        first_s, last_s = part.split("-", 1)
        first, last = int(first_s), int(last_s)
        if last < first:
            raise SystemExit(f"empty layout range {part}")
        chunk = list(range(first, last + 1))
    else:
        chunk = [int(part)]
    for layout_id in chunk:
        if not 0 <= layout_id < count:
            raise SystemExit(f"layout {layout_id} outside 0..{count - 1}")
        if layout_id not in seen:
            seen.add(layout_id)
            wanted_list.append(layout_id)
if not wanted_list:
    raise SystemExit(f"empty layout spec {layout_spec!r}")
wanted = set(wanted_list)

payload = {
    "run_id": run_id,
    "save_dir": f"{result_dir}/{run_id}",
    "task_name": task_name,
    "policy_name": policy_name,
    "config_name": env_cfg,
    "eval_seed": int(eval_seed),
    "additional_info": f"ckpt_name={ckpt_name},action_type={action_type}",
    "success_nums": 0,
    "fail_nums": 0,
    "unstable_nums": 0,
    "total_score": 0.0,
    "completed_layout_ids": [],
    "abandoned_layout_ids": [i for i in range(count) if i not in wanted],
    "details": {},
    "restart_count": 0,
}
Path(manifest_path).write_text(json.dumps(payload, indent=2))
print(f"[layout-select] {manifest_path}: keeping {wanted_list} of {count}")
PY

layout_count="$(python3 -c "
spec = '${layout_spec}'
ids = []
seen = set()
for part in spec.split(','):
    part = part.strip()
    if not part:
        continue
    if '-' in part:
        first, last = (int(x) for x in part.split('-', 1))
        chunk = range(first, last + 1)
    else:
        chunk = [int(part)]
    for layout_id in chunk:
        if layout_id not in seen:
            seen.add(layout_id)
            ids.append(layout_id)
print(len(ids))
")"

exec bash "${repo_root}/scripts/run_robodojo_sim_eval.sh" eval "${policy_name}" \
  --task "${task_name}" --eval-num "${layout_count}" --seed "${eval_seed}" "$@"
