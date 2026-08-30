#!/bin/bash
# Start the P1 Qwen locator and export P1_LOCATOR_URL. Caller owns cleanup via LOCATOR_PID.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

if [[ "${EVAL_ENV_TYPE:-sim}" == "debug" ]]; then
  return 0 2>/dev/null || exit 0
fi

locator_port=$(bash "${UTILS_DIR}/get_free_port.sh")
locator_python="${P1_LOCATOR_PYTHON:-${SCRIPT_DIR}/.locator-venv/bin/python}"
locator_gpu="${P1_LOCATOR_GPU:-2}"
locator_model="${P1_LOCATOR_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"

if [[ ! -x "${locator_python}" ]]; then
  echo "[MAIN][ERROR] Locator environment missing: ${locator_python}" >&2
  echo "[MAIN][ERROR] Run: bash ${SCRIPT_DIR}/install.sh" >&2
  exit 1
fi

echo "[MAIN] start locator on GPU ${locator_gpu}, port ${locator_port}"
setsid env CUDA_VISIBLE_DEVICES="${locator_gpu}" \
  "${locator_python}" "${SCRIPT_DIR}/locator_server.py" \
  --model-path "${locator_model}" \
  --port "${locator_port}" &
LOCATOR_PID=$!
export P1_LOCATOR_URL="http://127.0.0.1:${locator_port}"

"${locator_python}" - "${P1_LOCATOR_URL}" "${LOCATOR_PID}" <<'PY'
import sys
import time
from urllib.request import urlopen

url, pid = sys.argv[1], int(sys.argv[2])
deadline = time.time() + 1200
while time.time() < deadline:
    try:
        with urlopen(f"{url}/health", timeout=2) as response:
            if response.status == 200:
                print(f"[MAIN] locator ready at {url}")
                break
    except Exception:
        pass
    try:
        import os
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"Locator exited before becoming ready: {exc}")
    time.sleep(2)
else:
    raise SystemExit("Timed out waiting for locator.")
PY
