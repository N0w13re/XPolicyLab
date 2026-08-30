#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_DIR="${SCRIPT_DIR}/../Pi_05"
LOCATOR_VENV="${SCRIPT_DIR}/.locator-venv"

if [[ ! -x "${PI05_DIR}/openpi/.venv/bin/python" ]]; then
    echo "[INSTALL] Pi_05 environment is missing; installing it first."
    bash "${PI05_DIR}/install.sh"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[INSTALL][ERROR] uv is required." >&2
    exit 1
fi

uv venv "${LOCATOR_VENV}" --python 3.11
uv pip install --python "${LOCATOR_VENV}/bin/python" \
    torch torchvision transformers accelerate pillow

if [[ "${P1_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
    # Some managed hosts include a bare "::1" entry in no_proxy. New httpx
    # rejects that entry while constructing proxy mounts, before any request.
    env -u no_proxy -u NO_PROXY "${LOCATOR_VENV}/bin/hf" download \
        "${P1_LOCATOR_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
fi

echo "[INSTALL] P1-gaze locator environment is ready at ${LOCATOR_VENV}."
