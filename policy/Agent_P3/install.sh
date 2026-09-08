#!/usr/bin/env bash
# P3 needs no VLA checkpoint or policy GPU. It deliberately installs the
# published policy rather than carrying a behaviorally drifting fork.
set -euo pipefail

python_bin=${1:-python}
"${python_bin}" -m pip install \
  "inspect-robots==0.58.0" \
  "inspect-robots-agent==0.26.0"

echo "[INSTALL] Agent_P3 uses inspect-robots-agent 0.26.0 / core 0.58.0."
echo "[INSTALL] Set P3_MODEL and the matching provider key before running."
