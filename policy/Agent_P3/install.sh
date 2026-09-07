#!/usr/bin/env bash
# P3 needs no VLA environment: no checkpoint, no JAX, no policy GPU. The wires
# use urllib, so the only optional dependency is Pillow, and policy.py falls
# back to a stdlib PNG encoder when it is absent.
set -euo pipefail

echo "[INSTALL] Agent_P3 has no VLA and no extra dependencies."
echo "[INSTALL] Set P3_MODEL and the matching provider key before running."
