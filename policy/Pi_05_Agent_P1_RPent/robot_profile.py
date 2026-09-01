"""Robot/environment capability configuration, separate from task recipes."""

from __future__ import annotations

import numpy as np


_PREGRASP_QUATERNIONS = {
    "left": np.array(
        [-0.61239, 0.353523, -0.61239, -0.353524], dtype=np.float32
    ),
    "right": np.array(
        [-0.353523, 0.61239, -0.353524, -0.61239], dtype=np.float32
    ),
}


def default_clearance() -> float:
    import os

    return float(os.environ.get("RPENT_APPROACH_CLEARANCE_M", "0.20"))


def pregrasp_quaternion(arm: str) -> np.ndarray:
    try:
        quaternion = _PREGRASP_QUATERNIONS[arm].copy()
    except KeyError as exc:
        raise ValueError("arm must be 'left' or 'right'") from exc
    return quaternion / np.linalg.norm(quaternion)
