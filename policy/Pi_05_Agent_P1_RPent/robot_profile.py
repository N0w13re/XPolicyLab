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


PREGRASP_CLEARANCE_MIN_M = 0.12
PREGRASP_CLEARANCE_MAX_M = 0.30
# arx_x5 robot_config.yml: ee_link=link6, gripper_bias=0.145. Under the
# top-down pregrasp quaternion the fingers hang along world -Z, so this
# offset must be added to EEF z or clearance_m is measured at the flange.
DEFAULT_EEF_TCP_OFFSET_M = 0.145


def eef_tcp_offset() -> float:
    """Link6/EEF to TCP distance used when converting surface clearance to EEF z."""
    import os

    return float(os.environ.get("RPENT_EEF_TCP_OFFSET_M", str(DEFAULT_EEF_TCP_OFFSET_M)))


def clamp_pregrasp_clearance(clearance_m: float) -> float:
    """Keep pre-grasp hover in the prompt range of 0.12-0.30 m."""
    return min(
        PREGRASP_CLEARANCE_MAX_M,
        max(PREGRASP_CLEARANCE_MIN_M, float(clearance_m)),
    )


def default_pregrasp_clearance() -> float:
    """Height held above a measured object before Pi_05 takes the contact.

    The added hover is 0.12-0.30 m at the fingertips, according to the
    object's own height. The default is the floor for a short object. The
    tool also adds the EEF-to-TCP offset so the flange does not sit at that
    height.
    """
    import os

    return clamp_pregrasp_clearance(
        float(os.environ.get("RPENT_PREGRASP_CLEARANCE_M", "0.12"))
    )


def pregrasp_quaternion(arm: str) -> np.ndarray:
    try:
        quaternion = _PREGRASP_QUATERNIONS[arm].copy()
    except KeyError as exc:
        raise ValueError("arm must be 'left' or 'right'") from exc
    return quaternion / np.linalg.norm(quaternion)
