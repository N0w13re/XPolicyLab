"""P2 primitive surface built from the tested RPent environment executor."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.tools import (
    RpentPrimitives,
    enable_camera_calibration,
)


class P2Primitives(RpentPrimitives):
    """Allow only explicit atomic motion; learned contact helpers are disabled."""

    def move_to(
        self,
        xyz: list[float] | None = None,
        arm: str | None = None,
        gripper: float | None = None,
        quat: list[float] | None = None,
        substeps: int = 25,
    ) -> dict[str, Any]:
        if quat is None:
            raise ValueError("P2 move_to requires an explicit quat")
        quaternion = np.asarray(quat, dtype=np.float32).reshape(-1)
        if quaternion.size != 4 or not np.isfinite(quaternion).all():
            raise ValueError("quat must contain four finite values")
        if float(np.linalg.norm(quaternion)) <= 1e-8:
            raise ValueError("quat must have non-zero norm")
        return super().move_to(
            xyz=xyz,
            arm=arm,
            gripper=gripper,
            quat=(quaternion / np.linalg.norm(quaternion)).tolist(),
            substeps=substeps,
        )

    def pregrasp(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("pregrasp is disabled in the P2 atomic executor")

    def pi05_act(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Pi_05 actions are disabled in the P2 atomic executor")

    def pi05_pick(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Pi_05 actions are disabled in the P2 atomic executor")
