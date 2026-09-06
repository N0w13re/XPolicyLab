"""P2 primitive surface built from the tested RPent environment executor."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.tools import (
    RpentPrimitives,
    enable_camera_calibration,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.geometry import bbox_to_pixels


def normalized_xy_to_pixel_rc(
    points: list[float] | list[list[float]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[list[int]]:
    """Convert Qwen's 0..1000 [x,y] points into image [row,col] pixels."""
    array = np.asarray(points, dtype=np.float64)
    if array.shape == (2,):
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("points must be one [x,y] pair or a list of finite pairs")
    height, width = int(image_shape[0]), int(image_shape[1])
    xy = np.clip(array, 0.0, 1000.0)
    cols = np.clip(np.rint(xy[:, 0] * width / 1000.0), 0, width - 1)
    rows = np.clip(np.rint(xy[:, 1] * height / 1000.0), 0, height - 1)
    return np.stack([rows, cols], axis=1).astype(int).tolist()


class P2Primitives(RpentPrimitives):
    """Allow only explicit atomic motion; learned contact helpers are disabled."""

    def sample_world_xyz(
        self,
        view: str,
        pixels: list[float] | list[list[float]],
        step: int = -1,
        radius: int = 2,
    ) -> dict[str, Any]:
        record = self.env_states.get(step)
        pixel_rc = normalized_xy_to_pixel_rc(
            pixels, record.views[view].world_xyz.shape
        )
        result = super().sample_world_xyz(view, pixel_rc, step, radius)
        result["input_points_1000_xy"] = np.asarray(pixels).tolist()
        return result

    def query_world_map(
        self,
        view: str,
        bbox: list[int],
        step: int = -1,
    ) -> dict[str, Any]:
        record = self.env_states.get(step)
        bbox_rc = bbox_to_pixels(bbox, record.views[view].world_xyz.shape)
        result = super().query_world_map(view, list(bbox_rc), step)
        result["input_bbox_1000_xyxy"] = list(bbox)
        return result

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
