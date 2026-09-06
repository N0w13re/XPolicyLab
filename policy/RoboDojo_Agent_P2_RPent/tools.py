"""P2 primitive surface built from the tested RPent environment executor."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.tools import (
    RpentPrimitives,
    enable_camera_calibration,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.geometry import bbox_to_pixels

# Approximate arx_x5 flange-to-fingertip length along a top-down tool axis.
FLANGE_TO_FINGERTIP_M = 0.145
DEFAULT_HOVER_CLEARANCE_M = 0.12


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


def top_down_eef_targets_from_surface(
    surface_xyz: list[float] | np.ndarray,
    *,
    flange_to_fingertip_m: float = FLANGE_TO_FINGERTIP_M,
    hover_clearance_m: float = DEFAULT_HOVER_CLEARANCE_M,
) -> dict[str, list[float]]:
    """Suggest flange XYZ for top-down approach from a measured surface point."""
    surface = np.asarray(surface_xyz, dtype=np.float64).reshape(-1)[:3]
    if surface.size != 3 or not np.isfinite(surface).all():
        raise ValueError("surface_xyz must contain three finite values")
    contact = surface + np.asarray([0.0, 0.0, flange_to_fingertip_m])
    hover = contact + np.asarray([0.0, 0.0, hover_clearance_m])
    return {
        "surface_xyz": surface.round(5).tolist(),
        "suggested_contact_eef_xyz": contact.round(5).tolist(),
        "suggested_hover_eef_xyz": hover.round(5).tolist(),
        "flange_to_fingertip_m": float(flange_to_fingertip_m),
        "hover_clearance_m": float(hover_clearance_m),
    }


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
        samples = result.get("samples") or []
        annotated = []
        for sample in samples:
            item = dict(sample)
            xyz = item.get("xyz")
            if xyz is not None:
                item.update(top_down_eef_targets_from_surface(xyz))
            annotated.append(item)
        if annotated:
            result["samples"] = annotated
            result["eef_target_note"] = (
                "samples[*].xyz is an object/table surface point. For top-down "
                "arx_x5 grasping, move_to must use suggested_hover_eef_xyz first, "
                "then suggested_contact_eef_xyz; never pass surface xyz as the "
                "flange target."
            )
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
        result = super().move_to(
            xyz=xyz,
            arm=arm,
            gripper=gripper,
            quat=(quaternion / np.linalg.norm(quaternion)).tolist(),
            substeps=substeps,
        )
        if result.get("success") is False and xyz is not None:
            advice = top_down_eef_targets_from_surface(xyz)
            result["remediation"] = {
                **advice,
                "note": (
                    "plan_failed often means the commanded flange pose collides "
                    "or is unreachable. If xyz was a surface sample, retry with "
                    "suggested_hover_eef_xyz (then descend to "
                    "suggested_contact_eef_xyz) and a documented top-down quat. "
                    "Do not repeat the identical failed target."
                ),
            }
        return result

    def pregrasp(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("pregrasp is disabled in the P2 atomic executor")

    def pi05_act(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Pi_05 actions are disabled in the P2 atomic executor")

    def pi05_pick(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Pi_05 actions are disabled in the P2 atomic executor")
