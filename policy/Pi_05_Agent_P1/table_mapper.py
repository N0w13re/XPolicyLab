"""Shared head-camera pixel to RoboDojo table mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


HEAD_CAMERA_CANDIDATES = ("cam_head", "cam_high", "head_camera", "top_camera")


@dataclass(frozen=True)
class TableMapConfig:
    table_z: float = 0.765
    hover_height: float = 0.12
    x_limits: tuple[float, float] = (-0.46, 0.46)
    y_limits: tuple[float, float] = (-0.30, 0.10)


class TablePlaneMapper:
    """Intersect an OpenCV pixel ray with a horizontal world-frame table."""

    def __init__(self, config: TableMapConfig):
        self.config = config

    def pixel_to_table(
        self,
        pixel_xy: Sequence[float],
        intrinsic_matrix: Any,
        camera_to_world: Any,
        y_limits: tuple[float, float] | None = None,
    ) -> np.ndarray:
        intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
        extrinsic = np.asarray(camera_to_world, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 intrinsic matrix, got {intrinsic.shape}.")
        if extrinsic.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera-to-world matrix, got {extrinsic.shape}.")

        pixel = np.array([float(pixel_xy[0]), float(pixel_xy[1]), 1.0], dtype=np.float64)
        ray_opencv = np.linalg.solve(intrinsic, pixel)

        # Isaac/Usd cameras look along -Z with +Y up. Image pixels use the
        # OpenCV convention (+Z forward, +Y down).
        ray_opengl = np.array(
            [ray_opencv[0], -ray_opencv[1], -ray_opencv[2]],
            dtype=np.float64,
        )
        ray_world = extrinsic[:3, :3] @ ray_opengl
        origin_world = extrinsic[:3, 3]

        if ray_world[2] >= -1e-6:
            raise ValueError(f"Pixel ray does not point toward the table: dz={ray_world[2]:.6f}.")
        distance = (self.config.table_z - origin_world[2]) / ray_world[2]
        if distance <= 0:
            raise ValueError(f"Table intersection is behind the camera: distance={distance:.6f}.")

        point = origin_world + distance * ray_world
        self._validate_workspace(point, y_limits=y_limits)
        point[2] = self.config.table_z
        return point

    def hover_target(
        self,
        pixel_xy: Sequence[float],
        intrinsic_matrix: Any,
        camera_to_world: Any,
        hover_height: float | None = None,
        y_limits: tuple[float, float] | None = None,
    ) -> np.ndarray:
        target = self.pixel_to_table(
            pixel_xy,
            intrinsic_matrix,
            camera_to_world,
            y_limits=y_limits,
        )
        height = self.config.hover_height if hover_height is None else hover_height
        target[2] += height
        return target

    def _validate_workspace(
        self,
        point: np.ndarray,
        y_limits: tuple[float, float] | None = None,
    ) -> None:
        xmin, xmax = self.config.x_limits
        ymin, ymax = self.config.y_limits if y_limits is None else y_limits
        if not xmin <= point[0] <= xmax or not ymin <= point[1] <= ymax:
            raise ValueError(
                "Mapped point is outside the shared tabletop workspace: "
                f"xyz={point.tolist()}, x_limits={self.config.x_limits}, "
                f"y_limits={(ymin, ymax)}."
            )


def get_head_camera(observation: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    vision = observation.get("vision", {})
    for name in HEAD_CAMERA_CANDIDATES:
        camera = vision.get(name)
        if isinstance(camera, Mapping):
            return name, camera
    raise KeyError(f"No head camera found; tried {HEAD_CAMERA_CANDIDATES}.")


def get_camera_calibration(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = camera.get("intrinsic_matrix")
    extrinsic = camera.get("extrinsics_matrix")
    if extrinsic is None:
        # RoboDojo's current ObsManager uses the singular spelling. The shared
        # XPolicyLab runtime format uses the plural spelling.
        extrinsic = camera.get("extrinsic_matrix")
    if intrinsic is None or extrinsic is None:
        raise KeyError("Head-camera intrinsic/extrinsic matrices are required for P1-gaze.")
    return np.asarray(intrinsic), np.asarray(extrinsic)


def bbox_to_table_anchor(bbox_1000: Sequence[float], image_shape: Sequence[int]) -> np.ndarray:
    """Use a lower-center box point, which better approximates table contact."""
    if len(bbox_1000) != 4:
        raise ValueError(f"Expected [x0, y0, x1, y1], got {bbox_1000}.")
    x0, y0, x1, y1 = np.clip(np.asarray(bbox_1000, dtype=np.float64), 0.0, 1000.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid normalized bounding box: {bbox_1000}.")
    height, width = int(image_shape[0]), int(image_shape[1])
    x = 0.5 * (x0 + x1) * width / 1000.0
    y = (0.15 * y0 + 0.85 * y1) * height / 1000.0
    return np.array([x, y], dtype=np.float64)


def bbox_to_basket_anchor(bbox_1000: Sequence[float], image_shape: Sequence[int]) -> np.ndarray:
    """Anchor on the basket's front rim, which is the part resting on the table."""
    if len(bbox_1000) != 4:
        raise ValueError(f"Expected [x0, y0, x1, y1], got {bbox_1000}.")
    x0, y0, x1, y1 = np.clip(np.asarray(bbox_1000, dtype=np.float64), 0.0, 1000.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid normalized bounding box: {bbox_1000}.")
    height, width = int(image_shape[0]), int(image_shape[1])
    x = 0.5 * (x0 + x1) * width / 1000.0
    y = y1 * height / 1000.0
    return np.array([x, y], dtype=np.float64)
