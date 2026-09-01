"""RPent-style primitives for one RoboDojo episode.

Planner-facing calls return after the primitive's internal post-condition.
The simulator still only sees per-step `take_action` dictionaries.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .env_state import EnvStateStore
from .geometry import (
    bbox_to_pixels,
    get_camera,
    query_world_map as summarize_world_map,
    sample_world_xyz as sample_xyz,
)
from .manipulation import ManipulationLedger
from .robot_profile import (
    pregrasp_quaternion,
)
from .trace import EpisodeTrace


GROUND_PROMPT = """You ground one requested robot-manipulation target in an image.
Locate exactly the visual target requested by the user. Do not choose a task
order or silently substitute another target. The "label" should concisely
describe the located target. Return one JSON object only:
{"label": "target description", "bbox_2d": [x0, y0, x1, y1]}
Coordinates are relative integers from 0 to 1000. Return
{"label": null, "bbox_2d": null} if the requested target is not visible."""

VERIFY_PROMPT = """You verify one observable robot-manipulation transition from
the supplied fresh head and wrist images. Never infer a hold from gripper
closure alone. For grasp, the named target must have left its source and move
with the requested TCP. For transport, the same target must remain with the
TCP. For placement or release, it must be separated from the gripper and
visibly supported in/on the requested destination. Return JSON only:
{"passed": true, "evidence": "concise visible evidence"}
or {"passed": false, "evidence": "first unmet observable condition"}."""


def _gripper_open_threshold() -> float:
    return float(os.environ.get("RPENT_GRIPPER_OPEN_THRESH", "0.8"))


def _gripper_open_command() -> float:
    return float(os.environ.get("RPENT_GRIPPER_OPEN_CMD", "1.0"))


def _hover_max_steps() -> int:
    return max(8, int(os.environ.get("RPENT_MOVE_MAX_STEPS", "80")))


def _hover_tolerance_m() -> float:
    return max(0.0, float(os.environ.get("RPENT_MOVE_TOLERANCE_M", "0.03")))


def _orientation_tolerance_rad() -> float:
    return max(
        0.0,
        float(os.environ.get("RPENT_MOVE_ORIENTATION_TOLERANCE_RAD", "0.15")),
    )


def _hover_stall_steps() -> int:
    return max(1, int(os.environ.get("RPENT_MOVE_STALL_STEPS", "8")))


def _state_vector(
    state: Mapping[str, Any], key: str, size: int, default: float = 0.0
) -> np.ndarray:
    value = state.get(key)
    if value is None:
        return np.full(size, default, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != size:
        raise ValueError(
            f"Observation state {key!r} has size {result.size}, expected {size}."
        )
    return result.copy()


def as_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional RGB image, got {array.shape}.")
    if array.shape[-1] == 3:
        image_hwc = array
    elif array.shape[0] == 3:
        image_hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Expected three RGB channels, got {array.shape}.")
    if np.issubdtype(image_hwc.dtype, np.floating):
        image_hwc = (np.clip(image_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image_hwc.dtype != np.uint8:
        image_hwc = image_hwc.astype(np.uint8)
    return np.ascontiguousarray(image_hwc)


def encode_image_data_url(image: np.ndarray) -> str:
    rgb = as_hwc_uint8(image)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def closed_arm(observation: Mapping[str, Any]) -> str | None:
    state = observation.get("state", {})
    threshold = _gripper_open_threshold()
    for arm in ("left", "right"):
        value = state.get(f"{arm}_ee_joint_state")
        if value is None:
            continue
        if float(np.asarray(value, dtype=np.float32).reshape(-1)[0]) < threshold:
            return arm
    return None


def enable_camera_calibration(task_env: Any) -> bool:
    obs_manager = getattr(task_env, "obs_manager", None)
    if obs_manager is None:
        return False
    obs_manager.collect_intrinsic_matrix = True
    obs_manager.collect_extrinsic_matrix = True
    obs_manager.collect_depth = True
    return True


def env_origin(task_env: Any, env_idx: int) -> np.ndarray:
    scene = getattr(task_env, "scene", None)
    origins = getattr(scene, "env_origins", None)
    if origins is None:
        scene_manager = getattr(task_env, "scene_manager", None)
        origins = getattr(scene_manager, "env_origins", None)
    if origins is None:
        return np.zeros(3, dtype=np.float64)
    origin = origins[env_idx]
    if hasattr(origin, "detach"):
        origin = origin.detach()
    if hasattr(origin, "cpu"):
        origin = origin.cpu()
    return np.asarray(origin, dtype=np.float64).reshape(3)


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Model did not return JSON: {text!r}")
    return json.loads(match.group(0))


def freeze_policy_actions(actions: Any) -> Any:
    return copy.deepcopy(actions)


def quaternion_angular_error(first: Any, second: Any) -> float:
    first_quaternion = np.asarray(first, dtype=np.float64).reshape(4)
    second_quaternion = np.asarray(second, dtype=np.float64).reshape(4)
    first_norm = float(np.linalg.norm(first_quaternion))
    second_norm = float(np.linalg.norm(second_quaternion))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        raise ValueError("Cannot compare zero-norm quaternions.")
    dot = abs(
        float(
            np.dot(
                first_quaternion / first_norm,
                second_quaternion / second_norm,
            )
        )
    )
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def assert_policy_action_unchanged(
    action: Mapping[str, Any],
    frozen_action: Mapping[str, Any],
) -> None:
    if action.keys() != frozen_action.keys():
        raise RuntimeError("pi05_pick must forward every Pi_05 action field unchanged")
    for key in action:
        current = np.asarray(action[key])
        frozen = np.asarray(frozen_action[key])
        if (
            current.shape != frozen.shape
            or current.dtype != frozen.dtype
            or current.tobytes() != frozen.tobytes()
        ):
            raise RuntimeError(
                f"pi05_pick modified Pi_05 action field {key!r} inside a chunk"
            )


class RpentPrimitives:
    def __init__(
        self,
        task_env: Any,
        model_client: Any,
        qwen: Any,
        env_idx: int = 0,
        trace: EpisodeTrace | None = None,
    ) -> None:
        self.task_env = task_env
        self.model_client = model_client
        self.qwen = qwen
        self.env_idx = env_idx
        self.trace = trace or EpisodeTrace()
        self.env_states = EnvStateStore()
        self.ledger = ManipulationLedger()
        self.reset_poses: dict[str, np.ndarray] | None = None
        self.last_label: str | None = None
        self.last_grounding: dict[str, Any] | None = None
        self.finished = False
        self.last_tool: str | None = None
        self.last_reached_move: dict[str, dict[str, Any]] = {}

    def _obs(self) -> dict[str, Any]:
        if hasattr(self.task_env, "get_obs"):
            return self.task_env.get_obs()
        running = self.task_env.get_running_env_idx_list()
        env_idx = running[0] if running else self.env_idx
        self.env_idx = env_idx
        return self.task_env.get_obs_batch([env_idx])[0]

    def _cache_start(self, observation: Mapping[str, Any]) -> None:
        if self.reset_poses is not None:
            return
        state = observation["state"]
        self.reset_poses = {
            arm: _state_vector(state, f"{arm}_ee_pose", 7).copy()
            for arm in ("left", "right")
        }

    def _take(self, action: dict[str, np.ndarray]) -> None:
        if hasattr(self.task_env, "take_action"):
            self.task_env.take_action(action)
            return
        self.task_env.take_action_batch([action], [self.env_idx])

    def video_frame_counts(self) -> dict[str, int]:
        writers_by_env = getattr(self.task_env, "video_writers", {})
        writers = (
            writers_by_env.get(self.env_idx, {})
            if isinstance(writers_by_env, Mapping)
            else {}
        )
        camera_names = {
            "cam_head": "head",
            "cam_left_wrist": "left_wrist",
            "cam_right_wrist": "right_wrist",
        }
        return {
            camera_names.get(str(camera), str(camera)): int(writer.n_frames)
            for camera, writer in writers.items()
            if hasattr(writer, "n_frames")
        }

    def env_step(self) -> int | None:
        counts = getattr(self.task_env, "take_action_cnt", None)
        if counts is None:
            return None
        try:
            return int(np.asarray(counts)[self.env_idx])
        except (IndexError, TypeError, ValueError):
            return None

    def snapshot(self, observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        observation = observation or self._obs()
        self._cache_start(observation)
        state = observation.get("state", {})
        step = self.env_step()
        step_limit = getattr(self.task_env, "step_lim", None)
        official = self.episode_status()
        return {
            "instruction": observation.get("instruction")
            or observation.get("instructions"),
            "env_idx": self.env_idx,
            "step": step,
            "remaining_steps": (
                max(0, int(step_limit) - step)
                if step is not None and step_limit is not None
                else None
            ),
            "episode_end": bool(official["episode_end"]),
            "eval_success": official["eval_success"],
            "gripper_state": {
                arm: (
                    "open"
                    if float(
                        _state_vector(
                            state, f"{arm}_ee_joint_state", 1, default=1.0
                        )[0]
                    )
                    >= _gripper_open_threshold()
                    else "closed"
                )
                for arm in ("left", "right")
            },
            "manipulation": self.ledger.as_dict(),
            "carrying_arm": self.ledger.holding_arm,
            "last_grounded_label": self.last_label,
            "left_ee_xyz": _state_vector(state, "left_ee_pose", 7)[:3].round(4).tolist(),
            "right_ee_xyz": _state_vector(state, "right_ee_pose", 7)[:3].round(4).tolist(),
            "left_gripper": float(_state_vector(state, "left_ee_joint_state", 1, 1.0)[0]),
            "right_gripper": float(
                _state_vector(state, "right_ee_joint_state", 1, 1.0)[0]
            ),
            "last_tool": self.last_tool,
        }

    def episode_status(self) -> dict[str, Any]:
        end_flags = getattr(self.task_env, "end_flag", None)
        successes = getattr(self.task_env, "success", None)
        ended = (
            bool(end_flags[self.env_idx])
            if end_flags is not None and len(end_flags) > self.env_idx
            else bool(self.task_env.is_episode_end())
        )
        success = (
            bool(successes[self.env_idx])
            if ended and successes is not None and len(successes) > self.env_idx
            else False
        )
        return {
            "episode_end": ended,
            "eval_success": success,
            "native_steps": self.env_step(),
        }

    def observation_images(
        self, observation: Mapping[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        observation = observation or self._obs()
        images: dict[str, np.ndarray] = {}
        vision = observation.get("vision", {})
        for name, key in (
            ("head", None),
            ("left_wrist", "cam_left_wrist"),
            ("right_wrist", "cam_right_wrist"),
        ):
            if key is None:
                try:
                    _, camera = get_camera(observation, "head")
                except Exception:
                    continue
            else:
                camera = vision.get(key)
                if not isinstance(camera, Mapping) or camera.get("color") is None:
                    continue
            try:
                images[name] = as_hwc_uint8(camera["color"])
            except Exception:
                continue
        return images

    def image_parts(self, observation: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for name, image in self.observation_images(observation).items():
            parts.append({"type": "text", "text": f"[{name} camera]"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_url(image),
                        "detail": "auto",
                    },
                }
            )
        return parts

    def record_tool_result(
        self,
        name: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        observation = self._obs()
        rgbd_views = None
        try:
            record = self._capture_env_state(observation)
            rgbd_views = record.views
        except (KeyError, ValueError):
            pass
        trace_info = self.trace.record_observation(
            tool=name,
            arguments=arguments,
            result=result,
            observation=self.snapshot(observation),
            images=self.observation_images(observation),
            rgbd_views=rgbd_views,
        )
        self.last_tool = name
        return {**dict(result), **trace_info}

    def _capture_env_state(
        self, observation: Mapping[str, Any] | None = None
    ) -> Any:
        observation = observation or self._obs()
        return self.env_states.capture(
            observation,
            rgb_images=self.observation_images(observation),
            episode_status=self.episode_status(),
            env_origin=env_origin(self.task_env, self.env_idx),
        )

    def _vision_json(
        self,
        image: np.ndarray,
        prompt: str,
        user_text: str,
    ) -> dict[str, Any]:
        result = self.qwen.chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                        "url": encode_image_data_url(image),
                        "detail": "auto",
                    },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            tools=None,
            tool_choice=None,
        )
        text, _ = self.qwen.message_text_and_tools(result)
        return _extract_json(text)

    def observe(self) -> dict[str, Any]:
        observation = self._obs()
        result = self.snapshot(observation)
        try:
            record = self._capture_env_state(observation)
            result["env_state_step"] = record.step
            result["views"] = {
                name: {
                    "shape": list(view.rgb.shape),
                    "depth_unit": "metres",
                    "valid_world_points": int(
                        np.isfinite(view.world_xyz).all(axis=2).sum()
                    ),
                }
                for name, view in record.views.items()
            }
        except (KeyError, ValueError) as exc:
            result["rgbd_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def view_env_state(self, step: int = -1) -> dict[str, Any]:
        record = self.env_states.get(step)
        return {
            "env_state_step": record.step,
            "instruction": record.instruction,
            "views": {
                name: {
                    "shape": list(view.rgb.shape),
                    "depth_unit": "metres",
                    "valid_world_points": int(
                        np.isfinite(view.world_xyz).all(axis=2).sum()
                    ),
                }
                for name, view in record.views.items()
            },
            "episode_status": dict(record.episode_status),
            "manipulation": self.ledger.as_dict(),
        }

    def sample_world_xyz(
        self,
        view: str,
        pixels: list[list[float]],
        step: int = -1,
        radius: int = 2,
    ) -> dict[str, Any]:
        record = self.env_states.get(step)
        camera = record.views[view]
        return {
            "env_state_step": record.step,
            "view": view,
            "samples": [
                sample_xyz(camera.world_xyz, pixel, radius=radius)
                for pixel in pixels
            ],
        }

    def query_world_map(
        self,
        view: str,
        bbox: list[int],
        step: int = -1,
    ) -> dict[str, Any]:
        record = self.env_states.get(step)
        return {
            "env_state_step": record.step,
            "view": view,
            **summarize_world_map(record.views[view].world_xyz, bbox),
        }

    def ground(
        self,
        query: str,
        camera: str = "head",
    ) -> dict[str, Any]:
        self.last_grounding = None
        self.last_label = None
        if camera != "head":
            raise ValueError(
                "ground is head-only for semantic target binding (camera='head'); "
                "refine wrist geometry with sample_world_xyz or query_world_map "
                "at the exact step, view, and resolution for the same candidate."
            )
        observation = self._obs()
        instruction = str(
            observation.get("instruction") or observation.get("instructions") or ""
        )
        record = self._capture_env_state(observation)
        image = record.views[camera].rgb
        parsed = self._vision_json(
            image,
            GROUND_PROMPT,
            f"Robot instruction: {instruction}\nLocate this requested target: {query}",
        )
        bbox = parsed.get("bbox_2d")
        label = parsed.get("label")
        if bbox is None:
            self.last_label = None
            return {
                "query": query,
                "label": None,
                "bbox_2d": None,
                "env_state_step": record.step,
                "bbox_rc": None,
                "image_shape": list(image.shape[:2]),
                "carrying_arm": self.ledger.holding_arm,
            }
        bbox_rc = bbox_to_pixels(bbox, image.shape)
        self.last_label = None if label is None else str(label)
        result = {
            "query": query,
            "label": label,
            "bbox_2d": bbox,
            "env_state_step": record.step,
            "bbox_rc": list(bbox_rc),
            "image_shape": list(image.shape[:2]),
            "carrying_arm": self.ledger.holding_arm,
        }
        self.last_grounding = result
        return result

    def _build_ee_action(
        self,
        observation: Mapping[str, Any],
        xyz: np.ndarray,
        arm: str,
        gripper: float | None = None,
        orientation: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        state = observation["state"]
        left_pose = _state_vector(state, "left_ee_pose", 7)
        right_pose = _state_vector(state, "right_ee_pose", 7)
        if not np.any(left_pose[3:]):
            left_pose[3] = 1.0
        if not np.any(right_pose[3:]):
            right_pose[3] = 1.0
        if orientation is not None:
            if arm == "left":
                left_pose[3:] = orientation
            else:
                right_pose[3:] = orientation
        if arm == "left":
            left_pose[:3] = xyz
        else:
            right_pose[:3] = xyz
        left_grip = _state_vector(state, "left_ee_joint_state", 1, default=1.0)
        right_grip = _state_vector(state, "right_ee_joint_state", 1, default=1.0)
        if gripper is not None:
            if arm == "left":
                left_grip[:] = gripper
            else:
                right_grip[:] = gripper
        return {
            "left_ee_pose": left_pose,
            "right_ee_pose": right_pose,
            "left_ee_joint_state": left_grip,
            "right_ee_joint_state": right_grip,
        }

    def _build_joint_action(
        self,
        observation: Mapping[str, Any],
        *,
        arm: str,
        arm_qpos: Any,
        gripper: float | None,
    ) -> dict[str, np.ndarray]:
        state = observation["state"]
        action = {
            selected: _state_vector(state, selected, 6)
            for selected in ("left_arm_joint_state", "right_arm_joint_state")
        }
        action.update(
            {
                selected: _state_vector(state, selected, 1, default=1.0)
                for selected in ("left_ee_joint_state", "right_ee_joint_state")
            }
        )
        action[f"{arm}_arm_joint_state"] = np.asarray(
            arm_qpos, dtype=np.float32
        ).reshape(-1)
        if gripper is not None:
            action[f"{arm}_ee_joint_state"][:] = float(gripper)
        return action

    def _plan_arm_path(
        self,
        *,
        arm: str,
        target_pose: np.ndarray,
    ) -> dict[str, Any]:
        manager = getattr(self.task_env, "robot_manager", None)
        if manager is None:
            return {"status": "Unavailable"}
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            return {"status": "Unavailable"}
        current = manager.get_joint(robot, env_idx_list=[self.env_idx])[self.env_idx]
        return planner.plan_path(
            current,
            target_pose,
            real_robot_pose=copy.deepcopy(robot.entity_origin_pose),
        )

    def _execute_planned_path(
        self,
        *,
        arm: str,
        path: np.ndarray,
        gripper: float | None,
    ) -> int:
        executed = 0
        for waypoint in path:
            if self.task_env.is_episode_end():
                break
            observation = self._obs()
            self._take(
                self._build_joint_action(
                    observation,
                    arm=arm,
                    arm_qpos=waypoint,
                    gripper=gripper,
                )
            )
            executed += 1
        return executed

    def _servo_to(
        self,
        xyz: np.ndarray,
        arm: str,
        gripper: float | None = None,
        orientation: np.ndarray | None = None,
    ) -> dict[str, Any]:
        observation = self._obs()
        action = self._build_ee_action(
            observation, xyz, arm, gripper=gripper, orientation=orientation
        )
        best = float("inf")
        best_orientation_error = float("inf")
        stalled = 0
        steps = 0
        error = float("inf")
        orientation_error = None
        for steps in range(1, _hover_max_steps() + 1):
            self._take(action)
            if self.task_env.is_episode_end():
                break
            observation = self._obs()
            actual_pose = _state_vector(
                observation["state"], f"{arm}_ee_pose", 7
            )
            error = float(np.linalg.norm(actual_pose[:3] - xyz))
            orientation_error = (
                None
                if orientation is None
                else quaternion_angular_error(actual_pose[3:], orientation)
            )
            position_reached = error <= _hover_tolerance_m()
            orientation_reached = (
                orientation_error is None
                or orientation_error <= _orientation_tolerance_rad()
            )
            if position_reached and orientation_reached:
                break
            improved = error < best - 1e-4
            if orientation_error is not None:
                improved = (
                    improved
                    or orientation_error < best_orientation_error - 1e-3
                )
                best_orientation_error = min(
                    best_orientation_error, orientation_error
                )
            if improved:
                best = error
                stalled = 0
            else:
                stalled += 1
                if stalled >= _hover_stall_steps():
                    break
            action = self._build_ee_action(
                observation, xyz, arm, gripper=gripper, orientation=orientation
            )
        print(
            f"[P1-RPent] move_to arm={arm} target={np.asarray(xyz).round(3).tolist()} "
            f"error_m={error:.4f} orientation_error_rad={orientation_error} "
            f"steps={steps}",
            flush=True,
        )
        position_reached = error <= _hover_tolerance_m()
        orientation_reached = (
            orientation_error is None
            or orientation_error <= _orientation_tolerance_rad()
        )
        return {
            "arm": arm,
            "target_xyz": np.asarray(xyz, dtype=np.float32).round(4).tolist(),
            "final_error_m": round(error, 4),
            "final_orientation_error_rad": (
                None if orientation_error is None else round(orientation_error, 4)
            ),
            "steps_used": steps,
            "reached": position_reached and orientation_reached,
            "stalled": (not position_reached or not orientation_reached)
            and stalled >= _hover_stall_steps(),
            "episode_end": bool(self.task_env.is_episode_end()),
            **self.snapshot(),
        }

    def move_to(
        self,
        xyz: list[float] | None = None,
        arm: str | None = None,
        gripper: float | None = None,
        quat: list[float] | None = None,
        substeps: int = 25,
    ) -> dict[str, Any]:
        observation = self._obs()
        if xyz is None:
            raise ValueError("move_to requires xyz")
        hover_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1)[:3]
        if arm is None:
            raise ValueError("move_to requires an explicit arm")
        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        selected = arm
        if (
            self.ledger.holding_arm == selected
            and self.ledger.hold_state != "verified"
        ):
            return {
                "error": "transport_requires_verified_hold",
                "arm": selected,
                "target_xyz": hover_xyz.round(4).tolist(),
                "precondition": {
                    "required": "hold_state=verified",
                    "actual": self.ledger.hold_state,
                },
                "recovery_hint": "Call verify_state(gate='grasp') before transport.",
                **self.snapshot(observation),
            }
        current_pose = _state_vector(
            observation["state"], f"{selected}_ee_pose", 7
        )
        orientation = (
            current_pose[3:]
            if quat is None
            else np.asarray(quat, dtype=np.float32).reshape(4)
        )
        target_pose = np.concatenate([hover_xyz, orientation])
        planned = self._plan_arm_path(arm=selected, target_pose=target_pose)
        if planned.get("status") == "Unavailable":
            result = self._servo_to(
                hover_xyz,
                selected,
                gripper=gripper,
                orientation=orientation,
            )
            result["plan_status"] = "Unavailable"
            result["execution_mode"] = "ee_servo_fallback"
            return result
        if planned.get("status") != "Success" or planned.get("position") is None:
            return {
                "completed": False,
                "requested_steps": 0,
                "executed_steps": 0,
                "stop_reason": "plan_failed",
                "success": False,
                "plan_status": planned.get("status", "Fail"),
                "hint": "target may be unreachable or in collision",
                "arm": selected,
                "target_xyz": hover_xyz.round(4).tolist(),
                **self.snapshot(observation),
            }
        path = np.asarray(planned["position"], dtype=np.float32)
        if int(substeps) < 1:
            raise ValueError("substeps must be at least 1")
        if len(path) > int(substeps):
            indices = np.linspace(0, len(path) - 1, int(substeps)).astype(int)
            path = path[indices]
        executed = self._execute_planned_path(
            arm=selected,
            path=path,
            gripper=gripper,
        )
        final_observation = self._obs()
        final_pose = _state_vector(
            final_observation["state"], f"{selected}_ee_pose", 7
        )
        position_error = float(np.linalg.norm(final_pose[:3] - hover_xyz))
        orientation_error = quaternion_angular_error(final_pose[3:], orientation)
        reached = (
            position_error <= _hover_tolerance_m()
            and orientation_error <= _orientation_tolerance_rad()
        )
        result = {
            "completed": executed == len(path),
            "requested_steps": int(len(path)),
            "executed_steps": executed,
            "stop_reason": (
                "reached" if reached else "residual_too_large"
            ),
            "success": reached,
            "reached": reached,
            "plan_status": planned["status"],
            "execution_mode": "curobo_joint_path",
            "waypoints": int(len(path)),
            "arm": selected,
            "target_xyz": hover_xyz.round(4).tolist(),
            "target_quat": orientation.round(5).tolist(),
            "final_eef_pose": final_pose.round(5).tolist(),
            "final_error_m": round(position_error, 5),
            "final_orientation_error_rad": round(orientation_error, 5),
            **self.snapshot(final_observation),
        }
        if result["reached"]:
            self.last_reached_move[selected] = {
                "target_xyz": result["target_xyz"],
                "sim_step": result["step"],
                "carrying": (
                    self.ledger.holding_arm == selected
                    and self.ledger.hold_state == "verified"
                ),
            }
        return result

    def rotate_wrist(
        self,
        arm: str,
        delta_yaw_deg: float,
        gripper: float | None = None,
        substeps: int = 25,
    ) -> dict[str, Any]:
        observation = self._obs()
        pose = _state_vector(observation["state"], f"{arm}_ee_pose", 7)
        yaw = np.deg2rad(float(delta_yaw_deg))
        rotation = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        )
        w1, x1, y1, z1 = rotation
        w2, x2, y2, z2 = pose[3:]
        quaternion = [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
        result = self.move_to(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=quaternion,
            gripper=gripper,
            substeps=substeps,
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def pi05_act(
        self,
        focus: str | None = None,
        max_chunks: int = 1,
        execution_horizon: int | None = None,
    ) -> dict[str, Any]:
        if not focus:
            raise ValueError("pi05_act requires a concise current focus")
        execution_horizon = max(
            1,
            int(
                execution_horizon
                or os.environ.get("RPENT_PI05_EXECUTION_HORIZON", "20")
            ),
        )
        self.last_reached_move.clear()
        start = self._obs()
        start_z = {
            arm: float(_state_vector(start["state"], f"{arm}_ee_pose", 7)[2])
            for arm in ("left", "right")
        }
        min_z = dict(start_z)
        post_min_peak_z = dict(start_z)
        descent_done = {"left": False, "right": False}
        chunks_used = 0
        actions_executed = 0
        candidate_arm: str | None = None
        for _ in range(max(1, int(max_chunks))):
            if self.task_env.is_episode_end():
                break
            observation = self._obs()
            local_observation = copy.deepcopy(observation)
            self.model_client.call(func_name="update_obs", obs=local_observation)
            actions = self.model_client.call(func_name="get_action")
            frozen = freeze_policy_actions(actions)
            chunks_used += 1
            for action_idx, action in enumerate(actions[:execution_horizon]):
                assert_policy_action_unchanged(action, frozen[action_idx])
                self._take(action)
                actions_executed += 1
                if os.environ.get("RPENT_RECORD_EVERY_PI05_ACTION") == "1":
                    self._obs()
                if self.task_env.is_episode_end():
                    break
            observation = self._obs()
            for arm in ("left", "right"):
                z = float(
                    _state_vector(observation["state"], f"{arm}_ee_pose", 7)[2]
                )
                if z < min_z[arm]:
                    min_z[arm] = z
                    post_min_peak_z[arm] = z
                else:
                    post_min_peak_z[arm] = max(post_min_peak_z[arm], z)
                if start_z[arm] - min_z[arm] >= float(
                    os.environ.get("RPENT_DESCENT_THRESH_M", "0.04")
                ):
                    descent_done[arm] = True
            candidate_arm = closed_arm(observation)
            if candidate_arm is not None:
                lift = post_min_peak_z[candidate_arm] - min_z[candidate_arm]
                if descent_done[candidate_arm] and lift >= float(
                    os.environ.get("RPENT_LIFT_THRESH_M", "0.04")
                ):
                    break
        candidate_evidence = bool(
            candidate_arm
            and descent_done[candidate_arm]
            and post_min_peak_z[candidate_arm] - min_z[candidate_arm]
            >= float(os.environ.get("RPENT_LIFT_THRESH_M", "0.04"))
        )
        self.ledger.candidate(
            arm=candidate_arm if candidate_evidence else None,
            target=focus,
            step=self.env_step(),
        )
        print(
            f"[P1-RPent] pi05_act candidate={candidate_evidence} chunks={chunks_used} "
            f"actions={actions_executed} horizon={execution_horizon} "
            f"arm={candidate_arm}",
            flush=True,
        )
        return {
            "focus": focus,
            "model_instruction": str(
                start.get("instruction") or start.get("instructions") or ""
            ),
            "chunks_used": chunks_used,
            "execution_horizon": execution_horizon,
            "actions_executed": actions_executed,
            "stop_reason": (
                "candidate_grasp_evidence"
                if candidate_evidence
                else "execution_prefix_complete"
            ),
            "candidate_evidence": candidate_evidence,
            "candidate_arm": candidate_arm,
            "descent_done": descent_done,
            "post_descent_lift_m": {
                arm: round(post_min_peak_z[arm] - min_z[arm], 4)
                for arm in ("left", "right")
            },
            **self.snapshot(),
        }

    def pi05_pick(self, prompt: str | None = None, max_chunks: int = 1) -> dict[str, Any]:
        """Compatibility alias for historical callers and traces."""
        result = self.pi05_act(focus=prompt, max_chunks=max_chunks)
        return {
            **result,
            "prompt": prompt,
            "candidate_success": result["candidate_evidence"],
            "carrying_arm": result["candidate_arm"],
        }

    def verify_state(
        self,
        gate: str,
        target: str,
        arm: str | None = None,
    ) -> dict[str, Any]:
        allowed = {
            "grasp",
            "transport",
            "support_placement",
            "container_placement",
            "handover_receive",
            "release",
        }
        if gate not in allowed:
            raise ValueError(f"Unsupported verification gate: {gate}")
        if not target.strip():
            raise ValueError("verify_state requires the target/relation to verify")
        observation = self._obs()
        parts = self.image_parts(observation)
        parts.append(
            {
                "type": "text",
                "text": (
                    f"Gate: {gate}\nTarget/relation: {target}\n"
                    f"Arm: {arm or self.ledger.holding_arm}\n"
                    f"Ledger before check: {json.dumps(self.ledger.as_dict())}"
                ),
            }
        )
        response = self.qwen.chat(
            [
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": parts},
            ],
            tools=None,
            tool_choice=None,
        )
        text, _ = self.qwen.message_text_and_tools(response)
        verification = _extract_json(text)
        passed = verification.get("passed") is True
        evidence = str(verification.get("evidence") or "")
        if gate in {"grasp", "transport", "handover_receive"} and passed:
            selected = arm or self.ledger.holding_arm
            if selected not in {"left", "right"}:
                raise ValueError(f"{gate} verification requires an arm")
            self.ledger.holding_arm = selected
        self.ledger.verify(gate=gate, passed=bool(passed), step=self.env_step())
        if gate == "handover_receive" and passed:
            self.ledger.holding_arm = arm
        return {
            "gate": gate,
            "passed": bool(passed),
            "target": target,
            "evidence": evidence,
            "transition": self.ledger.as_dict(),
            **self.snapshot(),
        }

    def set_gripper(
        self,
        arm: str,
        state: str,
        steps: int = 8,
    ) -> dict[str, Any]:
        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        if state not in {"open", "closed"}:
            raise ValueError("state must be 'open' or 'closed'")
        command = (
            _gripper_open_command()
            if state == "open"
            else float(os.environ.get("RPENT_GRIPPER_CLOSE_CMD", "0.0"))
        )
        used = 0
        for used in range(1, max(1, int(steps)) + 1):
            if self.task_env.is_episode_end():
                break
            observation = self._obs()
            xyz = _state_vector(observation["state"], f"{arm}_ee_pose", 7)[:3]
            action = self._build_ee_action(
                observation,
                xyz,
                arm,
                gripper=command,
            )
            self._take(action)
            actual = float(
                _state_vector(
                    self._obs()["state"],
                    f"{arm}_ee_joint_state",
                    1,
                    default=1.0,
                )[0]
            )
            is_open = actual >= _gripper_open_threshold()
            if (state == "open" and is_open) or (state == "closed" and not is_open):
                break
        observation = self._obs()
        actual = float(
            _state_vector(
                observation["state"], f"{arm}_ee_joint_state", 1, default=1.0
            )[0]
        )
        return {
            "arm": arm,
            "requested_state": state,
            "steps_used": used,
            "gripper": actual,
            "opened": actual >= _gripper_open_threshold(),
            "reached": (
                actual >= _gripper_open_threshold()
                if state == "open"
                else actual < _gripper_open_threshold()
            ),
            **self.snapshot(observation),
        }

    def release(self, arm: str | None = None, max_steps: int = 20) -> dict[str, Any]:
        observation = self._obs()
        arm = arm or self.ledger.holding_arm
        if arm is None:
            return {"error": "no_verified_hold", **self.snapshot()}
        move = self.last_reached_move.get(arm)
        if (
            move is None
            or not move["carrying"]
            or self.ledger.holding_arm != arm
            or self.ledger.hold_state != "verified"
        ):
            return {
                "error": "release_requires_reached_carry_move",
                "arm": arm,
                "precondition": {
                    "required": "verified hold and successful transport move",
                    "hold_state": self.ledger.hold_state,
                },
                **self.snapshot(observation),
            }
        result = self.set_gripper(arm, "open", max_steps)
        self.last_reached_move.pop(arm, None)
        self.ledger.verify(gate="release", passed=True, step=self.env_step())
        print(
            f"[P1-RPent] release arm={arm} steps={result['steps_used']} "
            f"open={result['opened']}",
            flush=True,
        )
        return {
            **result,
            "opened": result["opened"],
            "all_grippers_open": all(
                value == "open"
                for value in self.snapshot()["gripper_state"].values()
            ),
        }

    def return_home(self, arm: str = "both") -> dict[str, Any]:
        observation = self._obs()
        self._cache_start(observation)
        assert self.reset_poses is not None
        arms = ("left", "right") if arm == "both" else (arm,)
        results: dict[str, Any] = {}
        for selected in arms:
            if selected not in {"left", "right"}:
                raise ValueError("arm must be 'left', 'right', or 'both'")
            if self.task_env.is_episode_end():
                break
            results[selected] = self.move_to(
                self.reset_poses[selected][:3].tolist(),
                selected,
                quat=self.reset_poses[selected][3:].tolist(),
            )
        return {"arms": results, **self.snapshot()}

    def finish(self, status: str, summary: str) -> dict[str, Any]:
        self.finished = True
        native = self.episode_status()
        requested_success = status.lower() == "success"
        verified_success = native["eval_success"] is True
        return {
            "_finish": True,
            "status": (
                "success"
                if verified_success
                else "failure"
                if requested_success
                else status
            ),
            "summary": summary,
            "requested_success": requested_success,
            "success": verified_success,
            "episode_status": native,
            **self.snapshot(),
        }
