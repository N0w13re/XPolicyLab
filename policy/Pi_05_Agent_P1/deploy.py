"""RoboDojo loop with repeated P1-gaze approach priming."""

from __future__ import annotations

import base64
import copy
import json
import os
from typing import Any, Mapping
from urllib import request

import numpy as np

from .instruction import BASKET_SIDES, destination_side
from .table_mapper import (
    TableMapConfig,
    TablePlaneMapper,
    bbox_to_basket_anchor,
    bbox_to_table_anchor,
    get_camera_calibration,
    get_head_camera,
)


def _reprime_enabled() -> bool:
    if os.environ.get("P1_APPROACH", "1").lower() in {"0", "false", "no"}:
        return False
    return os.environ.get("P1_REPRIME", "1").lower() not in {"0", "false", "no"}


def _identify_held_enabled() -> bool:
    return os.environ.get("P1_IDENTIFY_HELD", "1").lower() not in {"0", "false", "no"}


def _gripper_open_threshold() -> float:
    return float(os.environ.get("P1_GRIPPER_OPEN_THRESH", "0.8"))


def _reprime_mode() -> str:
    return os.environ.get("P1_REPRIME_MODE", "release").lower()


def _grippers_open(observation: Mapping[str, Any]) -> bool:
    """Return True when both grippers look open enough to start a new pick."""
    state = observation.get("state", {})
    threshold = _gripper_open_threshold()
    for key in ("left_ee_joint_state", "right_ee_joint_state"):
        value = state.get(key)
        if value is None:
            continue
        gripper = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
        if gripper < threshold:
            return False
    return True


def _hover_dwell_steps() -> int:
    return max(1, int(os.environ.get("P1_HOVER_DWELL", "8")))


def _hover_max_steps() -> int:
    return max(_hover_dwell_steps(), int(os.environ.get("P1_HOVER_MAX_STEPS", "40")))


def _hover_tolerance_m() -> float:
    return max(0.0, float(os.environ.get("P1_HOVER_TOLERANCE_M", "0.03")))


def _reprime_min_steps() -> int:
    return max(0, int(os.environ.get("P1_REPRIME_MIN_STEPS", "80")))


def _grasp_min_steps() -> int:
    return max(1, int(os.environ.get("P1_GRASP_MIN_STEPS", "15")))


def _reprime_stop_margin() -> int:
    return max(0, int(os.environ.get("P1_REPRIME_STOP_MARGIN", "80")))


def decide_reprime(
    *,
    reprime_enabled: bool,
    grippers_open: bool,
    step_count: int,
    env_idx: int,
    mode: str,
    gripper_was_closed: dict[int, bool],
    gripper_closed_step: dict[int, int],
    episode_started: set[int],
    last_reprime_step: dict[int, int],
    min_gap: int,
    grasp_min_steps: int,
) -> bool:
    """Pure release/chunk re-prime policy used by the live deploy loop."""
    if not reprime_enabled:
        return False

    if not grippers_open:
        if not gripper_was_closed.get(env_idx, False):
            gripper_closed_step[env_idx] = step_count
        gripper_was_closed[env_idx] = True
        return False

    if mode in {"chunk", "every", "every_chunk"}:
        if min_gap > 0 and step_count - last_reprime_step.get(env_idx, -10**9) < min_gap:
            return False
        last_reprime_step[env_idx] = step_count
        return True

    if env_idx not in episode_started and step_count == 0:
        episode_started.add(env_idx)
        last_reprime_step[env_idx] = step_count
        return True

    if not gripper_was_closed.get(env_idx, False):
        return False

    closed_at = gripper_closed_step.get(env_idx, step_count)
    closed_duration = step_count - closed_at
    gripper_was_closed[env_idx] = False
    if closed_duration >= grasp_min_steps:
        last_reprime_step[env_idx] = step_count
        return True

    if min_gap > 0 and step_count - last_reprime_step.get(env_idx, -10**9) < min_gap:
        return False
    last_reprime_step[env_idx] = step_count
    return True


def _prime_state(
    task_env: Any,
) -> tuple[dict[int, bool], set[int], dict[int, int], dict[int, int]]:
    if not hasattr(task_env, "_p1_gripper_was_closed"):
        task_env._p1_gripper_was_closed = {}
    if not hasattr(task_env, "_p1_episode_started"):
        task_env._p1_episode_started = set()
    if not hasattr(task_env, "_p1_last_reprime_step"):
        task_env._p1_last_reprime_step = {}
    if not hasattr(task_env, "_p1_gripper_closed_step"):
        task_env._p1_gripper_closed_step = {}
    return (
        task_env._p1_gripper_was_closed,
        task_env._p1_episode_started,
        task_env._p1_last_reprime_step,
        task_env._p1_gripper_closed_step,
    )


def _should_reprime(observation: Mapping[str, Any], env_idx: int, task_env: Any) -> bool:
    gripper_was_closed, episode_started, last_reprime_step, gripper_closed_step = _prime_state(task_env)
    step_count = task_env.take_action_cnt[env_idx]
    step_limit = int(getattr(task_env, "step_lim", 0))
    if step_limit > 0 and step_count >= step_limit - _reprime_stop_margin():
        return False
    should_reprime = decide_reprime(
        reprime_enabled=_reprime_enabled(),
        grippers_open=_grippers_open(observation),
        step_count=step_count,
        env_idx=env_idx,
        mode=_reprime_mode(),
        gripper_was_closed=gripper_was_closed,
        gripper_closed_step=gripper_closed_step,
        episode_started=episode_started,
        last_reprime_step=last_reprime_step,
        min_gap=_reprime_min_steps(),
        grasp_min_steps=_grasp_min_steps(),
    )
    if should_reprime:
        print(
            f"[P1-gaze] env={env_idx} step={step_count} pre_chunk_reprime "
            f"mode={_reprime_mode()}",
            flush=True,
        )
    return should_reprime


def _chunk_interrupt(
    observation: Mapping[str, Any],
    env_idx: int,
    task_env: Any,
) -> str | None:
    """Reason to abandon the rest of the current Pi_05 chunk, if any.

    Both intervention points are only useful while they are still actionable: a
    new target has to be aimed at before the next grasp, and a carried object
    has to be steered before Pi_05 releases it. Waiting for the 50-step chunk
    boundary misses both.
    """
    released = _track_gripper_transition(observation, env_idx, task_env)
    if released and _reprime_enabled():
        return "release"
    if _should_deliver(observation, env_idx, task_env):
        return "grasp"
    return None


def _track_gripper_transition(
    observation: Mapping[str, Any],
    env_idx: int,
    task_env: Any,
) -> bool:
    """Track within-chunk closure and report the first observed release."""
    gripper_was_closed, _, _, gripper_closed_step = _prime_state(task_env)
    open_now = _grippers_open(observation)
    was_closed = gripper_was_closed.get(env_idx, False)
    if not open_now:
        if not was_closed:
            gripper_closed_step[env_idx] = task_env.take_action_cnt[env_idx]
        gripper_was_closed[env_idx] = True
        return False
    return was_closed


def _reset_prime_tracking(task_env: Any) -> None:
    task_env._p1_gripper_was_closed = {}
    task_env._p1_episode_started = set()
    task_env._p1_last_reprime_step = {}
    task_env._p1_gripper_closed_step = {}
    task_env._p1_neutral_quaternions = {}
    task_env._p1_held_category = {}
    task_env._p1_delivered_closure = {}
    task_env._p1_basket_targets = {}


def _delivery_enabled() -> bool:
    return os.environ.get("P1_DELIVER", "1").lower() not in {"0", "false", "no"}


def _deliver_height() -> float:
    # Measured reach limit: at the outer basket x, commanding 0.18 above the
    # table stalled the arm near z=0.88, so the release pose has to sit lower.
    return float(os.environ.get("P1_DELIVER_HEIGHT", "0.13"))


def _basket_depth_offset() -> float:
    # Aim slightly past the mapped front rim. The arm undershoots in depth at
    # the outer baskets, so aiming a little inside lands the release inside the
    # rim rather than on it.
    return float(os.environ.get("P1_BASKET_DEPTH_OFFSET_M", "0.05"))


def _deliver_mode() -> str:
    return os.environ.get("P1_DELIVER_MODE", "basket").lower()


def _deliver_z_margin() -> float:
    return float(os.environ.get("P1_DELIVER_Z_MARGIN", "0.06"))


def _deliver_depth_step() -> float:
    return float(os.environ.get("P1_DELIVER_DEPTH_STEP", "0.08"))


def _relay_enabled() -> bool:
    return os.environ.get("P1_RELAY", "1").lower() not in {"0", "false", "no"}


def relay_target(
    current_xyz: np.ndarray,
    destination_x: float,
    cross_reach: float,
    min_z: float,
    z_margin: float = 0.06,
) -> np.ndarray:
    """Where to carry an object whose basket the holding arm cannot reach.

    Leaving these to Pi_05 is not neutral: the reward also requires that no
    foreign category sits in a basket, so one wrong drop makes that category
    permanently unscorable. Moving the object to the gap between baskets, inside
    both arms' reach, lets the correct-side arm take it later while keeping it
    away from any basket mouth.
    """
    target = np.asarray(current_xyz, dtype=np.float32).copy()
    # Stay a little inside the reach limit rather than exactly on it, so the
    # commanded pose is comfortably reachable.
    target[0] = float(np.sign(destination_x) * cross_reach * 0.9)
    target[2] = float(np.clip(target[2], min_z, min_z + z_margin))
    return target


def deliver_target(
    current_xyz: np.ndarray,
    basket_xyz: np.ndarray,
    mode: str,
    min_z: float,
    z_margin: float = 0.06,
    depth_step: float = 0.08,
) -> np.ndarray:
    """Where to move the carrying arm so Pi_05 releases over the right basket.

    Three axes behave differently, and each limit below is measured:

    - The basket-selecting axis is fully corrected; those hovers converge to
      about 0.1 mm.
    - Depth is corrected by at most `depth_step`. Commanding the basket depth
      outright left a median residual of 0.07-0.23 m even on the near side,
      because a fixed wrist orientation cannot reach as far as Pi_05's joint
      control; keeping the carried depth instead released 0.10-0.18 m in front
      of the rim. Stepping part-way keeps the pose reachable while still moving
      the release toward the basket.
    - Height is clamped into a band above the release clearance, since Pi_05
      often carries 0.25 m above the table and releasing from there bounces the
      object out.

    `basket_pose` commands the mapped point unchanged, for diagnosis.
    """
    if mode in {"basket_pose", "full"}:
        return np.asarray(basket_xyz, dtype=np.float32).copy()

    target = np.asarray(current_xyz, dtype=np.float32).copy()
    target[0] = float(basket_xyz[0])
    if mode != "lateral":
        depth_gap = float(basket_xyz[1]) - float(target[1])
        target[1] += float(np.clip(depth_gap, -depth_step, depth_step))
    target[2] = float(np.clip(target[2], min_z, min_z + z_margin))
    return target


def _basket_y_limits() -> tuple[float, float]:
    return (-0.40, float(os.environ.get("P1_BASKET_Y_MAX", "0.30")))


def _delivery_state(task_env: Any) -> tuple[dict[int, str], dict[int, int], dict[int, dict]]:
    if not hasattr(task_env, "_p1_held_category"):
        task_env._p1_held_category = {}
    if not hasattr(task_env, "_p1_delivered_closure"):
        task_env._p1_delivered_closure = {}
    if not hasattr(task_env, "_p1_basket_targets"):
        task_env._p1_basket_targets = {}
    return (
        task_env._p1_held_category,
        task_env._p1_delivered_closure,
        task_env._p1_basket_targets,
    )


def _should_deliver(observation: Mapping[str, Any], env_idx: int, task_env: Any) -> bool:
    """True once a confirmed grasp still needs to be steered to its basket."""
    if not _delivery_enabled():
        return False
    if _grippers_open(observation):
        return False

    gripper_was_closed, _, _, gripper_closed_step = _prime_state(task_env)
    if not gripper_was_closed.get(env_idx, False):
        return False

    step_count = task_env.take_action_cnt[env_idx]
    step_limit = int(getattr(task_env, "step_lim", 0))
    if step_limit > 0 and step_count >= step_limit - _reprime_stop_margin():
        return False

    closed_at = gripper_closed_step.get(env_idx)
    if closed_at is None or step_count - closed_at < _grasp_min_steps():
        return False

    held_category, delivered_closure, _ = _delivery_state(task_env)
    if held_category.get(env_idx) is None and not _identify_held_enabled():
        # Without a wrist-camera lookup the only clue to the carried category is
        # the pick we hovered for.
        return False
    return delivered_closure.get(env_idx) != closed_at


def _reset_poses(
    task_env: Any,
    env_idx: int,
    observation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Cache the episode's start pose, which is the origin the reward checks."""
    if not hasattr(task_env, "_p1_reset_poses"):
        task_env._p1_reset_poses = {}
    if env_idx not in task_env._p1_reset_poses:
        state = observation["state"]
        task_env._p1_reset_poses[env_idx] = {
            arm: _state_vector(state, f"{arm}_ee_pose", 7).copy()
            for arm in ("left", "right")
        }
    return task_env._p1_reset_poses[env_idx]


def _make_reset_action(
    task_env: Any,
    env_idx: int,
    observation: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str, np.ndarray]:
    """Return both arms to the start pose, closing out the instruction.

    The instruction ends with "then reset the robot arm", and success also
    requires `all_robot_back_to_origin`, which the partial score ignores. The
    locator reporting an empty table is the visual cue that this step is due.
    """
    poses = _reset_poses(task_env, env_idx, observation)
    state = observation["state"]
    action = {
        "left_ee_pose": poses["left"].copy(),
        "right_ee_pose": poses["right"].copy(),
        # The agent only guides poses. Opening or closing a gripper is Pi_05's
        # decision, so the observed values are passed through unchanged.
        "left_ee_joint_state": _state_vector(state, "left_ee_joint_state", 1, default=1.0),
        "right_ee_joint_state": _state_vector(state, "right_ee_joint_state", 1, default=1.0),
    }
    print(
        f"[P1-gaze] env={env_idx} table_clear reset_to_origin "
        f"left={poses['left'][:3].round(3).tolist()} "
        f"right={poses['right'][:3].round(3).tolist()}",
        flush=True,
    )
    return action, "left", poses["left"][:3].copy()


def _neutral_quaternions(
    task_env: Any,
    env_idx: int,
    observation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not hasattr(task_env, "_p1_neutral_quaternions"):
        task_env._p1_neutral_quaternions = {}
    if env_idx not in task_env._p1_neutral_quaternions:
        state = observation["state"]
        task_env._p1_neutral_quaternions[env_idx] = {
            arm: _state_vector(state, f"{arm}_ee_pose", 7)[3:].copy()
            for arm in ("left", "right")
        }
    return task_env._p1_neutral_quaternions[env_idx]


def _mapper_from_env() -> TablePlaneMapper:
    config = TableMapConfig(
        table_z=float(os.environ.get("P1_TABLE_Z", "0.765")),
        hover_height=float(os.environ.get("P1_HOVER_HEIGHT", "0.10")),
    )
    return TablePlaneMapper(config)


def _enable_camera_calibration(task_env: Any) -> bool:
    obs_manager = getattr(task_env, "obs_manager", None)
    if obs_manager is None:
        return False
    obs_manager.collect_intrinsic_matrix = True
    obs_manager.collect_extrinsic_matrix = True
    return True


def _env_origin(task_env: Any, env_idx: int) -> np.ndarray:
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


def _as_hwc_uint8(image: Any) -> np.ndarray:
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


def _locate_target(image: np.ndarray, instruction: str) -> Mapping[str, Any]:
    locator_url = os.environ.get("P1_LOCATOR_URL")
    if not locator_url:
        raise RuntimeError("P1_LOCATOR_URL is not set.")
    payload = json.dumps(
        {
            "shape": list(image.shape),
            "rgb_u8": base64.b64encode(image.tobytes()).decode("ascii"),
            "instruction": instruction,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{locator_url.rstrip('/')}/locate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=float(os.environ.get("P1_LOCATOR_TIMEOUT_S", "180"))) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result


def _post_locator(path: str, payload: dict[str, Any]) -> Mapping[str, Any]:
    locator_url = os.environ.get("P1_LOCATOR_URL")
    if not locator_url:
        raise RuntimeError("P1_LOCATOR_URL is not set.")
    req = request.Request(
        f"{locator_url.rstrip('/')}/{path.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=float(os.environ.get("P1_LOCATOR_TIMEOUT_S", "180"))) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result


def _classify_held(image: np.ndarray, instruction: str) -> Mapping[str, Any]:
    return _post_locator(
        "classify_held",
        {
            "shape": list(image.shape),
            "rgb_u8": base64.b64encode(image.tobytes()).decode("ascii"),
            "instruction": instruction,
        },
    )


def _locate_baskets(image: np.ndarray) -> Mapping[str, Any]:
    locator_url = os.environ.get("P1_LOCATOR_URL")
    if not locator_url:
        raise RuntimeError("P1_LOCATOR_URL is not set.")
    payload = json.dumps(
        {
            "shape": list(image.shape),
            "rgb_u8": base64.b64encode(image.tobytes()).decode("ascii"),
        }
    ).encode("utf-8")
    req = request.Request(
        f"{locator_url.rstrip('/')}/locate_baskets",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=float(os.environ.get("P1_LOCATOR_TIMEOUT_S", "180"))) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result


def _locate_targets(
    images: list[np.ndarray],
    instructions: list[str],
) -> list[Mapping[str, Any]]:
    if len(images) != len(instructions):
        raise ValueError("images and instructions must have equal length.")
    if len(images) == 1:
        return [_locate_target(images[0], instructions[0])]
    locator_url = os.environ.get("P1_LOCATOR_URL")
    if not locator_url:
        raise RuntimeError("P1_LOCATOR_URL is not set.")
    items = [
        {
            "shape": list(image.shape),
            "rgb_u8": base64.b64encode(image.tobytes()).decode("ascii"),
            "instruction": instruction,
        }
        for image, instruction in zip(images, instructions)
    ]
    req = request.Request(
        f"{locator_url.rstrip('/')}/locate_batch",
        data=json.dumps({"items": items}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=float(os.environ.get("P1_LOCATOR_TIMEOUT_S", "180"))) as response:
        payload = json.loads(response.read())
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(images):
        raise RuntimeError(f"Locator returned an invalid batch: {payload!r}")
    return results


def _state_vector(state: Mapping[str, Any], key: str, size: int, default: float = 0.0) -> np.ndarray:
    value = state.get(key)
    if value is None:
        return np.full(size, default, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != size:
        raise ValueError(f"Observation state {key!r} has size {result.size}, expected {size}.")
    return result.copy()


def _cross_reach_limit() -> float:
    return float(os.environ.get("P1_CROSS_REACH_X", "0.15"))


def arm_can_reach(arm: str, target_x: float, cross_reach: float) -> bool:
    """Each arm covers its own half of the table plus a small inner overlap."""
    if arm == "left":
        return target_x <= cross_reach
    return target_x >= -cross_reach


def select_pick_arm(object_x: float) -> str:
    """Use the arm on the object's side of the table.

    Pi_05 picks its arm from the object's position, so any other choice puts two
    arms on one object with the hovered one empty. Choosing by destination
    basket was tried and produced exactly that: the left arm was hovered over an
    object at x=+0.06 while Pi_05 closed the right gripper.
    """
    return "left" if object_x < 0.0 else "right"


def closed_arm(observation: Mapping[str, Any]) -> str | None:
    """Return the arm whose gripper is closed, i.e. the one holding an object."""
    state = observation.get("state", {})
    threshold = _gripper_open_threshold()
    for arm in ("left", "right"):
        value = state.get(f"{arm}_ee_joint_state")
        if value is None:
            continue
        if float(np.asarray(value, dtype=np.float32).reshape(-1)[0]) < threshold:
            return arm
    return None


def _build_hover_action(
    observation: Mapping[str, Any],
    hover_xyz: np.ndarray,
    neutral_quaternions: Mapping[str, np.ndarray] | None = None,
    arm: str | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    state = observation["state"]
    left_pose = _state_vector(state, "left_ee_pose", 7)
    right_pose = _state_vector(state, "right_ee_pose", 7)
    if not np.any(left_pose[3:]):
        left_pose[3] = 1.0
    if not np.any(right_pose[3:]):
        right_pose[3] = 1.0
    if neutral_quaternions is not None:
        left_pose[3:] = neutral_quaternions["left"]
        right_pose[3:] = neutral_quaternions["right"]

    selected = arm or ("left" if hover_xyz[0] < 0.0 else "right")
    if selected == "left":
        left_pose[:3] = hover_xyz
    else:
        right_pose[:3] = hover_xyz

    action = {
        "left_ee_pose": left_pose,
        "right_ee_pose": right_pose,
        "left_ee_joint_state": _state_vector(state, "left_ee_joint_state", 1, default=1.0),
        "right_ee_joint_state": _state_vector(state, "right_ee_joint_state", 1, default=1.0),
    }
    print(
        f"[P1-gaze] arm={selected}, hover_xyz={hover_xyz.round(4).tolist()}",
        flush=True,
    )
    return action, selected


def _log_hover_error(
    observation: Mapping[str, Any],
    arm: str,
    hover_xyz: np.ndarray,
    *,
    env_idx: int,
) -> float:
    actual_xyz = _state_vector(observation["state"], f"{arm}_ee_pose", 7)[:3]
    error_m = float(np.linalg.norm(actual_xyz - hover_xyz))
    print(
        f"[P1-gaze] env={env_idx} hover_reached arm={arm} "
        f"actual_xyz={actual_xyz.round(4).tolist()} error_m={error_m:.4f}",
        flush=True,
    )
    return error_m


def _hover_error(
    observation: Mapping[str, Any],
    arm: str,
    hover_xyz: np.ndarray,
) -> float:
    actual_xyz = _state_vector(observation["state"], f"{arm}_ee_pose", 7)[:3]
    return float(np.linalg.norm(actual_xyz - hover_xyz))


def _make_hover_action(
    observation: Mapping[str, Any],
    mapper: TablePlaneMapper,
    env_origin: np.ndarray | None = None,
    grounding: Mapping[str, Any] | None = None,
    neutral_quaternions: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], str, np.ndarray]:
    camera_name, camera = get_head_camera(observation)
    image = _as_hwc_uint8(camera["color"])
    if grounding is None:
        grounding = _locate_target(image, str(observation.get("instruction", "")))
    bbox = grounding.get("bbox_2d")
    if bbox is None:
        raise ValueError(f"Locator found no target: {grounding!r}")
    pixel = bbox_to_table_anchor(bbox, image.shape)
    intrinsic, extrinsic = get_camera_calibration(camera)
    intrinsic = intrinsic.astype(np.float64, copy=True)
    intrinsic[1, 1] *= float(os.environ.get("P1_HEAD_FY_SCALE", "1.167741"))
    extrinsic = extrinsic.astype(np.float64, copy=True)
    if env_origin is not None:
        # RoboDojo reports camera-to-Isaac-world poses, while EE actions and
        # tabletop limits use coordinates relative to each replicated env.
        extrinsic[:3, 3] -= env_origin
    hover_xyz = mapper.hover_target(pixel, intrinsic, extrinsic)
    print(
        f"[P1-gaze] camera={camera_name}, label={grounding.get('label')!r}, "
        f"bbox_1000={bbox}, pixel={pixel.round(1).tolist()}",
        flush=True,
    )
    action, arm = _build_hover_action(
        observation,
        hover_xyz,
        neutral_quaternions=neutral_quaternions,
        arm=select_pick_arm(float(hover_xyz[0])),
    )
    return action, arm, hover_xyz


def _camera_projection(
    observation: Mapping[str, Any],
    env_origin: np.ndarray | None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    camera_name, camera = get_head_camera(observation)
    image = _as_hwc_uint8(camera["color"])
    intrinsic, extrinsic = get_camera_calibration(camera)
    intrinsic = intrinsic.astype(np.float64, copy=True)
    intrinsic[1, 1] *= float(os.environ.get("P1_HEAD_FY_SCALE", "1.167741"))
    extrinsic = extrinsic.astype(np.float64, copy=True)
    if env_origin is not None:
        extrinsic[:3, 3] -= env_origin
    return camera_name, image, intrinsic, extrinsic


def _basket_targets(
    task_env: Any,
    env_idx: int,
    observation: Mapping[str, Any],
    mapper: TablePlaneMapper,
) -> dict[str, np.ndarray]:
    """Locate the three baskets once per episode and cache their drop points."""
    _, _, basket_targets = _delivery_state(task_env)
    cached = basket_targets.get(env_idx)
    if cached is not None:
        return cached

    _, image, intrinsic, extrinsic = _camera_projection(
        observation,
        _env_origin(task_env, env_idx),
    )
    grounding = _locate_baskets(image)
    depth_offset = _basket_depth_offset()
    targets: dict[str, np.ndarray] = {}
    for side in BASKET_SIDES:
        pixel = bbox_to_basket_anchor(grounding[side], image.shape)
        rim_xyz = mapper.hover_target(
            pixel,
            intrinsic,
            extrinsic,
            hover_height=_deliver_height(),
            y_limits=_basket_y_limits(),
        )
        # The visible rim sits at the basket's near edge; step inward so the
        # object is released over the opening rather than onto the lip.
        rim_xyz[1] += depth_offset
        targets[side] = rim_xyz
    basket_targets[env_idx] = targets
    print(
        f"[P1-gaze] env={env_idx} baskets="
        + ", ".join(f"{side}:{targets[side].round(3).tolist()}" for side in BASKET_SIDES),
        flush=True,
    )
    return targets


def _held_category(
    task_env: Any,
    env_idx: int,
    observation: Mapping[str, Any],
    arm: str,
) -> str | None:
    """Ask the wrist camera what the gripper holds, falling back to our pick.

    Pi_05 chooses its own grasping arm and sometimes its own object, so the
    category recorded when we hovered is not reliable evidence of what is
    actually being carried.
    """
    held_category, _, _ = _delivery_state(task_env)
    recorded = held_category.get(env_idx)
    if not _identify_held_enabled():
        return recorded

    camera = observation.get("vision", {}).get(f"cam_{arm}_wrist")
    if not isinstance(camera, Mapping) or camera.get("color") is None:
        return recorded
    try:
        result = _classify_held(
            _as_hwc_uint8(camera["color"]),
            str(observation.get("instruction", "")),
        )
    except Exception as exc:
        print(
            f"[P1-gaze] env={env_idx} held lookup failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return recorded

    label = result.get("label")
    if not label:
        return recorded
    if label != recorded:
        print(
            f"[P1-gaze] env={env_idx} held={label!r} (hovered {recorded!r})",
            flush=True,
        )
    held_category[env_idx] = label
    return label


def _make_deliver_action(
    task_env: Any,
    env_idx: int,
    observation: Mapping[str, Any],
    mapper: TablePlaneMapper,
) -> tuple[dict[str, np.ndarray], str, np.ndarray]:
    arm = closed_arm(observation)
    if arm is None:
        raise ValueError("No gripper is closed, so nothing is being carried.")

    instruction = str(observation.get("instruction", ""))
    category = _held_category(task_env, env_idx, observation, arm)
    side = destination_side(instruction, category)
    if side is None:
        raise ValueError(f"Instruction does not bind {category!r} to a basket.")

    target_xyz = _basket_targets(task_env, env_idx, observation, mapper)[side]
    mode = _deliver_mode()
    cross_reach = _cross_reach_limit()
    cross_body = not arm_can_reach(arm, float(target_xyz[0]), cross_reach)
    current_xyz = _state_vector(observation["state"], f"{arm}_ee_pose", 7)[:3]

    if cross_body:
        # Commanding the basket itself measured a 0.58-0.61 m median residual
        # across the body, so the basket is genuinely out of reach for this arm.
        if not _relay_enabled():
            raise ValueError(
                f"The {arm} arm cannot reach the {side} basket at "
                f"x={target_xyz[0]:.3f}; leaving the transfer to Pi_05."
            )
        hover_xyz = relay_target(
            current_xyz,
            float(target_xyz[0]),
            cross_reach,
            min_z=float(target_xyz[2]),
            z_margin=_deliver_z_margin(),
        )
        action, selected = _build_hover_action(observation, hover_xyz, arm=arm)
        print(
            f"[P1-gaze] env={env_idx} relay category={category!r} "
            f"side={side} arm={selected} target={hover_xyz.round(3).tolist()}",
            flush=True,
        )
        return action, selected, hover_xyz

    hover_xyz = deliver_target(
        current_xyz,
        target_xyz,
        mode,
        min_z=float(target_xyz[2]),
        z_margin=_deliver_z_margin(),
        depth_step=_deliver_depth_step(),
    )
    # Keep the wrist orientation Pi_05 adopted for carrying. Forcing the episode
    # reset orientation here over-constrains the IK at the outer baskets.
    action, selected = _build_hover_action(observation, hover_xyz, arm=arm)
    print(
        f"[P1-gaze] env={env_idx} deliver category={category!r} "
        f"side={side} arm={selected} mode={mode} "
        f"cross_body={cross_body} target={hover_xyz.round(3).tolist()}",
        flush=True,
    )
    return action, selected, hover_xyz


def _record_pick(task_env: Any, env_idx: int, grounding: Mapping[str, Any]) -> None:
    """Remember the category just targeted so delivery knows its basket."""
    held_category, delivered_closure, _ = _delivery_state(task_env)
    held_category[env_idx] = grounding.get("label")
    delivered_closure.pop(env_idx, None)


def _mark_delivered(task_env: Any, env_idx: int) -> None:
    gripper_closed_step = _prime_state(task_env)[3]
    _, delivered_closure, _ = _delivery_state(task_env)
    delivered_closure[env_idx] = gripper_closed_step.get(env_idx)


def _hover_stall_steps() -> int:
    return max(1, int(os.environ.get("P1_HOVER_STALL_STEPS", "6")))


def _run_hover(
    task_env: Any,
    action: dict[str, np.ndarray],
    arm: str,
    hover_xyz: np.ndarray,
    env_idx: int,
) -> None:
    """Hold the pose until the error clears the tolerance or stops improving.

    Some commanded poses are simply out of reach for a fixed wrist orientation.
    Holding those for the full budget spends simulator steps the episode needs
    elsewhere, so give up once the error stops falling.
    """
    observation = None
    best_error = float("inf")
    stalled = 0
    for step_idx in range(_hover_max_steps()):
        task_env.take_action(action)
        if step_idx + 1 < _hover_dwell_steps():
            continue
        observation = task_env.get_obs()
        error = _hover_error(observation, arm, hover_xyz)
        if error <= _hover_tolerance_m():
            break
        if error < best_error - 1e-4:
            best_error = error
            stalled = 0
        else:
            stalled += 1
            if stalled >= _hover_stall_steps():
                break
    _log_hover_error(observation or task_env.get_obs(), arm, hover_xyz, env_idx=env_idx)


def _prime_single(task_env: Any, mapper: TablePlaneMapper | None = None) -> None:
    observation = task_env.get_obs()
    running = task_env.get_running_env_idx_list()
    env_idx = running[0] if running else 0
    mapper = mapper or _mapper_from_env()

    if _should_reprime(observation, env_idx, task_env):
        try:
            camera = get_head_camera(observation)[1]
            grounding = _locate_target(
                _as_hwc_uint8(camera["color"]),
                str(observation.get("instruction", "")),
            )
            if grounding.get("bbox_2d") is None and "bbox_2d" in grounding:
                action, arm, hover_xyz = _make_reset_action(
                    task_env, env_idx, observation
                )
                _run_hover(task_env, action, arm, hover_xyz, env_idx)
                return
            action, arm, hover_xyz = _make_hover_action(
                observation,
                mapper,
                env_origin=_env_origin(task_env, env_idx),
                grounding=grounding,
                neutral_quaternions=_neutral_quaternions(task_env, env_idx, observation),
            )
            _record_pick(task_env, env_idx, grounding)
        except Exception as exc:
            print(f"[P1-gaze] skip pre-move: {type(exc).__name__}: {exc}", flush=True)
            return
    elif _should_deliver(observation, env_idx, task_env):
        try:
            action, arm, hover_xyz = _make_deliver_action(
                task_env, env_idx, observation, mapper
            )
            _mark_delivered(task_env, env_idx)
        except Exception as exc:
            print(f"[P1-gaze] skip delivery: {type(exc).__name__}: {exc}", flush=True)
            _mark_delivered(task_env, env_idx)
            return
    else:
        return

    _run_hover(task_env, action, arm, hover_xyz, env_idx)


def _prime_batch(task_env: Any, env_idx_list: list[int] | None = None, mapper: TablePlaneMapper | None = None) -> None:
    if env_idx_list is None:
        env_idx_list = task_env.get_running_env_idx_list()
    if not env_idx_list:
        return
    observations = task_env.get_obs_batch(env_idx_list)
    mapper = mapper or _mapper_from_env()

    approach_envs = []
    approach_observations = []
    deliver_envs = []
    deliver_observations = []
    for env_idx, observation in zip(env_idx_list, observations):
        if _should_reprime(observation, env_idx, task_env):
            approach_envs.append(env_idx)
            approach_observations.append(observation)
        elif _should_deliver(observation, env_idx, task_env):
            deliver_envs.append(env_idx)
            deliver_observations.append(observation)

    if not approach_envs and not deliver_envs:
        return

    actions = []
    primed_envs = []
    hover_metadata = []

    if approach_envs:
        try:
            images = [
                _as_hwc_uint8(get_head_camera(observation)[1]["color"])
                for observation in approach_observations
            ]
            groundings = _locate_targets(
                images,
                [str(observation.get("instruction", "")) for observation in approach_observations],
            )
        except Exception as exc:
            print(f"[P1-gaze] skip batch pre-move: {type(exc).__name__}: {exc}", flush=True)
            groundings = []

        for env_idx, observation, grounding in zip(
            approach_envs,
            approach_observations,
            groundings,
        ):
            if grounding.get("bbox_2d") is None and "bbox_2d" in grounding:
                action, arm, hover_xyz = _make_reset_action(
                    task_env, env_idx, observation
                )
                actions.append(action)
                primed_envs.append(env_idx)
                hover_metadata.append((arm, hover_xyz))
                continue
            try:
                action, arm, hover_xyz = _make_hover_action(
                    observation,
                    mapper,
                    env_origin=_env_origin(task_env, env_idx),
                    grounding=grounding,
                    neutral_quaternions=_neutral_quaternions(task_env, env_idx, observation),
                )
                actions.append(action)
                primed_envs.append(env_idx)
                hover_metadata.append((arm, hover_xyz))
                _record_pick(task_env, env_idx, grounding)
            except Exception as exc:
                print(
                    f"[P1-gaze] env={env_idx} skip pre-move: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    for env_idx, observation in zip(deliver_envs, deliver_observations):
        try:
            action, arm, hover_xyz = _make_deliver_action(
                task_env, env_idx, observation, mapper
            )
            actions.append(action)
            primed_envs.append(env_idx)
            hover_metadata.append((arm, hover_xyz))
        except Exception as exc:
            print(
                f"[P1-gaze] env={env_idx} skip delivery: {type(exc).__name__}: {exc}",
                flush=True,
            )
        # Mark either way so a failed delivery does not retry every chunk.
        _mark_delivered(task_env, env_idx)

    if actions:
        pending = list(range(len(actions)))
        tolerance = _hover_tolerance_m()
        stall_limit = _hover_stall_steps()
        best_error = {index: float("inf") for index in pending}
        stalled = {index: 0 for index in pending}
        for step_idx in range(_hover_max_steps()):
            task_env.take_action_batch(
                [actions[index] for index in pending],
                [primed_envs[index] for index in pending],
            )
            if step_idx + 1 < _hover_dwell_steps():
                continue
            pending_observations = task_env.get_obs_batch(
                [primed_envs[index] for index in pending]
            )
            still_pending = []
            for index, observation in zip(pending, pending_observations):
                error = _hover_error(
                    observation,
                    hover_metadata[index][0],
                    hover_metadata[index][1],
                )
                if error <= tolerance:
                    continue
                if error < best_error[index] - 1e-4:
                    best_error[index] = error
                    stalled[index] = 0
                else:
                    stalled[index] += 1
                    if stalled[index] >= stall_limit:
                        continue
                still_pending.append(index)
            pending = still_pending
            if not pending:
                break
        reached_observations = task_env.get_obs_batch(primed_envs)
        for env_idx, observation, (arm, hover_xyz) in zip(
            primed_envs,
            reached_observations,
            hover_metadata,
        ):
            _log_hover_error(observation, arm, hover_xyz, env_idx=env_idx)


def _freeze_policy_actions(actions: Any) -> Any:
    """Snapshot a Pi_05 chunk at the model boundary.

    P1 may interrupt a chunk, but it must never rewrite an action inside it.
    """
    return copy.deepcopy(actions)


def _assert_policy_action_unchanged(
    action: Mapping[str, Any],
    frozen_action: Mapping[str, Any],
) -> None:
    """Fail closed if P1 rewrites any field produced by Pi_05."""
    if action.keys() != frozen_action.keys():
        raise RuntimeError("P1 must forward every Pi_05 action field unchanged")
    for key in action:
        current = np.asarray(action[key])
        frozen = np.asarray(frozen_action[key])
        if (
            current.shape != frozen.shape
            or current.dtype != frozen.dtype
            or current.tobytes() != frozen.tobytes()
        ):
            raise RuntimeError(
                f"P1 modified Pi_05 action field {key!r} inside a policy chunk"
            )


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    calibration_enabled = _enable_camera_calibration(TASK_ENV)
    mapper = _mapper_from_env() if calibration_enabled else None
    _reset_prime_tracking(TASK_ENV)

    while not TASK_ENV.is_episode_end():
        if calibration_enabled and _reprime_enabled():
            _prime_single(TASK_ENV, mapper=mapper)

        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        frozen_actions = _freeze_policy_actions(actions)
        for action_idx, action in enumerate(actions):
            _assert_policy_action_unchanged(action, frozen_actions[action_idx])
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            observation = TASK_ENV.get_obs()
            reason = _chunk_interrupt(observation, 0, TASK_ENV)
            if reason is not None:
                print(
                    f"[P1-gaze] env=0 step={TASK_ENV.take_action_cnt[0]} "
                    f"chunk_interrupt reason={reason}",
                    flush=True,
                )
                break
            model_client.call(func_name="update_obs", obs=observation)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    calibration_enabled = _enable_camera_calibration(TASK_ENV)
    mapper = _mapper_from_env() if calibration_enabled else None
    _reset_prime_tracking(TASK_ENV)

    intervene = calibration_enabled and (_reprime_enabled() or _delivery_enabled())

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        if intervene:
            _prime_batch(TASK_ENV, env_idx_list=env_idx_list, mapper=mapper)
            # A hover may consume the last remaining simulator steps for only a
            # subset of the batch. Refresh before requesting and executing the
            # next Pi_05 chunk so actions and environment indices stay aligned.
            env_idx_list = TASK_ENV.get_running_env_idx_list()
            if not env_idx_list:
                continue

        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        frozen_actions = _freeze_policy_actions(actions)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            frozen_action_list = [
                env_actions[action_idx] for env_actions in frozen_actions
            ]
            for action, frozen_action in zip(
                current_action_list, frozen_action_list
            ):
                _assert_policy_action_unchanged(action, frozen_action)
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active_batch_idx]
            frozen_actions = [frozen_actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            if not env_idx_list:
                break
            observations = TASK_ENV.get_obs_batch(env_idx_list)
            interrupts = {
                env_idx: reason
                for env_idx, observation in zip(env_idx_list, observations)
                if (reason := _chunk_interrupt(observation, env_idx, TASK_ENV))
            }
            if interrupts:
                print(f"[P1-gaze] chunk_interrupt {interrupts}", flush=True)
                break
            model_client.call(
                func_name="update_obs_batch",
                obs=observations,
            )
