"""Run the upstream inspect-robots-agent policy inside RoboDojo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import yaml

from .policy import LlmPolicy, RoboDojoActionSpec
from .types import Observation


def _observation(task_env: Any, policy_step: int) -> Observation:
    raw = task_env.get_obs()
    camera_views = raw.get("vision") or {}
    images = {
        name: np.asarray(view["color"])
        for name, view in camera_views.items()
        if isinstance(view, dict) and "color" in view
    }
    depth = {
        f"{name}_depth": np.asarray(view["depth"])
        for name, view in camera_views.items()
        if isinstance(view, dict) and "depth" in view
    }
    extra = dict(raw.get("extra") or {})
    extra.update(depth)
    for key in ("additional_info", "data_format_version", "env_idx"):
        if key in raw:
            extra[key] = raw[key]
    env_idx = raw.get("env_idx", 0)
    env_seeds = getattr(task_env, "env_seeds", None)
    if isinstance(env_idx, int) and env_seeds is not None:
        try:
            layout_id = env_seeds[env_idx]
        except (IndexError, KeyError, TypeError):
            layout_id = None
        if layout_id is not None:
            extra["layout_id"] = int(layout_id)
    return Observation(
        images=images,
        state=raw.get("state") or {},
        instruction=(
            raw.get("instruction")
            or raw.get("instructions")
            or getattr(task_env, "instruction", None)
        ),
        step=policy_step,
        extra=extra,
    )


def _action_spec(task_env: Any) -> RoboDojoActionSpec:
    """Read bounds and control rate from the live RoboDojo embodiment.

    inspect-robots-agent builds its tools from EmbodimentInfo; using the live
    Isaac articulation here is the equivalent boundary. Guessed constants are
    deliberately forbidden.
    """
    if os.environ.get("P3_ACTION_TYPE", "joint") != "joint":
        raise ValueError(
            "Agent_P3 supports only joint control: RoboDojo's EE action is "
            "quaternion+IK, while inspect-robots-agent rejects quaternion pose "
            "spaces as unsafe for per-dimension interpolation"
        )
    manager = getattr(task_env, "robot_manager", None)
    if manager is None:
        if os.environ.get("EVAL_ENV_TYPE") != "debug":
            raise RuntimeError("RoboDojo robot_manager is required to derive action bounds")
        return _debug_action_spec()

    labels: list[str] = []
    low: list[float] = []
    high: list[float] = []
    sides: list[str] = []
    for index, robot in enumerate(manager.robot_list):
        if robot.type != "target":
            continue
        side = str(robot.arm_name).removesuffix("_arm")
        sides.append(side)
        asset = manager.robot_key[index]
        limits = (
            asset.data.soft_joint_pos_limits[0, robot.arm_joint_indices]
            .detach()
            .cpu()
            .numpy()
        )
        names = tuple(str(name) for name in robot.arm_joints_name)
        if limits.shape != (len(names), 2):
            raise RuntimeError(
                f"{side} joint limits have shape {limits.shape}, expected {(len(names), 2)}"
            )
        labels.extend(f"{side}_{name}" for name in names)
        low.extend(float(value) for value in limits[:, 0])
        high.extend(float(value) for value in limits[:, 1])
        labels.append(f"{side}_gripper")
        low.append(0.0)
        high.append(1.0)

    control_hz = float(getattr(task_env.obs_manager, "collect_freq", 0.0))
    docs = (
        "RoboDojo dual ARX X5 joint-position control. Dimensions are ordered "
        + ", ".join(labels)
        + ". Arm values are absolute radians. Grippers are normalized: "
        "0 means closed and 1 means open. Each arm is mounted on the table; "
        "its joint axes are in that arm's own base frame."
    )
    spec = RoboDojoActionSpec(
        labels=tuple(labels),
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        control_hz=control_hz,
        docs=docs,
    )
    if len(sides) != 2 or len(spec.labels) != 14:
        raise RuntimeError(
            f"Agent_P3 expected dual 6-DoF ARX X5, got sides={sides}, "
            f"dimensions={len(spec.labels)}"
        )
    return spec


def _debug_action_spec() -> RoboDojoActionSpec:
    """Build the debug-only spec from RoboDojo's URDF and env configuration."""
    root_value = os.environ.get("ROBODOJO_ROOT")
    if not root_value:
        raise RuntimeError("ROBODOJO_ROOT is required for Agent_P3 debug evaluation")
    root = Path(root_value)
    urdf = ElementTree.parse(root / "Assets/Robots/x5/X5A.urdf")
    arm_joints = []
    for joint in urdf.getroot().findall("joint"):
        name = joint.get("name", "")
        if name not in {f"joint{i}" for i in range(1, 7)}:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"URDF joint {name} has no limit")
        arm_joints.append(
            (name, float(limit.get("lower", "")), float(limit.get("upper", "")))
        )
    arm_joints.sort(key=lambda item: int(item[0].removeprefix("joint")))
    if len(arm_joints) != 6:
        raise RuntimeError(f"expected six X5 arm joints in URDF, got {arm_joints}")
    env_config = yaml.safe_load(
        (root / "env_cfg/arx_x5.yml").read_text(encoding="utf-8")
    )
    control_hz = float(env_config["observation"]["collect_freq"])
    labels: list[str] = []
    low: list[float] = []
    high: list[float] = []
    for side in ("left", "right"):
        labels.extend(f"{side}_{name}" for name, _, _ in arm_joints)
        low.extend(lower for _, lower, _ in arm_joints)
        high.extend(upper for _, _, upper in arm_joints)
        labels.append(f"{side}_gripper")
        low.append(0.0)
        high.append(1.0)
    return RoboDojoActionSpec(
        labels=tuple(labels),
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        control_hz=control_hz,
        docs=(
            "RoboDojo debug dual ARX X5 joint-position control. Bounds and "
            "control rate were loaded from Assets/Robots/x5/X5A.urdf and "
            "env_cfg/arx_x5.yml. Grippers are normalized 0 closed to 1 open."
        ),
    )


def _mark_incomplete_episode_failed(task_env: Any) -> None:
    """A policy that stops early is a failure, never an implicit success."""
    if task_env.is_episode_end():
        return
    running = (
        task_env.get_running_env_idx_list()
        if hasattr(task_env, "get_running_env_idx_list")
        else [0]
    )
    success = getattr(task_env, "success", None)
    if success is None:
        raise RuntimeError(
            "P3 policy stopped before official termination and the "
            "environment does not expose a failure state."
        )
    for env_idx in running:
        success[env_idx] = False
    task_env.is_episode_end()
    print(f"[P3] policy stopped early; marked envs failed: {running}", flush=True)


def _write_transcript(
    policy: LlmPolicy,
    task_env: Any,
    *,
    inspect_metadata: dict[str, Any],
    termination_reason: str | None,
) -> None:
    directory = os.environ.get("P3_TRACE_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    policy_config = policy.audit_config()
    result = {
        "task": getattr(task_env, "task_name", None),
        "layout_id": policy_config["scene"]["init_seed"],
        "llm_calls": policy.calls,
        "usage": policy.usage_totals,
        "hindsight": policy.hindsight,
        "termination_reason": termination_reason,
        "official_success": list(getattr(task_env, "success", [])),
        "policy_config": policy_config,
        "inspect_metadata": inspect_metadata,
        "transcript": policy.transcript(),
    }
    (path / "p3_transcript.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (path / "p3_config.json").write_text(
        json.dumps(
            policy_config,
            indent=2,
        ),
        encoding="utf-8",
    )


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    del model_client  # P3 serves no VLA
    if os.environ.get("P3_DEPTH", "render") == "render":
        TASK_ENV.obs_manager.collect_depth = True
    policy = LlmPolicy(action_spec=_action_spec(TASK_ENV))
    policy.reset()
    policy_step = 0
    stopped = False
    termination_reason: str | None = None
    inspect_metadata: dict[str, Any] = {}
    trace_dir = os.environ.get("P3_TRACE_DIR")
    run_id = os.environ.get("ROBODOJO_RUN_ID", "agent-p3")
    if not trace_dir:
        print(
            "[P3] P3_TRACE_DIR is unset; inspect wire capture and transcript "
            "are not archived for this episode",
            flush=True,
        )
    try:
        first = _observation(TASK_ENV, policy_step)
        policy.prepare(first)
        if trace_dir:
            policy.start_capture(trace_dir, run_id)
        while not TASK_ENV.is_episode_end():
            observation = first if policy_step == 0 else _observation(TASK_ENV, policy_step)
            chunk = policy.act(observation)
            policy_step += 1
            for action in chunk.actions:
                TASK_ENV.take_action(dict(action.data))
                if action.meta.get("request_stop"):
                    stopped = True
                    reason = action.meta.get("stop_reason")
                    termination_reason = str(reason)
                    print(f"[P3] policy {reason}: {action.meta.get('stop_detail')}", flush=True)
                    break
                if TASK_ENV.is_episode_end():
                    break
            if stopped:
                break
    except Exception as error:
        print(f"[P3] {error}", flush=True)
        stopped = True
        termination_reason = "policy_error"
    finally:
        if stopped or not TASK_ENV.is_episode_end():
            _mark_incomplete_episode_failed(TASK_ENV)
        ended = TASK_ENV.is_episode_end()
        policy_requested_stop = termination_reason in {"done", "give_up"}
        truncated = bool(
            policy_requested_stop
            or (
                ended
                and not any(bool(value) for value in getattr(TASK_ENV, "success", []))
                and termination_reason is None
            )
        )
        if trace_dir:
            inspect_metadata = policy.finish_capture(
                trace_dir,
                run_id,
                terminated=ended and not truncated and termination_reason is None,
                truncated=truncated,
                termination_reason=termination_reason,
            )
        print(
            f"[P3] {policy.calls} llm calls, usage={policy.usage_totals}",
            flush=True,
        )
        _write_transcript(
            policy,
            TASK_ENV,
            inspect_metadata=inspect_metadata,
            termination_reason=termination_reason,
        )


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    """P3 is one conversation per environment; RoboDojo already forces num_envs=1."""
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise RuntimeError(
            "Agent_P3 is one conversation per environment; "
            "keep eval_batch=false so RoboDojo runs a single env"
        )
    eval_one_episode(TASK_ENV, model_client)
