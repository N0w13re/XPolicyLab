"""Speed-limited motion tools, ported from inspect-robots-agent.

The LLM names partial absolute targets. This module interpolates them from the
observed state at a declared fraction of each dimension's range per second,
with the same 5%-of-range per-step ceiling and 10 s playout cap as
`plugins/inspect-robots-agent`. The model never emits a raw 14-D dump.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .types import Action, ActionChunk, ActionSpace, Observation

_FALLBACK_HZ = 10.0
_MAX_DURATION_S = 10.0
_BACKSTOP_STEP_FRAC = 0.05
_DEFAULT_SPEED_FRAC = 0.1
_RELATIVE_HEADROOM = 1e-6

_HINDSIGHT_DESCRIPTION = (
    "What do you know now that you wish you had known at the start of this episode? "
    "Concrete, transferable facts about this rig, task, or embodiment (camera mounting "
    "and extrinsics, table and base geometry, gripper axis and offsets, controller "
    "behavior, metric scale), written as advice to a future agent attempting the same "
    "task. Say 'none' if nothing qualifies."
)

# Named dimensions the model sees. Order matches JOINT_CHANNELS / EE_CHANNELS.
JOINT_LABELS: tuple[str, ...] = (
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
)

EE_LABELS: tuple[str, ...] = (
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
)

# Radians for arm joints; grippers are 0 closed – 1 open.
_JOINT_LOW = np.array([-math.pi] * 6 + [0.0] + [-math.pi] * 6 + [0.0], dtype=np.float64)
_JOINT_HIGH = np.array([math.pi] * 6 + [1.0] + [math.pi] * 6 + [1.0], dtype=np.float64)
_EE_LOW = np.array(
    [-2.0, -2.0, -0.5, -1.0, -1.0, -1.0, -1.0, 0.0] * 2, dtype=np.float64
)
_EE_HIGH = np.array(
    [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2, dtype=np.float64
)

EMBODIMENT_NOTES = (
    "Bimanual ARX X5. Joint order is left_j0..left_j5, left_gripper, "
    "right_j0..right_j5, right_gripper. Gripper channels are 0 closed to 1 open. "
    "Coordinates are absolute servo targets. Head and wrist RGB cameras look at "
    "the table; unnamed dimensions keep their current value."
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    chunk: ActionChunk | None = None
    error: str | None = None
    note: str = ""
    target: npt.NDArray[np.float64] | None = None


class Toolset:
    """Schemas and execution for one RoboDojo action space."""

    def __init__(
        self,
        *,
        action_space: ActionSpace,
        labels: tuple[str, ...],
        low: npt.NDArray[np.float64],
        high: npt.NDArray[np.float64],
        step_limits: npt.NDArray[np.float64],
        control_hz: float | None,
        bounds_text: str,
        pose: bool = False,
    ) -> None:
        if len(labels) != action_space.width:
            raise ValueError("label count must match the action-space width")
        self.action_space = action_space
        self._labels = labels
        self._index_by_label = {label: i for i, label in enumerate(labels)}
        self._low = low
        self._high = high
        self._step_limits = step_limits
        self._hz = control_hz
        self._resolved_hz = control_hz if control_hz is not None else _FALLBACK_HZ
        self._max_steps = math.ceil(_MAX_DURATION_S * self._resolved_hz)
        self._move_tool = "move_to" if pose else "move_joints"
        self._bounds_text = bounds_text
        self._pose = pose

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def move_tool(self) -> str:
        return self._move_tool

    def schemas(self) -> list[dict[str, Any]]:
        if self._pose:
            move_description = (
                "Move to absolute Cartesian end-effector targets (meters for "
                "positions, radians for rotations, per the dimension labels). "
                "The motion is a straight line interpolated at a fixed safe "
                "speed and the result reports its step count. Unnamed "
                "dimensions hold their current value. " + self._bounds_text
            )
        else:
            move_description = (
                "Move to absolute joint/dimension targets. The motion is smoothly "
                "interpolated at a fixed safe speed and the result reports its step "
                "count. Unnamed dimensions hold their current value. " + self._bounds_text
            )
        move = {
            "type": "function",
            "function": {
                "name": self._move_tool,
                "description": move_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "object",
                            "description": (
                                "Map of dimension name to value. Valid names: "
                                + ", ".join(self._labels)
                            ),
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "What you observe right now in the observation (images, if any, "
                                "and state), and why you chose this motion. The user reads these "
                                "notes live and in the saved transcript to follow what you see "
                                "and what you decide. Write for them, in one or two plain "
                                "sentences."
                            ),
                        },
                    },
                    "required": ["targets", "note"],
                },
            },
        }
        done = {
            "type": "function",
            "function": {
                "name": "done",
                "description": (
                    "Declare the task finished. The trial ends; a scorer judges success."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "hindsight": {
                            "type": "string",
                            "description": _HINDSIGHT_DESCRIPTION,
                        },
                    },
                    "required": ["summary", "hindsight"],
                },
            },
        }
        give_up = {
            "type": "function",
            "function": {
                "name": "give_up",
                "description": "Stop trying; the task cannot be completed. The trial ends.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "hindsight": {
                            "type": "string",
                            "description": _HINDSIGHT_DESCRIPTION,
                        },
                    },
                    "required": ["reason", "hindsight"],
                },
            },
        }
        return [move, done, give_up]

    def execute(self, call: ToolCall, observation: Observation) -> ToolResult:
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, ValueError):
            return ToolResult(error=f"arguments for {call.name} are not valid JSON")
        if not isinstance(arguments, dict):
            return ToolResult(error=f"arguments for {call.name} must be a JSON object")
        if call.name in ("done", "give_up"):
            return self._stop(call.name, arguments, observation)
        if call.name != self._move_tool:
            available = f"{self._move_tool}, done, give_up"
            return ToolResult(error=f"unknown tool {call.name!r}; available: {available}")
        return self._move(arguments, observation)

    def current_state(self, observation: Observation) -> npt.NDArray[np.float64]:
        return np.asarray(self.action_space.encode(observation.state), dtype=np.float64)

    def _decode(self, vector: npt.NDArray[np.float64], meta: dict[str, Any] | None = None) -> Action:
        action = self.action_space.decode(vector.tolist())
        if not meta:
            return action
        return Action(data=action.data, meta=meta)

    def _stop(self, name: str, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        data = self.current_state(observation)
        detail = str(arguments.get("summary") or arguments.get("reason") or "")
        meta: dict[str, Any] = {
            "request_stop": True,
            "stop_reason": name,
            "stop_detail": detail,
        }
        hindsight = arguments.get("hindsight")
        if isinstance(hindsight, str) and hindsight.strip() and hindsight.strip().lower() != "none":
            meta["stop_hindsight"] = hindsight.strip()
        action = self._decode(data, meta)
        return ToolResult(
            chunk=ActionChunk(actions=[action]),
            note=f"{name}: {detail}",
        )

    def _move(self, arguments: dict[str, Any], observation: Observation) -> ToolResult:
        current = self.current_state(observation)
        if not bool(np.all(np.isfinite(current))):
            raise ValueError("proprioceptive reference contains a non-finite value")

        call_note = arguments.get("note")
        if not isinstance(call_note, str) or not call_note.strip():
            return ToolResult(
                error="note is required: describe what you observe and why you chose this motion"
            )

        values = arguments.get("targets")
        if not isinstance(values, dict) or not values:
            return ToolResult(error="targets must be a non-empty object of name: value")

        vector = np.zeros(len(self._labels))
        named_indices: list[int] = []
        for label, raw in values.items():
            index = self._index_by_label.get(str(label))
            if index is None:
                return ToolResult(
                    error=f"unknown dimension {label!r}; valid names: {', '.join(self._labels)}"
                )
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            try:
                coerced = float(raw)
            except OverflowError:
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            if not np.isfinite(coerced):
                return ToolResult(error=f"value for {label!r} must be a finite number, got {raw!r}")
            vector[index] = coerced
            named_indices.append(index)

        target = current.copy()
        for label, index in zip(values, named_indices, strict=True):
            value = vector[index]
            if self._step_limits[index] == 0:
                if value != self._low[index]:
                    return ToolResult(
                        error=f"dimension {label} is fixed at {float(self._low[index])!r}"
                    )
            elif value < self._low[index] or value > self._high[index]:
                return ToolResult(
                    error=(
                        f"target for {label} is outside "
                        f"[{float(self._low[index])!r}, {float(self._high[index])!r}]"
                    )
                )
            target[index] = value

        ratios: list[np.float64] = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index in named_indices:
                distance = np.abs(np.subtract(target[index], current[index]))
                limit = self._step_limits[index]
                if distance > 0 and limit > 0:
                    ratios.append(np.divide(distance, limit))
            ratio = max(ratios, default=np.float64(0.0))
            headed_ratio = np.divide(ratio, 1.0 - _RELATIVE_HEADROOM)
        if headed_ratio > self._max_steps:
            return ToolResult(
                error=(
                    f"requested motion exceeds the {_MAX_DURATION_S:g}s playout cap; "
                    "split the move into smaller motions"
                )
            )
        steps = max(1, math.ceil(float(headed_ratio)))
        fractions = np.linspace(1.0 / steps, 1.0, steps)
        actions = [
            self._decode(np.clip(current + (target - current) * fraction, self._low, self._high))
            for fraction in fractions
        ]
        clipped = np.clip(target.copy(), self._low, self._high)
        actions[-1] = self._decode(clipped, {"chunk_final": True})
        note = f"executing {self._move_tool} over {steps} steps"
        if self._hz is not None:
            note += f" ({steps / self._hz:.1f}s)"
        return ToolResult(
            chunk=ActionChunk(actions=actions),
            note=note,
            target=clipped,
        )


def build_toolset(
    action_space: ActionSpace,
    *,
    control_hz: float | None = _FALLBACK_HZ,
    max_speed_frac: float = _DEFAULT_SPEED_FRAC,
    pose: bool = False,
) -> Toolset:
    dim = action_space.width
    labels = EE_LABELS if pose else JOINT_LABELS
    if len(labels) != dim:
        raise ValueError(f"labels length {len(labels)} does not match action width {dim}")
    low = (_EE_LOW if pose else _JOINT_LOW).copy()
    high = (_EE_HIGH if pose else _JOINT_HIGH).copy()
    resolved_hz = control_hz if control_hz is not None else _FALLBACK_HZ
    native_backstop = _BACKSTOP_STEP_FRAC * (high - low)
    step_frac = min(max_speed_frac / resolved_hz, _BACKSTOP_STEP_FRAC)
    step_limits = np.minimum(step_frac * (high - low), native_backstop)
    pairs = ", ".join(
        f"{label}: [{lo:.4g}, {hi:.4g}]"
        for label, lo, hi in zip(labels, low.tolist(), high.tolist(), strict=True)
    )
    bounds_text = f"Per-dimension bounds: {pairs}."
    return Toolset(
        action_space=action_space,
        labels=labels,
        low=low,
        high=high,
        step_limits=step_limits,
        control_hz=control_hz,
        bounds_text=bounds_text,
        pose=pose,
    )
