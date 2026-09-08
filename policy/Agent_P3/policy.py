"""The LLM as the policy, on the inspect-robots-agent protocol.

`LlmPolicy.act` occupies the same slot a served VLA occupies: observation in,
`ActionChunk` out. Inside the call the model never sees raw actuation. It names
partial targets through `move_joints` / `move_to`; this module interpolates
them, feeds tool results back, and ends the trial on `done` / `give_up`. That
is the inspect-robots-agent condition, running on the RoboDojo websocket.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .motion import (
    EMBODIMENT_NOTES,
    ToolCall,
    Toolset,
    build_toolset,
)
from .types import ActionChunk, ActionSpace, EE_CHANNELS, JOINT_CHANNELS, Observation
from .wire import Reply, create_client, resolve_provider

_UNSET = object()
_MAX_CONSECUTIVE_FAILURES = 3
_CAMERA_LABEL_PREFIX = "camera "
_DEFAULT_IMAGE_HORIZON = 2
_DEFAULT_MAX_LLM_CALLS = 100
_DEFAULT_SPEED_FRAC = 0.1
_DEFAULT_CONTROL_HZ = 10.0
_EMBODIMENT_NAME = "arx_x5"

SYSTEM_TEMPLATE = """You are controlling a real robot embodiment named {name!r} \
through tool calls. Each observation message gives you the current \
proprioceptive state and camera images. Work toward the user's goal in \
small, deliberate motions; re-check the observation after every motion. \
Every move tool call must include a `note`: in one or two sentences, say what \
you observe in the current observation and why you chose this motion. The user \
is watching these notes to see what you see and what you decide, so write them \
for a human reader. \
Safety approvers clamp out-of-bounds and too-fast actions below you. \
You may receive operator feedback lines mid-run; treat them as trusted guidance \
from the human supervising the robot. \
Respond with exactly one tool call per turn. When the goal is achieved call \
done; if it cannot be achieved call give_up. Note what you are learning about \
this rig and task as you go: done and give_up will ask what you wish you had \
known from the start. You have a budget of \
{budget} LLM calls for the whole trial."""


def _png_data_url(image: np.ndarray) -> str:
    """Encode an RGB array as a PNG data URL.

    Pillow when it is importable, otherwise a stdlib zlib PNG, because the eval
    environments do not all ship Pillow and a missing image is worse than a
    slower encode.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError:
        raw = _encode_png(array)
    else:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG", optimize=False)
        raw = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _encode_png(array: np.ndarray) -> bytes:
    import struct
    import zlib

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    height, width, _ = array.shape
    rows = b"".join(b"\x00" + array[y, :, :3].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 6))
        + chunk(b"IEND", b"")
    )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_optional_int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = env.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"", "none"}:
        return None
    return int(raw)


class LlmPolicy:
    """An LLM in the slot a VLA occupies, on the inspect-robots-agent loop."""

    def __init__(
        self,
        *,
        provider=None,
        action_space: ActionSpace | None = None,
        system_prompt: str | None = None,
        max_llm_calls: int | None = None,
        max_speed_frac: float | None = None,
        control_hz: float | None = None,
        image_horizon: Any = _UNSET,
        client: Any = None,
        env: Mapping[str, str] | None = None,
        toolset: Toolset | None = None,
    ) -> None:
        env = os.environ if env is None else env
        pose = env.get("P3_ACTION_TYPE", "joint") == "ee"
        self.action_space = action_space or ActionSpace(
            EE_CHANNELS if pose else JOINT_CHANNELS
        )
        self._max_llm_calls = (
            max_llm_calls
            if max_llm_calls is not None
            else _env_int(env, "P3_MAX_LLM_CALLS", _DEFAULT_MAX_LLM_CALLS)
        )
        speed = (
            max_speed_frac
            if max_speed_frac is not None
            else _env_float(env, "P3_MAX_SPEED_FRAC", _DEFAULT_SPEED_FRAC)
        )
        hz = (
            control_hz
            if control_hz is not None
            else _env_float(env, "P3_CONTROL_HZ", _DEFAULT_CONTROL_HZ)
        )
        if image_horizon is _UNSET:
            self.image_horizon = _env_optional_int(
                env, "P3_IMAGE_HORIZON", _DEFAULT_IMAGE_HORIZON
            )
        else:
            self.image_horizon = image_horizon
        self._toolset = toolset or build_toolset(
            self.action_space,
            control_hz=hz,
            max_speed_frac=speed,
            pose=pose,
        )
        formatted = SYSTEM_TEMPLATE.format(
            name=_EMBODIMENT_NAME, budget=self._max_llm_calls
        )
        notes = f"\n\nEmbodiment notes:\n{EMBODIMENT_NOTES}"
        prior = env.get("P3_PRIOR_LEARNINGS", "")
        if prior:
            prior_path = Path(prior)
            extra = prior_path.read_text(encoding="utf-8")
            notes += "\n\nPrior learnings:\n" + extra
        self.system_prompt = system_prompt if system_prompt is not None else formatted + notes
        self._provider = provider
        self._client = client
        self._messages: list[dict[str, Any]] = []
        self.usage_totals: dict[str, int] = {}
        self.calls = 0
        self.transcript: list[dict[str, Any]] = []
        self._hindsight: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            provider = self._provider or resolve_provider()
            self._client = create_client(provider)
        return self._client

    def reset(self) -> None:
        self._messages.clear()
        self._hindsight = None

    def act(self, observation: Observation) -> ActionChunk:
        toolset = self._toolset
        self._messages.append({"role": "user", "content": _observation_content(observation, toolset)})
        started = time.monotonic()
        failures = 0
        last_error: str | None = None
        usage: dict[str, int] = {}

        while True:
            if self.calls >= self._max_llm_calls:
                stopped = self._forced_give_up(toolset, observation, "LLM call budget exhausted")
                return ActionChunk(
                    actions=stopped.actions,
                    reasoning=stopped.reasoning,
                    latency_s=time.monotonic() - started,
                    usage=dict(self.usage_totals),
                )

            outgoing = self._messages
            if self.image_horizon is not None:
                outgoing = _evicted_view(self._messages, self.image_horizon)

            reply: Reply = self.client.complete(
                outgoing, toolset.schemas(), system=self.system_prompt
            )
            self.calls += 1
            for key, value in reply.usage.items():
                usage[key] = usage.get(key, 0) + value
                self.usage_totals[key] = self.usage_totals.get(key, 0) + value

            self._messages.append(_assistant_message(reply))
            calls = _tool_calls(reply)
            if not calls:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"LLM produced no tool call in {failures} consecutive turns"
                    )
                self._messages.append(
                    {"role": "user", "content": "Respond with exactly one tool call."}
                )
                continue

            chunk: ActionChunk | None = None
            closed = False
            for call in calls:
                if closed:
                    result_text = "ignored: one tool call per turn"
                else:
                    result = toolset.execute(call, observation)
                    if result.error is not None:
                        result_text = result.error
                        failures += 1
                        last_error = result.error
                        closed = True
                    else:
                        result_text = result.note
                        if result.chunk is not None:
                            chunk = result.chunk
                            stopped = bool(
                                chunk.actions[0].meta.get("request_stop")
                            )
                            if stopped:
                                self._hindsight = chunk.actions[0].meta.get(
                                    "stop_hindsight"
                                )
                            closed = True
                            failures = 0
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_text,
                    }
                )

            if chunk is not None:
                filled = ActionChunk(
                    actions=chunk.actions,
                    reasoning=reply.text,
                    latency_s=time.monotonic() - started,
                    usage=dict(usage),
                )
                self.transcript.append(
                    {
                        "step": observation.step,
                        "actions": len(filled.actions),
                        "note": filled.actions[0].meta.get("stop_reason")
                        or (calls[0].name if calls else None),
                        "reasoning": reply.text or "",
                        "usage": dict(usage),
                        "hindsight": self._hindsight,
                    }
                )
                return filled
            if failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"LLM tool calls kept failing; last error: {last_error or 'unknown'}"
                )

    def _forced_give_up(
        self, toolset: Toolset, observation: Observation, why: str
    ) -> ActionChunk:
        synthetic = ToolCall(
            id="budget",
            name="give_up",
            arguments=json.dumps({"reason": why}),
        )
        result = toolset.execute(synthetic, observation)
        if result.chunk is None:
            raise RuntimeError(f"forced give_up failed: {result.error}")
        return result.chunk


def _tool_calls(reply: Reply) -> list[ToolCall]:
    if reply.tool_calls:
        return [
            ToolCall(id=call.id, name=call.name, arguments=call.arguments)
            for call in reply.tool_calls
        ]
    if reply.tool_name:
        return [
            ToolCall(
                id="call_0",
                name=reply.tool_name,
                arguments=reply.tool_arguments or "{}",
            )
        ]
    return []


def _assistant_message(reply: Reply) -> dict[str, Any]:
    if reply.assistant_message:
        return dict(reply.assistant_message)
    calls = _tool_calls(reply)
    message: dict[str, Any] = {"role": "assistant", "content": reply.text}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ]
    return message


def _state_line(observation: Observation, toolset: Toolset) -> str:
    values = toolset.current_state(observation)
    rounded = np.round(values, 4).tolist()
    labeled = " ".join(
        f"{label}={item}" for label, item in zip(toolset.labels, rounded, strict=True)
    )
    return f"state[joints]: {labeled}"


def _observation_content(observation: Observation, toolset: Toolset) -> list[dict[str, Any]]:
    lines = ["Current observation."]
    lines.append(f"step {observation.step}")
    if observation.remaining_steps is not None:
        lines.append(f"{observation.remaining_steps} steps remain")
    if observation.instruction:
        lines.append(f"Instruction: {observation.instruction}")
    lines.append(_state_line(observation, toolset))
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    for name, image in observation.images.items():
        parts.append({"type": "text", "text": f"{_CAMERA_LABEL_PREFIX}{name!r}:"})
        parts.append({"type": "image_url", "image_url": {"url": _png_data_url(image)}})
    return parts


def _evicted_view(messages: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    """Stub camera frames older than the image horizon, matching inspect-robots-agent."""
    image_message_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance((content := message.get("content")), list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
    ]
    stubbed_indices = image_message_indices[:-horizon]
    if not stubbed_indices:
        return list(messages)

    view = list(messages)
    for message_index in stubbed_indices:
        message = messages[message_index]
        content = message["content"]
        assert isinstance(content, list)
        image_indices = [
            index
            for index, part in enumerate(content)
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        removed_indices = set(image_indices)
        for image_index in image_indices:
            if image_index == 0:
                continue
            label = content[image_index - 1]
            if isinstance(label, dict) and label.get("type") == "text":
                removed_indices.add(image_index - 1)
        stubbed_content = [
            part for index, part in enumerate(content) if index not in removed_indices
        ]
        stubbed_content.append(
            {
                "type": "text",
                "text": f"[{len(image_indices)} camera frame(s) elided]",
            }
        )
        view[message_index] = {**message, "content": stubbed_content}
    return view
