"""The LLM as the policy.

`LlmPolicy.act` is the whole of P3: an observation goes in, a request goes out
on the provider's native wire, and the tool-call arguments are decoded into the
action chunk the harness executes. Nothing in here plans, retargets, or
interpolates; if it did, the condition would be P2.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .types import (
    EE_CHANNELS,
    JOINT_CHANNELS,
    Action,
    ActionChunk,
    ActionSpace,
    Observation,
)
from .wire import Provider, Reply, create_client, resolve_provider

SYSTEM_PROMPT = """\
You are controlling a bimanual robot directly. Every turn you receive the
current camera images and joint state, and you answer with the next actions by
calling `act`. There is no motion planner, no inverse kinematics, and no
scripted skill between your numbers and the robot: the values you emit are the
targets the arms servo to.

Read the images to find the objects and to judge whether your last action did
what you intended. Read the state vector for where the arms are now; it is
given in exactly the order and units `act` expects, so you can start from it
and change only what should move.

Move in small increments. A large jump between the current state and your
target makes the arm swing through whatever is in the way, and you will not see
the result until the whole chunk has run. Approach, then descend, then close.

Gripper channels run 0 closed to 1 open. Close before lifting and confirm in
the next image that the object came with the gripper before transporting it.
"""


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
    rows = b"".join(
        b"\x00" + array[y, :, :3].tobytes() for y in range(height)
    )

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


def _round(values: Sequence[float], places: int = 4) -> list[float]:
    return [round(float(v), places) for v in values]


class ActionDecodeError(ValueError):
    """The model's call could not be turned into actions; it is told why."""


class LlmPolicy:
    """An LLM in the slot a VLA occupies.

    Conversation state is deliberately shallow. The robot's state is fully
    visible in every request, so carrying the whole dialogue would spend
    context on information the images already carry; `history_turns` keeps a
    short window for continuity of intent and nothing more.
    """

    def __init__(
        self,
        *,
        provider: Provider | None = None,
        action_space: ActionSpace | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        history_turns: int = 4,
        max_repairs: int = 2,
        client: Any = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if env is None else env
        self.action_space = action_space or _action_space_from_env(env)
        self.system_prompt = system_prompt
        self.history_turns = max(0, int(history_turns))
        self.max_repairs = max(0, int(max_repairs))
        self._provider = provider
        self._client = client
        self._history: list[dict[str, Any]] = []
        self.usage_totals: dict[str, int] = {}
        self.calls = 0
        self.transcript: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            provider = self._provider or resolve_provider()
            self._client = create_client(provider)
        return self._client

    def reset(self) -> None:
        self._history.clear()

    def act(self, observation: Observation) -> ActionChunk:
        request = self._observation_turn(observation)
        turns = [*self._history, request]
        tools = [self.action_space.tool_schema()]
        started = time.monotonic()
        usage: dict[str, int] = {}
        reply: Reply | None = None
        failure = ""

        for attempt in range(self.max_repairs + 1):
            reply = self.client.complete(
                turns, tools, system=self.system_prompt
            )
            self.calls += 1
            for key, value in reply.usage.items():
                usage[key] = usage.get(key, 0) + value
                self.usage_totals[key] = self.usage_totals.get(key, 0) + value
            try:
                actions = self._decode(reply)
            except ActionDecodeError as error:
                failure = str(error)
                if attempt == self.max_repairs:
                    break
                # Show the model its own rejected call so the repair turn is
                # about the mistake rather than about the observation again.
                turns = [
                    *turns,
                    {
                        "role": "assistant",
                        "content": f"(rejected call: {reply.tool_arguments})",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"That call was not executable: {failure} "
                            "Call `act` again with the correction."
                        ),
                    },
                ]
                continue

            chunk = ActionChunk(
                actions=actions,
                reasoning=reply.text,
                latency_s=time.monotonic() - started,
                usage=usage,
            )
            self._remember(request, reply, len(actions))
            self.transcript.append(
                {
                    "step": observation.step,
                    "actions": len(actions),
                    "reasoning": reply.text,
                    "usage": dict(usage),
                }
            )
            return chunk

        raise ActionDecodeError(
            f"The model did not produce an executable action in "
            f"{self.max_repairs + 1} attempts — {failure}"
        )

    def _decode(self, reply: Reply) -> list[Action]:
        if reply.tool_name is None:
            raise ActionDecodeError(
                "No `act` call was made; reply with a tool call, not prose."
            )
        if reply.tool_name != "act":
            raise ActionDecodeError(
                f"Called {reply.tool_name!r}; the only tool is `act`."
            )
        try:
            arguments = json.loads(reply.tool_arguments or "{}")
        except json.JSONDecodeError as error:
            raise ActionDecodeError(f"Arguments were not valid JSON: {error}") from error
        rows = arguments.get("actions")
        if not isinstance(rows, list) or not rows:
            raise ActionDecodeError("`actions` must be a non-empty list of arrays.")
        if len(rows) > self.action_space.max_chunk:
            raise ActionDecodeError(
                f"{len(rows)} actions exceeds the limit of "
                f"{self.action_space.max_chunk}."
            )
        actions: list[Action] = []
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)):
                raise ActionDecodeError(f"Action {index} is not an array of numbers.")
            try:
                actions.append(self.action_space.decode(row))
            except ValueError as error:
                raise ActionDecodeError(f"Action {index}: {error}") from error
        return actions

    def _observation_turn(self, observation: Observation) -> dict[str, Any]:
        state = _round(self.action_space.encode(observation.state))
        lines = [
            f"step {observation.step}",
        ]
        if observation.remaining_steps is not None:
            lines.append(f"{observation.remaining_steps} steps remain")
        if observation.instruction:
            lines.append(f"task: {observation.instruction}")
        lines.append(
            "current state, in the order `act` expects "
            f"({', '.join(self.action_space.flat_labels())}):"
        )
        lines.append(json.dumps(state))

        content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
        for camera, image in observation.images.items():
            content.append({"type": "text", "text": f"camera {camera}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(image)},
                }
            )
        return {"role": "user", "content": content}

    def _remember(
        self, request: Mapping[str, Any], reply: Reply, count: int
    ) -> None:
        """Keep a short text-only window.

        The images are dropped on the way in: the next request carries fresh
        ones, and stale frames are the single largest way this conversation
        grows without telling the model anything it cannot already see.
        """
        if self.history_turns == 0:
            return
        summary = next(
            (
                part["text"]
                for part in request["content"]
                if part.get("type") == "text"
            ),
            "",
        )
        self._history.append({"role": "user", "content": summary})
        self._history.append(
            {
                "role": "assistant",
                "content": (reply.text or "") + f"\n(emitted {count} action(s))",
            }
        )
        excess = len(self._history) - 2 * self.history_turns
        if excess > 0:
            del self._history[:excess]


def _action_space_from_env(env: Mapping[str, str]) -> ActionSpace:
    channels = (
        EE_CHANNELS if env.get("P3_ACTION_TYPE", "joint") == "ee" else JOINT_CHANNELS
    )
    return ActionSpace(channels, max_chunk=int(env.get("P3_MAX_CHUNK", "1")))
