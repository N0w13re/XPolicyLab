"""The P3 contract: the slot a VLA fills, filled by an LLM instead.

`Policy` is deliberately the same two calls a served VLA answers, so the
harness cannot tell from the contract which one it is driving. Everything that
makes the LLM an LLM lives behind `act`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# The arx_x5 websocket action dict, in the order the tool schema presents it.
JOINT_CHANNELS: tuple[tuple[str, int], ...] = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)

EE_CHANNELS: tuple[tuple[str, int], ...] = (
    ("left_ee_pose", 7),
    ("left_ee_joint_state", 1),
    ("right_ee_pose", 7),
    ("right_ee_joint_state", 1),
)


@dataclass(frozen=True)
class Observation:
    """One environment observation as the policy sees it."""

    images: Mapping[str, np.ndarray]
    state: Mapping[str, np.ndarray]
    instruction: str | None = None
    step: int = 0
    remaining_steps: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """One environment action, already in websocket channel form."""

    data: Mapping[str, np.ndarray]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionChunk:
    """Actions executed open-loop before the policy is consulted again.

    For the inspect-robots-agent protocol a chunk is one speed-limited
    interpolant (or a stop hold), not a raw per-control-step dump. `reasoning`
    carries whatever the model said alongside the call, for the transcript only.
    """

    actions: Sequence[Action]
    reasoning: str | None = None
    latency_s: float | None = None
    control_hz: float | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("An action chunk must contain at least one action.")

    def __len__(self) -> int:
        return len(self.actions)


@runtime_checkable
class Policy(Protocol):
    """What the harness drives. A served VLA satisfies this; so does P3."""

    def reset(self) -> None:
        """Clear per-episode state before the first observation."""
        ...

    def act(self, observation: Observation) -> ActionChunk:
        """Return a non-empty open-loop action chunk for this observation."""
        ...


class ActionSpace:
    """Channels, bounds, and the tool schema the model calls to emit actions.

    The schema is generated from the channel table rather than written out, so
    an action the model can express and an action the websocket accepts cannot
    drift apart.
    """

    def __init__(
        self,
        channels: Sequence[tuple[str, int]] = JOINT_CHANNELS,
        *,
        max_chunk: int = 1,
    ) -> None:
        if max_chunk < 1:
            raise ValueError("max_chunk must be at least 1.")
        self.channels = tuple(channels)
        self.max_chunk = int(max_chunk)

    @property
    def width(self) -> int:
        return sum(size for _, size in self.channels)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.channels)

    def flat_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for name, size in self.channels:
            if size == 1:
                labels.append(name)
            else:
                labels.extend(f"{name}[{i}]" for i in range(size))
        return tuple(labels)

    def tool_schema(self) -> dict[str, Any]:
        """The action space as a function the model calls.

        A flat array rather than one property per channel: models emit long
        numeric vectors far more reliably as a list than as a nested object,
        and the width is fixed so a wrong length is caught immediately.
        """
        labels = ", ".join(self.flat_labels())
        return {
            "type": "function",
            "function": {
                "name": "act",
                "description": (
                    "Emit the next actions for the robot. Each action is "
                    f"{self.width} numbers in this fixed order: {labels}. "
                    "Gripper channels are 0 closed to 1 open. Send absolute "
                    "targets, not deltas."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": self.max_chunk,
                            "description": (
                                "Actions executed open-loop in order. You do "
                                "not see the robot again until the last one "
                                "has run, so send more than one only where the "
                                "motion is already determined."
                            ),
                            "items": {
                                "type": "array",
                                "minItems": self.width,
                                "maxItems": self.width,
                                "items": {"type": "number"},
                            },
                        },
                    },
                    "required": ["actions"],
                },
            },
        }

    def decode(self, vector: Sequence[float]) -> Action:
        """Split one flat vector into the websocket action dict."""
        values = np.asarray(list(vector), dtype=np.float32).reshape(-1)
        if values.size != self.width:
            raise ValueError(
                f"Action has {values.size} numbers, expected {self.width} "
                f"({', '.join(self.flat_labels())})."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Action contains a non-finite number.")
        data: dict[str, np.ndarray] = {}
        offset = 0
        for name, size in self.channels:
            data[name] = values[offset : offset + size].copy()
            offset += size
        return Action(data=data)

    def encode(self, state: Mapping[str, Any]) -> list[float]:
        """Read the current state back out in the same flat order.

        Used to show the model where the arm is now in the units it must
        answer in, so it never has to convert between two layouts.
        """
        values: list[float] = []
        for name, size in self.channels:
            channel = np.asarray(state.get(name, np.zeros(size)), dtype=np.float32)
            channel = channel.reshape(-1)
            if channel.size < size:
                channel = np.pad(channel, (0, size - channel.size))
            values.extend(float(v) for v in channel[:size])
        return values
