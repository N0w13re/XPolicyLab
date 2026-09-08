"""Thin RoboDojo bridge around the published inspect-robots-agent policy.

All strategy-bearing behavior -- prompts, tools, motion interpolation, history,
repair, image modes, call budget, hindsight, and wire capture -- stays in the
pinned upstream package. This module only maps RoboDojo arrays/configuration to
Inspect Robots types and maps its returned vector chunks back to XPolicyLab
action dictionaries.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
from inspect_robots.approver import ClampApprover, DeltaLimitApprover
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.rollout import TrialRecord
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation as InspectObservation
from inspect_robots_agent import LLMAgentPolicy

from .types import Action, ActionChunk, ActionSpace, JOINT_CHANNELS, Observation
from .wire import AzureChatTransport

_DEFAULT_MAX_LLM_CALLS = 100
_DEFAULT_MAX_SPEED_FRAC = 0.1


@dataclass(frozen=True)
class RoboDojoActionSpec:
    """The semantics RoboDojo exposes to the upstream agent policy."""

    labels: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    control_hz: float
    docs: str

    def __post_init__(self) -> None:
        width = len(self.labels)
        if self.low.shape != (width,) or self.high.shape != (width,):
            raise ValueError("action bounds must be flat and match labels")
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("action bounds must be finite")
        if np.any(self.low > self.high):
            raise ValueError("action lower bounds must not exceed upper bounds")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")


def _optional_number(env: Mapping[str, str], key: str, kind: type[int] | type[float]) -> Any:
    raw = env.get(key)
    return None if raw is None or raw == "" else kind(raw)


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be true or false, got {raw!r}")


def _agent_kwargs(
    env: Mapping[str, str],
    *,
    transport: Any = None,
    pre_check: Any = None,
) -> tuple[dict[str, Any], str]:
    model = env.get("P3_MODEL", "")
    if not model:
        raise ValueError("P3_MODEL is required")
    requested_wire = env.get("P3_WIRE", "")
    kwargs: dict[str, Any] = {
        "model": model,
        "max_llm_calls": int(env.get("P3_MAX_LLM_CALLS", _DEFAULT_MAX_LLM_CALLS)),
        "max_speed_frac": float(
            env.get("P3_MAX_SPEED_FRAC", _DEFAULT_MAX_SPEED_FRAC)
        ),
        "transcript_echo": _bool(env, "P3_TRANSCRIPT_ECHO", False),
        "images": env.get("P3_IMAGES", "always"),
        "depth": env.get("P3_DEPTH", "render"),
        "wire_capture": _bool(env, "P3_WIRE_CAPTURE", True),
        "env": dict(env),
        "pre_check": pre_check,
    }
    aliases = {
        "P3_BASE_URL": "base_url",
        "P3_API_KEY_ENV": "api_key_env",
        "P3_SPEED": "speed",
        "P3_PRIOR_LEARNINGS": "prior_learnings",
    }
    for source, destination in aliases.items():
        if value := env.get(source):
            kwargs[destination] = value
    for source, destination, kind in (
        ("P3_MAX_OUTPUT_TOKENS", "max_output_tokens", int),
        ("P3_TEMPERATURE", "temperature", float),
    ):
        if (value := _optional_number(env, source, kind)) is not None:
            kwargs[destination] = value
    if "P3_EFFORT" in env:
        effort = env["P3_EFFORT"].strip()
        if effort == "":
            kwargs["effort"] = None
        else:
            try:
                kwargs["effort"] = float(effort)
            except ValueError:
                kwargs["effort"] = effort
    if "P3_IMAGE_HORIZON" in env:
        horizon = env["P3_IMAGE_HORIZON"].strip()
        kwargs["image_horizon"] = None if horizon.lower() == "none" else int(horizon)

    if requested_wire == "azure-chat":
        base_url = env.get("P3_BASE_URL")
        api_version = env.get("P3_API_VERSION")
        if not base_url or not api_version:
            raise ValueError("azure-chat requires P3_BASE_URL and P3_API_VERSION")
        deployment = model.split("/", 1)[-1]
        key_env = env.get("P3_API_KEY_ENV", "P3_API_KEY")
        kwargs.update(
            model=deployment,
            base_url=base_url,
            api_key_env=key_env,
            wire="chat",
            transport=transport
            or AzureChatTransport(
                base_url=base_url,
                deployment=deployment,
                api_version=api_version,
            ),
        )
    elif requested_wire:
        kwargs["wire"] = requested_wire
        if transport is not None:
            kwargs["transport"] = transport
    elif transport is not None:
        kwargs["transport"] = transport
    return kwargs, requested_wire or "auto"


class LlmPolicy:
    """XPolicyLab-shaped facade over the exact upstream LLMAgentPolicy."""

    def __init__(
        self,
        *,
        action_spec: RoboDojoActionSpec,
        env: Mapping[str, str] | None = None,
        transport: Any = None,
        pre_check: Any = None,
    ) -> None:
        self._env = dict(os.environ if env is None else env)
        kwargs, self.requested_wire = _agent_kwargs(
            self._env, transport=transport, pre_check=pre_check
        )
        self.inner = LLMAgentPolicy(**kwargs)
        self.action_spec = action_spec
        self.action_space = ActionSpace(JOINT_CHANNELS)
        if len(action_spec.labels) != self.action_space.width:
            raise ValueError(
                f"RoboDojo joint action spec has {len(action_spec.labels)} dimensions; "
                f"expected {self.action_space.width}"
            )
        self._bound = False
        self._scene: Scene | None = None
        self._approvers: tuple[Any, ...] = ()
        self._approver_store: dict[str, Any] = {}
        self._pending_approvals: list[dict[str, Any]] = []
        self._env_action_step = 0

    def _bind(self, observation: Observation) -> None:
        cameras = tuple(
            CameraSpec(
                name=name,
                height=int(np.asarray(image).shape[0]),
                width=int(np.asarray(image).shape[1]),
                channels=int(np.asarray(image).shape[2]),
            )
            for name, image in observation.images.items()
        )
        semantics = ActionSemantics(
            control_mode="joint_pos",
            rotation_repr="none",
            gripper="continuous",
            frame="base",
            dim_labels=self.action_spec.labels,
        )
        box = Box(
            shape=(self.action_space.width,),
            low=np.asarray(self.action_spec.low, dtype=np.float64),
            high=np.asarray(self.action_spec.high, dtype=np.float64),
            semantics=semantics,
        )
        state = StateSpec(
            fields=(
                StateField(
                    key="joint_pos",
                    shape=(self.action_space.width,),
                    unit="rad+normalized_gripper",
                    dtype="float64",
                ),
            )
        )
        info = EmbodimentInfo(
            name="robodojo-arx-x5",
            action_space=box,
            observation_space=ObservationSpace(cameras=cameras, state=state),
            control_hz=self.action_spec.control_hz,
            is_simulated=True,
            docs=self.action_spec.docs,
        )
        self.inner.bind(info)
        self._approvers = (ClampApprover(box), DeltaLimitApprover(box))
        self._approver_store.clear()
        self._pending_approvals.clear()
        self._env_action_step = 0
        layout_id = observation.extra.get("layout_id")
        if not isinstance(layout_id, int):
            layout_id = _optional_number(self._env, "P3_LAYOUT_ID", int)
        scene = Scene(
            id=f"{self._env.get('P3_TASK_NAME', 'robodojo')}-layout-"
            f"{layout_id if layout_id is not None else 'unknown'}",
            instruction=observation.instruction or "",
            init_seed=layout_id,
        )
        self.inner.reset(scene)
        self._scene = scene
        self._bound = True

    def reset(self) -> None:
        """Defer upstream reset until the first observation supplies the scene."""
        self._bound = False
        self._scene = None

    def prepare(self, observation: Observation) -> None:
        """Bind/reset before capture starts, without issuing an LLM request."""
        if not self._bound:
            self._bind(observation)

    def start_capture(self, log_dir: str, run_id: str) -> None:
        scene_id = self._scene.id if self._scene is not None else "robodojo"
        self.inner.on_trial_start(scene_id, 0, log_dir, run_id)

    def finish_capture(
        self,
        log_dir: str,
        run_id: str,
        *,
        terminated: bool,
        truncated: bool,
        termination_reason: str | None,
    ) -> dict[str, Any]:
        scene = self._scene or Scene(id="robodojo", instruction="")
        record = TrialRecord(
            scene_id=scene.id,
            epoch=0,
            seed=scene.init_seed,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
        )
        self.inner.on_trial_end(record, log_dir, run_id)
        return dict(record.metadata)

    def act(self, observation: Observation) -> ActionChunk:
        if not self._bound:
            self._bind(observation)
        vector = np.asarray(self.action_space.encode(observation.state), dtype=np.float64)
        inspect_observation = InspectObservation(
            images={
                name: np.asarray(image, dtype=np.uint8)
                for name, image in observation.images.items()
            },
            state={"joint_pos": vector},
            instruction=observation.instruction,
            extra={
                **dict(observation.extra),
                "env_step": self._env_action_step,
                "approvals": list(self._pending_approvals),
            },
        )
        self._pending_approvals.clear()
        chunk = self.inner.act(inspect_observation)
        actions = []
        for upstream_action in chunk.actions:
            reviewed = upstream_action
            modifications: list[str] = []
            for approver in self._approvers:
                candidate = approver.review(reviewed, self._approver_store)
                if candidate is not reviewed:
                    if candidate.meta.get("clamped"):
                        modifications.append("clamped")
                    if candidate.meta.get("delta_clamped"):
                        modifications.append("delta_clamped")
                reviewed = candidate
            if modifications:
                self._pending_approvals.append(
                    {
                        "t": self._env_action_step,
                        "detail": ", ".join(modifications),
                    }
                )
            decoded = self.action_space.decode(
                np.asarray(reviewed.data, dtype=np.float64).tolist()
            )
            actions.append(
                Action(data=decoded.data, meta=dict(reviewed.meta))
            )
            self._env_action_step += 1
        return ActionChunk(
            actions=actions,
            latency_s=chunk.inference_latency_s,
            control_hz=chunk.control_hz,
            meta=dict(chunk.meta),
        )

    @property
    def calls(self) -> int:
        return int(self.inner._calls_used)

    @property
    def usage_totals(self) -> dict[str, int]:
        return dict(self.inner._usage_totals)

    @property
    def hindsight(self) -> str | None:
        return self.inner._hindsight

    def transcript(self) -> list[dict[str, Any]] | None:
        return self.inner.transcript()

    def audit_config(self) -> dict[str, Any]:
        scene = self._scene
        return {
            "adapter": "inspect-robots-agent",
            "inspect_robots_version": version("inspect-robots"),
            "inspect_robots_agent_version": version("inspect-robots-agent"),
            "requested_wire": self.requested_wire,
            "azure_api_version": self._env.get("P3_API_VERSION"),
            "upstream_policy_config": asdict(self.inner.config),
            "scene": {
                "id": scene.id if scene is not None else None,
                "instruction": scene.instruction if scene is not None else None,
                "init_seed": scene.init_seed if scene is not None else None,
            },
            "embodiment": {
                "labels": list(self.action_spec.labels),
                "low": self.action_spec.low.tolist(),
                "high": self.action_spec.high.tolist(),
                "control_hz": self.action_spec.control_hz,
                "docs": self.action_spec.docs,
            },
        }
