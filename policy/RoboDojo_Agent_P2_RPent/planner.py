"""Restricted planner for the RoboDojo P2 atomic-executor condition."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner import (
    DEFAULT_PLANNER_CONTEXT_MODE,
    SUPPORTED_PLANNER_CONTEXT_MODES,
    RpentPlanner,
)

from .prompts import SYSTEM_PROMPT, opening_prompt
from .tools import P2Primitives


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS_SPEC = [
    _function(
        "view_env_state",
        "Read one previously captured immutable RGB-D state.",
        {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": (
                        "env_state_step of a recorded state, not the "
                        "snapshot's env_steps count. -1 is the latest."
                    ),
                },
            },
        },
    ),
    _function(
        "sample_world_xyz",
        (
            "Sample world XYZ at Qwen visual points. pixels accepts one [x,y] "
            "pair or a nested list of pairs, each normalized to 0..1000; the "
            "runtime converts them to the captured image resolution."
        ),
        {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["head", "left_wrist", "right_wrist"],
                },
                "pixels": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": (
                        "env_state_step of a recorded state, not the "
                        "snapshot's env_steps count. -1 is the latest."
                    ),
                },
                "radius": {"type": "integer", "minimum": 0, "default": 2},
            },
            "required": ["view", "pixels"],
        },
    ),
    _function(
        "query_world_map",
        (
            "Summarize world XYZ inside a Qwen visual bbox normalized to "
            "0..1000 as [x0,y0,x1,y1]."
        ),
        {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["head", "left_wrist", "right_wrist"],
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": (
                        "env_state_step of a recorded state, not the "
                        "snapshot's env_steps count. -1 is the latest."
                    ),
                },
            },
            "required": ["view", "bbox"],
        },
    ),
    _function(
        "move_to",
        (
            "Move one arm to an explicit world-frame flange pose. xyz must be a "
            "flange target (use sample_world_xyz suggested_hover_eef_xyz / "
            "suggested_contact_eef_xyz), never a raw surface sample. quat is "
            "[qw,qx,qy,qz] and is mandatory. The observed gripper is preserved "
            "unless gripper is given."
        ),
        {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "arm": {"type": "string", "enum": ["left", "right"]},
                "quat": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "gripper": {"type": "number"},
                "substeps": {"type": "integer", "minimum": 1, "default": 25},
            },
            "required": ["xyz", "arm", "quat"],
        },
    ),
    _function(
        "set_gripper",
        "Hold the current EEF pose and explicitly open or close one gripper.",
        {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "state": {"type": "string", "enum": ["open", "closed"]},
                "steps": {"type": "integer", "minimum": 1, "default": 8},
            },
            "required": ["arm", "state"],
        },
    ),
    _function(
        "return_home",
        "Return one or both arms to their episode-start poses with open grippers.",
        {
            "type": "object",
            "properties": {
                "arm": {
                    "type": "string",
                    "enum": ["left", "right", "both"],
                    "default": "both",
                }
            },
        },
    ),
    _function(
        "finish",
        "Stop planning without overriding official RoboDojo scoring.",
        {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
    ),
]

ALLOWED_TOOLS = frozenset(
    tool["function"]["name"] for tool in TOOLS_SPEC
)


class P2Planner(RpentPlanner):
    """Reuse the planner loop while replacing its prompt and complete tool surface."""

    def __init__(self, primitives: P2Primitives, qwen: Any = None) -> None:
        self.primitives = primitives
        self.qwen = qwen or primitives.qwen
        self.max_turns = max(1, int(os.environ.get("RPENT_MAX_TURNS", "120")))
        self.prompt_version = "p2-v2"
        self.context_mode = os.environ.get(
            "RPENT_PLANNER_CONTEXT", DEFAULT_PLANNER_CONTEXT_MODE
        ).strip().lower()
        if self.context_mode not in SUPPORTED_PLANNER_CONTEXT_MODES:
            raise ValueError(
                "RPENT_PLANNER_CONTEXT must be one of: "
                + ", ".join(SUPPORTED_PLANNER_CONTEXT_MODES)
            )
        self.session_id = os.environ.get("RPENT_GPT_SESSION_ID", "").strip() or (
            f"p2-{self.context_mode}-{uuid4().hex}"
        )
        self.instruction_contract_enabled = False
        self.tools_spec = TOOLS_SPEC
        self.successful_mutations: list[dict[str, Any]] = []
        self.measurements: list[dict[str, Any]] = []
        self.instruction_contract = None
        self._last_tool_memory = None
        self._base_guidance_sources: list[str] = []
        self._embedded_documents = set()
        self._guidance_memory: dict[str, dict[str, Any]] = {}

    def _prompt_config(self) -> dict[str, Any]:
        task_env = self.primitives.task_env
        task_name = self._task_name()
        seed = str(
            getattr(task_env, "seed", None)
            or os.environ.get("EVAL_SEED", "0")
        )
        instruction = self.primitives.snapshot().get("instruction")
        return {
            "prompt_version": self.prompt_version,
            "prompt_source": "XPolicyLab RoboDojo P2 atomic executor",
            "system_prompt": SYSTEM_PROMPT,
            "opening_prompt": opening_prompt(
                task_name=task_name,
                seed=seed,
                instruction=instruction,
            ),
        }

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in ALLOWED_TOOLS:
            return self.primitives.record_tool_result(
                name,
                arguments,
                {"error": f"tool {name!r} is not available in P2 atomic mode"},
            )
        return super()._dispatch(name, arguments)

    def _observation_suffix(self, *, include_memory: bool) -> dict[str, Any]:
        suffix = super()._observation_suffix(include_memory=include_memory)
        content = suffix.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = part["text"].replace(
                        "ACTIONS COMPLETED THIS EPISODE (includes pregrasp, so an "
                        "active target with no later pregrasp entry is not staged):",
                        "ATOMIC ACTIONS COMPLETED THIS EPISODE:",
                    )
        return suffix
