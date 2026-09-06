"""RoboDojo loop for the P2 atomic non-learned executor."""

from __future__ import annotations

from typing import Any

from XPolicyLab.policy.Pi_05_Agent_P1_RPent.deploy import (
    _mark_incomplete_episode_failed,
)
from XPolicyLab.policy.Pi_05_Agent_P1_RPent.planner_llm import (
    create_planner_llm,
)

from .planner import P2Planner
from .tools import P2Primitives, enable_camera_calibration


def _run(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    enable_camera_calibration(TASK_ENV)
    planner_llm = create_planner_llm()
    if not planner_llm.available():
        raise RuntimeError(
            "P2 requires an available planner backend; refusing to fall back "
            "to Pi_05 or a scripted grasp"
        )
    primitives = P2Primitives(TASK_ENV, model_client, planner_llm)
    primitives.trace.append(
        {
            "type": "episode_start",
            "condition": "P2-atomic",
            "task": getattr(TASK_ENV, "task_name", None),
            "layout_id": getattr(TASK_ENV, "seed", None),
        }
    )
    P2Planner(primitives, planner_llm).run()
    _mark_incomplete_episode_failed(TASK_ENV)


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    _run(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    """P2 currently supports one active environment per planner."""
    _run(TASK_ENV, model_client)
