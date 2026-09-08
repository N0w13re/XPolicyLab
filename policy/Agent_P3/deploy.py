"""RoboDojo loop for P3: the LLM names targets, the motion layer interpolates them.

There is no `model_client` call here because P3 serves no VLA. The harness
still passes one so every adapter has the same signature; it stays unused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .policy import LlmPolicy
from .types import Observation


def _remaining_steps(task_env: Any) -> int | None:
    limit = getattr(task_env, "step_lim", None)
    if limit is None:
        return None
    counts = np.asarray(getattr(task_env, "take_action_cnt", 0)).reshape(-1)
    used = int(counts[0]) if counts.size else 0
    return max(0, int(limit) - used)


def _observation(task_env: Any, policy_step: int) -> Observation:
    raw = task_env.get_obs()
    images = {
        name: np.asarray(view["rgb"])
        for name, view in (raw.get("observation") or {}).items()
        if isinstance(view, dict) and "rgb" in view
    }
    return Observation(
        images=images,
        state=raw.get("state") or {},
        instruction=getattr(task_env, "instruction", None),
        step=policy_step,
        remaining_steps=_remaining_steps(task_env),
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


def _write_transcript(policy: LlmPolicy, task_env: Any) -> None:
    directory = os.environ.get("P3_TRACE_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "p3_transcript.json").write_text(
        json.dumps(
            {
                "task": getattr(task_env, "task_name", None),
                "layout_id": getattr(task_env, "seed", None),
                "llm_calls": policy.calls,
                "usage": policy.usage_totals,
                "hindsight": policy._hindsight,
                "turns": policy.transcript,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    del model_client  # P3 serves no VLA
    policy = LlmPolicy()
    policy.reset()
    policy_step = 0
    stopped = False
    try:
        while not TASK_ENV.is_episode_end():
            chunk = policy.act(_observation(TASK_ENV, policy_step))
            policy_step += 1
            for action in chunk.actions:
                TASK_ENV.take_action(dict(action.data))
                if action.meta.get("request_stop"):
                    stopped = True
                    reason = action.meta.get("stop_reason")
                    print(f"[P3] policy {reason}: {action.meta.get('stop_detail')}", flush=True)
                    break
                if TASK_ENV.is_episode_end():
                    break
            if stopped:
                break
    except RuntimeError as error:
        print(f"[P3] {error}", flush=True)
        stopped = True
    finally:
        print(
            f"[P3] {policy.calls} llm calls, usage={policy.usage_totals}",
            flush=True,
        )
        _write_transcript(policy, TASK_ENV)
    if stopped or not TASK_ENV.is_episode_end():
        _mark_incomplete_episode_failed(TASK_ENV)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    """P3 is one conversation per environment, so batching runs a single env."""
    eval_one_episode(TASK_ENV, model_client)
