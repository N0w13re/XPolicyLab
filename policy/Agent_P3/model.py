"""P3 serves no VLA.

The harness expects every adapter to expose a `Model`, so this one exists to
answer that contract and to fail loudly if anything ever tries to ask it for an
action: at P3 the only thing that produces actions is the LLM in `policy.py`.
"""

from __future__ import annotations

from typing import Any

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg

    def reset(self) -> None:
        return None

    def update_obs(self, obs: dict[str, Any]) -> None:
        del obs

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        del obs_list

    def get_action(self) -> Any:
        raise RuntimeError(
            "P3 has no VLA to call. Actions come from the LLM in policy.py; a "
            "condition that calls a served policy here is P0 or P1, not P3."
        )

    def get_action_batch(self, env_idx_list: list[int] | None = None) -> Any:
        del env_idx_list
        raise RuntimeError(
            "P3 has no batched VLA action path. Each RoboDojo environment needs "
            "its own inspect-robots-agent conversation."
        )
