"""P3 serves no VLA.

The harness expects every adapter to expose a `Model`, so this one exists to
answer that contract and to fail loudly if anything ever tries to ask it for an
action: at P3 the only thing that produces actions is the LLM in `policy.py`.
"""

from __future__ import annotations

from typing import Any


class Model:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def reset(self) -> None:
        return None

    def update_obs(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def get_action(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "P3 has no VLA to call. Actions come from the LLM in policy.py; a "
            "condition that calls a served policy here is P0 or P1, not P3."
        )
