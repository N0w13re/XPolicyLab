"""Observable manipulation state and transition gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ManipulationLedger:
    phase: str = "observe"
    hold_state: str = "empty"
    holding_arm: str | None = None
    held_target: str | None = None
    last_verified_step: int | None = None
    last_gate: str | None = None
    last_gate_passed: bool | None = None

    def candidate(self, *, arm: str | None, target: str, step: int | None) -> None:
        self.phase = "verify_grasp"
        self.hold_state = "candidate" if arm else "unknown"
        self.holding_arm = arm
        self.held_target = target
        self.last_verified_step = step

    def verify(self, *, gate: str, passed: bool, step: int | None) -> None:
        self.last_gate = gate
        self.last_gate_passed = passed
        self.last_verified_step = step
        if gate in {"grasp", "transport", "handover_receive"}:
            self.hold_state = "verified" if passed else "lost"
            if not passed:
                self.holding_arm = None
        elif gate in {"support_placement", "container_placement", "release"}:
            if passed:
                self.hold_state = "empty"
                self.holding_arm = None
                self.held_target = None
                self.phase = "observe"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
