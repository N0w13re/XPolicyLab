# P2 Atomic Executor Design

## Goal

Build the thinnest P2 condition for RoboDojo `general_pickup`: a low-frequency
planner must complete contact using only explicit Cartesian motion and gripper
commands. No Pi_05 action, pregrasp helper, or pick/place macro is available.

## Boundary

- Reuse P1-RPent observation, RGB-D geometry, trace, and Qwen/Azure client code.
- Expose only state/resource reads, `render`, `sample_world_xyz`,
  `query_world_map`, `move_to`, `set_gripper`, `return_home`, and `finish`.
- `move_to` requires `xyz`, `arm`, and quaternion `quat`.
- The planner must compose open, hover, descend, close, and lift itself.
- The sampled object point is not an end-effector target. The planner must add
  clearance and account for the gripper/TCP geometry.
- Official RoboDojo termination remains the only success signal.

## Adapter

The policy is `RoboDojo_Agent_P2_RPent`. It is evaluation-only and uses a
no-action policy-server model because all environment actions originate from
the executor in `deploy.py`. Debug mode must not silently fall back to Pi_05
when no planner backend is available.

## First Experiment

Run one `general_pickup` episode with `arx_x5` and `action_type=joint`. Unit
tests first verify the restricted tool surface, mandatory quaternion, dispatch
rejection for removed tools, and absence of a Pi_05 debug fallback. The debug
client then verifies adapter startup and RPC wiring; a simulator run is needed
to measure grasp success.
