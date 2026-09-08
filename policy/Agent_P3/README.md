# Agent_P3 — L5, aligned with inspect-robots-agent

This directory is the **L5** condition (`policy_name` stays `Agent_P3`). The
LLM occupies the same `reset` / `act` slot a served VLA occupies. The
**strategy** of that `act` matches
[`inspect-robots-agent`](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent):
the model never emits a raw 14-D dump. It names partial targets; a motion
layer interpolates them.

The comparison table, deferred items, and test map live in
[`docs/l5_inspect_robots_alignment.md`](../../docs/l5_inspect_robots_alignment.md).

| Level | System | What stands between the model and the robot |
| --- | --- | --- |
| L1 / P0 | VLA / WAM | Everything; there is no LLM |
| L2 / P1 | LLM + task-level VLA | The LLM instructs, then hands off |
| L4 / P2 | LLM + harness primitives | `pick` / `place` / IK we wrote |
| L5 / P3 | Direct LLM API | inspect-robots-agent tools + interpolation |

## How an action gets made

1. `deploy.py` reads RGB, joint state, remaining steps, and the task instruction.
2. The observation is labeled state (`left_j0=…`) plus camera PNGs, matching
   inspect's `_observation_content`.
3. The model calls exactly one of `move_joints` (or `move_to` in EE mode),
   `done`, or `give_up`. Every move requires a human-readable `note`.
4. `motion.py` interpolates named absolute targets from the current state at
   `max_speed_frac=0.1` with a 5%-of-range per-step ceiling and a 10 s cap —
   the same numbers as inspect-robots-agent.
5. The interpolated chunk is played on the RoboDojo websocket. `done` /
   `give_up` end the trial; the official scorer still judges success.

Malformed calls come back as tool results. Three consecutive failures abort.
`P3_MAX_LLM_CALLS` (default 100) forces `give_up`. Camera frames older than
`P3_IMAGE_HORIZON` (default 2) are elided; the text history stays.

## Wires

| Wire | Endpoint |
| --- | --- |
| `messages` | Anthropic `POST /messages` |
| `chat` | OpenAI-compatible `POST /chat/completions` |
| `azure-chat` | Azure `.../openai/deployments/<model>/chat/completions` |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `P3_MODEL` | required | `provider/model` |
| `P3_MAX_LLM_CALLS` | 100 | Trial LLM budget |
| `P3_MAX_SPEED_FRAC` | 0.1 | Inspect interpolant speed |
| `P3_CONTROL_HZ` | 10 | Used to turn speed into steps |
| `P3_IMAGE_HORIZON` | 2 | Keep this many image-bearing turns (`none` = all) |
| `P3_ACTION_TYPE` | `joint` | `joint` → `move_joints`; `ee` → `move_to` |
| `P3_PRIOR_LEARNINGS` | unset | UTF-8 notes file appended to the system prompt |
| `P3_WIRE` / `P3_BASE_URL` / `P3_API_VERSION` | | Endpoint override |
| `P3_TRACE_DIR` | | Writes `p3_transcript.json` |

## Running

```bash
export P3_MODEL=anthropic/claude-sonnet-4-20250514
export ANTHROPIC_API_KEY=...
export P3_TRACE_DIR=/tmp/xpolicylab-p3/$RUN_ID

bash policy/Agent_P3/run_fixed_layout.sh <layout> <env_gpu> general_pickup
```

No VLA checkpoint. `deploy.yml` borrows `policy/Pi_05/openpi` only for the
shared harness packages.
