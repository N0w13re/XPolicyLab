# Agent_P3 — L5, aligned with inspect-robots-agent

**Contributor:** XPolicyLab | **Paper:** N/A | **arXiv:** N/A |
**Original code:** [robocurve/inspect-robots](https://github.com/robocurve/inspect-robots)

This directory is the **L5** condition (`policy_name` stays `Agent_P3`). It
uses the published, pinned
[`inspect-robots-agent`](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent):
`inspect-robots-agent==0.26.0` with `inspect-robots==0.58.0`. Prompt, tool
schema, motion interpolation, history, repair, image modes, budget, hindsight,
provider wires, usage accounting, and replay-grade capture execute in that
package rather than in an XPolicyLab fork.

The source-level comparison and harness mapping live in
[`docs/l5_inspect_robots_alignment.md`](../../docs/l5_inspect_robots_alignment.md).
Shared XPolicyLab conventions are in the [root README](../../README.md);
official results are published on the
[RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

| Level | System | What stands between the model and the robot |
| --- | --- | --- |
| L1 / P0 | VLA / WAM | Everything; there is no LLM |
| L2 / P1 | LLM + task-level VLA | The LLM instructs, then hands off |
| L4 / P2 | LLM + harness primitives | `pick` / `place` / IK we wrote |
| L5 / P3 | Direct LLM API | inspect-robots-agent tools + interpolation |

## How an action gets made

1. `deploy.py` reads RoboDojo's public `vision.*.color`, `state` and
   `instruction` fields, plus the live Isaac joint limits and actual
   observation/control rate (25 Hz).
2. `policy.py` presents those as an Inspect `EmbodimentInfo` and `Observation`.
3. The upstream policy emits `move_joints`, `done`, or `give_up`; it also owns
   optional `images=on_demand`, depth rendering, all supported provider wires,
   and `prior_learnings`.
4. The upstream interpolated chunk is converted losslessly to RoboDojo's four
   action dictionary channels and played in full. `done` /
   `give_up` end the trial; the official scorer still judges success.

EE mode is rejected. RoboDojo exposes EE actions as quaternion poses followed
by IK; upstream intentionally rejects quaternion pose spaces because linear
per-dimension interpolation is unsafe. Silently advertising `move_to` would
not be an aligned condition.

## Wires

| Wire | Endpoint |
| --- | --- |
| `messages`, `responses`, `gemini-live`, `interactions`, `chat` | Upstream implementation |
| `azure-chat` | Azure `.../openai/deployments/<model>/chat/completions` |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `P3_MODEL` | required | `provider/model` |
| `P3_MAX_LLM_CALLS` | 100 | Trial LLM budget |
| `P3_MAX_SPEED_FRAC` | 0.1 | Inspect interpolant speed |
| control rate | live RoboDojo value | `obs_manager.collect_freq` (25 Hz for `arx_x5`) |
| `P3_IMAGE_HORIZON` | upstream wire default | 2 for replayed HTTP wires; server-side history for Live/Interactions; `none` = full replayable history |
| `P3_IMAGES` | `always` | `always` or upstream `on_demand` |
| `P3_DEPTH` | `render` | `render` or `off`; renders only when depth exists |
| `P3_ACTION_TYPE` | `joint` | `ee` is rejected for unsafe quaternion semantics |
| `P3_PRIOR_LEARNINGS` | unset | UTF-8 notes file appended to the system prompt |
| `P3_WIRE` / `P3_BASE_URL` / `P3_API_KEY_ENV` | upstream resolution | Endpoint override |
| `P3_API_VERSION` | required for `azure-chat` | Azure API version |
| `P3_TEMPERATURE`, `P3_EFFORT`, `P3_MAX_OUTPUT_TOKENS`, `P3_SPEED` | upstream defaults | Provider controls |
| `P3_WIRE_CAPTURE` | `true` | Replay-grade request/response JSONL + deduplicated PNGs |
| `P3_TRACE_DIR` | unset | Transcript, full config, usage, hindsight, capture |

## Installation

Install the exact upstream policy into the Python environment that runs the
RoboDojo client:

```bash
bash policy/Agent_P3/install.sh /path/to/robodojo/.venv/bin/python
```

## Data Processing

Unsupported: this is an eval-only API policy and has no training dataset.

## Training

Unsupported: there is no checkpoint or trainable VLA in this condition.

## Evaluation

```bash
export P3_MODEL=anthropic/claude-sonnet-4-20250514
export ANTHROPIC_API_KEY=...
export P3_TRACE_DIR=/tmp/xpolicylab-p3/$RUN_ID

bash policy/Agent_P3/run_fixed_layout.sh <layout> <env_gpu> general_pickup
```

No VLA checkpoint. The Python process running RoboDojo must contain the two
pinned Inspect packages; `install.sh` installs them.
