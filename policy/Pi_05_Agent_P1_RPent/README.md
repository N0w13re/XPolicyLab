# Pi_05_Agent_P1_RPent

Evaluation-only Harness-VLA adapter for RoboDojo. Qwen is the low-frequency
planner, frozen Pi_05 handles contact-rich motion, and generic RGB-D/CuRobo
tools handle localization, free-space motion, verification, and recovery.

This is a new RPent-style experimental condition, not the earlier
`Pi_05_Agent_P1` gaze-only condition. In particular, the planner may give Pi_05
the original full episode instruction and may explicitly control a gripper between Pi_05
calls. Pi_05 weights and returned action chunks remain unchanged.

## Supported Configuration

- Benchmark: RoboDojo
- Initial task: `classify_objects_by_language`
- Robot: `arx_x5`
- Pi_05 action type: `joint`
- Integration: evaluation only; there is no data conversion or training entry
- Policy environment: `policy/Pi_05/openpi/.venv`, selected with `uv`

The checkpoint is resolved by the inherited `policy/Pi_05/model.py`. The
expected directory is:

```text
policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999/
```

## Architecture

The runtime Python contains no task-named primitive or pre-written task
procedure. The planner derives the episode plan from the live instruction,
camera observations, and generic tool results; the tool implementation does
not know about categories, baskets, or benchmark rewards.

The planner calls exactly one structured tool per turn:

- `view_env_state`: inspect one immutable RGB-D state
- `render`: capture a fresh RGB-D state without moving the robot
- `sample_world_xyz` / `query_world_map`: derive robust metric geometry from
  planner-selected head pixels and wrist-view refinement for the same candidate
  at the exact step/view
- `move_to`: execute a CuRobo collision-checked joint path to an EEF pose
- `pregrasp`: open one gripper and hold it one clearance above a measured object
  point, using the top-down pre-grasp orientation and the arm on the object's
  side, so Pi_05 sees the intended object rather than a distractor
- `rotate_wrist`: rotate one wrist at fixed EEF position
- `pi05_act`: run a short prefix of frozen Pi_05 with the full instruction
- `set_gripper`: explicitly firm or open one gripper
- `release`: open one gripper without transporting it
- `return_home`: return one or both arms to their episode-start poses
- `finish`: stop the planner without overriding official scoring

Metric depth and camera calibration are converted into a dense per-view
`world_xyz` map. Fixed support-plane projection and rectangular arm
reachability are not used as execution authority. Following upstream RoboTwin
RPent, no tool adjudicates an observable gate and no primitive is blocked on a
recorded verdict: tool results report gripper values and motion residuals, and
the planner judges grasp, transport, and placement from the fresh images it
already receives.

Every tool result is followed by fresh labeled camera images. A persistent
trace is written under `RPENT_TRACE_DIR`, containing `transcript.jsonl` and one
directory per tool call with state JSON and JPEG observations.

Planner prompts are explicitly versioned:

- `v0` is the verbatim upstream RPent RoboTwin system/user prompt from commit
  `f29a69c9ab42876cf876f749a0eb3c216a470a2f`. It is an archival baseline:
  RPent has no RoboDojo prompt, and this text refers to RoboTwin, LingBot-VLA,
  recipes, memory, and upstream-only tools.
- `v1` is the first XPolicyLab RoboDojo/Pi_05 adaptation. It preserves the
  upstream RPent section structure with RoboDojo, Pi_05, local resource, and
  `pi05_act` horizon substitutions, and treats head-view `ground` as identity
  plus bbox pixels.
- `v2` keeps v1 resource loading and aligns the
  registered tools with upstream RoboTwin RPent: no SAM3, no `segment`, no
  `ground`, and no `verify_state`. Identity comes from the head RGB; the
  planner picks `[row,col]` pixels and queries `sample_world_xyz` or
  `query_world_map` before using metric xyz; grasp with `pi05_act`; `move_to`
  is for transport after the planner has seen a hold, with planner-added
  EEF/TCP clearance. Gate discipline lives in the prompt and guide, not in a
  runtime state machine. It loads the generic guide, exact task/seed curated
  resources when present, the legacy task recipe as an experimental prior, and
  the memory index. Missing curated resources remain supported and are
  recorded in trace.
- `v3` is the current default and differs from v2 only in the grasp. Pi_05 never
  receives our measured coordinates and binds its own target, so on RoboDojo
  layouts with several plausible objects it regularly grasps a distractor. v3
  therefore requires the measured geometry to position the arm before the
  contact: sample the object xyz, call `pregrasp`, confirm on the fresh wrist
  image that the intended object is centred under the open gripper, and only
  then run `pi05_act` for the descent and closure. Transport after a verified
  hold is unchanged.

Select a version with `RPENT_PLANNER_PROMPT_VERSION=v0|v1|v2|v3`. Each new trace
records a `planner_config` event containing the selected version, exact system
and opening prompts, recipe text/path when applicable, and provenance; every
`planner_turn` repeats the version.
Add a new version rather than changing an existing version's text.

## Offline Trace Viewer

New rollouts record an exact half-open video frame range `[start, end)` for
each planner tool call. After the rollout finishes, launch the local viewer
with the trace directory and the matching RoboDojo result directory:

```bash
cd XPolicyLab
uv run --active --no-sync python \
  policy/Pi_05_Agent_P1_RPent/trace_viewer.py \
  --trace-dir /tmp/xpolicylab-rpent/<run-id> \
  --video-dir /path/to/RoboDojo-eval/eval_result/<run-id> \
  --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`. The page shows synchronized head,
left-wrist, and right-wrist videos, a global timeline segmented by tool call,
an ordered tool sidebar, frame stepping, and the selected tool's arguments,
result, environment-step range, and video-frame range. The viewer streams the
original MP4 files with HTTP byte-range support and does not export sampled
frames.

On a Merlin GPU devbox, bind the viewer to IPv6 and use one of the instance's
reserved ports before creating an instance link:

```bash
... trace_viewer.py --host :: --port <reserved-port>
merlin-cli gpu-devbox instance-links create \
  --json '{"trial_sid":"<ARNOLD_TRIAL_ID>","port":<reserved-port>,"is_public":false}'
```

Only rollouts created after `tool_frame_range` tracing was added can be aligned
exactly. The trace directory and video directory must come from the same run.

## Install

```bash
cd XPolicyLab/policy/Pi_05_Agent_P1_RPent
bash install.sh
```

The adapter has no additional model dependencies beyond Pi_05 and, for the
Azure GPT backend, the `openai` package. Qwen is called through the standard
library HTTP client; GPT uses `AzureOpenAI`.

## Planner LLM

Planner calls use one backend. Default
is Qwen. Set `RPENT_LLM_BACKEND=gpt` (or only a GPT key) to use ByteDance AIDP
Azure OpenAI for the same loop.

By default, when neither Qwen nor GPT credentials are present, `eval.sh` starts
the already-installed local checkpoint at
`../Qwen3-VL-4B-Instruct` through an OpenAI-compatible localhost endpoint. It
uses `policy/G05/G05/.venv` and GPU 2 by default:

```bash
export RPENT_QWEN_MODEL_PATH=/mnt/bn/robotics-data-mx/wenbo/Qwen3-VL-4B-Instruct
export RPENT_QWEN_PYTHON=policy/G05/G05/.venv/bin/python
export RPENT_QWEN_GPU=2
```

Remote Qwen:

```bash
export DASHSCOPE_API_KEY=...
# or
export QWEN_API_KEY=...
```

Remote GPT (planning + vision). Put the key in the environment only; do not
commit it:

```bash
export RPENT_LLM_BACKEND=gpt
export RPENT_GPT_API_KEY=...
export RPENT_GPT_ENDPOINT=https://aidp.bytedance.net/api/modelhub/online/v2/crawl
export RPENT_GPT_API_VERSION=2024-03-01-preview
export RPENT_GPT_MODEL=gpt-5.5-2026-04-24
# optional
export RPENT_GPT_LOGID=...
export RPENT_GPT_MAX_TOKENS=4096
export RPENT_GPT_MAX_RETRIES=12
export RPENT_GPT_RETRY_CAP_S=120
export RPENT_GPT_RETRY_BUDGET_S=1800
```

`AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are also accepted. GPT mode
does not start the local Qwen server. HTTP 429/5xx and transient transport
errors use capped exponential backoff, honor a numeric `Retry-After` header,
and stop once either `RPENT_GPT_MAX_RETRIES` or the total
`RPENT_GPT_RETRY_BUDGET_S` wait budget is exhausted. Set
`RPENT_GPT_RETRY_JITTER=0` only for deterministic diagnostics.

Optional Qwen / loop settings:

```bash
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export QWEN_MODEL=qwen3-vl-plus
export QWEN_TIMEOUT_S=120
export QWEN_TEMPERATURE=0.1
export RPENT_MAX_TURNS=120
export RPENT_APPROACH_CLEARANCE_M=0.20
export RPENT_PI05_EXECUTION_HORIZON=20
export RPENT_RECORD_EVERY_PI05_ACTION=1
export RPENT_STOP_AFTER_FIRST_PI05=1
```

`RPENT_APPROACH_CLEARANCE_M` is the offset the planner should add above
explicitly sampled geometry before `move_to`. `RPENT_PREGRASP_CLEARANCE_M` is
the lower hover `pregrasp` holds above a measured object, chosen so the object
still fills the wrist view when Pi_05 takes the contact.
`RPENT_PI05_EXECUTION_HORIZON` limits how many
actions are executed from each Pi_05-generated chunk before re-observing and
requesting a new chunk; the model still generates its native 50-action chunk.
`RPENT_RECORD_EVERY_PI05_ACTION=1` is a diagnostic option that records a camera
frame after every Pi_05 action; leave it unset for normal evaluation.
`RPENT_STOP_AFTER_FIRST_PI05=1` ends a diagnostic rollout immediately after the
first complete `pi05_act` call returns.

## Fixed Layout 16

Layout 16 is the first target because bare Pi_05 previously achieved partial
score `0.4` there. Run one deterministic-layout development episode:

```bash
cd XPolicyLab
ROBODOJO_RUN_ID=rpent-layout16-attempt1 \
  bash policy/Pi_05_Agent_P1_RPent/run_fixed_layout.sh 16 0 1 uv
```

Arguments are `layout policy_gpu env_gpu eval_env [task_name]`. The task name
defaults to `classify_objects_by_language`; for example, append
`general_pickup` to run that task. `uv` resolves to
`RoboDojo-eval/.venv`. The default trace is
written to `/tmp/xpolicylab-rpent/$ROBODOJO_RUN_ID`.

For repeated fixed-layout validation, use a unique run id each time and require
official RoboDojo `success: true` for every run. Planner `finish(status=...)`
is only a claim and is not counted as success.

## Standard Evaluation

```bash
cd XPolicyLab/policy/Pi_05_Agent_P1_RPent
bash eval.sh RoboDojo classify_objects_by_language sim arx_x5 joint 0 \
  0 1 uv RoboDojo
```

## Debug Wiring

Without a Qwen key, debug mode falls back to frozen Pi_05 passthrough so server
and action-shape wiring can still be checked:

```bash
cd XPolicyLab/policy/Pi_05_Agent_P1_RPent
EVAL_ENV_TYPE=debug \
  bash eval.sh RoboDojo classify_objects_by_language sim arx_x5 joint 0 \
  0 0 uv base
```

## Known Limitations

- Grasp and placement verification is the planner's own reading of fresh RGB
  views. It avoids privileged simulator poses but can still produce perception
  errors, and nothing in the runtime catches a wrong call.
- The planner is single-environment. Use `--num-envs 1` for fixed-layout
  development.
- A fixed-layout single-environment result is not directly comparable with the
  historical five-environment Pi_05 baseline because Pi_05 sampling depends on
  batch shape.
- Successful trace-to-recipe export and automatic memory promotion are not yet
  enabled; failed or merely anticipated procedures are never treated as
  supported recipes.
