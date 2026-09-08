# L5 / inspect-robots-agent source-level alignment

Authority: PyPI `inspect-robots-agent==0.26.0`,
`inspect-robots==0.58.0`, corresponding to
[`robocurve/inspect-robots`](https://github.com/robocurve/inspect-robots)
`main` on 2026-09-08. `Agent_P3` imports `LLMAgentPolicy`; it does not copy its
strategy implementation.

## Alignment boundary

The policy boundary and the benchmark boundary are intentionally separate:

- `inspect-robots-agent` owns every model-facing decision condition: prompt,
  provider request/retry, conversation, tools, validation, interpolation and
  stop requests.
- RoboDojo owns task code, saved layouts, reset, simulation, action counting,
  reward, termination, success/process score, videos and `_result.json`.
- `Agent_P3` only converts the values crossing
  `get_obs() -> policy -> take_action()`. It does not patch or import private
  RoboDojo task/reward implementations.

Therefore “aligned” means policy-protocol equivalence, not replacing
RoboDojo's benchmark with Inspect's scenes or scorer.

## Complete policy-protocol matrix

| Area | Upstream v0.26.0 behavior/default | Agent_P3 mapping and evidence | Status |
| --- | --- | --- | --- |
| System prompt, one-call rule, small-motion guidance, notes, call budget | `inspect_robots_agent.policy` | Executed by imported `LLMAgentPolicy`; prompt test | Exact upstream |
| Initial `Goal:` turn and repeated `Instruction:` | upstream `reset` / `_observation_content` | `Scene` + converted `Observation`; goal-turn test | Exact upstream |
| Observation | instruction, labeled state, approver/operator lines, then images | RoboDojo `instruction`, 14-D state, and reserved `extra` channels are mapped losslessly | Equivalent |
| Camera PNG encoding and step labels | upstream `_png` / observation formatter | RoboDojo `vision.*.color` RGB arrays cross the bridge; camera test | Equivalent |
| `images=always`; `on_demand` and `take_pic` | upstream policy/toolset | `P3_IMAGES` passed without reimplementation | Exact upstream |
| Metric depth rendering / `depth=off` | `render`; per-camera `<name>_depth`, 2-D metres | `vision.*.depth` is forwarded under that exact key and rendered by upstream; `off` resolves nothing; both depth tests | Equivalent, with the benchmark-side caveat below |
| State labels and partial named targets | upstream `ActionSemantics.dim_labels` | labels come from the live articulation | Equivalent embodiment |
| Tool schema | `move_joints(targets,note)`, `done(summary,hindsight)`, `give_up(reason,hindsight)`; `take_pic` only on demand | schemas are returned by imported `build_toolset`; schema/stop/image tests | Exact upstream |
| Action semantics | finite 1-D `Box`, `joint_pos`, absolute named dimensions, continuous gripper | live dual-X5 bounds and ordered labels create `EmbodimentInfo` | Equivalent embodiment |
| Bounds and numeric validation | unknown/non-finite/out-of-box/fixed dimensions return structured tool errors | no local parser; upstream receives live `Box` | Exact upstream |
| Chunk/interpolation | `max_speed_frac=.1`; per-step `min(.1/hz, 5%) * range`; 10 s cap; partial targets hold current state | upstream chunk is converted action-for-action and fully played unless RoboDojo ends | Equivalent |
| Safety approvers | Inspect CLI `_build_approver` applies clamp then delta limit after policy output (the lower-level Python `eval()` fallback alone is permissive) | bridge applies the same `ClampApprover` and `DeltaLimitApprover` before `take_action` | Equivalent to standard CLI evaluation |
| History | canonical full chat history; `image_horizon=2` except Live/Interactions server-side history; old image parts are stubbed | constructor default is left to upstream; explicit `P3_IMAGE_HORIZON` is forwarded | Exact upstream |
| Tool repair | errors become tool results; no-tool response gets a nudge; three consecutive failures raise | no local loop; malformed/no-tool/cap tests execute upstream | Exact upstream |
| Provider retry | Chat: at most 3 attempts for transport, 429 and 5xx, sleeping 1 s then 2 s; non-429 4xx fails immediately | Azure changes only URL/auth/model and retains `ChatClient.complete`; retry test | Exact upstream |
| Call budget | `max_llm_calls=100`; retry attempts do not spend extra policy calls; exhaustion synthesizes `give_up` | `P3_MAX_LLM_CALLS`; call-budget and retry tests | Exact upstream |
| Stop/hindsight | `done`/`give_up` return a one-action hold with request metadata; non-`none` hindsight is harvested | hold goes through `take_action`; early stop is explicitly finalized as RoboDojo failure unless its reward already succeeded | Equivalent scorer boundary |
| `prior_learnings`, size/hash validation | upstream policy | path forwarded; prompt and config contain hash | Exact upstream |
| Operator/pre-check | well-formed `extra.operator_messages` are prompt lines; callable pre-check sees read-only interpolated waypoints | operator channel forwarded; bridge-reserved step/approval keys cannot be overwritten; callable constructor hook forwarded; production default `None` | Exact upstream / absent RoboDojo operator UI |
| Provider controls | wire-specific validation for temperature, named/fractional effort, Messages max tokens and fast mode | `P3_TEMPERATURE`, `P3_EFFORT`, `P3_MAX_OUTPUT_TOKENS`, `P3_SPEED` passed directly | Exact upstream |
| Wires | Chat, Messages, Responses, Gemini Live, Interactions | `P3_WIRE` forwards all native wires | Exact upstream |
| Azure Chat | not an upstream endpoint resolver | transport changes only deployment URL, API version, API-key header and deployment model; upstream owns body/retry/parse/capture | Transport-only exception |
| Trial record | `terminated`/`truncated`/`termination_reason`; a policy exception is `status="error"` with the exception string and neither flag set | the bridge fills the same `TrialRecord` fields from the RoboDojo outcome, including the error status; give-up and policy-failure tests | Exact upstream |
| Capture/config | one row per wire attempt, deduplicated PNGs, sanitized transcript, policy config, hindsight and LLM calls | `on_trial_start/end`; `p3_config.json` adds package versions, Azure version, scene and live embodiment | Equivalent plus benchmark metadata |
| Token metrics | Chat v0.26.0 does not populate `AssistantMessage.usage`; raw provider usage remains in capture. Other clients normalize supported usage | report never invents missing totals; `llm_calls` and raw per-attempt responses remain auditable | Exact known upstream limitation |

This means an upstream behavior change is adopted by changing the two pinned
versions, then rerunning the differential bridge tests. There is no local
prompt or interpolation implementation to drift.

### The depth condition depends on the RoboDojo checkout

`P3_DEPTH=render` only decides what the policy does with depth it receives. Whether
depth exists at all is a RoboDojo camera setting: the `distance_to_image_plane_capture`
annotator ships commented out in `env_cfg/camera/camera_config.yml`, annotators are
built before the environment is created, and a policy `deploy.yml` cannot inject them.
The runs archived here use a `ROBODOJO_ENABLE_METRIC_DEPTH=1` hook in the evaluation
checkout that adds exactly that annotator for the three standard cameras; uncommenting
the config entries is equivalent. This changes the observation the benchmark renders,
not the policy strategy, so both conditions remain valid L5 runs.

Because that switch lives outside this repo, the adapter never assumes it worked:
`eval_one_episode` warns when `render` is requested and no depth arrives, and every
transcript records `depth_cameras`, the cameras whose metric depth actually reached the
model. An RGB-only run therefore cannot be filed as a depth run by accident.

## Embodiment and harness mapping

| Inspect concept | RoboDojo equivalent | Comparability decision |
| --- | --- | --- |
| `EmbodimentInfo.action_space` | live dual-X5 Isaac articulation + normalized grippers | Exact run-time bounds, not guessed `[-pi, pi]` |
| `control_hz` | `obs_manager.collect_freq` | 25 Hz in `env_cfg/arx_x5.yml`; old 10 Hz value was wrong |
| `Observation.state["joint_pos"]` | ordered left arm/gripper/right arm/gripper dict | Lossless 14-D conversion |
| `DefaultController(replan_interval=None)` | play every action in returned chunk | Same full open-loop chunk |
| embodiment `step()` | `TASK_ENV.take_action()` | One RoboDojo observation/control step; its internal servo interpolation is embodiment dynamics |
| `Scene.instruction/init_seed` | task instruction/layout id | Supplied to upstream reset |
| task scenes / epochs | RoboDojo task registry + saved `Assets/Eval_Layout/.../<eval_seed>/<task>_<layout>.json` | Dataset remains RoboDojo; Agent_P3 adds or removes no task/layout |
| seed protocol | RoboDojo `eval_seed` selects a layout directory; `SeedManager` enumerates numeric layout ids, handles resume/abandon and passes each id to reset | actual `env_seeds[env_idx]`, not a guessed CLI value, becomes Inspect `Scene.init_seed` and audit `layout_id` |
| scorer | `reward_manager.get_reward(final_check=...)`; optional process `get_score()` | environment `success` remains authoritative; model `done` cannot force success |
| horizon | RoboDojo `step_lim` and `take_action_cnt` | every interpolated waypoint uses public `take_action`; RoboDojo may terminate a chunk |
| operator messages / approval records | `Observation.extra` | Forwarded when supplied; default RoboDojo has no operator channel |
| generic clamp/delta approvers | upstream motion already emits in-box, step-limited absolute targets | Box/step invariants are checked before conversion; RoboDojo applies final embodiment control |
| reporting | RoboDojo `_result.json`: `success_rate`, `eval_time`, percentage `score`, per-episode `layout_id/success/score`; videos tagged success/fail; unstable samples excluded | archived beside `p3_config.json`, transcript and wire JSONL; no Inspect scorer replaces these values |
| multi-env batch | Inspect eval can run many scenes; RoboDojo `eval_batch=false` already forces `num_envs=1` | `eval_one_episode_batch` refuses `num_envs>1` rather than silently driving only env 0 |

### RoboDojo non-regression invariants

1. The only mutating calls are the existing public `take_action()` and, for an
   explicit `done`/`give_up`/policy error, setting the active `success` entry
   false before calling `is_episode_end()` so RoboDojo finalizes it normally.
2. No model response can set `success=true`, a process score, layout id, action
   counter or end flag.
3. `is_episode_end()` is checked before every policy turn and after every
   interpolated waypoint; RoboDojo can truncate a chunk at its own horizon or
   reward success.
4. Audit output is written only after early-stop failure finalization, so
   `official_success` agrees with RoboDojo's `_result.json`.

## Deliberate refusal, not a missing mode

`action_type=ee` is rejected. RoboDojo's EE schema is
`xyz + quaternion_wxyz + gripper` and invokes IK. Upstream
`inspect-robots-agent` refuses quaternion absolute-pose tool interpolation;
advertising local `move_to` with component-wise quaternion interpolation would
be unsafe and behaviorally different. Joint control is therefore the aligned
L5 condition.

## Verification gates

- `tests/test_agent_p3.py` executes the installed upstream policy through an
  HTTP mock and checks the bridge, not a local clone.
- The trace records both exact package versions, every upstream policy config
  field, live labels/bounds/rate, transcript, usage, hindsight, and capture
  pointer.
- Two integration runs are committed under
  `experiments/l5-inspect-agent/general_pickup/layout-0/`, each with wire
  capture, sanitized transcript, full config and a copy of RoboDojo's
  authoritative `_result.json`:
  - `l5-inspect-v026-final-depth-layout0/` — frontier model over the Azure Chat
    transport, three RGB and three metric-depth frames.
  - `l5-inspect-v026-depthaudit-layout0/` — current revision against a local
    model, no external credentials required. It records `depth_cameras` and an
    upstream-shaped `status="error"` trial after the repair loop gave up.
