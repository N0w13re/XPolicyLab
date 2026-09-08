# L5 / inspect-robots-agent source-level alignment

Authority: PyPI `inspect-robots-agent==0.26.0`,
`inspect-robots==0.58.0`, corresponding to
[`robocurve/inspect-robots`](https://github.com/robocurve/inspect-robots)
`main` on 2026-09-08. `Agent_P3` imports `LLMAgentPolicy`; it does not copy its
strategy implementation.

## Policy protocol

| Behavior that changes model decisions | Upstream authority | Agent_P3 evidence |
| --- | --- | --- |
| System prompt, one-call rule, small-motion guidance, notes, call budget | `inspect_robots_agent.policy` | Executed by imported `LLMAgentPolicy`; prompt test |
| Initial `Goal:` turn and repeated `Instruction:` | upstream `reset` / `_observation_content` | `Scene` + converted `Observation`; goal-turn test |
| Camera PNG encoding and step labels | upstream `_png` / observation formatter | Raw RGB arrays cross the bridge; camera test |
| `images=always`; `on_demand` and `take_pic` | upstream policy/toolset | `P3_IMAGES` passed without reimplementation |
| Metric depth rendering / `depth=off` | upstream `_depth` | RoboDojo depth extra may be supplied; absent depth is valid |
| State labels and partial named targets | upstream `ActionSemantics.dim_labels` | labels come from the live articulation |
| `move_joints`, `done`, `give_up`, required `note` / hindsight | upstream `_tools` | tool-schema and stop tests |
| Bounds, pinned dimensions, numeric validation | upstream `_tools`, input `Box` | `Box.low/high` come from Isaac `soft_joint_pos_limits`; grippers are `[0,1]` |
| 0.1 range/s, 5%-range step backstop, 10 s playout cap | upstream `_tools` | live 25 Hz enters `EmbodimentInfo`; interpolation test |
| Full history, `image_horizon=2`, image stubbing | upstream policy | env config forwarded; horizon test |
| Structured repair, no-tool nudge, multiple-call closure, 3 strikes | upstream policy | malformed/no-tool tests |
| Trial-wide `max_llm_calls=100`, forced `give_up` | upstream policy | budget test |
| `prior_learnings`, size/hash validation | upstream policy | path forwarded; prompt and config contain hash |
| temperature, effort, Messages max tokens/speed | upstream clients | `P3_*` values forwarded directly |
| chat, Messages, Responses, Gemini Live, Interactions | upstream clients | `P3_WIRE` forwarded directly |
| retries, usage normalization, transcript, replay wire capture | upstream clients / `_capture` | `on_trial_start/end`; trace metadata and versioned config |
| Azure Chat | not upstream | transport-only URL/auth adapter; upstream owns body/retry/parse/capture |

This means an upstream behavior change is adopted by changing the two pinned
versions, then rerunning the differential bridge tests. There is no local
prompt or interpolation implementation to drift.

## Embodiment and harness mapping

| Inspect concept | RoboDojo equivalent | Comparability decision |
| --- | --- | --- |
| `EmbodimentInfo.action_space` | live dual-X5 Isaac articulation + normalized grippers | Exact run-time bounds, not guessed `[-pi, pi]` |
| `control_hz` | `obs_manager.collect_freq` | 25 Hz in `env_cfg/arx_x5.yml`; old 10 Hz value was wrong |
| `Observation.state["joint_pos"]` | ordered left arm/gripper/right arm/gripper dict | Lossless 14-D conversion |
| `DefaultController(replan_interval=None)` | play every action in returned chunk | Same full open-loop chunk |
| embodiment `step()` | `TASK_ENV.take_action()` | One RoboDojo observation/control step; its internal servo interpolation is embodiment dynamics |
| `Scene.instruction/init_seed` | task instruction/layout id | Supplied to upstream reset |
| task scenes / epochs | RoboDojo official task/layout/episode set | Dataset remains RoboDojo by requirement |
| scorer | RoboDojo official `success` | `done` requests termination; environment success remains authoritative |
| horizon | RoboDojo `step_lim` | Harness cutoff remains authoritative |
| operator messages / approval records | no RoboDojo channel | Empty because the harness produces none, not silently discarded |
| generic clamp/delta approvers | upstream motion already emits in-box, step-limited absolute targets | Box/step invariants are checked before conversion; RoboDojo applies final embodiment control |
| EvalLog/Rerun | official RoboDojo result + mp4 + `p3_config.json`, transcript, wire JSONL | Different report container, same policy audit evidence |

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
- Integration completion additionally requires one official RoboDojo run whose
  trace and official result are archived together.
